"""The project's own sequence network (PLAN T7.3): GRU over 60 sessions x 12 channels plus a tabular branch.

`X` for this model carries two trailing key columns (`add_keys`): the sid and the date as days since the
Unix epoch (exact in float32), so the model can look up each row's sequence in a `SeqProvider`. The
walk-forward runner must append them (see docs/DECISIONS.md, T7.3). Training batches are one date each,
the loss is `1 - Pearson(pred, z_target)`, one cosine schedule over the epochs, early stop on the
inner-validation mean daily rank IC; every epoch is checkpointed and a restart resumes from the newest.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from stocksense.models.base import FitInfo, Model, ModelSpec, mean_daily_rank_ic, register_model

N_KEYS = 2
EPOCH_DAYS = np.datetime64("1970-01-01", "D")


def date_to_days(dates: np.ndarray) -> np.ndarray:
    """Days since 1970-01-01 as int64 for a datetime64 array."""
    return (np.asarray(dates).astype("datetime64[D]") - EPOCH_DAYS).astype(np.int64)


def add_keys(X: np.ndarray, sids: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """X with the two key columns (sid, days) appended, Float32."""
    k = np.column_stack([np.asarray(sids, dtype=np.float64), date_to_days(dates).astype(np.float64)])
    return np.hstack([np.asarray(X, dtype=np.float32), k.astype(np.float32)])


class SeqProvider(Protocol):
    def get(self, days: np.ndarray, sids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sequences float16/32 [n, 60, 12], valid bool [n]) for (days-since-epoch, sid) pairs."""
        ...


class MemmapSeqProvider:
    """Reads features/seq/v<N>/X_<year>.npy (memory-mapped) through the index parquets."""

    def __init__(self, root: Path) -> None:
        import polars as pl

        self._maps: list[np.ndarray] = []
        keys, which, rows = [], [], []
        for k, p in enumerate(sorted(Path(root).glob("index_*.parquet"))):
            year = p.stem.split("_")[1]
            idx = pl.read_parquet(p)
            self._maps.append(np.load(Path(root) / f"X_{year}.npy", mmap_mode="r"))
            keys.append(
                date_to_days(idx["date"].to_numpy()) * (1 << 22) + idx["sid"].to_numpy().astype(np.int64)
            )
            which.append(np.full(idx.height, k))
            rows.append(idx["row"].to_numpy())
        self._keys = np.concatenate(keys) if keys else np.empty(0, dtype=np.int64)
        o = np.argsort(self._keys)
        self._keys = self._keys[o]
        self._which = np.concatenate(which)[o] if which else np.empty(0, dtype=int)
        self._rows = np.concatenate(rows)[o] if rows else np.empty(0, dtype=int)

    def get(self, days: np.ndarray, sids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(days, dtype=np.int64) * (1 << 22) + np.asarray(sids, dtype=np.int64)
        pos = np.searchsorted(self._keys, q)
        pos = np.minimum(pos, max(len(self._keys) - 1, 0))
        valid = (self._keys[pos] == q) if len(self._keys) else np.zeros(len(q), dtype=bool)
        out = np.zeros((len(q), 60, 12), dtype=np.float16)
        for k, m in enumerate(self._maps):
            sel = np.flatnonzero(valid & (self._which[pos] == k))
            if len(sel):
                r = self._rows[pos[sel]]
                o = np.argsort(r)
                out[sel[o]] = m[r[o]]
        return out, valid


class SeqNet(nn.Module):
    """GRU(12 -> H, 2 layers) on the sequence, MLP on [x, isnan(x)], joint head to one score."""

    def __init__(self, n_tab: int, hidden: int = 64, dropout: float = 0.2, n_ch: int = 12) -> None:
        super().__init__()
        self.n_tab = n_tab
        self.gru = nn.GRU(n_ch, hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.tab = nn.Sequential(
            nn.Linear(2 * n_tab, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 64, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        seq = torch.nan_to_num(seq.float(), nan=0.0, posinf=0.0, neginf=0.0)
        tab = tab.float()
        nan = torch.isnan(tab).float()
        tab = torch.cat([torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0), nan], dim=1)
        _, h = self.gru(seq)
        z = torch.cat([h[-1], self.tab(tab)], dim=1)
        return self.head(z).squeeze(-1)


def ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - Pearson correlation (0 for perfectly correlated, 2 for perfectly anti-correlated)."""
    p = pred.float() - pred.float().mean()
    t = target.float() - target.float().mean()
    den = torch.sqrt((p * p).sum() * (t * t).sum()) + eps
    return 1.0 - (p * t).sum() / den


class _Halt(Exception):
    pass


class SeqNetModel(Model):
    target_prefix = "z_"

    def __init__(
        self,
        provider: SeqProvider | None = None,
        hidden: int = 64,
        dropout: float = 0.2,
        max_epochs: int = 30,
        patience: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        min_batch: int = 50,
        device: str | None = None,
        work_dir: Path | str | None = None,
        pred_chunk: int = 4096,
    ) -> None:
        self.provider = provider
        self.hidden, self.dropout = int(hidden), float(dropout)
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.lr, self.weight_decay, self.min_batch = lr, weight_decay, int(min_batch)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.work_dir = Path(work_dir) if work_dir else None
        self.pred_chunk = pred_chunk
        self.on_epoch_end: Callable[[int], None] | None = None  # test hook (called after each checkpoint)
        self.net: SeqNet | None = None
        self.n_tab = 0
        self._chunk_div = 1  # OOM: split every date batch into this many pieces

    # ---- data ----------------------------------------------------------------------------------
    def _need_provider(self) -> SeqProvider:
        if self.provider is None:
            raise RuntimeError("SeqNetModel needs a SeqProvider (attach one with .provider = ...)")
        return self.provider

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return X[:, :-N_KEYS], X[:, -2], X[:, -1]

    def _batches(self, X: np.ndarray, dates: np.ndarray, y: np.ndarray | None) -> list[np.ndarray]:
        """Row index arrays, one per date, rows whose sequence exists and (if y) whose label is finite."""
        tab, sid, days = self._split(X)
        _, valid = self._need_provider().get(days.astype(np.int64), sid.astype(np.int64))
        ok = valid.copy()
        if y is not None:
            ok &= np.isfinite(y)
        order = np.argsort(dates, kind="stable")
        sd = dates[order]
        cuts = np.flatnonzero(sd[1:] != sd[:-1]) + 1
        groups = [g[ok[g]] for g in np.split(order, cuts)]
        return [g for g in groups if len(g) >= self.min_batch]

    def _tensors(self, X: np.ndarray, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        tab, sid, days = self._split(X[rows])
        seq, _ = self._need_provider().get(days.astype(np.int64), sid.astype(np.int64))
        return (
            torch.as_tensor(np.asarray(seq, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(tab, dtype=np.float32), device=self.device),
        )

    # ---- training ------------------------------------------------------------------------------
    def _train_batch(
        self, opt: Any, scaler: Any, seq: torch.Tensor, tab: torch.Tensor, y: torch.Tensor, wt: float
    ) -> float:
        """One optimizer step over a date batch; on CUDA OOM the batch is split in halves repeatedly."""
        assert self.net is not None
        while True:
            parts = self._chunk_div
            try:
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for idx in torch.arange(len(y), device=y.device).chunk(parts):
                    if len(idx) < 2:
                        continue
                    with torch.autocast(
                        self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
                    ):
                        pred = self.net(seq[idx], tab[idx])
                    loss = ic_loss(pred, y[idx]) * wt / parts
                    scaler.scale(loss).backward()
                    total += float(loss.detach())
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                return total
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if parts >= len(y) // 2:
                    raise
                self._chunk_div = parts * 2

    def _predict_rows(self, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        out = np.zeros(len(rows), dtype=np.float64)
        with torch.no_grad():
            for a in range(0, len(rows), self.pred_chunk):
                sl = rows[a : a + self.pred_chunk]
                seq, tab = self._tensors(X, sl)
                with torch.autocast(
                    self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
                ):
                    out[a : a + len(sl)] = self.net(seq, tab).float().cpu().numpy()
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _ckpt_dir(self, phase: str) -> Path | None:
        return None if self.work_dir is None else self.work_dir / phase

    @staticmethod
    def _latest(d: Path | None) -> Path | None:
        if d is None or not d.is_dir():
            return None
        files = sorted(d.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        return files[-1] if files else None

    def _run(
        self,
        phase: str,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        dates: np.ndarray,
        epochs: int,
        val: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        seed: int,
    ) -> tuple[int, float]:
        """Train for up to `epochs` epochs (resuming from a checkpoint); returns (best_epoch, best_ic)."""
        self.n_tab = X.shape[1] - N_KEYS
        torch.manual_seed(seed)
        self.net = SeqNet(self.n_tab, self.hidden, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        st = {"epoch": 0, "best_ic": -math.inf, "best_epoch": 0, "bad": 0, "best_state": None, "done": False}
        ck_dir = self._ckpt_dir(phase)
        last = self._latest(ck_dir)
        if last is not None:
            ck = torch.load(last, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            st = ck["state"]
        batches = self._batches(X, dates, y)
        if not batches:
            raise ValueError("no date batch with enough usable rows to train on")
        wd = np.array([float(np.mean(w[g])) for g in batches])
        wd = wd / wd.mean()
        vb = self._batches(val[0], val[2], val[1]) if val is not None else []
        while st["epoch"] < epochs and not st["done"]:
            ep = st["epoch"] + 1
            torch.manual_seed(seed * 100003 + ep)
            order = np.random.default_rng(seed * 100003 + ep).permutation(len(batches))
            self.net.train()
            for bi in order:
                rows = batches[bi]
                seq, tab = self._tensors(X, rows)
                yt = torch.as_tensor(y[rows], dtype=torch.float32, device=self.device)
                self._train_batch(opt, scaler, seq, tab, yt, float(wd[bi]))
            sched.step()
            st["epoch"] = ep
            if val is not None:
                vrows = np.concatenate(vb) if vb else np.empty(0, dtype=int)
                pv = self._predict_rows(val[0], vrows)
                ic = mean_daily_rank_ic(pv, val[1][vrows], val[2][vrows], min_pairs=self.min_batch)
                if np.isfinite(ic) and ic > st["best_ic"]:
                    st.update(
                        best_ic=float(ic),
                        best_epoch=ep,
                        bad=0,
                        best_state=copy.deepcopy(self.net.state_dict()),
                    )
                else:
                    st["bad"] += 1
                    if st["bad"] >= self.patience:
                        st["done"] = True
            else:
                st["best_epoch"] = ep
            if ck_dir is not None:
                ck_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": self.net.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep, "best_ic": st["best_ic"], "best_state": st["best_state"], "state": st,
                    },
                    ck_dir / f"epoch_{ep}.pt",
                )  # fmt: skip
            if self.on_epoch_end is not None:
                self.on_epoch_end(ep)
        if st["best_state"] is not None:
            self.net.load_state_dict(st["best_state"])
        return int(st["best_epoch"]), float(st["best_ic"])

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed: int = 0) -> FitInfo:  # type: ignore[no-untyped-def]
        X, y, w, dates = (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float64),
            np.asarray(w),
            np.asarray(dates),
        )
        val = None
        if X_val is not None and y_val is not None and dates_val is not None:
            val = (
                np.asarray(X_val, dtype=np.float32),
                np.asarray(y_val, dtype=np.float64),
                np.asarray(dates_val),
            )
        best_epoch, best_ic = self._run("fit", X, y, w, dates, self.max_epochs, val, seed)
        return FitInfo(
            best_iteration=max(best_epoch, 1),
            val_ic=None if not np.isfinite(best_ic) else best_ic,
            params={"hidden": self.hidden, "dropout": self.dropout, "best_epoch": best_epoch},
        )

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        epochs = math.ceil(1.1 * int(fit_info.best_iteration or 1))
        self._run(
            "refit", np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64), np.asarray(w),
            np.asarray(dates), epochs, None, seed,
        )  # fmt: skip

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("SeqNetModel is not fitted")
        X = np.asarray(X, dtype=np.float32)
        return self._predict_rows(X, np.arange(len(X)))

    def explain(self, X, feature_names):  # type: ignore[no-untyped-def]
        return None

    def save(self, dir: Path) -> None:
        if self.net is None:
            raise RuntimeError("SeqNetModel is not fitted")
        dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), dir / "weights.pt")
        (dir / "seqnet_meta.json").write_text(
            json.dumps({"n_tab": self.n_tab, "hidden": self.hidden, "dropout": self.dropout}),
            encoding="utf-8",
        )

    def load(self, dir: Path) -> None:
        meta = json.loads((dir / "seqnet_meta.json").read_text(encoding="utf-8"))
        self.n_tab, self.hidden, self.dropout = meta["n_tab"], meta["hidden"], meta["dropout"]
        self.net = SeqNet(self.n_tab, self.hidden, self.dropout).to(self.device)
        self.net.load_state_dict(torch.load(dir / "weights.pt", map_location=self.device))


def _factory(spec: ModelSpec) -> Model:
    p = dict(spec.params)
    root = p.pop("seq_dir", None)
    work = p.pop("work_dir", None)
    keys = ("hidden", "dropout", "max_epochs", "patience", "lr", "weight_decay", "min_batch", "device")
    kw = {k: p[k] for k in keys if k in p}
    return SeqNetModel(MemmapSeqProvider(Path(root)) if root else None, work_dir=work, **kw)


register_model("seqnet", _factory)

# Grid of T7.3 (inner validation at anchors 2019 and 2022 only), selected by the T7.3 run script.
GRID = [{"hidden": h, "dropout": p} for h in (32, 64) for p in (0.2, 0.4)]
