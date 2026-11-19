"""Shared base for the cross-sectional torch families of Program G2 (card D0.4).

`TorchCSModel` trains one optimizer step per date (one cross-section per batch), early-stops on the
inner-validation mean daily rank IC, refits for `ceil(1.1 * best_epoch)` epochs, checkpoints every epoch
to `work_dir/{fit|refit}/epoch_N.pt` (a restart resumes from the newest), halves the date batch on CUDA
out-of-memory, and stops at the best epoch so far when `max_minutes` of wall time are used
(`FitInfo.params["stopped"] == "wall"`). A family supplies `build_net` and `batch_inputs` (and
`forward_net` if the net is not simply `net(**inputs)`). Float32 unless `amp=True`.
"""

from __future__ import annotations

import copy
import json
import math
import time
from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from stocksense.models.base import FitInfo, Model, mean_daily_rank_ic
from stocksense.models.seqnet import ic_loss

__all__ = ["DateBatches", "TorchCSModel", "ic_loss", "listnet_loss", "mse_loss"]


class DateBatches:
    """Row-index arrays, one per date with at least `min_names` rows; `epoch(seed)` shuffles their order."""

    def __init__(self, dates: np.ndarray, min_names: int = 50, rows: np.ndarray | None = None) -> None:
        dates = np.asarray(dates)
        idx = np.arange(len(dates)) if rows is None else np.asarray(rows)
        order = idx[np.argsort(dates[idx], kind="stable")]
        sd = dates[order]
        cuts = np.flatnonzero(sd[1:] != sd[:-1]) + 1
        self.batches: list[np.ndarray] = [g for g in np.split(order, cuts) if len(g) >= min_names]

    def __len__(self) -> int:
        return len(self.batches)

    def order(self, seed: int) -> np.ndarray:
        """The seeded shuffled order of batch numbers for one epoch."""
        return np.random.default_rng(seed).permutation(len(self.batches))

    def epoch(self, seed: int) -> list[np.ndarray]:
        return [self.batches[i] for i in self.order(seed)]


def mse_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred.float() - y.float()) ** 2).mean()


def listnet_loss(pred: torch.Tensor, y: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Cross-entropy between softmax(y / tau) and softmax(pred) over the names of one date."""
    target = torch.softmax(y.float() / tau, dim=0)
    return -(target * torch.log_softmax(pred.float(), dim=0)).sum()


LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "ic": ic_loss,
    "mse": mse_loss,
    "listnet": listnet_loss,
}


class TorchCSModel(Model):
    target_prefix = "z_"

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        patience: int = 5,
        loss: str = "ic",
        seed_offset: int = 0,
        device: str | None = None,
        work_dir: Path | str | None = None,
        amp: bool = False,
        min_names: int = 50,
        pred_chunk: int = 4096,
        max_minutes: float | None = None,
    ) -> None:
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {sorted(LOSSES)}")
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.loss_name, self.seed_offset = loss, int(seed_offset)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.work_dir = Path(work_dir) if work_dir else None
        self.amp = bool(amp)
        self.min_names, self.pred_chunk = int(min_names), int(pred_chunk)
        self.max_minutes = max_minutes
        self.on_epoch_end: Callable[[int], None] | None = None  # test hook, called after each checkpoint
        self.net: nn.Module | None = None
        self.feature_names: list[str] = []
        self.kinds: list[str] = []
        self.n_feat = 0
        self.n_mkt = 0
        self._X: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._chunk_div = 1
        torch.backends.cudnn.benchmark = False

    # ---- what a family supplies ----------------------------------------------------------------
    @abstractmethod
    def build_net(self, n_feat: int, n_mkt: int) -> nn.Module:
        """The network for `n_feat` feature columns, `n_mkt` of them market-state ('mkt') columns."""

    @abstractmethod
    def batch_inputs(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        """Tensors (on `self.device`) for rows `idx` of the bound X (see `tab_tensor`)."""

    def forward_net(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.net is not None
        return self.net(**inputs)  # type: ignore[no-any-return]

    # ---- helpers -------------------------------------------------------------------------------
    def configure(self, feature_names: Sequence[str], kinds: Sequence[str]) -> None:
        self.feature_names, self.kinds = list(feature_names), list(kinds)
        self.n_mkt = sum(k == "mkt" for k in self.kinds)

    def tab_tensor(self, idx: np.ndarray) -> torch.Tensor:
        """Feature columns (not the key columns) of rows `idx`, NaN -> 0, float32."""
        x = np.nan_to_num(self._X[idx][:, : self.n_feat], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def key_columns(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sid, days-since-epoch) of rows `idx` (needs_keys families only)."""
        k = self._X[idx][:, -2:]
        return k[:, 0].astype(np.int64), k[:, 1].astype(np.int64)

    def _bind(self, X: np.ndarray) -> None:
        self._X = np.asarray(X, dtype=np.float32)
        self.n_feat = self._X.shape[1] - (2 if self.needs_keys else 0)

    def _groups(self, n: int) -> list[np.ndarray]:
        """Row groups for prediction: one per date when the keys are present, else fixed chunks."""
        if self.needs_keys:
            days = self._X[:, -1]
            order = np.argsort(days, kind="stable")
            cuts = np.flatnonzero(days[order][1:] != days[order][:-1]) + 1
            return [g for g in np.split(order, cuts) if len(g)]
        rows = np.arange(n)
        return [rows[a : a + self.pred_chunk] for a in range(0, n, self.pred_chunk)]

    def _predict_bound(self, rows: np.ndarray | None = None) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        n = len(self._X)
        out = np.zeros(n, dtype=np.float64)
        sel = np.ones(n, dtype=bool)
        if rows is not None:
            sel[:] = False
            sel[rows] = True
        with torch.no_grad():
            for g in self._groups(n):
                g = g[sel[g]]
                if not len(g):
                    continue
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self._amp_on()):
                    out[g] = self.forward_net(self.batch_inputs(g)).float().cpu().numpy()
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _amp_on(self) -> bool:
        return self.amp and self.device.type == "cuda"

    # ---- training ------------------------------------------------------------------------------
    def _train_batch(self, opt: Any, scaler: Any, idx: np.ndarray, y: np.ndarray, wt: float) -> float:
        """One optimizer step on one date; on CUDA OOM the date batch is split in halves repeatedly and the
        loss is averaged over the pieces."""
        assert self.net is not None
        loss_fn = LOSSES[self.loss_name]
        while True:
            parts = self._chunk_div
            try:
                opt.zero_grad(set_to_none=True)
                total = 0.0
                pieces = [p for p in np.array_split(np.arange(len(idx)), parts) if len(p) >= 2]
                for p in pieces:
                    rows = idx[p]
                    with torch.autocast(self.device.type, dtype=torch.float16, enabled=self._amp_on()):
                        pred = self.forward_net(self.batch_inputs(rows))
                    yt = torch.as_tensor(y[rows], dtype=torch.float32, device=self.device)
                    loss = loss_fn(pred, yt) * wt / len(pieces)
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
                if parts >= len(idx) // 2:
                    raise
                self._chunk_div = parts * 2

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
    ) -> tuple[int, float, str | None]:
        """Train up to `epochs` epochs (resuming from a checkpoint): (best_epoch, best_ic, stopped)."""
        t0 = time.monotonic()
        self._bind(X)
        seed = seed + self.seed_offset
        torch.manual_seed(seed)
        self.net = self.build_net(self.n_feat, self.n_mkt).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=self._amp_on())
        st: dict[str, Any] = {
            "epoch": 0, "best_ic": -math.inf, "best_epoch": 0, "bad": 0, "best_state": None, "done": False,
            "stopped": None,
        }  # fmt: skip
        ck_dir = self._ckpt_dir(phase)
        last = self._latest(ck_dir)
        if last is not None:
            ck = torch.load(last, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            st = ck["state"]
        train = DateBatches(dates, self.min_names, rows=np.flatnonzero(np.isfinite(y)))
        if not len(train):
            raise ValueError("no date batch with enough usable rows to train on")
        wd = np.array([float(np.mean(w[g])) for g in train.batches])
        wd = wd / wd.mean()
        vrows = np.empty(0, dtype=int)
        if val is not None:
            vb = DateBatches(val[2], self.min_names, rows=np.flatnonzero(np.isfinite(val[1])))
            vrows = np.concatenate(vb.batches) if len(vb) else vrows
        while st["epoch"] < epochs and not st["done"]:
            ep = st["epoch"] + 1
            torch.manual_seed(seed * 100003 + ep)
            self.net.train()
            for bi in train.order(seed * 100003 + ep):
                self._train_batch(opt, scaler, train.batches[bi], y, float(wd[bi]))
            sched.step()
            st["epoch"] = ep
            if val is not None:
                self._bind(val[0])
                pv = self._predict_bound(vrows)[vrows]
                self._bind(X)
                ic = mean_daily_rank_ic(pv, val[1][vrows], val[2][vrows], min_pairs=self.min_names)
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
            if self.max_minutes is not None and (time.monotonic() - t0) / 60.0 >= self.max_minutes:
                st["done"], st["stopped"] = True, "wall"
            if ck_dir is not None:
                ck_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": self.net.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep, "state": st,
                    },
                    ck_dir / f"epoch_{ep}.pt",
                )  # fmt: skip
            if self.on_epoch_end is not None:
                self.on_epoch_end(ep)
        if st["best_state"] is not None:
            self.net.load_state_dict(st["best_state"])
        return max(int(st["best_epoch"]), 1), float(st["best_ic"]), st["stopped"]

    def fit(  # type: ignore[no-untyped-def]
        self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed: int = 0
    ) -> FitInfo:
        X = np.asarray(X, dtype=np.float32)
        val = None
        if X_val is not None and y_val is not None and dates_val is not None:
            val = (
                np.asarray(X_val, dtype=np.float32),
                np.asarray(y_val, dtype=np.float64),
                np.asarray(dates_val),
            )
        best_epoch, best_ic, stopped = self._run(
            "fit", X, np.asarray(y, dtype=np.float64), np.asarray(w), np.asarray(dates), self.max_epochs,
            val, seed,
        )  # fmt: skip
        params: dict[str, Any] = {"best_epoch": best_epoch, "loss": self.loss_name}
        if stopped:
            params["stopped"] = stopped
        return FitInfo(
            best_iteration=best_epoch, val_ic=None if not np.isfinite(best_ic) else best_ic, params=params
        )

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        epochs = math.ceil(1.1 * int(fit_info.best_iteration or 1))
        self._run(
            "refit", np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64), np.asarray(w),
            np.asarray(dates), epochs, None, seed,
        )  # fmt: skip

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        self._bind(X)
        return self._predict_bound()

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, dir: Path) -> None:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), dir / "weights.pt")
        meta = {"n_feat": self.n_feat, "n_mkt": self.n_mkt, "kinds": self.kinds, "names": self.feature_names}
        (dir / "torch_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, dir: Path) -> None:
        meta = json.loads((dir / "torch_meta.json").read_text(encoding="utf-8"))
        self.n_feat, self.n_mkt = int(meta["n_feat"]), int(meta["n_mkt"])
        self.kinds, self.feature_names = list(meta["kinds"]), list(meta["names"])
        self.net = self.build_net(self.n_feat, self.n_mkt).to(self.device)
        self.net.load_state_dict(torch.load(dir / "weights.pt", map_location=self.device))
