"""DoubleEnsemble (Zhang et al., 2020; Qlib `DEnsembleModel`), the upgraded GBM baseline (G2 card D11).

K LightGBM sub-models are trained one after another. After sub-model k the next one gets
  * sample reweighting by the training loss curve of the sub-model just fitted: every training row has a
    loss at a few snapshots of the boosting run; the normalised loss ranks give a slope (end minus start
    thirds) and the ensemble-so-far final loss gives a level; both are pct-ranked and averaged into h
    (small = easy and well learned). Rows are cut into `bins` bins by h and each bin gets the weight
    1 / (decay^k * mean_h(bin) + 0.1), rescaled to mean 1 (Qlib's form: decay shrinks the preference as k
    grows);
  * shuffle-based feature selection: the importance of a feature is the rise of the squared error of the
    sub-model on a row sample when that column is permuted. The ranked features are cut into `fs_bins`
    groups and the next sub-model's `ratios[k]` x F features are drawn evenly from the groups.
The score is the mean of the sub-model predictions. Early stopping uses the per-date mean rank IC of
`make_mean_ic_feval` (through `GbmModel`); `refit_full` re-runs the whole chain on all rows with the stored
feature sets and ceil(1.1 x best_iteration) trees per sub-model. Fixed LightGBM parameters are those of
`GbmModel`. `explain` returns None.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from stocksense.models.base import FitInfo, Model, ModelSpec, register_model
from stocksense.models.gbm import GbmModel

DEFAULT_RATIOS = (0.8, 0.7, 0.6, 0.5, 0.4)
H_OFFSET = 0.1  # the additive constant of the weight formula


def _pct_rank(a: np.ndarray) -> np.ndarray:
    return rankdata(a, method="average") / len(a)


def loss_curve_slope_weights(
    loss_curve: np.ndarray, loss_values: np.ndarray, k: int, decay: float, bins: int = 10
) -> np.ndarray:
    """Sample weights (mean 1) for sub-model `k` from `loss_curve` [n, T] (per-row loss at T snapshots of
    the previous sub-model) and `loss_values` [n] (loss of the ensemble so far). k = 0 gives all ones."""
    n = loss_curve.shape[0]
    if k <= 0 or n == 0:
        return np.ones(n)
    curve = np.asarray(loss_curve, dtype=np.float64)
    norm = np.column_stack([_pct_rank(curve[:, t]) for t in range(curve.shape[1])])
    third = max(1, curve.shape[1] // 3)
    slope = norm[:, -third:].mean(axis=1) - norm[:, :third].mean(axis=1)  # negative: ranks improve
    h = 0.5 * (_pct_rank(np.asarray(loss_values, dtype=np.float64)) + _pct_rank(slope))
    r = rankdata(h, method="ordinal")
    b = np.minimum((r - 1) * bins // n, bins - 1).astype(np.int64)
    sums = np.bincount(b, weights=h, minlength=bins)
    cnt = np.bincount(b, minlength=bins)
    h_avg = np.divide(sums, cnt, out=np.zeros(bins), where=cnt > 0)
    w = 1.0 / (decay**k * h_avg[b] + H_OFFSET)
    return w / w.mean()


def select_features(
    importance: np.ndarray, cols: np.ndarray, n_sel: int, n_bins: int, rng: np.random.Generator
) -> np.ndarray:
    """`n_sel` columns of `cols` drawn evenly from `n_bins` importance-ranked groups (sorted result)."""
    n_sel = int(min(max(n_sel, 1), len(cols)))
    order = cols[np.argsort(-importance, kind="stable")]
    groups = np.array_split(order, min(n_bins, len(order)))
    base, extra = divmod(n_sel, len(groups))
    chosen: list[int] = []
    for i, g in enumerate(groups):
        take = min(len(g), base + (1 if i < extra else 0))
        chosen += list(rng.choice(g, size=take, replace=False))
    if len(chosen) < n_sel:  # small groups could not give their share: top up by importance
        rest = [c for c in order if c not in set(chosen)]
        chosen += rest[: n_sel - len(chosen)]
    return np.sort(np.asarray(chosen, dtype=np.int64))


class DensModel(Model):
    def __init__(
        self,
        k: int = 6,
        decay: float = 0.5,
        bins: int = 10,
        ratios: Sequence[float] = DEFAULT_RATIOS,
        fs_bins: int = 5,
        n_snapshots: int = 6,
        importance_rows: int = 100_000,
        num_leaves: int = 31,
        min_child_samples: int = 200,
        colsample_bytree: float = 0.7,
        n_estimators: int = 1500,
        n_jobs: int = 8,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k, self.decay, self.bins, self.fs_bins = int(k), float(decay), int(bins), int(fs_bins)
        r = [float(x) for x in ratios] or [1.0]
        self.ratios = tuple(r + [r[-1]] * max(0, self.k - 1 - len(r)))
        self.n_snapshots, self.importance_rows = int(n_snapshots), int(importance_rows)
        self.num_leaves, self.min_child_samples = int(num_leaves), int(min_child_samples)
        self.colsample_bytree, self.n_estimators, self.n_jobs = (
            float(colsample_bytree),
            int(n_estimators),
            n_jobs,
        )
        self._members: list[tuple[np.ndarray, GbmModel]] = []

    # ---- helpers ---------------------------------------------------------------------------

    def _sub(self) -> GbmModel:
        return GbmModel(
            "regression", self.num_leaves, self.min_child_samples, self.colsample_bytree,
            self.n_estimators, self.n_jobs,
        )  # fmt: skip

    def _loss_curve(self, gm: GbmModel, Xc: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Squared error of every row at `n_snapshots` cumulative tree counts (one pass of prediction)."""
        booster = gm._need()
        total = booster.num_trees()
        ends = np.unique(np.linspace(total / self.n_snapshots, total, self.n_snapshots).round().astype(int))
        ends = ends[ends >= 1]
        pred = np.zeros(len(y))
        prev = 0
        cols = []
        for e in ends:
            pred = pred + np.asarray(
                booster.predict(Xc, start_iteration=int(prev), num_iteration=int(e - prev)), dtype=np.float64
            )
            prev = int(e)
            cols.append((y - pred) ** 2)
        return np.column_stack(cols)

    def _importance(
        self, gm: GbmModel, X: np.ndarray, cols: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Rise of the squared error when one column of the sub-model's feature set is permuted."""
        n = len(y)
        rows = np.sort(rng.choice(n, size=min(self.importance_rows, n), replace=False))
        Xs = np.ascontiguousarray(X[np.ix_(rows, cols)], dtype=np.float32)
        ys = y[rows]
        base = float(np.mean((ys - gm.predict(Xs)) ** 2))
        out = np.zeros(len(cols))
        for j in range(len(cols)):
            keep = Xs[:, j].copy()
            Xs[:, j] = keep[rng.permutation(len(rows))]
            out[j] = float(np.mean((ys - gm.predict(Xs)) ** 2)) - base
            Xs[:, j] = keep
        return out

    # ---- training --------------------------------------------------------------------------

    def _chain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        dates: np.ndarray,
        seed: int,
        val: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        feature_sets: list[np.ndarray] | None,
        best_iters: list[int] | None,
    ) -> tuple[list[tuple[np.ndarray, GbmModel]], list[np.ndarray], list[int], float | None]:
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(int(seed) + 7919)
        cols = np.arange(X.shape[1], dtype=np.int64)
        sub_w = np.ones(len(y))
        ens = np.zeros(len(y))
        members: list[tuple[np.ndarray, GbmModel]] = []
        sets: list[np.ndarray] = []
        bests: list[int] = []
        for k in range(self.k):
            if feature_sets is not None:
                cols = feature_sets[k]
            sets.append(cols)
            gm = self._sub()
            sk = int(seed) + 1000 * k
            Xc = np.ascontiguousarray(X[:, cols])
            wk = w * sub_w
            if val is not None:
                Xv, yv, dv = val
                info = gm.fit(Xc, y, wk, dates, np.asarray(Xv, dtype=np.float32)[:, cols], yv, dv, seed=sk)
            else:
                assert best_iters is not None
                info = FitInfo(best_iteration=best_iters[k])
                gm.refit_full(Xc, y, wk, dates, info, sk)
            bests.append(int(info.best_iteration or self.n_estimators))
            members.append((cols, gm))
            if k == self.k - 1:
                break
            ens = (ens * k + gm.predict(Xc)) / (k + 1)
            curve = self._loss_curve(gm, Xc, y)
            sub_w = loss_curve_slope_weights(curve, (y - ens) ** 2, k + 1, self.decay, self.bins)
            if feature_sets is None:
                imp = self._importance(gm, X, cols, y, rng)
                n_sel = round(self.ratios[k] * X.shape[1])
                cols = select_features(imp, cols, n_sel, self.fs_bins, rng)
        return members, sets, bests, None

    def fit(  # type: ignore[no-untyped-def]
        self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed: int = 0
    ) -> FitInfo:
        if X_val is None or y_val is None or dates_val is None:
            raise ValueError("DensModel.fit needs inner-validation rows for early stopping")
        from stocksense.models.base import mean_daily_rank_ic

        val = (X_val, np.asarray(y_val, dtype=np.float64), np.asarray(dates_val))
        self._members, sets, bests, _ = self._chain(X, y, w, np.asarray(dates), seed, val, None, None)
        ic = mean_daily_rank_ic(self.predict(X_val), val[1], val[2])
        return FitInfo(
            best_iteration=int(round(float(np.mean(bests)))),
            val_ic=float(ic),
            params={**self._params(), "features": [s.tolist() for s in sets], "best_iterations": bests},
        )

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        sets = [np.asarray(s, dtype=np.int64) for s in fit_info.params["features"]]
        bests = [int(b) for b in fit_info.params["best_iterations"]]
        self._members, _, _, _ = self._chain(X, y, w, np.asarray(dates), seed, None, sets, bests)

    # ---- inference and persistence ---------------------------------------------------------

    def _params(self) -> dict[str, Any]:
        return {
            "k": self.k, "decay": self.decay, "bins": self.bins, "ratios": list(self.ratios),
            "fs_bins": self.fs_bins, "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples, "colsample_bytree": self.colsample_bytree,
        }  # fmt: skip

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._members:
            raise RuntimeError("DensModel is not fitted")
        X = np.asarray(X, dtype=np.float32)
        return np.mean([gm.predict(X[:, cols]) for cols, gm in self._members], axis=0)

    def save(self, dir: Path) -> None:
        if not self._members:
            raise RuntimeError("DensModel is not fitted")
        dir.mkdir(parents=True, exist_ok=True)
        for i, (_, gm) in enumerate(self._members):
            gm._need().save_model(str(dir / f"weights.{i}.txt"))
        meta = {**self._params(), "features": [c.tolist() for c, _ in self._members]}
        (dir / "dens_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, dir: Path) -> None:
        import lightgbm as lgb

        meta = json.loads((dir / "dens_meta.json").read_text(encoding="utf-8"))
        self.k, self.decay = int(meta["k"]), float(meta["decay"])
        self._members = []
        for i, cols in enumerate(meta["features"]):
            gm = self._sub()
            gm._booster = lgb.Booster(model_file=str(dir / f"weights.{i}.txt"))
            self._members.append((np.asarray(cols, dtype=np.int64), gm))


_KEYS = (
    "k", "decay", "bins", "ratios", "fs_bins", "n_snapshots", "importance_rows", "num_leaves",
    "min_child_samples", "colsample_bytree", "n_estimators", "n_jobs",
)  # fmt: skip


def _factory(spec: ModelSpec) -> Model:
    return DensModel(**{k: spec.params[k] for k in _KEYS if k in spec.params})


register_model("dens", _factory)
