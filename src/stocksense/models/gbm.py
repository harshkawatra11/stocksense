"""LightGBM (PLAN T7.2): regression or lambdarank on the cross-sectional features.

Early stopping uses a custom per-date Spearman (`mean_ic`, metric "None"); two-step fit: inner-train /
inner-validation finds `best_iteration`, `refit_full` retrains on all rows with ceil(1.1 x best_iteration)
trees. `select_hyperparams` picks one of the 12 grid configurations by inner-validation IC only.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import SystemClock
from stocksense.core.config import Settings
from stocksense.core.state import State
from stocksense.features.build import load_features
from stocksense.labels.forward import load_labels
from stocksense.models.base import FitInfo, Model, ModelSpec, mean_daily_rank_ic, register_model
from stocksense.validation.splits import Fold, inner_validation_split, train_mask, walk_forward_folds
from stocksense.validation.walkforward import END_COLUMN

GRID: list[dict[str, Any]] = [
    {"num_leaves": nl, "min_child_samples": mc, "colsample_bytree": cs}
    for nl, mc, cs in itertools.product((15, 31, 63), (200, 1000), (0.4, 0.7))
]
MIN_NAMES = 10  # a date with fewer names does not enter the IC


def per_date_slices(dates: np.ndarray) -> list[np.ndarray]:
    """Index arrays, one per distinct date (dates need not be sorted)."""
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    cuts = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    return [s for s in np.split(order, cuts) if len(s)]


def make_mean_ic_feval(dates_val: np.ndarray, y_val: np.ndarray) -> Callable[..., tuple[str, float, bool]]:
    """LightGBM eval metric: mean over dates of Spearman(prediction, y_val). Slices are computed once;
    the dataset label passed by LightGBM is ignored (for lambdarank it is the decile, not the return)."""
    slices = [s for s in per_date_slices(np.asarray(dates_val)) if len(s) >= MIN_NAMES]
    yv = np.asarray(y_val, dtype=np.float64)
    ry = [rankdata(yv[s]) for s in slices]

    def feval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        ics = []
        for s, r in zip(slices, ry, strict=True):
            rp = rankdata(y_pred[s])
            a, b = rp - rp.mean(), r - r.mean()
            den = math.sqrt(float(a @ a) * float(b @ b))
            if den > 0:
                ics.append(float(a @ b) / den)
        return "mean_ic", (float(np.mean(ics)) if ics else 0.0), True

    return feval


def lambdarank_groups(dates_sorted: np.ndarray) -> np.ndarray:
    """Row counts per date in date order (rows must be sorted by date)."""
    _, counts = np.unique(dates_sorted, return_counts=True)
    return counts


def _deciles(y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Decile index 0-9 of y within each date."""
    out = np.zeros(len(y), dtype=np.int32)
    for s in per_date_slices(dates):
        r = rankdata(y[s], method="ordinal")
        out[s] = np.minimum(((r - 1) * 10 // len(s)).astype(np.int32), 9)
    return out


class GbmModel(Model):
    def __init__(
        self,
        objective: str = "regression",
        num_leaves: int = 31,
        min_child_samples: int = 200,
        colsample_bytree: float = 0.7,
        n_estimators: int = 1500,
        n_jobs: int = 8,
    ) -> None:
        if objective not in ("regression", "lambdarank"):
            raise ValueError(f"objective {objective!r}")
        self.objective = objective
        self.num_leaves, self.min_child_samples = int(num_leaves), int(min_child_samples)
        self.colsample_bytree, self.n_estimators, self.n_jobs = (
            float(colsample_bytree),
            int(n_estimators),
            n_jobs,
        )
        self._est: Any = None
        self._booster: lgb.Booster | None = None

    def _make(self, seed: int, n_estimators: int) -> Any:
        kw: dict[str, Any] = {
            "boosting_type": "gbdt", "learning_rate": 0.05, "n_estimators": n_estimators,
            "subsample": 0.7, "subsample_freq": 1, "reg_lambda": 10.0, "max_bin": 127,
            "min_child_samples": self.min_child_samples, "colsample_bytree": self.colsample_bytree,
            "num_leaves": self.num_leaves, "deterministic": True, "force_row_wise": True,
            "n_jobs": self.n_jobs, "verbose": -1, "random_state": int(seed), "metric": "None",
        }  # fmt: skip
        if self.objective == "lambdarank":
            return lgb.LGBMRanker(
                objective="lambdarank", lambdarank_truncation_level=50,
                label_gain=[2**i - 1 for i in range(10)], **kw,
            )  # fmt: skip
        return lgb.LGBMRegressor(objective="regression", **kw)

    @staticmethod
    def _sorted(X, y, w, dates):  # type: ignore[no-untyped-def]
        o = np.argsort(dates, kind="stable")
        return (
            X[o],
            np.asarray(y, dtype=np.float64)[o],
            np.asarray(w, dtype=np.float64)[o],
            np.asarray(dates)[o],
        )

    def _fit_args(self, y: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.objective == "lambdarank":
            return _deciles(y, d), {"group": lambdarank_groups(d)}
        return y, {}

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed: int = 0) -> FitInfo:  # type: ignore[no-untyped-def]
        X, y, w, d = self._sorted(X, y, w, dates)
        label, grp = self._fit_args(y, d)
        est = self._make(seed, self.n_estimators)
        if X_val is None or y_val is None or dates_val is None:
            est.fit(X, label, sample_weight=w, **grp)
            self._est, self._booster = est, est.booster_
            return FitInfo(best_iteration=self.n_estimators, params=self._params())
        Xv, yv, _, dv = self._sorted(X_val, y_val, np.ones(len(y_val)), dates_val)
        lab_v, grp_v = self._fit_args(yv, dv)
        eval_kw: dict[str, Any] = {"eval_group": [grp_v["group"]]} if grp_v else {}
        est.fit(
            X, label, sample_weight=w, eval_set=[(Xv, lab_v)], eval_metric=make_mean_ic_feval(dv, yv),
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)], **grp, **eval_kw,
        )  # fmt: skip
        self._est, self._booster = est, est.booster_
        best = int(est.best_iteration_ or self.n_estimators)
        val_ic = mean_daily_rank_ic(self.predict(Xv), yv, dv)
        return FitInfo(best_iteration=best, val_ic=float(val_ic), params=self._params())

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        X, y, w, d = self._sorted(X, y, w, dates)
        label, grp = self._fit_args(y, d)
        est = self._make(seed, math.ceil(1.1 * int(fit_info.best_iteration or self.n_estimators)))
        est.fit(X, label, sample_weight=w, **grp)
        self._est, self._booster = est, est.booster_

    def _params(self) -> dict[str, Any]:
        return {
            "objective": self.objective, "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples, "colsample_bytree": self.colsample_bytree,
        }  # fmt: skip

    def _need(self) -> lgb.Booster:
        if self._booster is None:
            raise RuntimeError("GbmModel is not fitted")
        return self._booster

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._need().predict(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def explain(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray | None:
        c = self._need().predict(np.asarray(X, dtype=np.float32), pred_contrib=True)
        return np.asarray(c)[:, :-1]  # the last column is the bias

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        self._need().save_model(str(dir / "weights.txt"))
        (dir / "gbm_meta.json").write_text(json.dumps(self._params()), encoding="utf-8")

    def load(self, dir: Path) -> None:
        self._booster = lgb.Booster(model_file=str(dir / "weights.txt"))
        meta = json.loads((dir / "gbm_meta.json").read_text(encoding="utf-8"))
        self.objective = meta["objective"]


def _factory(spec: ModelSpec) -> Model:
    keys = ("objective", "num_leaves", "min_child_samples", "colsample_bytree", "n_estimators", "n_jobs")
    return GbmModel(**{k: spec.params[k] for k in keys if k in spec.params})


register_model("gbm", _factory)


# ---- hyper-parameter selection (inner validation only) ---------------------------------------------


def select_hyperparams(
    target: str,
    objective: str,
    settings: Settings,
    state: State,
    anchors: Sequence[int | Fold] = (2016, 2019, 2022),
    date_frac: float = 0.3,
    seed: int = 0,
    *,
    folds: Sequence[Fold] | None = None,
    calendar: TradingCalendar | None = None,
    inner_months: int = 12,
    embargo_sessions: int = 5,
    grid: Sequence[dict[str, Any]] | None = None,
    base_params: dict[str, Any] | None = None,
    universe: str = "u500",
    entry: str | None = None,
    purpose: str = "research",
) -> dict[str, Any]:
    """Best grid configuration (by mean inner-validation IC over the anchor folds) for (target, objective).

    Only rows whose date is before each anchor's test start are used (the loaders are asked for nothing
    later than the last anchor's train end), a seeded `date_frac` subsample of dates, and no trial is
    registered (`state` is accepted for symmetry and never written). Returns
    {"best": config, "mean_ic": float, "table": [{config, ic_by_anchor, mean_ic}]}.
    """
    del state
    cal = calendar if calendar is not None else TradingCalendar.from_store(settings, SystemClock())
    pool = list(folds) if folds is not None else walk_forward_folds(settings, cal)
    chosen = [a if isinstance(a, Fold) else next(f for f in pool if f.name == str(a)) for a in anchors]
    last_train_end = max(f.train_end for f in chosen)
    entry = entry or ("open" if target == "y_oc" else "close")
    label_col, end_col, blocked = f"r_{target}", END_COLUMN[target], f"entry_blocked_{entry}"
    lab = (
        load_labels(settings, purpose, end=last_train_end, columns=[label_col, end_col, blocked])
        .collect()
        .filter(pl.col(label_col).is_not_null() & ~pl.col(blocked).fill_null(False))
    )
    if universe != "u500":
        from stocksense.data import io

        uni = io.read(settings, "panel_universe", columns=["date", "sid", universe]).collect()
        lab = lab.join(uni.filter(pl.col(universe)).select("date", "sid"), on=["date", "sid"])
    feats = load_features(settings, purpose, end=last_train_end).collect()
    names = [c for c in feats.columns if c not in ("date", "sid")]
    df = lab.join(feats, on=["date", "sid"], how="inner").sort("date", "sid")
    rng = np.random.default_rng(seed)
    data: list[tuple[np.ndarray, ...]] = []
    for fold in chosen:
        sub = df.filter(pl.col("date") < fold.test_start)
        tr = train_mask(
            sub[end_col], fold, embargo_sessions, calendar=cal, dates=sub["date"],
            dev_start=settings.research.dev_start,
        ).to_numpy()  # fmt: skip
        sub = sub.filter(pl.Series(tr))
        a, b = inner_validation_split(sub["date"], sub[end_col], inner_months, embargo_sessions, calendar=cal)
        parts = []
        for m in (a.to_numpy(), b.to_numpy()):
            ud = sub["date"].filter(pl.Series(m)).unique().sort().to_numpy()
            keep = rng.choice(ud, size=max(1, round(len(ud) * date_frac)), replace=False)
            sel = m & np.isin(sub["date"].to_numpy(), keep)
            s2 = sub.filter(pl.Series(sel))
            parts += [
                s2.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32),
                s2[label_col].to_numpy().astype(np.float64),
                s2["date"].to_numpy(),
            ]
        data.append(tuple(parts))
    table = []
    for cfg in grid if grid is not None else GRID:
        ics = []
        for Xa, ya, da, Xb, yb, db in data:
            m = GbmModel(objective=objective, **{**(base_params or {}), **cfg})
            info = m.fit(Xa, ya, np.ones(len(ya)), da, Xb, yb, db, seed=seed)
            ics.append(float(info.val_ic) if info.val_ic is not None else float("nan"))
        ok = [i for i in ics if np.isfinite(i)]
        mean_ic = float(np.mean(ok)) if ok else float("-inf")
        table.append({"config": dict(cfg), "ic_by_anchor": ics, "mean_ic": mean_ic})
    best = max(table, key=lambda r: r["mean_ic"])
    return {"best": best["config"], "mean_ic": best["mean_ic"], "table": table}
