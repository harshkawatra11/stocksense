"""Meta-labeling for the S1 book (PLAN T9.4; kept in portfolio/ because models/ is not this track's).

A small LightGBM classifier learns, from walk-forward S1 entries the policy would have taken, whether a
trade's net return ends positive. `S1Policy` uses its probability p two ways: a size multiplier
`clip(2p - 1, 0, 1)` on the raw weight and a veto when p < 0.5. A fold's model is trained only on trades
that CLOSED strictly before the fold's first test date, so no label from the test window (or from a trade
still open) leaks in. Without enough closed history the model stays unfitted, predicts NaN, and the layer is
neutral (multiplier 1, no veto).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from stocksense.validation.splits import Fold

META_FEATURES: tuple[str, ...] = (
    "score",
    "score_rank",
    "disagreement",
    "vol_21",
    "turnover_rank",
    "mkt_ret_21",
    "breadth_200",
    "days_since_result",
    "gross_target",
)
VETO_BELOW = 0.5
INNER_VAL_FRAC = 0.2
MAX_TREES = 500
EARLY_STOP = 30


def size_multiplier(p: float) -> float:
    """clip(2p - 1, 0, 1); a missing probability (NaN) leaves the size unchanged."""
    if p is None or math.isnan(p):
        return 1.0
    return min(max(2.0 * p - 1.0, 0.0), 1.0)


def veto(p: float) -> bool:
    """True when p < 0.5 (a missing probability never vetoes)."""
    if p is None or math.isnan(p):
        return False
    return p < VETO_BELOW


def build_meta_samples(entry_log: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Join the policy's entry log (sid, date = entry session, META_FEATURES) to its CLOSED trades.

    Partial exits of one position are pooled: ret = summed net P&L / summed entry value, exit date = the last
    exit; label = 1 if ret > 0. Entries with no closed trade are dropped (their outcome is unknown)."""
    pooled = (
        trades.with_columns((pl.col("entry_price") * pl.col("qty")).alias("_basis"))
        .group_by("sid", "entry_date")
        .agg(
            pl.col("exit_date").max().alias("exit_date"),
            pl.col("net_pnl_inr").sum().alias("_pnl"),
            pl.col("_basis").sum().alias("_basis"),
        )
        .with_columns((pl.col("_pnl") / pl.col("_basis")).alias("ret"))
        .with_columns((pl.col("ret") > 0).cast(pl.Int8).alias("label"))
        .select("sid", "entry_date", "exit_date", "ret", "label")
    )
    feats = entry_log.select("sid", pl.col("date").alias("entry_date"), *META_FEATURES)
    return feats.join(pooled, on=["sid", "entry_date"], how="inner").sort("entry_date", "sid")


@dataclass(frozen=True)
class MetaFitInfo:
    fitted: bool
    n_train: int
    max_exit_date: date | None
    best_iteration: int | None = None
    val_logloss: float | None = None


class MetaModel:
    """LightGBM binary classifier (num_leaves 7, min_child_samples 200, learning rate 0.05)."""

    def __init__(
        self,
        *,
        num_leaves: int = 7,
        min_child_samples: int = 200,
        learning_rate: float = 0.05,
        n_jobs: int = 4,
    ) -> None:
        self.num_leaves, self.min_child_samples = int(num_leaves), int(min_child_samples)
        self.learning_rate, self.n_jobs = float(learning_rate), int(n_jobs)
        self._est = None
        self.fitted = False

    # ---- training data
    @staticmethod
    def training_rows(samples: pl.DataFrame, test_start: date) -> pl.DataFrame:
        """The samples whose trade closed strictly before `test_start`."""
        return samples.filter(pl.col("exit_date") < test_start)

    def fit(self, samples: pl.DataFrame, test_start: date, seed: int = 0) -> MetaFitInfo:
        import lightgbm as lgb

        self._est, self.fitted = None, False
        tr = self.training_rows(samples, test_start).sort("exit_date", "entry_date")
        n = tr.height
        last = tr["exit_date"].max() if n else None
        n_val = int(n * INNER_VAL_FRAC)
        y_all = tr["label"].to_numpy() if n else np.array([])
        enough = n - n_val >= 2 * self.min_child_samples and n_val >= 20
        if not enough or len(np.unique(y_all)) < 2:
            return MetaFitInfo(False, n, last)
        x = tr.select(META_FEATURES).to_numpy().astype(np.float32)
        y = y_all.astype(int)
        # chronological inner validation: the latest-closing 20% of the training trades
        xt, yt, xv, yv = x[: n - n_val], y[: n - n_val], x[n - n_val :], y[n - n_val :]
        if len(np.unique(yt)) < 2:
            return MetaFitInfo(False, n, last)
        est = lgb.LGBMClassifier(
            objective="binary", n_estimators=MAX_TREES, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples, subsample=0.8,
            subsample_freq=1, reg_lambda=10.0, deterministic=True, force_row_wise=True,
            n_jobs=self.n_jobs, verbose=-1, random_state=int(seed),
        )  # fmt: skip
        est.fit(
            xt, yt, eval_set=[(xv, yv)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )  # fmt: skip
        self._est, self.fitted = est, True
        best = int(est.best_iteration_ or MAX_TREES)
        ll = est.best_score_["valid_0"]["binary_logloss"]
        return MetaFitInfo(True, n, last, best, float(ll))

    # ---- prediction
    def _matrix(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        return np.array(
            [[float(r.get(k, float("nan"))) for k in META_FEATURES] for r in rows], dtype=np.float32
        ).reshape(len(rows), len(META_FEATURES))

    def predict_rows(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        """P(trade ends positive) per row dict (missing keys are NaN); NaN for every row if unfitted."""
        if not self.fitted or self._est is None:
            return np.full(len(rows), np.nan)
        return np.asarray(self._est.predict_proba(self._matrix(rows))[:, 1], dtype=float)

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.predict_rows(frame.select(META_FEATURES).to_dicts())

    def size_multiplier(self, p: float) -> float:
        return size_multiplier(p)

    def veto(self, p: float) -> bool:
        return veto(p)


def oof_meta_predictions(
    samples: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    factory: Callable[[], MetaModel] = MetaModel,
    seed: int = 0,
) -> pl.DataFrame:
    """Out-of-fold p for the entries inside each fold's test window, one model per fold fitted on the trades
    that closed before that fold's test start. Columns: fold, entry_date, sid, p (NaN where unfitted)."""
    parts = []
    for fold in folds:
        model = factory()
        model.fit(samples, fold.test_start, seed=seed)
        test = samples.filter(
            (pl.col("entry_date") >= fold.test_start) & (pl.col("entry_date") <= fold.test_end)
        )
        if test.height == 0:
            continue
        parts.append(
            test.select("entry_date", "sid").with_columns(
                pl.lit(fold.name).alias("fold"), pl.Series("p", model.predict(test), dtype=pl.Float64)
            )
        )
    if not parts:
        return pl.DataFrame(schema={"fold": pl.Utf8, "entry_date": pl.Date, "sid": pl.Int64, "p": pl.Float64})
    return pl.concat(parts).select("fold", "entry_date", "sid", "p")
