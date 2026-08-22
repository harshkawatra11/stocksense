"""
Cross-sectional alpha ranker.

Per the build plan's critique of v1: a binary classifier at a fixed
threshold fires on ~16% of ticker-days unconditionally, which makes
turnover emergent rather than controlled. A regression against relative
forward return, consumed by taking the top-N by predicted score on each
date (stocksense.portfolio.construct.target_weights_top_n), makes
selectivity an explicit, swept parameter instead.

LightGBM regression, per docs/04-model-brain.md's reasoning: tabular data
at this scale favors gradient-boosted trees over deep learning, and tree
models remain debuggable via feature importance when a prediction looks
wrong — important for a solo-operated system.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class RankerConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


class CrossSectionalRanker:
    """Wraps a LightGBM regressor trained to predict cross-sectional
    relative forward return. `.score()` returns raw predicted relative
    return, which stocksense.portfolio.construct ranks and top-N selects.
    """

    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CrossSectionalRanker":
        mask = y.notna() & X.notna().all(axis=1)
        X_train, y_train = X.loc[mask], y.loc[mask]
        if len(X_train) < 100:
            raise ValueError(f"too few clean training rows: {len(X_train)}")

        self.model = lgb.LGBMRegressor(
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            verbosity=-1,
        )
        self.model.fit(X_train, y_train)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("call .fit() before .predict()")
        X_aligned = X[self.feature_names_]
        valid = X_aligned.notna().all(axis=1)
        preds = pd.Series(np.nan, index=X.index)
        if valid.any():
            preds.loc[valid] = self.model.predict(X_aligned.loc[valid])
        return preds

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("call .fit() before requesting importance")
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)


QUANTILES = (0.1, 0.5, 0.9)  # p10/p50/p90 -- an 80% interval, wide enough to be honest at this sample size


class QuantileRanker:
    """Phase G3: the point-estimate CrossSectionalRanker above answers
    "which names rank highest" but not "how big a move, and how sure are
    we" -- the user's own "expected movements" requirement
    (predictions.predicted_return/confidence exist in the schema and are
    written NULL by every caller today). This wraps three independent
    LightGBM quantile regressors (p10/p50/p90, objective='quantile') so
    a caller gets a genuine interval rather than a single score dressed
    up as one.

    Deliberately a SEPARATE class from CrossSectionalRanker, not a mode
    switch on it: every existing caller (train_and_score_fold,
    simulate_portfolio, the walk-forward gate, the whole Phase 0/Run 4
    result) depends on CrossSectionalRanker's single-score `.predict()`
    contract unchanged. Three quantile models cost roughly 3x the fit
    time of one point model -- an explicit, opt-in choice for the
    handful of callers (Phase G3's record_predictions path) that
    actually need a band, not a cost every walk-forward fold pays.
    """

    def __init__(self, config: RankerConfig | None = None, quantiles: tuple[float, ...] = QUANTILES):
        self.config = config or RankerConfig()
        self.quantiles = quantiles
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileRanker":
        mask = y.notna() & X.notna().all(axis=1)
        X_train, y_train = X.loc[mask], y.loc[mask]
        if len(X_train) < 100:
            raise ValueError(f"too few clean training rows: {len(X_train)}")

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                num_leaves=self.config.num_leaves,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
            self.models[q] = model
        self.feature_names_ = list(X.columns)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns one column per fitted quantile (named 'q10', 'q50',
        'q90' for the default quantiles), NaN-preserving row-for-row like
        CrossSectionalRanker.predict(). Quantiles are fit independently
        (LightGBM has no built-in monotonicity guarantee across separate
        quantile models), so a caller relying on p10 <= p50 <= p90
        should sort per-row rather than assume fit order -- `bands()`
        below does exactly that."""
        if not self.models:
            raise RuntimeError("call .fit() before .predict_quantiles()")
        X_aligned = X[self.feature_names_]
        valid = X_aligned.notna().all(axis=1)
        out = pd.DataFrame(
            {f"q{int(q * 100)}": np.nan for q in self.quantiles}, index=X.index,
        )
        if valid.any():
            for q in self.quantiles:
                out.loc[valid, f"q{int(q * 100)}"] = self.models[q].predict(X_aligned.loc[valid])
        return out

    def predict_bands(self, X: pd.DataFrame) -> pd.DataFrame:
        """The band a caller actually wants to show: `predicted_return`
        (median, q50) and `confidence` (half-width of the outer
        interval, e.g. (q90-q10)/2) -- matching predictions.predicted_
        return/confidence's schema and the project's own standing rule
        (research/phase0_verdict.md) that alpha figures are quoted as a
        band, never a point estimate. Sorts each row's raw quantile
        predictions before computing the width, so an occasional
        crossed-quantile row (independently fit models, no monotonicity
        constraint) still yields a non-negative confidence rather than a
        silently-wrong negative one."""
        raw = self.predict_quantiles(X)
        cols = [f"q{int(q * 100)}" for q in sorted(self.quantiles)]
        mid_col = cols[len(cols) // 2]

        sorted_arr = np.sort(raw[cols].to_numpy(dtype=float), axis=1)  # NaN sorts last; an all-NaN row stays all-NaN
        lo = pd.Series(sorted_arr[:, 0], index=raw.index)
        hi = pd.Series(sorted_arr[:, -1], index=raw.index)

        return pd.DataFrame({
            "predicted_return": raw[mid_col],
            "confidence": ((hi - lo) / 2.0),
        })
