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
