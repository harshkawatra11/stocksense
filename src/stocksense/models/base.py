"""The common model contract (PLAN Phase 7 preamble): ModelSpec, FitInfo, the Model base class.

`X` is always a Float32 numpy array with NaN for missing. Scores only matter through their per-date ranks.
A model family registers a factory with `register_model`; `build_model(spec)` resolves a spec to an instance
(the family's last dash-separated segment is the key, so `s1-ridge` and `ridge` both resolve to `ridge`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ModelSpec:
    """What a trial is: family, target label, universe, features ('all' or names), parameters, entry."""

    family: str
    target: str
    universe: str = "u500"
    features: list[str] | str = "all"
    params: dict[str, Any] = field(default_factory=dict)
    entry: str = "close"  # 'open' or 'close': which entry_blocked_* flag removes rows

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(**d)


@dataclass
class FitInfo:
    """What the inner-validation fit found (best_iteration for boosting, the chosen alpha for ridge...)."""

    best_iteration: int | None = None
    val_ic: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitInfo:
        return cls(**d)


class Model(ABC):
    """Contract every model family implements (T7.1-T7.5)."""

    #: label prefix used for the training target: 'r_' (rank in [-0.5, 0.5]), 'z_' (robust z) or '' (raw)
    target_prefix: str = "r_"

    def configure(self, feature_names: Sequence[str], kinds: Sequence[str]) -> None:  # noqa: B027
        """Told by the runner which column is which feature and its kind ('xs' or 'mkt'); default: ignore."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        dates: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        dates_val: np.ndarray | None = None,
        seed: int = 0,
    ) -> FitInfo:
        """Fit on the inner-train rows, using the inner-validation rows to pick complexity."""

    @abstractmethod
    def refit_full(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, dates: np.ndarray, fit_info: FitInfo, seed: int
    ) -> None:
        """Refit on all training rows with the complexity found by `fit`."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """One score per row (Float64 is fine)."""

    @abstractmethod
    def save(self, dir: Path) -> None:
        """Write weights (files named `weights.*`) and anything needed to reload into `dir`."""

    @abstractmethod
    def load(self, dir: Path) -> None:
        """Restore what `save` wrote."""

    def explain(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray | None:
        """Per-row, per-feature contributions (n x n_features), or None when the model has none."""
        return None


# ---- registry ----------------------------------------------------------------------------------

ModelFactory = Callable[[ModelSpec], Model]
MODELS: dict[str, ModelFactory] = {}


def register_model(key: str, factory: ModelFactory) -> None:
    """Register `factory` for family key `key` (duplicates raise)."""
    if key in MODELS:
        raise ValueError(f"model family {key!r} already registered")
    MODELS[key] = factory


def build_model(spec: ModelSpec) -> Model:
    """Instance for `spec.family` (registered under the family's last '-' segment)."""
    import stocksense.models.linear  # noqa: F401  (registers the linear families)

    key = spec.family.rsplit("-", 1)[-1]
    if key not in MODELS:
        raise KeyError(f"no model registered for family {spec.family!r}")
    return MODELS[key](spec)


# ---- shared metric -----------------------------------------------------------------------------


def mean_daily_rank_ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray, min_pairs: int = 10) -> float:
    """Mean over dates of the Spearman correlation of pred and y (dates with fewer pairs are skipped);
    nan when no date qualifies."""
    df = pl.DataFrame({"d": dates, "p": np.asarray(pred, dtype=float), "y": np.asarray(y, dtype=float)})
    df = df.drop_nulls().filter(pl.col("p").is_finite() & pl.col("y").is_finite())
    out = (
        df.with_columns(
            pl.col("p").rank("average").over("d").alias("rp"),
            pl.col("y").rank("average").over("d").alias("ry"),
            pl.len().over("d").alias("n"),
        )
        .filter(pl.col("n") >= min_pairs)
        .group_by("d")
        .agg(pl.corr("rp", "ry").alias("ic"))
        .drop_nulls("ic")
    )
    return float(out["ic"].mean()) if out.height else float("nan")  # type: ignore[arg-type]
