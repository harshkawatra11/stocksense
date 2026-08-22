"""
Phase G3: calibration tracking for the graded prediction ledger
(`predictions`, written by harness/loops.py's reconcile loop). Per
docs/06-retraining-rigor.md, "calibration/confidence tracking (Brier
score, reliability curves per docs/06) also remains unbuilt" -- this is
that piece, operating on the ledger G2 makes real rather than on a
backtest, and matching research/phase0_verdict.md's own standing rule
that any forward-looking figure is quoted as a band, checked against
outcomes, not asserted.

Every function here takes a DataFrame of ALREADY-GRADED predictions
(graded_at not null, actual_return populated) with at minimum
`predicted_return`, `confidence`, `actual_return` columns -- the exact
shape `store.read_predictions()` returns once grade_matured_predictions
has run. Nothing here re-derives "actual" independently; that
computation lives exactly once, in labels.forward_return, per
grade_matured_predictions' own docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    """Standard quantile (pinball) loss: penalizes under-prediction and
    over-prediction asymmetrically per `quantile`. At quantile=0.5 this
    reduces to half the mean absolute error -- the metric a plain point
    forecast is implicitly optimizing, made explicit and comparable
    across quantiles."""
    diff = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


@dataclass(frozen=True)
class CoverageResult:
    n: int
    nominal_coverage: float
    observed_coverage: float
    mean_abs_error: float
    mean_confidence: float


def interval_coverage(graded: pd.DataFrame, nominal_coverage: float = 0.8) -> CoverageResult:
    """The core calibration check: does `actual_return` fall inside
    [predicted_return - confidence, predicted_return + confidence] as
    often as the band claims it should? `confidence` is written as the
    half-width of an 80% (p10/p90) interval (models.ranker.QuantileRanker
    .predict_bands), so nominal_coverage defaults to 0.8 to match --
    pass a different value only if the confidence figure being checked
    was constructed to mean something else.

    A band that is honest should show observed_coverage close to
    nominal_coverage: badly UNDER nominal means the band is too narrow
    (overconfident); badly OVER means it's too wide (underconfident,
    less useful than it could be but not dishonest in the dangerous
    direction)."""
    df = graded.dropna(subset=["predicted_return", "confidence", "actual_return"])
    n = len(df)
    if n == 0:
        return CoverageResult(n=0, nominal_coverage=nominal_coverage, observed_coverage=float("nan"),
                               mean_abs_error=float("nan"), mean_confidence=float("nan"))

    lo = df["predicted_return"] - df["confidence"]
    hi = df["predicted_return"] + df["confidence"]
    covered = (df["actual_return"] >= lo) & (df["actual_return"] <= hi)

    return CoverageResult(
        n=n,
        nominal_coverage=nominal_coverage,
        observed_coverage=float(covered.mean()),
        mean_abs_error=float((df["actual_return"] - df["predicted_return"]).abs().mean()),
        mean_confidence=float(df["confidence"].mean()),
    )


def reliability_table(graded: pd.DataFrame, n_buckets: int = 5, nominal_coverage: float = 0.8) -> pd.DataFrame:
    """The actual reliability curve, as a table rather than a plot: buckets
    graded predictions into `n_buckets` equal-sized groups by their OWN
    stated confidence (narrowest to widest) and reports observed coverage
    and mean absolute error per bucket. A well-calibrated model should
    show observed coverage near `nominal_coverage` in every bucket, not
    just on average -- averaging alone can hide a model that is
    overconfident on its "sure" predictions and underconfident on the
    rest, which cancel out in a single aggregate number but matter a
    great deal to a user deciding how much to trust one specific
    prediction."""
    df = graded.dropna(subset=["predicted_return", "confidence", "actual_return"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_confidence", "observed_coverage", "mean_abs_error"])

    n_buckets = min(n_buckets, len(df))
    df["_bucket"] = pd.qcut(df["confidence"], q=n_buckets, labels=False, duplicates="drop")

    rows = []
    for bucket, g in df.groupby("_bucket"):
        lo = g["predicted_return"] - g["confidence"]
        hi = g["predicted_return"] + g["confidence"]
        covered = (g["actual_return"] >= lo) & (g["actual_return"] <= hi)
        rows.append({
            "bucket": int(bucket),
            "n": len(g),
            "mean_confidence": float(g["confidence"].mean()),
            "observed_coverage": float(covered.mean()),
            "mean_abs_error": float((g["actual_return"] - g["predicted_return"]).abs().mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_confidence").reset_index(drop=True)
