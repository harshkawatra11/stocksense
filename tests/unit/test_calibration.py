"""Calibration tracking tests (Phase G3). Checks the load-bearing
property directly: a genuinely well-calibrated 80% band must show ~80%
observed coverage, an overconfident (too-narrow) band must show
observed coverage well BELOW nominal, and pinball loss must actually
penalize a prediction that misses on the wrong side more than one that
doesn't."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.evaluation.calibration import interval_coverage, pinball_loss, reliability_table


def _synthetic_graded(n=2000, band_scale=1.0, seed=0) -> pd.DataFrame:
    """actual_return ~ N(predicted_return, sigma); confidence set so that
    a band_scale of 1.0 gives ~80% true coverage (1.2816 sigma = the
    80% z-band half-width), band_scale < 1 makes the band too narrow
    (overconfident), > 1 too wide."""
    rng = np.random.default_rng(seed)
    predicted = rng.normal(0, 0.02, size=n)
    sigma = 0.01
    actual = predicted + rng.normal(0, sigma, size=n)
    confidence = np.full(n, 1.2816 * sigma * band_scale)
    return pd.DataFrame({"predicted_return": predicted, "confidence": confidence, "actual_return": actual})


def test_well_calibrated_band_shows_coverage_near_nominal() -> None:
    graded = _synthetic_graded(band_scale=1.0)
    result = interval_coverage(graded, nominal_coverage=0.8)
    assert result.n == 2000
    assert abs(result.observed_coverage - 0.8) < 0.03


def test_overconfident_band_shows_coverage_below_nominal() -> None:
    graded = _synthetic_graded(band_scale=0.3)  # much too narrow
    result = interval_coverage(graded, nominal_coverage=0.8)
    assert result.observed_coverage < 0.5  # well below the 80% the band claims


def test_underconfident_band_shows_coverage_above_nominal() -> None:
    graded = _synthetic_graded(band_scale=3.0)  # much too wide
    result = interval_coverage(graded, nominal_coverage=0.8)
    assert result.observed_coverage > 0.95


def test_interval_coverage_empty_input() -> None:
    graded = pd.DataFrame(columns=["predicted_return", "confidence", "actual_return"])
    result = interval_coverage(graded)
    assert result.n == 0
    assert result.observed_coverage != result.observed_coverage  # NaN


def test_interval_coverage_ignores_ungraded_rows() -> None:
    graded = _synthetic_graded(n=100)
    graded.loc[0:9, "actual_return"] = np.nan  # not yet graded
    result = interval_coverage(graded)
    assert result.n == 90


def test_reliability_table_buckets_by_confidence_and_reports_coverage() -> None:
    graded = _synthetic_graded(n=1000, band_scale=1.0)
    # give half the rows a much wider band -- two distinct confidence regimes
    graded.loc[500:, "confidence"] *= 3
    table = reliability_table(graded, n_buckets=2)

    assert len(table) == 2
    assert table["mean_confidence"].is_monotonic_increasing
    assert (table["n"] > 0).all()
    # the wide-confidence bucket should show higher observed coverage than
    # the narrow one, matching how the synthetic data was constructed
    assert table.iloc[-1]["observed_coverage"] >= table.iloc[0]["observed_coverage"]


def test_reliability_table_empty_input() -> None:
    graded = pd.DataFrame(columns=["predicted_return", "confidence", "actual_return"])
    table = reliability_table(graded)
    assert table.empty


def test_pinball_loss_at_median_is_half_mean_abs_error() -> None:
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    predicted = pd.Series([1.5, 1.5, 3.5, 3.5])
    mae = (actual - predicted).abs().mean()
    loss = pinball_loss(actual, predicted, quantile=0.5)
    assert loss == pytest.approx(mae / 2, abs=1e-9)


def test_pinball_loss_penalizes_underprediction_more_at_high_quantile() -> None:
    actual = pd.Series([10.0])
    under = pd.Series([5.0])  # predicted well below actual
    over = pd.Series([15.0])  # predicted well above actual

    # at quantile=0.9, under-predicting the true value should cost more
    # than over-predicting by the same margin
    loss_under = pinball_loss(actual, under, quantile=0.9)
    loss_over = pinball_loss(actual, over, quantile=0.9)
    assert loss_under > loss_over
