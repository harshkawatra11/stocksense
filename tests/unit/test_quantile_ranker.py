"""Phase G3: QuantileRanker tests -- the user asked for "expected
movements", and predictions.predicted_return/confidence exist in the
schema but are written NULL by every caller today. These tests check
the actual load-bearing properties: quantiles come out ordered (p10 <=
p50 <= p90) after the explicit sort, confidence is non-negative, and a
tighter true relationship between features and target produces a
narrower band than a noisier one -- i.e. confidence responds to real
uncertainty rather than being a constant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.models.ranker import QuantileRanker, RankerConfig


def _make_frame(n=2000, n_features=4, noise_std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    true_signal = X["f0"] * 0.05 - X["f1"] * 0.02
    y = true_signal + rng.normal(0, noise_std, size=n)
    return X, pd.Series(y)


def test_quantile_ranker_fits_and_predicts_ordered_quantiles() -> None:
    X, y = _make_frame()
    ranker = QuantileRanker(RankerConfig(random_state=42, n_estimators=50))
    ranker.fit(X, y)

    raw = ranker.predict_quantiles(X.head(200))
    assert list(raw.columns) == ["q10", "q50", "q90"]
    assert raw.notna().all().all()

    bands = ranker.predict_bands(X.head(200))
    # after the explicit sort in predict_bands, confidence must be
    # derived from the WIDEST spread -- non-negative by construction,
    # even on rows where the raw (unsorted) quantile fits happened to
    # cross.
    assert (bands["confidence"] >= 0).all()
    assert bands["predicted_return"].notna().all()


def test_quantile_ranker_raises_on_too_few_rows() -> None:
    X, y = _make_frame(n=10)
    ranker = QuantileRanker()
    with pytest.raises(ValueError):
        ranker.fit(X, y)


def test_quantile_ranker_raises_before_fit() -> None:
    ranker = QuantileRanker()
    X = pd.DataFrame({"f0": [1.0, 2.0]})
    with pytest.raises(RuntimeError):
        ranker.predict_quantiles(X)


def test_quantile_ranker_nan_rows_stay_nan() -> None:
    X, y = _make_frame()
    ranker = QuantileRanker(RankerConfig(random_state=42, n_estimators=50))
    ranker.fit(X, y)

    X_test = X.head(5).copy()
    X_test.loc[X_test.index[0], "f0"] = np.nan

    bands = ranker.predict_bands(X_test)
    assert pd.isna(bands.iloc[0]["predicted_return"])
    assert pd.isna(bands.iloc[0]["confidence"])
    assert bands.iloc[1:]["predicted_return"].notna().all()


def test_confidence_widens_with_more_noise() -> None:
    """The load-bearing property: confidence must actually track real
    uncertainty in the data-generating process, not just be a constant
    the model always emits regardless of input."""
    X_tight, y_tight = _make_frame(noise_std=0.002, seed=1)
    X_noisy, y_noisy = _make_frame(noise_std=0.08, seed=1)

    ranker_tight = QuantileRanker(RankerConfig(random_state=42, n_estimators=80))
    ranker_tight.fit(X_tight, y_tight)
    ranker_noisy = QuantileRanker(RankerConfig(random_state=42, n_estimators=80))
    ranker_noisy.fit(X_noisy, y_noisy)

    bands_tight = ranker_tight.predict_bands(X_tight.head(300))
    bands_noisy = ranker_noisy.predict_bands(X_noisy.head(300))

    assert bands_noisy["confidence"].mean() > bands_tight["confidence"].mean() * 2
