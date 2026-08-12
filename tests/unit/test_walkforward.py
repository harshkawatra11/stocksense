"""Purged/embargoed walk-forward split correctness."""

from __future__ import annotations

import pandas as pd

from stocksense.evaluation.walkforward import make_folds


def test_folds_are_strictly_forward_in_time() -> None:
    dates = pd.bdate_range("2015-01-01", periods=1500)
    folds = make_folds(dates, horizon_bars=5, test_window_bars=63, min_train_bars=500)
    assert len(folds) > 0
    for f in folds:
        assert f.train_start < f.train_end < f.test_start < f.test_end


def test_embargo_gap_is_respected() -> None:
    """No test window may start within (horizon + embargo_buffer) bars of
    its own fold's train_end — the purge/embargo requirement.

    Corrected per the audit: the gap is horizon_bars (purge, so no
    training label resolves inside the test window) plus a small fixed
    embargo buffer against residual serial correlation — NOT horizon
    plus the longest feature lookback. Test-period features legitimately
    depend on pre-test-period data (that's how the model runs live); that
    dependency is not a leakage vector and embargoing it only destroys
    fold count for no statistical benefit. See walkforward.py's docstring
    for the full reasoning.
    """
    from stocksense.evaluation.walkforward import DEFAULT_EMBARGO_BUFFER_BARS

    dates = pd.bdate_range("2015-01-01", periods=1500)
    horizon = 5
    folds = make_folds(dates, horizon_bars=horizon, test_window_bars=63, min_train_bars=500)

    pos = {d: i for i, d in enumerate(dates)}
    for f in folds:
        gap_bars = pos[f.test_start] - pos[f.train_end]
        # +1 because train_end is the last INCLUDED training bar (inclusive
        # indexing), so the position delta is one more than the configured
        # gap — conservative, not a leakage risk.
        assert gap_bars == horizon + DEFAULT_EMBARGO_BUFFER_BARS + 1


def test_corrected_embargo_yields_more_folds_than_lookback_based_embargo() -> None:
    """Direct regression test for the audit finding: the old (buggy)
    embargo of horizon+252 halved fold count relative to the corrected
    horizon+small-buffer embargo, for no leakage-prevention benefit."""
    dates = pd.bdate_range("2000-01-01", periods=6500)  # ~26 years, matching Phase 0's real history
    horizon = 20

    corrected_folds = make_folds(dates, horizon_bars=horizon, test_window_bars=240, min_train_bars=500)
    old_buggy_folds = make_folds(
        dates, horizon_bars=horizon, test_window_bars=240, min_train_bars=500,
        embargo_buffer_bars=252,  # the old MAX_FEATURE_LOOKBACK_BARS value, injected as the buffer
    )
    assert len(corrected_folds) > len(old_buggy_folds)


def test_no_fold_when_insufficient_history() -> None:
    dates = pd.bdate_range("2015-01-01", periods=100)  # far too short
    folds = make_folds(dates, horizon_bars=5, test_window_bars=63, min_train_bars=500)
    assert folds == []
