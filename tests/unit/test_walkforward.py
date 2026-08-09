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
    """No test window may start within (horizon + max_lookback) bars of
    its own fold's train_end — the purge/embargo requirement."""
    from stocksense.evaluation.walkforward import MAX_FEATURE_LOOKBACK_BARS

    dates = pd.bdate_range("2015-01-01", periods=1500)
    horizon = 5
    folds = make_folds(dates, horizon_bars=horizon, test_window_bars=63, min_train_bars=500)

    pos = {d: i for i, d in enumerate(dates)}
    for f in folds:
        gap_bars = pos[f.test_start] - pos[f.train_end]
        assert gap_bars >= horizon + MAX_FEATURE_LOOKBACK_BARS


def test_no_fold_when_insufficient_history() -> None:
    dates = pd.bdate_range("2015-01-01", periods=100)  # far too short
    folds = make_folds(dates, horizon_bars=5, test_window_bars=63, min_train_bars=500)
    assert folds == []
