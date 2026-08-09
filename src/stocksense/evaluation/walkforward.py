"""
Purged, embargoed, expanding-window walk-forward splitting.

Per docs/06-retraining-rigor.md: standard k-fold CV leaks the future into
training on time-series data. This splitter trains on [start, T), embargoes
a gap of (horizon + max_feature_lookback) bars, then tests on the bars
immediately after the embargo — never shuffled, never centered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_FEATURE_LOOKBACK_BARS = 252  # longest window in features.engine (252d high/low)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def make_folds(
    trading_dates: pd.DatetimeIndex,
    horizon_bars: int,
    test_window_bars: int = 63,  # ~1 quarter of trading bars per fold
    min_train_bars: int = 500,   # ~2 years minimum before the first fold
) -> list[Fold]:
    """Expanding-window folds: fold k trains on everything up to some
    point, embargoes `horizon_bars + MAX_FEATURE_LOOKBACK_BARS` bars, then
    tests on the next `test_window_bars`. Train start is always the
    beginning of history — expanding, not rolling, so later folds see
    strictly more history, matching how the system will actually retrain.
    """
    embargo = horizon_bars + MAX_FEATURE_LOOKBACK_BARS
    n = len(trading_dates)
    folds: list[Fold] = []

    train_end_pos = min_train_bars
    fold_id = 0
    while True:
        test_start_pos = train_end_pos + embargo
        test_end_pos = test_start_pos + test_window_bars
        if test_end_pos > n:
            break
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=trading_dates[0],
                train_end=trading_dates[train_end_pos - 1],
                test_start=trading_dates[test_start_pos],
                test_end=trading_dates[test_end_pos - 1],
            )
        )
        fold_id += 1
        train_end_pos = test_start_pos + test_window_bars  # expand: next train includes prior test+embargo gap collapses forward

    return folds


def split(df: pd.DataFrame, fold: Fold, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a (symbol, date, ...) frame into (train, test) for one fold.
    Boundaries are inclusive on both ends of each named range."""
    dates = pd.to_datetime(df[date_col])
    train = df.loc[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test = df.loc[(dates >= fold.test_start) & (dates <= fold.test_end)]
    return train, test
