"""
Purged, embargoed, expanding-window walk-forward splitting.

Per docs/06-retraining-rigor.md: standard k-fold CV leaks the future into
training on time-series data. This splitter trains on [start, T), purges
training samples whose labels would overlap the test window, embargoes a
small additional buffer against residual serial correlation, then tests
on the bars immediately after — never shuffled, never centered.

A methodology correction made after auditing the initial implementation:
the gap between train and test is `horizon_bars` (the purge) plus a small
fixed buffer (the embargo) — **not** `horizon_bars + the longest feature
lookback`, which the original version used. That was a real bug, not a
conservative choice, and it cost roughly half the fold count for no
statistical benefit. The reasoning:

- PURGE (must equal the label horizon): a training sample at date T has
  a label that resolves at T + horizon_bars. If that resolution date
  falls inside the test window, the sample's label was informed by data
  the test period is supposed to be "unseen" with respect to. This is
  the only leakage vector `horizon_bars` alone doesn't already prevent
  via the test period simply starting after training ends — it exists
  specifically for labels whose resolution *outlives* the training
  cutoff.
- FEATURE LOOKBACK IS NOT A LEAKAGE VECTOR HERE: a test-period feature on
  date D legitimately uses the preceding ~252 days of history, including
  days that fall inside the training window. That is not leakage — it is
  exactly how the model computes features live, every single day. Test
  features depending on pre-test data is required, not forbidden;
  embargoing it wastes fold count for a risk that does not exist.
- EMBARGO (a small buffer beyond the purge): real markets have short-term
  serial correlation that a bare purge does not fully account for near
  the boundary. A small fixed buffer, not scaled to the feature lookback,
  covers this without discarding data for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_EMBARGO_BUFFER_BARS = 10  # ~2 weeks; guards against residual serial correlation at the boundary


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
    embargo_buffer_bars: int = DEFAULT_EMBARGO_BUFFER_BARS,
) -> list[Fold]:
    """Expanding-window folds: fold k trains on everything up to some
    point, purges+embargoes a gap of (horizon_bars + embargo_buffer_bars)
    bars, then tests on the next `test_window_bars`. Train start is
    always the beginning of history — expanding, not rolling, so later
    folds see strictly more history, matching how the system will
    actually retrain.
    """
    gap = horizon_bars + embargo_buffer_bars
    n = len(trading_dates)
    folds: list[Fold] = []

    train_end_pos = min_train_bars
    fold_id = 0
    while True:
        test_start_pos = train_end_pos + gap
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
        train_end_pos = test_start_pos + test_window_bars  # expand: next train includes prior test+gap collapses forward

    return folds


def split(df: pd.DataFrame, fold: Fold, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a (symbol, date, ...) frame into (train, test) for one fold.
    Boundaries are inclusive on both ends of each named range."""
    dates = pd.to_datetime(df[date_col])
    train = df.loc[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test = df.loc[(dates >= fold.test_start) & (dates <= fold.test_end)]
    return train, test
