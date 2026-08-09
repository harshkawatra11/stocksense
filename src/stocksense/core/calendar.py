"""
Trading calendar and bar-indexing discipline.

Everything in this codebase that says "N days" actually means "N trading
bars." This module is the single place that converts between calendar time
and bar-index time, so a leakage bug caused by mixing the two has exactly
one place it could have been introduced.
"""

from __future__ import annotations

import pandas as pd


def trading_days_index(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sorted, deduplicated trading-day index from observed data.

    We do not hardcode an NSE holiday calendar — the trading calendar is
    derived empirically from which dates actually have data. This is more
    robust than a maintained holiday list going stale, and it is exactly
    correct by construction: a day with no observed candles is a day
    nothing could have traded.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    return idx.unique().sort_values()


def bar_shift(index: pd.DatetimeIndex, date: pd.Timestamp, bars: int) -> pd.Timestamp | None:
    """Shift `date` forward or backward by `bars` trading bars using `index`.

    Returns None if the shift falls outside the known calendar — callers
    must treat that as "insufficient future data," never coerce it to the
    nearest available date, which would silently change the horizon.
    """
    date = pd.Timestamp(date).normalize()
    pos = index.searchsorted(date)
    if pos >= len(index) or index[pos] != date:
        return None
    target = pos + bars
    if target < 0 or target >= len(index):
        return None
    return index[target]


def embargo_bars(horizon_bars: int, max_feature_lookback_bars: int) -> int:
    """Minimum gap (in bars) required between a training window's end and a
    test window's start, per docs/06-retraining-rigor.md ("purged, embargoed
    walk-forward validation"): the gap must cover both the label horizon
    (so no training label peeks into the test period) and the longest
    feature lookback (so no test-period feature bleeds from training data).
    """
    return horizon_bars + max_feature_lookback_bars
