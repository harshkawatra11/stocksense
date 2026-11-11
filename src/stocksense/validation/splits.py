from __future__ import annotations

import calendar as _pycal
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Settings
from stocksense.validation.lockbox import research_end

_HALF_LIFE_SESSIONS = 756  # three years of sessions


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: test = [test_start, test_end]; train_end = last session before test_start."""

    name: str
    train_end: date
    test_start: date
    test_end: date


def walk_forward_folds(settings: Settings, calendar: TradingCalendar) -> list[Fold]:
    """Calendar-year folds from wf_test_start to the research end (the last test ends at research_end)."""
    end = research_end(settings)
    folds = []
    for year in range(settings.research.wf_test_start.year, end.year + 1):
        test_start = max(date(year, 1, 1), settings.research.wf_test_start)
        test_end = min(date(year, 12, 31), end)
        folds.append(Fold(str(year), calendar.prev_session(test_start), test_start, test_end))
    return folds


def _first_session_on_or_after(calendar: TradingCalendar, d: date) -> date:
    return calendar.next_session(d - timedelta(days=1))


def train_mask(
    label_end: pl.Series,
    fold: Fold,
    embargo_sessions: int = 5,
    *,
    calendar: TradingCalendar,
    dates: pl.Series | None = None,
    dev_start: date | None = None,
) -> pl.Series:
    """Boolean mask of training rows: label end strictly before the session `embargo_sessions` before the
    first test session, and (when given) date >= dev_start."""
    cutoff = calendar.shift(_first_session_on_or_after(calendar, fold.test_start), -embargo_sessions)
    mask = label_end < cutoff
    if dates is not None and dev_start is not None:
        mask = mask & (dates >= dev_start)
    return mask.fill_null(False)


def sample_weights(dates: pl.Series, fold: Fold, calendar: TradingCalendar) -> np.ndarray:
    """0.5 ** (sessions back from the first test session / 756): 1.0 at the boundary, 0.5 at 3 years."""
    s0 = _first_session_on_or_after(calendar, fold.test_start)
    i0 = calendar.session_index(s0)
    uniq = dates.unique().sort().to_list()
    idx = {d: calendar.session_index(d) for d in uniq}
    gap = np.array([i0 - idx[d] for d in dates.to_list()], dtype=float)
    return 0.5 ** (gap / _HALF_LIFE_SESSIONS)


def _minus_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + (d.month - 1) - months, 12)
    return date(y, m + 1, min(d.day, _pycal.monthrange(y, m + 1)[1]))


def inner_validation_split(
    dates: pl.Series,
    label_end: pl.Series,
    months: int = 12,
    embargo_sessions: int = 5,
    *,
    calendar: TradingCalendar,
) -> tuple[pl.Series, pl.Series]:
    """(inner_train_mask, inner_val_mask): validation = the last `months` months of the given training
    rows; inner-train rows end their labels before the validation start minus the embargo."""
    last = dates.max()
    assert isinstance(last, date)
    val_start = _minus_months(last + timedelta(days=1), months)
    cutoff = calendar.shift(_first_session_on_or_after(calendar, val_start), -embargo_sessions)
    val = dates >= val_start
    train = (dates < val_start) & (label_end < cutoff)
    return train.fill_null(False), val


def cscv_blocks(n_rows: int, n_blocks: int = 16) -> list[tuple[int, int]]:
    """`n_blocks` contiguous (start, stop) ranges covering [0, n_rows); earlier blocks may be 1 row longer."""
    base, extra = divmod(n_rows, n_blocks)
    out, start = [], 0
    for i in range(n_blocks):
        stop = start + base + (1 if i < extra else 0)
        out.append((start, stop))
        start = stop
    return out
