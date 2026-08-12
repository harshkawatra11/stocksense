from __future__ import annotations

import pandas as pd

from stocksense.core.calendar import bar_shift, trading_days_index


def test_trading_days_index_dedupes_and_sorts() -> None:
    dates = pd.Series(["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-05"])
    idx = trading_days_index(dates)
    assert list(idx) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]))


def test_trading_days_index_derived_empirically_not_calendar() -> None:
    # A weekend or holiday with no observed data is simply absent — no
    # hardcoded holiday calendar involved.
    dates = pd.Series(["2024-01-01", "2024-01-02"])  # skips a gap deliberately
    idx = trading_days_index(dates)
    assert pd.Timestamp("2024-01-01") in idx
    assert len(idx) == 2


def test_bar_shift_forward() -> None:
    idx = trading_days_index(pd.Series(pd.bdate_range("2024-01-01", periods=10)))
    shifted = bar_shift(idx, idx[2], bars=3)
    assert shifted == idx[5]


def test_bar_shift_backward() -> None:
    idx = trading_days_index(pd.Series(pd.bdate_range("2024-01-01", periods=10)))
    shifted = bar_shift(idx, idx[5], bars=-3)
    assert shifted == idx[2]


def test_bar_shift_out_of_range_returns_none_not_nearest() -> None:
    """Must never silently coerce to the nearest available date — that
    would quietly change the horizon a caller asked for."""
    idx = trading_days_index(pd.Series(pd.bdate_range("2024-01-01", periods=10)))
    assert bar_shift(idx, idx[8], bars=5) is None  # would run past the end
    assert bar_shift(idx, idx[1], bars=-5) is None  # would run before the start


def test_bar_shift_date_not_in_index_returns_none() -> None:
    idx = trading_days_index(pd.Series(pd.bdate_range("2024-01-01", periods=10)))
    assert bar_shift(idx, pd.Timestamp("2024-06-01"), bars=1) is None
