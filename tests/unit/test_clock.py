"""Tests for core.clock.

trading_days is the anti-adjacency-fallacy tool used everywhere a day-over-day
computation must know whether two rows are genuinely consecutive sessions.
Regression: it used to query bhavcopy_eq as a live DuckDB table, which stopped
existing once storage split into Store (writer) + Reader (lock-free Parquet) --
it must go through Reader.sql() like every other read.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone, timedelta

import pandas as pd
import pytest

from stocksense.core.clock import (
    IST,
    in_session,
    is_weekend,
    session_bounds,
    today_ist,
    trading_days,
)
from stocksense.data.store import Reader, Store


def _bhav_row(symbol: str, d: date) -> dict:
    return dict(
        symbol=symbol, series="EQ", date=d, open=100.0, high=101.0, low=99.0,
        close=100.5, prev_close=100.0, last_price=100.5, volume=1000.0,
        turnover_inr=100_500.0, n_trades=10.0, era="udiff",
    )


@pytest.fixture()
def reader(tmp_path):
    db, pq = tmp_path / "hot.duckdb", tmp_path / "parquet"
    # NSE never trades weekends -- only weekday dates are seeded, mirroring
    # what a real backfill produces.
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9)]  # Tue-Fri, then Mon-Tue
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame([_bhav_row("X", d) for d in dates]))
    r = Reader(pq)
    yield r
    r.close()


def test_trading_days_reads_through_the_lock_free_reader(reader):
    """Uses Reader.sql(), not a raw DuckDB table -- proven by the fact this
    works at all against Parquet-only storage."""
    days = trading_days(reader, date(2024, 1, 1), date(2024, 1, 10))
    assert days == [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9)]


def test_trading_days_excludes_dates_outside_the_range(reader):
    days = trading_days(reader, date(2024, 1, 3), date(2024, 1, 4))
    assert days == [date(2024, 1, 3), date(2024, 1, 4)]


def test_trading_days_returns_plain_date_objects_not_timestamps(reader):
    """The dtype trap this codebase has been bitten by before: a Timestamp
    leaking out breaks `date == Timestamp` comparisons downstream."""
    days = trading_days(reader, date(2024, 1, 1), date(2024, 1, 10))
    assert all(type(d) is date for d in days)


def test_trading_days_on_empty_data_returns_empty_list(tmp_path):
    _, pq = tmp_path / "parquet", tmp_path / "parquet"
    r = Reader(tmp_path / "empty_parquet")
    assert trading_days(r, date(2024, 1, 1), date(2024, 1, 10)) == []
    r.close()


def test_trading_days_skips_weekends_because_no_data_exists_for_them(reader):
    """2024-01-06/07 are a Sat/Sun -- absent from the seeded data, so they
    must be absent from the calendar too (never assumed present or absent by
    a hardcoded weekday rule)."""
    days = trading_days(reader, date(2024, 1, 6), date(2024, 1, 7))
    assert days == []


# ------------------------------------------------------------- IST/session
def test_ist_is_utc_plus_5_30():
    ts = datetime(2024, 1, 1, 12, 0, tzinfo=IST)
    assert ts.utcoffset() == timedelta(hours=5, minutes=30)


def test_is_weekend_correct_for_known_dates():
    assert is_weekend(date(2024, 1, 6)) is True   # Saturday
    assert is_weekend(date(2024, 1, 7)) is True   # Sunday
    assert is_weekend(date(2024, 1, 8)) is False  # Monday


def test_session_bounds_are_915_to_1530_ist():
    open_dt, close_dt = session_bounds(date(2024, 1, 8))
    assert open_dt.time() == time(9, 15)
    assert close_dt.time() == time(15, 30)
    assert open_dt.tzinfo == IST


def test_in_session_false_on_a_weekend():
    weekend_ts = datetime(2024, 1, 6, 12, 0, tzinfo=IST)  # Saturday, mid-day
    assert in_session(weekend_ts) is False


def test_in_session_true_inside_market_hours_on_a_weekday():
    ts = datetime(2024, 1, 8, 11, 0, tzinfo=IST)  # Monday, 11:00
    assert in_session(ts) is True


def test_in_session_false_before_open_and_after_close():
    before = datetime(2024, 1, 8, 9, 0, tzinfo=IST)
    after = datetime(2024, 1, 8, 16, 0, tzinfo=IST)
    assert in_session(before) is False
    assert in_session(after) is False


def test_today_ist_returns_a_date():
    assert isinstance(today_ist(), date)
