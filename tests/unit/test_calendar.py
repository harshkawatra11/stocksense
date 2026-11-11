from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST
from stocksense.core.config import SpecialSession


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# 600 weekday sessions from 2018-01-01 plus one Saturday special at index 120 (as in the synth market)
_BASE = _weekdays(date(2018, 1, 1), 600)
_SAT = _BASE[119] + timedelta(days=1)  # Saturday right after the 120th weekday (a Friday)
_PAST = sorted([*_BASE, _SAT])
CAL = TradingCalendar.from_dates(_PAST)


def test_saturday_special_session_is_a_session():
    friday = _SAT - timedelta(days=1)
    assert _SAT.weekday() == 5 and friday.weekday() == 4
    assert CAL.is_session(_SAT)
    assert CAL.next_session(friday) == _SAT
    # next after the Saturday is the following Monday (2 days later)
    assert CAL.next_session(_SAT) == _SAT + timedelta(days=2)


def test_non_sessions():
    # a Sunday inside the range, and a date before the first session
    assert not CAL.is_session(date(2018, 1, 7))
    assert not CAL.is_session(date(2017, 12, 29))


@given(st.integers(0, len(_PAST) - 1), st.integers(1, 30))
@settings(max_examples=200, deadline=None)
def test_shift_roundtrip(i, n):
    d = _PAST[i]
    if i + n < len(_PAST):
        assert CAL.shift(CAL.shift(d, n), -n) == d
    if i - n >= 0:
        assert CAL.shift(CAL.shift(d, -n), n) == d
        assert CAL.shift(d, -n) == _PAST[i - n]  # exact index arithmetic


def test_shift_exact_and_index():
    assert CAL.session_index(_PAST[0]) == 0
    assert CAL.session_index(_PAST[10]) == 10
    assert CAL.shift(_PAST[10], 5) == _PAST[15]


def test_prev_of_next_is_identity():
    for d in _PAST[:-1]:
        assert CAL.prev_session(CAL.next_session(d)) == d


def test_sessions_range_inclusive_and_reversed():
    a, b = _PAST[3], _PAST[7]
    assert CAL.sessions(a, b) == _PAST[3:8]
    assert CAL.sessions(b, a) == []


def test_shift_on_non_session_raises():
    with pytest.raises(ValueError):
        CAL.shift(date(2018, 1, 7), 1)
    with pytest.raises(ValueError):
        CAL.session_index(date(2018, 1, 7))


def test_future_weekdays_are_sessions_and_holidays_skipped():
    past = [date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)]  # Mon Tue Wed
    cal = TradingCalendar.from_dates(past, holidays=[date(2026, 11, 13)])  # Fri holiday
    assert cal.is_session(date(2026, 11, 12))
    assert not cal.is_session(date(2026, 11, 13))
    assert not cal.is_session(date(2026, 11, 14))  # Saturday
    # Thu 12 -> next is Mon 16 (Fri holiday, weekend)
    assert cal.next_session(date(2026, 11, 12)) == date(2026, 11, 16)
    assert cal.sessions(date(2026, 11, 11), date(2026, 11, 17)) == [
        date(2026, 11, 11),
        date(2026, 11, 12),
        date(2026, 11, 16),
        date(2026, 11, 17),
    ]
    assert cal.shift(date(2026, 11, 11), 2) == date(2026, 11, 16)
    assert cal.shift(date(2026, 11, 16), -2) == date(2026, 11, 11)


def test_future_special_session_and_times():
    past = [date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)]
    sat = date(2026, 11, 14)
    muhurat = SpecialSession(date=sat, open=time(18, 15), close=time(19, 15), tradable=False)
    cal = TradingCalendar.from_dates(past, special_sessions=[muhurat])
    assert cal.is_session(sat)
    assert cal.session_times(sat) == (
        datetime(2026, 11, 14, 18, 15, tzinfo=IST),
        datetime(2026, 11, 14, 19, 15, tzinfo=IST),
    )
    assert cal.is_tradable(sat) is False
    assert cal.session_times(date(2026, 11, 11)) == (
        datetime(2026, 11, 11, 9, 15, tzinfo=IST),
        datetime(2026, 11, 11, 15, 30, tzinfo=IST),
    )
    with pytest.raises(ValueError):
        cal.session_times(date(2026, 11, 15))


def test_last_completed_session_boundary():
    d, prev = _PAST[50], _PAST[49]
    assert CAL.last_completed_session(datetime.combine(d, time(15, 29), IST)) == prev
    assert CAL.last_completed_session(datetime.combine(d, time(15, 30), IST)) == d
    # on a non-session day (Sunday) the latest session is the previous Friday/Saturday session
    sun = date(2018, 1, 14)
    assert CAL.last_completed_session(datetime.combine(sun, time(12, 0), IST)) == date(2018, 1, 12)


def test_from_store_missing_raises(tmp_data_root, fixed_clock):
    from stocksense.core.errors import DataNotAvailable

    with pytest.raises(DataNotAvailable):
        TradingCalendar.from_store(tmp_data_root, fixed_clock)


def test_from_store_reads_mixed_timestamp_units(tmp_data_root, fixed_clock):
    import polars as pl

    d = tmp_data_root.paths.data_root / "parquet" / "bhavcopy_eq"
    d.mkdir(parents=True)
    pl.DataFrame({"date": [datetime(2026, 11, 9), datetime(2026, 11, 9)]}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    ).write_parquet(d / "part-2026-11.parquet")
    pl.DataFrame({"date": [datetime(2026, 11, 10)]}).write_parquet(d / "part-2026-12.parquet")  # us
    cal = TradingCalendar.from_store(tmp_data_root, fixed_clock)
    assert cal.is_session(date(2026, 11, 9)) and cal.is_session(date(2026, 11, 10))
    assert cal.session_index(date(2026, 11, 10)) == 1


@pytest.mark.realdata
def test_real_store_calendar():
    from stocksense.core.clock import FixedClock
    from stocksense.core.config import load_settings

    s = load_settings()
    cal = TradingCalendar.from_store(s, FixedClock(datetime(2026, 11, 12, 10, 0, tzinfo=IST)))
    # PLAN B.2: 4,135 sessions, 2010-01-04 .. 2026-09-29
    assert cal.session_index(date(2026, 9, 29)) == 4134
    assert cal.sessions(date(2010, 1, 1), date(2026, 9, 29))[0] == date(2010, 1, 4)
    assert len(cal.sessions(date(2010, 1, 1), date(2026, 9, 29))) == 4135
    assert not cal.is_session(date(2026, 1, 26))  # Republic Day
