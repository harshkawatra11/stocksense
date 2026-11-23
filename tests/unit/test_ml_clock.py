from __future__ import annotations

from datetime import date, datetime

import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.marketlab.clock import market_status, minute_now, session_minute

DAY = date(2026, 11, 12)  # a Thursday
CAL = TradingCalendar.from_dates([date(2026, 11, 10), date(2026, 11, 11)])


def at(h: int, m: int, d: date = DAY) -> FixedClock:
    return FixedClock(datetime(d.year, d.month, d.day, h, m, tzinfo=IST))


@pytest.mark.parametrize(
    ("h", "m", "want"),
    [
        (8, 59, "closed"),
        (9, 0, "pre-open"),
        (9, 14, "pre-open"),
        (9, 15, "open"),
        (15, 29, "open"),
        (15, 30, "closed"),
    ],
)
def test_status_boundaries(h, m, want):
    assert market_status(at(h, m), CAL) == want


def test_non_session_is_closed():
    sat = date(2026, 11, 14)
    assert market_status(at(10, 0, sat), CAL) == "closed"


def test_minute_now_and_session_minute():
    assert minute_now(at(9, 15)) == 0 and minute_now(at(15, 30)) == 375 and minute_now(at(9, 0)) == -15
    # 2025-11-16 04:15 UTC is 09:45 IST, 30 minutes after the open
    assert session_minute(1763266500) == 30
