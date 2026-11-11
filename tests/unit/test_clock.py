from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from stocksense.core.clock import IST, FixedClock, SteppingClock, SystemClock


def test_fixed_clock_constant():
    dt = datetime(2026, 11, 12, 10, 0, tzinfo=IST)
    c = FixedClock(dt)
    assert c.now() == dt and c.now() == dt
    assert c.today() == date(2026, 11, 12)
    c.sleep(100)  # no-op, no waiting
    assert c.now() == dt


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        FixedClock(datetime(2026, 11, 12, 10, 0))
    with pytest.raises(ValueError):
        SteppingClock(datetime(2026, 11, 12, 10, 0))


def test_stepping_clock_sleep_advances_exactly():
    start = datetime(2026, 11, 12, 10, 0, tzinfo=IST)
    c = SteppingClock(start)
    assert c.now() == start
    c.sleep(90)
    # 10:00:00 + 90 s = 10:01:30
    assert c.now() == datetime(2026, 11, 12, 10, 1, 30, tzinfo=IST)
    c.advance(timedelta(hours=1))
    assert c.now() == datetime(2026, 11, 12, 11, 1, 30, tzinfo=IST)


def test_stepping_clock_default_step_is_60s_per_now_call_semantics():
    # Spec: each now() call returns the current value; time moves only via sleep/advance.
    start = datetime(2026, 11, 12, 10, 0, tzinfo=IST)
    c = SteppingClock(start)
    assert c.now() == c.now() == start
    assert c.step == timedelta(seconds=60)


def test_system_clock_is_aware_ist():
    n = SystemClock().now()
    assert n.tzinfo is not None and n.utcoffset() == timedelta(hours=5, minutes=30)
