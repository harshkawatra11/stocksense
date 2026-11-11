from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class Clock(Protocol):
    def now(self) -> datetime:
        """Current tz-aware IST datetime."""
        ...

    def today(self) -> date:
        """Current IST calendar date."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait (or pretend to) for the given number of seconds."""
        ...


def _require_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("clock datetimes must be timezone-aware")
    return dt.astimezone(IST)


class SystemClock:
    """The real wall clock."""

    def now(self) -> datetime:
        return datetime.now(IST)

    def today(self) -> date:
        return self.now().date()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FixedClock:
    """A clock that always returns the same instant; sleep is a no-op."""

    def __init__(self, dt: datetime) -> None:
        self._dt = _require_aware(dt)

    def now(self) -> datetime:
        return self._dt

    def today(self) -> date:
        return self._dt.date()

    def sleep(self, seconds: float) -> None:
        return None


class SteppingClock:
    """A fake clock that moves only via sleep() or advance(); sleep never waits."""

    def __init__(self, start: datetime, step: timedelta = timedelta(seconds=60)) -> None:
        self._dt = _require_aware(start)
        self.step = step

    def now(self) -> datetime:
        return self._dt

    def today(self) -> date:
        return self._dt.date()

    def sleep(self, seconds: float) -> None:
        self._dt += timedelta(seconds=seconds)

    def advance(self, td: timedelta) -> None:
        self._dt += td
