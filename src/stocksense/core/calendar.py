from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path

import polars as pl

from stocksense.core.clock import IST, Clock
from stocksense.core.config import Settings, SpecialSession
from stocksense.core.errors import DataNotAvailable

_log = logging.getLogger(__name__)

_DEFAULT_OPEN = time(9, 15)
_DEFAULT_CLOSE = time(15, 30)
_ONE_DAY = timedelta(days=1)
_MAX_WALK_DAYS = 3660  # safety bound when stepping over non-sessions


def _read_dates(folder: Path) -> list[date]:
    """Distinct values of the `date` column across a dataset's parquet files, as dates.

    TODO(T1.6): replace with data/io.py once it exists. Files are scanned one at a time and only
    the `date` column is read, because old partitions mix timestamp units (ms vs us).
    """
    out: set[date] = set()
    for f in sorted(folder.glob("*.parquet")):
        col = pl.scan_parquet(f).select(pl.col("date").cast(pl.Date).unique()).collect()["date"]
        out.update(d for d in col.to_list() if d is not None)
    return sorted(out)


class TradingCalendar:
    """Immutable NSE cash-market session calendar: stored past sessions plus a projected future."""

    def __init__(
        self,
        past_sessions: list[date],
        holidays: frozenset[date],
        specials: dict[date, SpecialSession],
        tz: tzinfo,
    ) -> None:
        self._past = past_sessions
        self._past_set = frozenset(past_sessions)
        self._holidays = holidays
        self._specials = specials
        self._tz = tz

    @classmethod
    def from_dates(
        cls,
        past_sessions: Iterable[date],
        holidays: Iterable[date] = (),
        special_sessions: Iterable[SpecialSession] = (),
        tz: tzinfo = IST,
    ) -> TradingCalendar:
        """Build from explicit past session dates (the truth), future holidays and special sessions."""
        past = sorted(set(past_sessions))
        if not past:
            raise ValueError("calendar needs at least one past session")
        return cls(past, frozenset(holidays), {s.date: s for s in special_sessions}, tz)

    @classmethod
    def from_store(cls, settings: Settings, clock: Clock) -> TradingCalendar:
        """Build from distinct dates in the bhavcopy_eq store plus config/schedule.toml specials."""
        parquet = settings.paths.data_root / "parquet"
        eq = parquet / "bhavcopy_eq"
        if not eq.is_dir() or not any(eq.glob("*.parquet")):
            raise DataNotAvailable(f"no bhavcopy_eq data under {eq}")
        past = _read_dates(eq)
        hol_dir = parquet / "holidays"
        if hol_dir.is_dir() and any(hol_dir.glob("*.parquet")):
            holidays = _read_dates(hol_dir)
        else:
            holidays = []
            _log.warning("holidays dataset missing: treating all future weekdays as sessions")
        return cls.from_dates(past, holidays, settings.schedule.special_sessions, tz=_tz_of(settings))

    # ---- membership -------------------------------------------------------------------------

    def is_session(self, d: date) -> bool:
        """True if NSE's cash market traded (past) or is expected to trade (future) on d."""
        if d <= self._past[-1]:
            return d in self._past_set
        if d in self._specials:
            return True
        return d.weekday() < 5 and d not in self._holidays

    def is_tradable(self, d: date) -> bool:
        """False for sessions the daemon must not trade (flagged special sessions such as Muhurat)."""
        self._require_session(d)
        s = self._specials.get(d)
        return True if s is None else s.tradable

    def _require_session(self, d: date) -> None:
        if not self.is_session(d):
            raise ValueError(f"{d} is not a session")

    # ---- navigation -------------------------------------------------------------------------

    def next_session(self, d: date) -> date:
        """First session strictly after d."""
        i = bisect_right(self._past, d)
        if i < len(self._past):
            return self._past[i]
        cur = max(d, self._past[-1]) + _ONE_DAY
        for _ in range(_MAX_WALK_DAYS):
            if self.is_session(cur):
                return cur
            cur += _ONE_DAY
        raise ValueError(f"no session found after {d}")

    def prev_session(self, d: date) -> date:
        """Last session strictly before d."""
        if d > self._past[-1]:
            cur = d - _ONE_DAY
            while cur > self._past[-1]:
                if self.is_session(cur):
                    return cur
                cur -= _ONE_DAY
            return self._past[-1]
        i = bisect_left(self._past, d)
        if i == 0:
            raise ValueError(f"no session before {d}")
        return self._past[i - 1]

    def sessions(self, a: date, b: date) -> list[date]:
        """All sessions in [a, b] inclusive; empty if a > b."""
        if a > b:
            return []
        out = self._past[bisect_left(self._past, a) : bisect_right(self._past, b)]
        cur = max(a, self._past[-1] + _ONE_DAY)
        while cur <= b:
            if self.is_session(cur):
                out.append(cur)
            cur += _ONE_DAY
        return out

    def shift(self, d: date, n: int) -> date:
        """The session n sessions after (n>0) or before (n<0) session d."""
        self._require_session(d)
        step = self.next_session if n >= 0 else self.prev_session
        for _ in range(abs(n)):
            d = step(d)
        return d

    def session_index(self, d: date) -> int:
        """0-based position of session d in the calendar (past sessions first, then projected)."""
        self._require_session(d)
        if d <= self._past[-1]:
            return bisect_left(self._past, d)
        return len(self._past) + len(self.sessions(self._past[-1] + _ONE_DAY, d)) - 1

    # ---- time -------------------------------------------------------------------------------

    def session_times(self, d: date) -> tuple[datetime, datetime]:
        """(open, close) tz-aware datetimes of session d (09:15-15:30 unless a special session)."""
        self._require_session(d)
        s = self._specials.get(d)
        o, c = (s.open, s.close) if s is not None else (_DEFAULT_OPEN, _DEFAULT_CLOSE)
        return datetime.combine(d, o, self._tz), datetime.combine(d, c, self._tz)

    def last_completed_session(self, now: datetime) -> date:
        """The latest session whose close time is <= now."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        d = now.astimezone(self._tz).date()
        while d >= self._past[0]:
            if self.is_session(d) and self.session_times(d)[1] <= now:
                return d
            d -= _ONE_DAY
        raise ValueError(f"no completed session before {now}")


def _tz_of(settings: Settings) -> tzinfo:
    from zoneinfo import ZoneInfo

    return ZoneInfo(settings.time.tz)
