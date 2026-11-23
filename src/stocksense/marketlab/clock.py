"""Ported from market-lab sm/market.py (MIT, ba3f771); changes: our TradingCalendar and Clock instead of
holiday-master files and datetime.now."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, Clock

Status = Literal["closed", "pre-open", "open"]

_PREOPEN_START = 9 * 60  # 09:00
_OPEN = 9 * 60 + 15  # 09:15
_CLOSE = 15 * 60 + 30  # 15:30


def market_status(clock: Clock, cal: TradingCalendar, t: datetime | None = None) -> Status:
    """Status: pre-open 09:00-09:14, open 09:15-15:29, else closed (sessions from our calendar)."""
    t = (t or clock.now()).astimezone(IST)
    if not cal.is_session(t.date()):
        return "closed"
    m = t.hour * 60 + t.minute
    if m < _OPEN:
        return "pre-open" if m >= _PREOPEN_START else "closed"
    return "open" if m < _CLOSE else "closed"


def minute_now(clock: Clock) -> int:
    """Minutes since 09:15 IST (0 at 09:15, 375 at 15:30); negative before the open."""
    t = clock.now().astimezone(IST)
    return t.hour * 60 + t.minute - _OPEN


def session_minute(ts_epoch_s: int) -> int:
    """Minutes since 09:15 IST of an epoch-second timestamp, as the Market Lab page computes it."""
    return int(((ts_epoch_s + 19800) % 86400) // 60) - _OPEN
