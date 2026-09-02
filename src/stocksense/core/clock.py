"""IST time, the trading session, and the market calendar.

Everything in this project is single-exchange and single-timezone, so timestamps
are stored and reasoned about as IST wall-clock -- the same thing Upstox returns
and the same thing a trader reads off a screen. No UTC conversion anywhere.

The one trap this module exists to prevent: `LAG(close)` per symbol is NOT
"yesterday". An illiquid NSE name can skip days entirely, so any day-over-day
computation must join against the real trading calendar (`trading_days`) rather
than rely on row adjacency. That bug silently fabricated multi-week "one-day
returns" in a previous build.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# NSE equity session.
PRE_OPEN_START = time(9, 0)
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def session_bounds(d: date) -> tuple[datetime, datetime]:
    """(open, close) as timezone-aware IST datetimes for a given date."""
    return (
        datetime.combine(d, SESSION_OPEN, tzinfo=IST),
        datetime.combine(d, SESSION_CLOSE, tzinfo=IST),
    )


def in_session(ts: datetime | None = None) -> bool:
    ts = ts or now_ist()
    if is_weekend(ts.date()):
        return False
    return SESSION_OPEN <= ts.timetz().replace(tzinfo=None) <= SESSION_CLOSE


def trading_days(store, start: date, end: date) -> list[date]:
    """The REAL trading calendar, read from ingested bhavcopy dates.

    Deliberately derived from data rather than from a holiday list: a hardcoded
    holiday table drifts and is wrong for exactly the historical dates research
    depends on. If bhavcopy has rows for a date, the market traded that date.
    """
    rows = store.con.execute(
        "SELECT DISTINCT date FROM bhavcopy_eq WHERE date BETWEEN ? AND ? ORDER BY date",
        [start, end],
    ).fetchall()
    return [r[0] for r in rows]
