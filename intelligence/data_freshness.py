"""
Data-freshness truth — answers "is this signal built on LIVE intraday data, or
PAST end-of-day daily closes?"

The pipeline always reads the latest candle in ``ohlcv_daily``. That candle is a
daily close refreshed once per evening (Angel One / NSE Bhavcopy). Until a live
intraday feed actually populates ``ohlcv_1min`` for *today*, every signal is
derived from a past daily close — and the UI should say so plainly.

Used by:
  - GET /api/live/data-status   (banner before a run)
  - run_pipeline_multi()        (stamped onto the run summary)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))


def _fmt_date(d) -> str:
    """'2026-06-05' style, tolerant of date/datetime.

    ohlcv_daily.time is stored as midnight-IST-in-UTC (trading day D is
    2026-07-(D-1) 18:30:00+00:00) — formatting the raw UTC value directly
    shows the wrong calendar date (D-1 instead of D). Convert to IST first.
    Found live 2026-07-13, same bug class already fixed in eod_review.py
    and market_runner.py's Upstox/Bhavcopy reconciliation.
    """
    try:
        if hasattr(d, "tzinfo"):
            dt = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            return dt.astimezone(_IST).strftime("%Y-%m-%d")
        return d.strftime("%Y-%m-%d")
    except Exception:
        return str(d)


async def get_data_freshness(conn) -> dict:
    """
    Inspect the freshest market data available and classify the run as
    ``live_intraday`` or ``eod_daily``.

    Returns a dict the frontend can render directly:
        {
          "latest_date": "2026-06-05",
          "age_hours": 18.4,
          "source": "eod_daily" | "live_intraday",
          "is_live": False,
          "label": "Based on PAST end-of-day data (last close: 2026-06-05)",
        }
    """
    latest = await conn.fetchval("SELECT MAX(time) FROM ohlcv_daily")

    # Has an intraday feed written any 1-min bars for today (UTC)? If so, and they
    # are recent, the run can legitimately call itself live. Today this is False
    # until the SmartWebSocket feed is wired — the honest default.
    intraday_latest = None
    try:
        intraday_latest = await conn.fetchval("SELECT MAX(time) FROM ohlcv_1min")
    except Exception:
        intraday_latest = None  # table may not exist yet

    now = datetime.now(timezone.utc)

    if latest is None:
        return {
            "latest_date": None,
            "age_hours": None,
            "source": "eod_daily",
            "is_live": False,
            "label": "No market data loaded yet — run the data pipeline first.",
        }

    # latest is stored as UTC-midnight of the close date.
    age_hours = round((now - latest.replace(tzinfo=timezone.utc)).total_seconds() / 3600, 1) \
        if latest.tzinfo is None else round((now - latest).total_seconds() / 3600, 1)

    is_live = False
    if intraday_latest is not None:
        intr = intraday_latest if intraday_latest.tzinfo else intraday_latest.replace(tzinfo=timezone.utc)
        # live only if a 1-min bar landed within the last 30 minutes
        if (now - intr).total_seconds() <= 1800:
            is_live = True

    if is_live:
        return {
            "latest_date": _fmt_date(latest),
            "age_hours": age_hours,
            "source": "live_intraday",
            "is_live": True,
            "label": "Live intraday data — signals reflect today's moving market.",
        }

    return {
        "latest_date": _fmt_date(latest),
        "age_hours": age_hours,
        "source": "eod_daily",
        "is_live": False,
        "label": f"Based on PAST end-of-day data (last close: {_fmt_date(latest)}). Not live intraday.",
    }
