"""
OHLCV data endpoints — serves chart data from TimescaleDB.
"""
import logging
from datetime import date, timedelta

import asyncpg
from fastapi import APIRouter, Query, HTTPException

from config import settings

router = APIRouter()
log = logging.getLogger(__name__)

DB_DSN = settings.DATABASE_DSN


@router.get("/{ticker}")
async def get_ohlcv(
    ticker: str,
    days: int = Query(365, ge=30, le=3650),
):
    """Return daily OHLCV for a ticker. Default last 365 days."""
    ticker = ticker.upper()
    since = date.today() - timedelta(days=days)
    try:
        conn = await asyncpg.connect(DB_DSN)
        rows = await conn.fetch(
            """
            SELECT time::date as date,
                   open::float, high::float, low::float,
                   close::float, volume::bigint
            FROM ohlcv_daily
            WHERE ticker = $1 AND time >= $2
            ORDER BY time ASC
            """,
            ticker, since,
        )
        await conn.close()
    except Exception as e:
        log.error("OHLCV fetch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    return {
        "ticker": ticker,
        "count": len(rows),
        "candles": [
            {
                "time": str(r["date"]),
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
            }
            for r in rows
        ],
    }


@router.get("/{ticker}/signals")
async def get_ticker_signals(ticker: str, limit: int = Query(20, le=100)):
    """Return recent signals for a ticker to overlay on the chart."""
    ticker = ticker.upper()
    try:
        conn = await asyncpg.connect(DB_DSN)
        rows = await conn.fetch(
            """
            SELECT fired_at::date as date, signal_type, price_at_signal::float,
                   target_price::float, stop_loss::float, final_confidence::float
            FROM signals
            WHERE ticker = $1
            ORDER BY fired_at DESC
            LIMIT $2
            """,
            ticker, limit,
        )
        await conn.close()
    except Exception as e:
        log.error("Ticker signals fetch error: %s", e)
        return []

    return [dict(r) for r in rows]
