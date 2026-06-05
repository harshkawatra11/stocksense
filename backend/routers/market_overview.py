"""
Market Overview endpoint — top movers, sector summary, signal stats from DB.
"""
import logging
from datetime import date, timedelta

import asyncpg
from fastapi import APIRouter

from config import settings

router = APIRouter()
log = logging.getLogger(__name__)
DB_DSN = settings.DATABASE_DSN


@router.get("/overview")
async def get_market_overview():
    """
    Returns:
    - top_buy_signals: top BUY signals by confidence (last 7 days)
    - top_sell_signals: top SELL signals by confidence (last 7 days)
    - sector_summary: signal count + accuracy per sector
    - stats: overall signal counts + accuracy
    - top_movers: biggest % movers from ohlcv_daily (yesterday vs day before)
    """
    since = date.today() - timedelta(days=7)
    conn = await asyncpg.connect(DB_DSN)
    try:
        buy_rows = await conn.fetch(
            """
            SELECT s.ticker, s.signal_type, s.price_at_signal::float,
                   s.target_price::float, s.stop_loss::float,
                   s.final_confidence::float, s.fired_at::date as date,
                   st.sector
            FROM signals s
            LEFT JOIN stocks st ON st.ticker = s.ticker
            WHERE s.signal_type = 'BUY' AND s.fired_at::date >= $1
            ORDER BY s.final_confidence DESC NULLS LAST
            LIMIT 10
            """,
            since,
        )

        sell_rows = await conn.fetch(
            """
            SELECT s.ticker, s.signal_type, s.price_at_signal::float,
                   s.target_price::float, s.stop_loss::float,
                   s.final_confidence::float, s.fired_at::date as date,
                   st.sector
            FROM signals s
            LEFT JOIN stocks st ON st.ticker = s.ticker
            WHERE s.signal_type = 'SELL' AND s.fired_at::date >= $1
            ORDER BY s.final_confidence DESC NULLS LAST
            LIMIT 10
            """,
            since,
        )

        sector_rows = await conn.fetch(
            """
            SELECT st.sector,
                   COUNT(*) as total,
                   COUNT(*) FILTER (WHERE s.signal_type = 'BUY') as buys,
                   COUNT(*) FILTER (WHERE s.signal_type = 'SELL') as sells,
                   ROUND(AVG(s.final_confidence)::numeric, 3)::float as avg_conf
            FROM signals s
            JOIN stocks st ON st.ticker = s.ticker
            WHERE s.fired_at::date >= $1
              AND st.sector IS NOT NULL AND st.sector != ''
            GROUP BY st.sector
            ORDER BY total DESC
            LIMIT 15
            """,
            since,
        )

        stats_row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE signal_type = 'BUY') as total_buys,
                COUNT(*) FILTER (WHERE signal_type = 'SELL') as total_sells,
                COUNT(*) FILTER (WHERE signal_type = 'HOLD') as total_holds,
                COUNT(*) as total,
                ROUND(AVG(final_confidence)::numeric, 3)::float as avg_conf
            FROM signals
            WHERE fired_at::date >= $1
            """,
            since,
        )

        # Top movers — latest available trading day vs previous
        movers_rows = await conn.fetch(
            """
            WITH latest_two AS (
                SELECT ticker, time::date as d, close::float,
                       LAG(close::float) OVER (PARTITION BY ticker ORDER BY time) as prev_close
                FROM ohlcv_daily
                WHERE time >= (
                    SELECT MAX(time) - INTERVAL '5 days' FROM ohlcv_daily
                )
            ),
            latest AS (
                SELECT ticker, d, close, prev_close,
                       CASE WHEN prev_close > 0
                            THEN ROUND(((close - prev_close) / prev_close * 100)::numeric, 2)::float
                            ELSE 0 END as chg_pct
                FROM latest_two
                WHERE d = (SELECT MAX(d) FROM latest_two)
                  AND prev_close IS NOT NULL AND prev_close > 0
            )
            (SELECT ticker, close, chg_pct, 'gainer' as mover_type FROM latest ORDER BY chg_pct DESC LIMIT 5)
            UNION ALL
            (SELECT ticker, close, chg_pct, 'loser' as mover_type FROM latest ORDER BY chg_pct ASC LIMIT 5)
            """
        )

    finally:
        await conn.close()

    return {
        "top_buy_signals": [dict(r) for r in buy_rows],
        "top_sell_signals": [dict(r) for r in sell_rows],
        "sector_summary": [dict(r) for r in sector_rows],
        "stats": dict(stats_row) if stats_row else {},
        "top_movers": [dict(r) for r in movers_rows],
        "period_days": 7,
    }
