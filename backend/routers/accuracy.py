from fastapi import APIRouter, Query
import asyncpg
import os
from config import settings
from intelligence.accuracy_tracker import compute_net_expectancy

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


@router.get("/summary")
async def accuracy_summary():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT
            COUNT(*) FILTER (WHERE status != 'active') as total_resolved,
            COUNT(*) FILTER (WHERE status = 'hit_target') as correct,
            COUNT(*) FILTER (WHERE signal_type = 'BUY') as total_buys,
            COUNT(*) FILTER (WHERE signal_type = 'SELL') as total_sells,
            AVG(final_confidence) as avg_confidence,
            COUNT(*) FILTER (
                WHERE (fired_at AT TIME ZONE 'Asia/Kolkata')::date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date
            ) as today_total
        FROM signals
        """
    )
    await conn.close()
    r = rows[0]
    total = r["total_resolved"] or 1
    return {
        "total_resolved": r["total_resolved"],
        "correct": r["correct"],
        "accuracy_pct": round((r["correct"] or 0) / total * 100, 2),
        "total_buys": r["total_buys"],
        "total_sells": r["total_sells"],
        "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
        "today_total": r["today_total"],
    }


@router.get("/net-expectancy")
async def net_expectancy(n: int = Query(60, le=1000)):
    """
    Net (cost-adjusted) expectancy over the trailing `n` resolved signals —
    subtracts intelligence.costs.ROUND_TRIP_COST_PCT from every realized
    outcome before averaging. This is the honest complement to /summary's
    raw win rate; see intelligence/accuracy_tracker.compute_net_expectancy.
    """
    conn = await asyncpg.connect(DB_DSN)
    try:
        return await compute_net_expectancy(conn, n=n)
    finally:
        await conn.close()


@router.get("/by-model")
async def accuracy_by_model():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT model_name, period_start, period_end,
               total_signals, correct_signals, accuracy, avg_confidence
        FROM model_accuracy
        ORDER BY period_start DESC
        LIMIT 100
        """
    )
    await conn.close()
    return [dict(r) for r in rows]


@router.get("/daily")
async def daily_accuracy():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT
            (fired_at AT TIME ZONE 'Asia/Kolkata')::date as date,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'hit_target') as correct,
            ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'hit_target') / NULLIF(COUNT(*),0), 1) as accuracy_pct
        FROM signals
        WHERE status != 'active'
        GROUP BY (fired_at AT TIME ZONE 'Asia/Kolkata')::date
        ORDER BY date DESC
        LIMIT 30
        """
    )
    await conn.close()
    return [dict(r) for r in rows]
