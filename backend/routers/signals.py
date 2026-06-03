from fastapi import APIRouter, Query
from data.db.database import AsyncSessionLocal
from sqlalchemy import select, desc, text
from data.db.database import Signal, SignalReasoning
import asyncpg
import os

router = APIRouter()
DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")


@router.get("/recent")
async def get_recent_signals(limit: int = Query(50, le=200)):
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT s.id, s.ticker, s.signal_type, s.timeframe,
               s.price_at_signal, s.target_price, s.stop_loss,
               s.ml_confidence, s.kronos_confidence, s.slm_confidence,
               s.claude_confidence, s.final_confidence,
               s.status, s.fired_at, s.actual_close,
               st.name as stock_name, st.sector
        FROM signals s
        JOIN stocks st ON st.ticker = s.ticker
        ORDER BY s.fired_at DESC
        LIMIT $1
        """,
        limit,
    )
    await conn.close()
    return [dict(r) for r in rows]


@router.get("/{signal_id}/reasoning")
async def get_signal_reasoning(signal_id: int):
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT model_name, reasoning, raw_output, created_at
        FROM signal_reasoning
        WHERE signal_id = $1
        ORDER BY created_at
        """,
        signal_id,
    )
    await conn.close()
    return [dict(r) for r in rows]


@router.get("/today")
async def get_todays_signals():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT s.*, st.name, st.sector
        FROM signals s
        JOIN stocks st ON st.ticker = s.ticker
        WHERE s.fired_at::date = CURRENT_DATE
        ORDER BY s.final_confidence DESC NULLS LAST
        """
    )
    await conn.close()
    return [dict(r) for r in rows]
