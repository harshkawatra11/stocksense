from fastapi import APIRouter, Query
import asyncpg
import os

router = APIRouter()
DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")


@router.get("/learnings")
async def get_learnings(limit: int = Query(50, le=200), ticker: str = None):
    conn = await asyncpg.connect(DB_DSN)
    if ticker:
        rows = await conn.fetch(
            "SELECT * FROM learnings WHERE ticker = $1 ORDER BY created_at DESC LIMIT $2",
            ticker, limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM learnings ORDER BY created_at DESC LIMIT $1", limit
        )
    await conn.close()
    return [dict(r) for r in rows]


@router.get("/learnings/today")
async def get_todays_learnings():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        "SELECT * FROM learnings WHERE learning_date = CURRENT_DATE ORDER BY created_at DESC"
    )
    await conn.close()
    return [dict(r) for r in rows]
