from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import asyncpg
import os
from config import settings

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


class AddPositionRequest(BaseModel):
    ticker: str
    quantity: int
    avg_price: float
    buy_date: datetime | None = None
    notes: str | None = None


@router.get("/")
async def get_portfolio():
    from intelligence.portfolio_guard import get_portfolio
    conn = await asyncpg.connect(DB_DSN)
    result = await get_portfolio(conn)
    await conn.close()
    return result


@router.post("/add")
async def add_position(req: AddPositionRequest):
    conn = await asyncpg.connect(DB_DSN)
    # Ensure stock exists
    exists = await conn.fetchval("SELECT 1 FROM stocks WHERE ticker = $1", req.ticker)
    if not exists:
        await conn.execute(
            "INSERT INTO stocks (ticker, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            req.ticker, req.ticker,
        )
    await conn.execute(
        """
        INSERT INTO portfolio (ticker, quantity, avg_price, buy_date, notes)
        VALUES ($1, $2, $3, $4, $5)
        """,
        req.ticker, req.quantity, req.avg_price,
        req.buy_date or datetime.now(), req.notes,
    )
    await conn.close()
    return {"status": "added", "ticker": req.ticker}


@router.delete("/{ticker}")
async def remove_position(ticker: str):
    conn = await asyncpg.connect(DB_DSN)
    result = await conn.execute(
        "UPDATE portfolio SET active = FALSE WHERE ticker = $1 AND active = TRUE",
        ticker,
    )
    await conn.close()
    if result == "UPDATE 0":
        raise HTTPException(404, f"{ticker} not found in portfolio")
    return {"status": "removed", "ticker": ticker}


@router.get("/pnl")
async def get_pnl_summary():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        """
        SELECT p.ticker, p.quantity, p.avg_price,
               (SELECT close FROM ohlcv_daily WHERE ticker = p.ticker ORDER BY time DESC LIMIT 1) as current_price
        FROM portfolio p WHERE p.active = TRUE
        """
    )
    await conn.close()
    total_invested = 0
    total_current = 0
    positions = []
    for r in rows:
        curr = float(r["current_price"] or r["avg_price"])
        avg = float(r["avg_price"])
        qty = r["quantity"]
        invested = avg * qty
        current_val = curr * qty
        total_invested += invested
        total_current += current_val
        positions.append({
            "ticker": r["ticker"],
            "pnl_pct": round((curr - avg) / avg * 100, 2),
            "pnl_abs": round(current_val - invested, 2),
        })
    return {
        "total_invested": round(total_invested, 2),
        "total_current": round(total_current, 2),
        "total_pnl": round(total_current - total_invested, 2),
        "total_pnl_pct": round((total_current - total_invested) / (total_invested + 1e-8) * 100, 2),
        "positions": positions,
    }
