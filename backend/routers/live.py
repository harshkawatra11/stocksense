"""
Live intelligence API — the multi-timeframe engine surfaced for the app.

Exposes the actionable BUY signals (with target + fractional ETA), the capital
account, the decision ledger, the activity lifecycle feed, and position
re-analysis — plus the user actions: rate a signal, buy, pass, and trigger a run.
"""
from __future__ import annotations

import asyncpg
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from intelligence.trading_account import (
    get_account, record_decision, get_actionable_signals, get_decisions,
)
from intelligence.activity import rate_signal, get_activity_feed
from intelligence.position_monitor import review_all_positions
from intelligence.data_freshness import get_data_freshness
from intelligence.trading_mode import get_trading_mode

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


# ----------------------------- status ---------------------------- #
@router.get("/data-status")
async def data_status():
    """Is the latest data LIVE intraday or PAST end-of-day? Drives the UI banner."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        return await get_data_freshness(conn)
    finally:
        await conn.close()


@router.get("/mode")
async def trading_mode():
    """PAPER vs LIVE gate + progress toward unlocking LIVE execution."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        return await get_trading_mode(conn)
    finally:
        await conn.close()


# ----------------------------- reads ----------------------------- #
@router.get("/signals")
async def actionable_signals(limit: int = 30, only_affordable: bool = False):
    """Latest BUY signals, affordable-first. Each has target + target_eta_days."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        sigs = await get_actionable_signals(conn, limit=limit, only_affordable=only_affordable)
        # asyncpg Decimals → floats for JSON
        for s in sigs:
            for k in ("price_at_signal", "target_price", "stop_loss", "final_confidence",
                      "macro_sector_score", "target_eta_days", "expected_move_pct"):
                if s.get(k) is not None:
                    s[k] = float(s[k])
        return sigs
    finally:
        await conn.close()


@router.get("/account")
async def account():
    conn = await asyncpg.connect(DB_DSN)
    try:
        acct = await get_account(conn)
        acct["deployable_note"] = f"₹{acct['cash_available']:.0f} to deploy, ₹{acct['cash_reserve']:.0f} reserve"
        return acct
    finally:
        await conn.close()


@router.get("/activity")
async def activity(limit: int = 100):
    conn = await asyncpg.connect(DB_DSN)
    try:
        return await get_activity_feed(conn, limit=limit)
    finally:
        await conn.close()


@router.get("/decisions")
async def decisions(limit: int = 50):
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await get_decisions(conn, limit=limit)
        for r in rows:
            for k in ("price", "cash_after", "pnl", "final_confidence"):
                if r.get(k) is not None:
                    r[k] = float(r[k])
        return rows
    finally:
        await conn.close()


@router.get("/positions/reviews")
async def position_reviews(limit: int = 50):
    """Latest position re-analysis rows (held stocks vs their original prediction)."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            """
            SELECT ticker, days_elapsed, eta_days, entry_price, target_price,
                   current_price, progress_pct, status, verdict, reasoning, reviewed_at
            FROM position_reviews ORDER BY reviewed_at DESC LIMIT $1
            """,
            limit,
        )
        out = []
        for r in rows:
            d = dict(r)
            for k in ("days_elapsed", "eta_days", "entry_price", "target_price",
                      "current_price", "progress_pct"):
                if d.get(k) is not None:
                    d[k] = float(d[k])
            out.append(d)
        return out
    finally:
        await conn.close()


# ----------------------------- actions --------------------------- #
class RateRequest(BaseModel):
    signal_id: int
    ticker: str
    like: bool
    reason: str = ""


class BuyRequest(BaseModel):
    signal_id: int | None = None
    ticker: str
    quantity: int
    price: float
    rationale: str = ""


class PassRequest(BaseModel):
    signal_id: int | None = None
    ticker: str
    reason: str = ""


@router.post("/rate")
async def rate(req: RateRequest):
    conn = await asyncpg.connect(DB_DSN)
    try:
        aid = await rate_signal(conn, req.signal_id, req.ticker, like=req.like, reason=req.reason)
        return {"status": "rated", "activity_id": aid, "rating": "LIKE" if req.like else "DISLIKE"}
    finally:
        await conn.close()


@router.post("/buy")
async def buy(req: BuyRequest):
    conn = await asyncpg.connect(DB_DSN)
    try:
        result = await record_decision(
            conn, action="BUY", ticker=req.ticker, signal_id=req.signal_id,
            quantity=req.quantity, price=req.price, rationale=req.rationale,
        )
        return {"status": "bought", **result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        await conn.close()


@router.post("/pass")
async def pass_signal(req: PassRequest):
    conn = await asyncpg.connect(DB_DSN)
    try:
        result = await record_decision(
            conn, action="PASS", ticker=req.ticker, signal_id=req.signal_id,
            rationale=req.reason,
        )
        return {"status": "passed", **result}
    finally:
        await conn.close()


@router.post("/positions/review")
async def trigger_review():
    """Re-analyze all held positions now. Returns the review summaries."""
    return await review_all_positions()


@router.post("/run")
async def run_pipeline(limit: int = 50):
    """
    Trigger a live multi-timeframe run over the top `limit` active tickers.
    Synchronous (macro SLM + Kronos make this take a couple of minutes) — intended
    for manual triggering; the scheduler runs it automatically during market hours.
    Returns a compact summary of generated BUY signals.
    """
    from intelligence.signal_pipeline import run_pipeline_multi
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            "SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker LIMIT $1", limit
        )
        tickers = [r["ticker"] for r in rows]
        freshness = await get_data_freshness(conn)
        mode = await get_trading_mode(conn)
    finally:
        await conn.close()

    sigs = await run_pipeline_multi(tickers, save=True)
    return {
        "generated": len(sigs),
        "tickers_scanned": len(tickers),
        "data_status": freshness,
        "mode": mode,
        "signals": [
            {"ticker": s["ticker"], "timeframe": s["timeframe"],
             "confidence": s["confidence"], "headline": s.get("headline"),
             "affordable": s["affordable"]}
            for s in sorted(sigs, key=lambda x: -x["confidence"])
        ],
    }
