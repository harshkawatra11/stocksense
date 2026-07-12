"""
Human-confirmed trade approval queue.

Rows in `pending_trade_confirmations` (data/db/schema_v7_confirmation_queue.sql,
data/db/schema_v8_sandbox_orders.sql) represent one specific proposed trade
awaiting a human yes/no. approve() places a real order — but ONLY against
Upstox's SANDBOX environment (data/pipeline/upstox_orders.py): no real money,
no real market impact, ever, from this endpoint. Nothing auto-executes —
every row only reaches an order attempt because a human clicked approve on
that specific trade. See WHAT_TO_DO_NEXT.txt Section 5 for why live-money
execution is deliberately not built here.
"""
from __future__ import annotations

import asyncpg
from fastapi import APIRouter, HTTPException

from config import settings
from data.pipeline.upstox_orders import place_sandbox_order
from intelligence.live_confirmation import queue_fresh_signals

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


@router.post("/queue-fresh")
async def queue_fresh():
    """
    Manually trigger intelligence/live_confirmation.py to scan fresh signals
    and queue qualifying ones for review. Deliberately NOT wired into the
    scheduler — even sandbox order flow should start on an explicit action,
    not silently on a timer the moment LIVE_CONFIRMATION_ENABLED is set.
    Returns {"queued": [], "skipped": [], "reason": "..."} if the flag is off.
    """
    return await queue_fresh_signals()


def _row_to_dict(row: asyncpg.Record) -> dict:
    d = dict(row)
    if d.get("price") is not None:
        d["price"] = float(d["price"])
    for key in ("created_at", "resolved_at"):
        if d.get(key) is not None:
            d[key] = d[key].isoformat()
    return d


@router.get("/pending")
async def list_pending():
    """List all trade confirmations currently awaiting a human decision."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            """
            SELECT id, signal_id, ticker, action, quantity, price, reasoning,
                   status, created_at, resolved_at, order_id, execution_status,
                   execution_detail, is_sandbox
            FROM pending_trade_confirmations
            WHERE status = 'PENDING'
            ORDER BY created_at DESC
            """
        )
        return [_row_to_dict(r) for r in rows]
    finally:
        await conn.close()


@router.post("/{confirmation_id}/approve")
async def approve(confirmation_id: int):
    """
    Flip a pending confirmation to APPROVED and place the order on Upstox
    SANDBOX (never live). If sandbox isn't configured (no UPSTOX_SANDBOX_TOKEN)
    or the order fails, the confirmation still moves to APPROVED — the human
    decision is recorded regardless — but execution_status/execution_detail
    report the real outcome instead of silently pretending it worked.
    """
    conn = await asyncpg.connect(DB_DSN)
    try:
        row = await conn.fetchrow(
            """
            UPDATE pending_trade_confirmations
            SET status = 'APPROVED', resolved_at = NOW()
            WHERE id = $1 AND status = 'PENDING'
            RETURNING id, signal_id, ticker, action, quantity, price, reasoning,
                      status, created_at, resolved_at
            """,
            confirmation_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Confirmation not found or not pending")

        result = await place_sandbox_order(
            ticker=row["ticker"], action=row["action"],
            quantity=row["quantity"], price=float(row["price"]),
        )
        updated = await conn.fetchrow(
            """
            UPDATE pending_trade_confirmations
            SET order_id = $2, execution_status = $3, execution_detail = $4
            WHERE id = $1
            RETURNING id, signal_id, ticker, action, quantity, price, reasoning,
                      status, created_at, resolved_at, order_id, execution_status,
                      execution_detail, is_sandbox
            """,
            confirmation_id, result.get("order_id"), result["status"], result.get("detail"),
        )
        return _row_to_dict(updated)
    finally:
        await conn.close()


@router.post("/{confirmation_id}/reject")
async def reject(confirmation_id: int):
    """Flip a pending confirmation to REJECTED. Never places an order."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        row = await conn.fetchrow(
            """
            UPDATE pending_trade_confirmations
            SET status = 'REJECTED', resolved_at = NOW()
            WHERE id = $1 AND status = 'PENDING'
            RETURNING id, signal_id, ticker, action, quantity, price, reasoning,
                      status, created_at, resolved_at, order_id, execution_status,
                      execution_detail, is_sandbox
            """,
            confirmation_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Confirmation not found or not pending")
        return _row_to_dict(row)
    finally:
        await conn.close()
