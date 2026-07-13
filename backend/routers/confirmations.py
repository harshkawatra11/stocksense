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
from data.pipeline.upstox_orders import get_order_status
from intelligence.confirmation_actions import (
    approve_confirmation,
    reject_confirmation,
    row_to_dict as _row_to_dict,
)
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


@router.get("/pending")
async def list_pending():
    """
    List all trade confirmations currently awaiting a human decision.

    Rows older than CONFIRMATION_EXPIRY_MINUTES are still included (so the
    human can see what they missed and reject/re-scan) but annotated with
    `is_stale: true` — the quoted price/reasoning shouldn't be trusted at
    that age. This is a display-time flag only; the authoritative expiry
    enforcement is the scheduler job (flips old rows to EXPIRED so they drop
    out of this list entirely) and approve()'s own re-check at click time.
    """
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            """
            SELECT id, signal_id, ticker, action, quantity, price, reasoning,
                   status, created_at, resolved_at, order_id, execution_status,
                   execution_detail, is_sandbox,
                   EXTRACT(EPOCH FROM (NOW() - created_at)) / 60.0 AS age_minutes
            FROM pending_trade_confirmations
            WHERE status = 'PENDING'
            ORDER BY created_at DESC
            """
        )
        out = []
        for r in rows:
            d = _row_to_dict(r)
            age = d.pop("age_minutes", 0) or 0
            d["is_stale"] = age > settings.CONFIRMATION_EXPIRY_MINUTES
            out.append(d)
        return out
    finally:
        await conn.close()


@router.post("/{confirmation_id}/approve")
async def approve(confirmation_id: int):
    """
    Flip a pending confirmation to APPROVED and place the order on Upstox
    SANDBOX (never live). The actual logic — atomic PENDING->APPROVED flip,
    stale-quote re-check, sandbox order placement, outcome recording — lives
    in intelligence/confirmation_actions.approve_confirmation(), shared
    verbatim with the Telegram bot (backend/services/telegram_bot.py) so
    both surfaces behave identically. See that module's docstrings for the
    concurrency/crash-safety reasoning. Response shape is unchanged from
    when this logic lived inline here.
    """
    result = await approve_confirmation(confirmation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Confirmation not found or not pending")
    return result


@router.get("/{confirmation_id}/order-status")
async def order_status(confirmation_id: int):
    """
    On-demand check of a placed order's exchange-side status via Upstox's
    order/details endpoint (data/pipeline/upstox_orders.get_order_status).
    Complements the scheduler-driven reconciliation job
    (scheduler/market_runner.py task_reconcile_sandbox_orders) with an
    immediate path the frontend can call right after approving, instead of
    waiting for the next scheduled poll. Read-only — never places or
    modifies an order, just refreshes execution_status/execution_detail for
    rows that already have an order_id.
    """
    conn = await asyncpg.connect(DB_DSN)
    try:
        row = await conn.fetchrow(
            "SELECT id, order_id, execution_status FROM pending_trade_confirmations WHERE id = $1",
            confirmation_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Confirmation not found")
        if not row["order_id"]:
            raise HTTPException(status_code=400, detail="No order has been placed for this confirmation")

        result = await get_order_status(row["order_id"])
        updated = await conn.fetchrow(
            """
            UPDATE pending_trade_confirmations
            SET execution_status = $2, execution_detail = $3
            WHERE id = $1
            RETURNING id, signal_id, ticker, action, quantity, price, reasoning,
                      status, created_at, resolved_at, order_id, execution_status,
                      execution_detail, is_sandbox
            """,
            confirmation_id, result["status"], result.get("detail"),
        )
        return _row_to_dict(updated)
    finally:
        await conn.close()


@router.post("/{confirmation_id}/reject")
async def reject(confirmation_id: int):
    """Flip a pending confirmation to REJECTED. Never places an order.
    Shared logic: intelligence/confirmation_actions.reject_confirmation()."""
    result = await reject_confirmation(confirmation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Confirmation not found or not pending")
    return result
