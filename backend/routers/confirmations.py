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
from data.pipeline.upstox_orders import get_order_status, place_sandbox_order
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
    SANDBOX (never live). If sandbox isn't configured (no UPSTOX_SANDBOX_TOKEN)
    or the order fails, the confirmation still moves to APPROVED — the human
    decision is recorded regardless — but execution_status/execution_detail
    report the real outcome instead of silently pretending it worked.

    Duplicate-click / retry safety: the `UPDATE ... WHERE status = 'PENDING'`
    below is a single atomic statement. Postgres row-locks the target row for
    its duration, so if two approve() requests for the SAME id race, exactly
    one UPDATE affects a row (returns it) and the other affects zero rows
    (returns None -> 404 here). There is no window between "read PENDING"
    and "write APPROVED" for a second request to slip through — the read and
    write are the same statement. This is safe under concurrency without any
    extra locking.

    The remaining risk isn't a duplicate order — it's the opposite: if this
    process crashes or the DB connection drops between the status flip and
    place_sandbox_order() actually running, the row would be stuck APPROVED
    with execution_status still NULL forever (looking neither pending nor
    resolved). We close that gap by always recording *some* execution_status
    in a finally-equivalent: place_sandbox_order() itself never raises (it
    catches and returns FAILED), and if it does somehow raise unexpectedly,
    the except below still records FAILED with the exception text rather
    than leaving the row in limbo. A stuck APPROVED-with-NULL-status row is
    still possible only if the process dies mid-UPDATE, which is the same
    residual risk any non-transactional two-step write has; that's an
    acceptable, visible failure mode (the row is easy to spot and re-drive
    manually) rather than a silent double-order risk.
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

        # Stale-quote guard: re-check age at approve time, not just at queue
        # time. A row can sit PENDING for a while with a browser tab open;
        # the quoted price/reasoning may no longer reflect reality by the
        # time the human actually clicks. Refuse to place the order (but
        # keep the APPROVED status — the human decision is still recorded)
        # and flip the row to EXPIRED instead so it's clearly distinguishable
        # from a real placement attempt.
        age_minutes = (row["resolved_at"] - row["created_at"]).total_seconds() / 60.0
        if age_minutes > settings.CONFIRMATION_EXPIRY_MINUTES:
            expired = await conn.fetchrow(
                """
                UPDATE pending_trade_confirmations
                SET status = 'EXPIRED',
                    execution_status = 'FAILED',
                    execution_detail = $2
                WHERE id = $1
                RETURNING id, signal_id, ticker, action, quantity, price, reasoning,
                          status, created_at, resolved_at, order_id, execution_status,
                          execution_detail, is_sandbox
                """,
                confirmation_id,
                f"confirmation was {age_minutes:.0f} min old (limit "
                f"{settings.CONFIRMATION_EXPIRY_MINUTES}) — quoted price likely stale, "
                f"order not placed",
            )
            return _row_to_dict(expired)

        try:
            result = await place_sandbox_order(
                ticker=row["ticker"], action=row["action"],
                quantity=row["quantity"], price=float(row["price"]),
            )
        except Exception as e:
            # place_sandbox_order() is documented to never raise, but if it
            # somehow does, still record a terminal execution_status instead
            # of leaving the row APPROVED with execution_status NULL forever.
            result = {"status": "FAILED", "detail": f"unexpected exception: {e}"}

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
