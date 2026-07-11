"""
Human-confirmed trade approval queue — STUB.

Execution wiring intentionally deferred — see WHAT_TO_DO_NEXT.txt Section 5.
These endpoints only record human approve/reject decisions on proposed trades
in `pending_trade_confirmations` (data/db/schema_v7_confirmation_queue.sql);
nothing places real orders yet. Per explicit user requirement, trades must
always be human approve/reject only, never auto-executed — this router does
NOT call any order-placement code (none exists yet in this codebase).
"""
from __future__ import annotations

import asyncpg
from fastapi import APIRouter, HTTPException

from config import settings

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


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
                   status, created_at, resolved_at
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
    """Flip a pending confirmation to APPROVED. Records the decision only —
    does NOT place any order. Order execution is a future stage."""
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
        return _row_to_dict(row)
    finally:
        await conn.close()


@router.post("/{confirmation_id}/reject")
async def reject(confirmation_id: int):
    """Flip a pending confirmation to REJECTED. Records the decision only —
    does NOT place any order."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        row = await conn.fetchrow(
            """
            UPDATE pending_trade_confirmations
            SET status = 'REJECTED', resolved_at = NOW()
            WHERE id = $1 AND status = 'PENDING'
            RETURNING id, signal_id, ticker, action, quantity, price, reasoning,
                      status, created_at, resolved_at
            """,
            confirmation_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Confirmation not found or not pending")
        return _row_to_dict(row)
    finally:
        await conn.close()
