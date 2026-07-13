"""
Shared approve/reject logic for the human trade-confirmation queue —
factored out of backend/routers/confirmations.py so the HTTP endpoints and
the Telegram bot (backend/services/telegram_bot.py) drive the exact same
code path with zero duplicated SQL.

approve_confirmation() places a real order — but ONLY against Upstox's
SANDBOX environment (data/pipeline/upstox_orders.py): no real money, no
real market impact, ever. Nothing auto-executes — every row only reaches an
order attempt because a human explicitly approved that specific trade (web
click or Telegram button tap). There is deliberately NO bulk/auto-approve
entry point in this module: both functions take exactly one confirmation id.

Both functions return the confirmation row as a plain dict (same shape the
router has always returned to the frontend), or None when the row doesn't
exist or is no longer PENDING — callers decide how to surface that (the
router raises 404; the Telegram bot answers "already handled").
"""
from __future__ import annotations

import asyncpg

from config import settings
from data.pipeline.upstox_orders import place_sandbox_order


def row_to_dict(row: asyncpg.Record) -> dict:
    """Normalize a pending_trade_confirmations row for JSON: price -> float,
    timestamps -> ISO strings. Shared by the router's other endpoints too."""
    d = dict(row)
    if d.get("price") is not None:
        d["price"] = float(d["price"])
    for key in ("created_at", "resolved_at"):
        if d.get(key) is not None:
            d[key] = d[key].isoformat()
    return d


async def approve_confirmation(confirmation_id: int) -> dict | None:
    """
    Flip a pending confirmation to APPROVED and place the order on Upstox
    SANDBOX (never live). Returns None if the row doesn't exist or was
    already decided (not PENDING). If sandbox isn't configured (no
    UPSTOX_SANDBOX_TOKEN) or the order fails, the confirmation still moves
    to APPROVED — the human decision is recorded regardless — but
    execution_status/execution_detail report the real outcome instead of
    silently pretending it worked.

    Duplicate-click / retry safety: the `UPDATE ... WHERE status = 'PENDING'`
    below is a single atomic statement. Postgres row-locks the target row for
    its duration, so if two approvals for the SAME id race (e.g. a web click
    and a Telegram tap), exactly one UPDATE affects a row (returns it) and
    the other affects zero rows (returns None here). There is no window
    between "read PENDING" and "write APPROVED" for a second request to slip
    through — the read and write are the same statement. This is safe under
    concurrency without any extra locking.

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
    conn = await asyncpg.connect(settings.DATABASE_DSN)
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
            return None

        # Stale-quote guard: re-check age at approve time, not just at queue
        # time. A row can sit PENDING for a while with a browser tab (or a
        # Telegram message) open; the quoted price/reasoning may no longer
        # reflect reality by the time the human actually acts. Refuse to
        # place the order (but keep the human decision recorded) and flip
        # the row to EXPIRED instead so it's clearly distinguishable from a
        # real placement attempt.
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
            return row_to_dict(expired)

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

        if row["action"] == "BUY" and result.get("status") not in ("FAILED", None):
            # Same reasoning as intelligence/auto_trader.py's paper-BUY path:
            # a sandbox-approved BUY is a real new position that needs fast
            # stop coverage immediately, not whenever the backend next
            # restarts. Best-effort — never let this break the confirmation.
            try:
                from data.pipeline.upstox_client import resolve_instrument_key
                from backend.services.quote_cache import add_to_watchlist

                key = await resolve_instrument_key(row["ticker"])
                if key is not None:
                    await add_to_watchlist(key, row["ticker"])
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "approve_confirmation: failed to add %s to live feed (order already placed, unaffected)",
                    row["ticker"],
                )

        return row_to_dict(updated)
    finally:
        await conn.close()


async def reject_confirmation(confirmation_id: int) -> dict | None:
    """Flip a pending confirmation to REJECTED. Never places an order.
    Returns None if the row doesn't exist or was already decided."""
    conn = await asyncpg.connect(settings.DATABASE_DSN)
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
            return None
        return row_to_dict(row)
    finally:
        await conn.close()
