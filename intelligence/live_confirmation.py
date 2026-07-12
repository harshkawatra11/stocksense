"""
Feeds the human-confirmation queue (pending_trade_confirmations) from fresh
qualifying signals — the "real order, but only if you say yes" path,
parallel to and independent of intelligence/auto_trader.py's fully-autonomous
PAPER trading (which is unaffected by anything in this module).

Gated behind config.LIVE_CONFIRMATION_ENABLED (default OFF, same pattern as
KRONOS_ENABLED) — nothing queues for your review until you explicitly opt
in. Even once enabled, this module ONLY ever inserts a PENDING row; nothing
here places an order. Order placement happens exclusively in
backend/routers/confirmations.py's approve() endpoint, exclusively against
Upstox's SANDBOX (data/pipeline/upstox_orders.py) — never live money, and
only after an explicit human click on that specific proposed trade.

Run standalone: python -m intelligence.live_confirmation
Or schedule alongside the signal pipeline once you're ready to opt in.
"""
from __future__ import annotations

import asyncio
import logging

import asyncpg

from config import settings
from intelligence.trading_account import get_actionable_signals
from intelligence.portfolio_guard import get_portfolio_tickers

log = logging.getLogger(__name__)


async def _already_queued(conn, ticker: str) -> bool:
    row = await conn.fetchrow(
        "SELECT 1 FROM pending_trade_confirmations WHERE ticker = $1 AND status = 'PENDING'",
        ticker,
    )
    return row is not None


async def queue_fresh_signals(conn=None, limit: int = 10) -> dict:
    """
    Insert PENDING confirmation rows for fresh, qualifying BUY signals not
    already held and not already queued. Returns {queued: [...], skipped: [...]}.

    No-ops entirely (returns {"queued": [], "skipped": [], "reason": "..."})
    if LIVE_CONFIRMATION_ENABLED is false — the flag is checked here, not just
    at the scheduler-registration level, so this is safe to call directly
    (e.g. from a manual trigger or a test) without accidentally bypassing the
    opt-in gate.
    """
    if not settings.LIVE_CONFIRMATION_ENABLED:
        return {"queued": [], "skipped": [], "reason": "LIVE_CONFIRMATION_ENABLED is false"}

    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        held = await get_portfolio_tickers(conn)
        candidates = await get_actionable_signals(conn, limit=limit, only_affordable=True)

        queued, skipped = [], []
        seen: set[str] = set()
        for s in candidates:
            ticker = s["ticker"]
            if ticker in held:
                skipped.append({"ticker": ticker, "why": "already holding"})
                continue
            if ticker in seen:
                skipped.append({"ticker": ticker, "why": "duplicate in batch"})
                continue
            seen.add(ticker)
            if await _already_queued(conn, ticker):
                skipped.append({"ticker": ticker, "why": "already queued for review"})
                continue

            price = float(s["price_at_signal"] or 0)
            qty = int(s.get("shares_affordable") or 0)
            if qty < 1 or price <= 0:
                skipped.append({"ticker": ticker, "why": "no affordable quantity"})
                continue

            reasoning = (
                f"conf={float(s['final_confidence'] or 0):.2f} tf={s.get('timeframe')} "
                f"target=₹{s.get('target_price')} eta={s.get('target_eta_days')}d"
            )
            row = await conn.fetchrow(
                """
                INSERT INTO pending_trade_confirmations
                    (signal_id, ticker, action, quantity, price, reasoning)
                VALUES ($1, $2, 'BUY', $3, $4, $5)
                RETURNING id
                """,
                s["id"], ticker, qty, price, reasoning,
            )
            queued.append({"ticker": ticker, "confirmation_id": row["id"], "qty": qty, "price": price})
            log.info("Queued for confirmation: %s x%d @ ₹%.2f (confirmation_id=%d)",
                      ticker, qty, price, row["id"])

        return {"queued": queued, "skipped": skipped}
    finally:
        if own_conn:
            await conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(queue_fresh_signals()))
