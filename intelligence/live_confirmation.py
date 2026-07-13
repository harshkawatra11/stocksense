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
from data.pipeline.upstox_orders import get_available_funds

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

    Sizing: unlike intelligence/auto_trader.py's PAPER path (which sizes off
    the fixed settings.CASH_AVAILABLE ledger by design — see trading_account.py
    module docstring on "epoch 2"), this is the human-approval path that
    actually places Upstox sandbox orders (API-parity rehearsal of the real
    account), so it sizes off REAL funds fetched live from Upstox via
    data/pipeline/upstox_orders.get_available_funds(). Fetched once per run
    (not once per candidate) and reused across the whole batch. If funds are
    unavailable for any reason, this does NOT fall back to shares_affordable
    or CASH_AVAILABLE — it skips queueing entirely for the run, consistent
    with the project's "no silent fallback to fake data" rule.
    """
    if not settings.LIVE_CONFIRMATION_ENABLED:
        return {"queued": [], "skipped": [], "reason": "LIVE_CONFIRMATION_ENABLED is false"}

    funds = await get_available_funds()
    if funds["status"] != "ok" or funds.get("available") is None:
        detail = funds.get("detail", "unknown error")
        log.warning("queue_fresh_signals: skipping run — Upstox funds unavailable: %s", detail)
        return {
            "queued": [], "skipped": [],
            "reason": f"Upstox funds unavailable: {detail}",
        }
    available_funds = float(funds["available"])

    from intelligence.brain_params import get_params

    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        params = await get_params(conn)
        max_position_pct = params["max_position_pct"]
        budget = available_funds * max_position_pct

        held = await get_portfolio_tickers(conn)
        # Not passing only_affordable=True: that flag filters on s.affordable,
        # a column pre-computed from the static CASH_AVAILABLE ledger (see
        # signal_pipeline.annotate_affordability) — using it here as a
        # pre-filter would let a stale static-derived flag silently exclude
        # trades that are actually affordable with real Upstox funds (or
        # include ones that aren't). Fetch the broader candidate list and do
        # real affordability filtering below using the live-fetched balance.
        candidates = await get_actionable_signals(conn, limit=limit, only_affordable=False)

        queued, skipped = [], []
        seen: set[str] = set()
        remaining_budget = budget
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
            qty = int(remaining_budget // price) if price > 0 else 0
            if qty < 1 or price <= 0:
                skipped.append({
                    "ticker": ticker,
                    "why": f"no affordable quantity (budget ₹{remaining_budget:.0f} @ ₹{price:.2f}, "
                           f"real Upstox funds ₹{available_funds:.0f})",
                })
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
            # Each position drawn from the same real Upstox balance for this
            # batch — decrement so later candidates in the batch don't
            # oversize against funds already earmarked by earlier ones.
            remaining_budget -= qty * price
            log.info("Queued for confirmation: %s x%d @ ₹%.2f (confirmation_id=%d)",
                      ticker, qty, price, row["id"])

            # Push to Telegram (approve/reject from the phone). Strictly
            # best-effort: a missing module or any Telegram failure must
            # never break queueing — the web UI remains the source of truth.
            try:
                from backend.services.telegram_bot import notify_new_confirmation
                await notify_new_confirmation({
                    "id": row["id"], "ticker": ticker, "action": "BUY",
                    "quantity": qty, "price": price, "reasoning": reasoning,
                })
            except ImportError:
                pass
            except Exception as e:
                log.warning("Telegram notify failed for %s (queueing unaffected): %s", ticker, e)

        return {"queued": queued, "skipped": skipped}
    finally:
        if own_conn:
            await conn.close()


async def queue_exit_confirmations(reviews: list[dict], conn=None) -> dict:
    """
    SELL-side twin of queue_fresh_signals(): for each EXIT verdict in the
    position_monitor review dicts, insert a PENDING SELL row into
    pending_trade_confirmations (quantity = the held position's quantity,
    price = the review's current_price, reasoning = the review's reasoning).

    Purely ADDITIVE to the autonomous PAPER auto_exit flow — this only queues
    a proposal for human review; the paper exit has already happened (or will)
    independently. Same gates as the BUY path: no-op unless
    LIVE_CONFIRMATION_ENABLED, skip if a PENDING SELL already exists for the
    ticker, skip if the position isn't actually held.
    """
    if not settings.LIVE_CONFIRMATION_ENABLED:
        return {"queued": [], "skipped": [], "reason": "LIVE_CONFIRMATION_ENABLED is false"}

    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        queued, skipped = [], []
        seen: set[str] = set()
        for r in reviews:
            ticker = r.get("ticker")
            if not ticker or r.get("verdict") != "EXIT":
                continue
            if ticker in seen:
                skipped.append({"ticker": ticker, "why": "duplicate in batch"})
                continue
            seen.add(ticker)

            pos = await conn.fetchrow(
                "SELECT quantity FROM portfolio WHERE ticker = $1 AND active = TRUE "
                "ORDER BY buy_date DESC LIMIT 1",
                ticker,
            )
            qty = int(pos["quantity"]) if pos and pos["quantity"] else 0
            if qty < 1:
                skipped.append({"ticker": ticker, "why": "no active held position"})
                continue

            already = await conn.fetchrow(
                "SELECT 1 FROM pending_trade_confirmations "
                "WHERE ticker = $1 AND action = 'SELL' AND status = 'PENDING'",
                ticker,
            )
            if already:
                skipped.append({"ticker": ticker, "why": "SELL already queued for review"})
                continue

            price = float(r.get("current_price") or 0)
            if price <= 0:
                skipped.append({"ticker": ticker, "why": "no current price in review"})
                continue

            reasoning = (r.get("reasoning") or f"{ticker}: EXIT verdict from position review")[:1000]
            row = await conn.fetchrow(
                """
                INSERT INTO pending_trade_confirmations
                    (signal_id, ticker, action, quantity, price, reasoning)
                VALUES ($1, $2, 'SELL', $3, $4, $5)
                RETURNING id
                """,
                r.get("signal_id"), ticker, qty, price, reasoning,
            )
            entry = {"ticker": ticker, "confirmation_id": row["id"], "qty": qty, "price": price}
            queued.append(entry)
            log.info("Queued SELL for confirmation: %s x%d @ ₹%.2f (confirmation_id=%d)",
                      ticker, qty, price, row["id"])
            # Push to Telegram — same defensive best-effort pattern as the BUY
            # path above: a missing module or Telegram failure never blocks queueing.
            try:
                from backend.services.telegram_bot import notify_new_confirmation
                await notify_new_confirmation({
                    "id": row["id"], "ticker": ticker, "action": "SELL",
                    "quantity": qty, "price": price, "reasoning": reasoning,
                })
            except ImportError:
                pass
            except Exception as e:
                log.warning("Telegram notify failed for %s (queueing unaffected): %s", ticker, e)

        return {"queued": queued, "skipped": skipped}
    finally:
        if own_conn:
            await conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(queue_fresh_signals()))
