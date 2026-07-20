"""
Feeds the human-confirmation queue (pending_trade_confirmations) from fresh
qualifying signals — the "real order, but only if you say yes" path,
parallel to and independent of intelligence/auto_trader.py's fully-autonomous
PAPER trading (which is unaffected by anything in this module).

Gated behind config.LIVE_CONFIRMATION_ENABLED (default OFF) — nothing queues
for your review until you explicitly opt in. Even once enabled, this module
ONLY ever inserts a PENDING row; nothing
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


async def _sandbox_net_position(conn, ticker: str) -> int:
    """
    Net sandbox-approved quantity for a ticker: sum of APPROVED BUY qty minus
    APPROVED SELL qty, excluding attempts that FAILED to place. This is the
    actual truth of "do we already hold this via the sandbox path" — the
    PAPER portfolio table is a separate, parallel ledger and must not be
    used to size or gate sandbox BUY/SELL decisions (mixing them was a real
    bug: found live on 2026-07-13, ALKALI got queued and approved as a fresh
    BUY three times in one session because the only dedup check was
    status='PENDING', which stops matching the instant a confirmation
    resolves to APPROVED — nothing tracked that the position was already
    open). Callers use this to skip a new BUY once net_qty > 0, and to size
    /gate a SELL to what's actually net-held instead of trusting the
    unrelated PAPER portfolio table.
    """
    row = await conn.fetchrow(
        """
        SELECT COALESCE(SUM(CASE WHEN action = 'BUY' THEN quantity ELSE -quantity END), 0) AS net_qty
        FROM pending_trade_confirmations
        WHERE ticker = $1 AND status = 'APPROVED'
          AND execution_status IS DISTINCT FROM 'FAILED'
        """,
        ticker,
    )
    return int(row["net_qty"] or 0)


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
    actually places Upstox sandbox orders, so it's SUPPOSED to size off REAL
    funds fetched live from Upstox via
    data/pipeline/upstox_orders.get_available_funds(). Fetched once per run
    (not once per candidate) and reused across the whole batch.

    TEMPORARY STOPGAP (see config.SANDBOX_VIRTUAL_CAPITAL's docstring for the
    full story): get_available_funds() currently ALWAYS returns unavailable —
    Upstox's sandbox has no funds endpoint at all, and the live endpoint needs
    a static IP that isn't registered yet. Rather than skip queueing entirely
    every single run (which would make the sandbox flow untestable until a
    static IP exists), this falls back to settings.SANDBOX_VIRTUAL_CAPITAL —
    a config number, NOT a real balance. Every reasoning string, Telegram
    message, and this function's returned dict is labeled
    "[SANDBOX VIRTUAL CAPITAL]" whenever this fallback is used, so it's never
    confused with a real-funds read. DELETE THIS FALLBACK once a static IP
    is registered and get_available_funds() starts returning "ok" — see the
    TODO on SANDBOX_VIRTUAL_CAPITAL in config.py.
    """
    if not settings.LIVE_CONFIRMATION_ENABLED:
        return {"queued": [], "skipped": [], "reason": "LIVE_CONFIRMATION_ENABLED is false"}

    funds = await get_available_funds()
    funds_label = ""
    if funds["status"] != "ok" or funds.get("available") is None:
        detail = funds.get("detail", "unknown error")
        log.warning(
            "queue_fresh_signals: real Upstox funds unavailable (%s) — falling back to "
            "SANDBOX_VIRTUAL_CAPITAL (₹%.0f), NOT your real balance. See config.py "
            "SANDBOX_VIRTUAL_CAPITAL docstring — delete this fallback once a static IP "
            "is registered.", detail, settings.SANDBOX_VIRTUAL_CAPITAL,
        )
        available_funds = settings.SANDBOX_VIRTUAL_CAPITAL
        funds_label = "[SANDBOX VIRTUAL CAPITAL — not your real Upstox balance] "
    else:
        available_funds = float(funds["available"])

    from intelligence.brain_params import get_params

    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        # SANDBOX_MAX_POSITION_PCT, not brain_params' max_position_pct — that
        # value is adaptive/tuned for the PAPER ledger's much larger capital
        # and multi-position diversification. Reusing it here on a tiny real
        # account silently capped every trade at a fixed small budget
        # regardless of signal quality (e.g. ₹210 on ₹1,000 funds), which is
        # why proposals were always cheap stocks — not a model bias, a
        # sizing bug. See config.py's docstring for the full story.
        max_position_pct = settings.SANDBOX_MAX_POSITION_PCT
        budget = available_funds * max_position_pct

        held = await get_portfolio_tickers(conn)
        # Distinct tickers currently net-long via sandbox approval — the cap
        # on genuinely concurrent positions. Previously unlimited (found live
        # 2026-07-20 alongside the budget-sizing bug above).
        open_sandbox_tickers = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT ticker) FROM (
                SELECT ticker, SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) AS net_qty
                FROM pending_trade_confirmations
                WHERE status = 'APPROVED' AND execution_status IS DISTINCT FROM 'FAILED'
                GROUP BY ticker
            ) t WHERE net_qty > 0
            """
        )
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
            if open_sandbox_tickers >= settings.SANDBOX_MAX_OPEN_POSITIONS:
                skipped.append({
                    "ticker": ticker,
                    "why": f"sandbox position cap reached ({open_sandbox_tickers}/{settings.SANDBOX_MAX_OPEN_POSITIONS})",
                })
                continue
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
            if await _sandbox_net_position(conn, ticker) > 0:
                skipped.append({"ticker": ticker, "why": "already holding via sandbox approval"})
                continue

            price = float(s["price_at_signal"] or 0)
            qty = int(remaining_budget // price) if price > 0 else 0
            if qty < 1 or price <= 0:
                skipped.append({
                    "ticker": ticker,
                    "why": f"{funds_label}no affordable quantity (budget ₹{remaining_budget:.0f} "
                           f"@ ₹{price:.2f}, funds ₹{available_funds:.0f})",
                })
                continue

            reasoning = (
                f"{funds_label}conf={float(s['final_confidence'] or 0):.2f} tf={s.get('timeframe')} "
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
            open_sandbox_tickers += 1
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

        result = {"queued": queued, "skipped": skipped}
        if funds_label:
            result["funds_source"] = "sandbox_virtual_capital"
            result["warning"] = (
                "Sized against SANDBOX_VIRTUAL_CAPITAL, not your real Upstox balance "
                "— see config.py for why and how to remove this fallback."
            )
        else:
            result["funds_source"] = "live_upstox_funds"
        return result
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
            paper_qty = int(pos["quantity"]) if pos and pos["quantity"] else 0
            if paper_qty < 1:
                skipped.append({"ticker": ticker, "why": "no active held position"})
                continue

            # Cap to what's actually held via a real sandbox-approved BUY —
            # the PAPER portfolio (paper_qty above) is a separate, parallel
            # ledger and must not by itself authorize a sandbox SELL. Found
            # live on 2026-07-13: three tickers got sandbox SELL orders
            # placed purely off PAPER-ledger EXIT verdicts despite never
            # having a corresponding sandbox BUY, doubling up across two
            # 30-min review cycles because nothing tracked net sandbox
            # exposure. See _sandbox_net_position()'s docstring.
            sandbox_qty = await _sandbox_net_position(conn, ticker)
            if sandbox_qty < 1:
                skipped.append({"ticker": ticker,
                                 "why": "no sandbox-held position to sell (PAPER-only exit)"})
                continue
            qty = min(paper_qty, sandbox_qty)

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
