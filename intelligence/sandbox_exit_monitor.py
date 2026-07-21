"""
Sandbox-native exit monitoring — independent of the PAPER portfolio table.

Real bug found live 2026-07-21: sandbox-approved BUY positions were only
ever reviewed for exit via intelligence/position_monitor.py's
review_all_positions(), which exclusively reads active PAPER portfolio
rows. A sandbox position either never had a matching PAPER row, or its
PAPER twin closed independently (PAPER and sandbox are deliberately
separate ledgers) — either way, once that happened, nothing ever
evaluated the sandbox position against its own target/stop again. Six
sandbox positions from 2026-07-13 sat open for 8 days with zero SELL
proposals ever queued, silently maxing out SANDBOX_MAX_OPEN_POSITIONS and
blocking every new trade since.

This module tracks sandbox positions on their own terms: for every ticker
with net sandbox quantity > 0, look up the ORIGINAL signal's target/stop
(from the BUY confirmation's signal_id), compare against the current
price, and queue a SELL confirmation the moment either is hit — same
human-approval gate as every other sandbox order, just no longer
dependent on the PAPER ledger's state.

Run standalone: python -m intelligence.sandbox_exit_monitor
"""
from __future__ import annotations

import logging

import asyncpg

from config import settings

log = logging.getLogger(__name__)


async def _current_price(conn, ticker: str) -> float | None:
    """Live tick if available, else the most recent EOD close."""
    try:
        from backend.services import quote_cache
        tick = quote_cache.get(ticker)
        if tick and tick.get("ltp") is not None:
            return float(tick["ltp"])
    except Exception:
        pass
    return await conn.fetchval(
        "SELECT close FROM ohlcv_daily WHERE ticker = $1 ORDER BY time DESC LIMIT 1", ticker
    )


async def check_sandbox_exits(conn=None) -> dict:
    """
    For every ticker with net sandbox BUY quantity > 0, check its original
    signal's target_price/stop_loss against the current price. Queues a
    SELL confirmation (same shape as live_confirmation.queue_exit_confirmations)
    the instant either is hit. Returns {"queued": [...], "checked": int}.
    """
    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        positions = await conn.fetch(
            """
            SELECT ticker,
                   SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) AS net_qty,
                   -- most recent BUY's signal_id carries this position's target/stop
                   (SELECT signal_id FROM pending_trade_confirmations p2
                    WHERE p2.ticker = p1.ticker AND p2.action = 'BUY'
                      AND p2.status = 'APPROVED' AND p2.execution_status IS DISTINCT FROM 'FAILED'
                    ORDER BY p2.resolved_at DESC LIMIT 1) AS signal_id
            FROM pending_trade_confirmations p1
            WHERE status = 'APPROVED' AND execution_status IS DISTINCT FROM 'FAILED'
            GROUP BY ticker
            HAVING SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) > 0
            """
        )

        queued = []
        for pos in positions:
            ticker, net_qty, signal_id = pos["ticker"], int(pos["net_qty"]), pos["signal_id"]

            already = await conn.fetchrow(
                "SELECT 1 FROM pending_trade_confirmations WHERE ticker = $1 AND action = 'SELL' AND status = 'PENDING'",
                ticker,
            )
            if already:
                continue

            target = stop = None
            if signal_id is not None:
                sig = await conn.fetchrow(
                    "SELECT target_price, stop_loss FROM signals WHERE id = $1", signal_id
                )
                if sig:
                    target = float(sig["target_price"]) if sig["target_price"] else None
                    stop = float(sig["stop_loss"]) if sig["stop_loss"] else None

            if target is None and stop is None:
                # No target/stop on record for this position — can't evaluate an
                # exit condition, but this itself is worth surfacing rather than
                # silently sitting forever; log once per check cycle.
                log.warning(
                    "sandbox_exit_monitor: %s has no target/stop on record (signal_id=%s) "
                    "— cannot evaluate exit condition", ticker, signal_id,
                )
                continue

            price = await _current_price(conn, ticker)
            if price is None:
                continue

            hit_target = target is not None and price >= target
            hit_stop = stop is not None and price <= stop
            if not (hit_target or hit_stop):
                continue

            reason = f"target ₹{target:.2f} reached" if hit_target else f"stop-loss ₹{stop:.2f} hit"
            reasoning = f"[SANDBOX EXIT] {ticker}: {reason} (current ₹{price:.2f})"
            row = await conn.fetchrow(
                """
                INSERT INTO pending_trade_confirmations
                    (signal_id, ticker, action, quantity, price, reasoning)
                VALUES ($1, $2, 'SELL', $3, $4, $5)
                RETURNING id
                """,
                signal_id, ticker, net_qty, price, reasoning,
            )
            queued.append({"ticker": ticker, "confirmation_id": row["id"], "qty": net_qty,
                            "price": price, "reason": reason})
            log.info("Queued sandbox exit: %s x%d @ ₹%.2f (%s, confirmation_id=%d)",
                      ticker, net_qty, price, reason, row["id"])

            try:
                from backend.services.telegram_bot import notify_new_confirmation
                await notify_new_confirmation({
                    "id": row["id"], "ticker": ticker, "action": "SELL",
                    "quantity": net_qty, "price": price, "reasoning": reasoning,
                })
            except ImportError:
                pass
            except Exception as e:
                log.warning("Telegram notify failed for %s sandbox exit (queueing unaffected): %s", ticker, e)

        return {"queued": queued, "checked": len(positions)}
    finally:
        if own_conn:
            await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(check_sandbox_exits()))
