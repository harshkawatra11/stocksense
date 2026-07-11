"""
Autonomous paper executor — the hands of the brain.

After every pipeline run the brain acts on its own signals: qualifying signals
become paper BUYs sized by brain_params; everything else in the batch gets an
explicit PASS with the rule that failed, so the learning loop can study both.
Position-monitor EXIT verdicts become paper SELLs that realize P&L.

There is no on/off switch. If the scheduler is running, the brain is trading.
Every action lands in decisions + activity_log + app.log.

Idempotency: one [AUTO] decision per signal_id, enforced by a pre-check here
and a partial unique index (uq_decisions_auto_signal) as the backstop.
"""
from __future__ import annotations

import logging

import asyncpg

from config import settings
from intelligence.activity import log_activity
from intelligence.brain_params import get_params
from intelligence.portfolio_guard import (
    get_portfolio_tickers,
    check_position_limit,
    check_sector_concentration,
    check_daily_loss_circuit_breaker,
)
from intelligence.trading_account import get_account, get_actionable_signals, record_decision

log = logging.getLogger(__name__)

# Only act on signals from the current batch — anything older was already
# considered (or deliberately skipped) on a previous cycle.
FRESH_WINDOW_MINUTES = 35


async def _already_decided(conn, signal_id: int) -> bool:
    return bool(await conn.fetchval(
        "SELECT 1 FROM decisions WHERE signal_id = $1 LIMIT 1", signal_id
    ))


async def _open_position_count(conn) -> int:
    return int(await conn.fetchval(
        "SELECT COUNT(*) FROM portfolio WHERE active = TRUE"
    ) or 0)


async def auto_trade(conn=None) -> dict:
    """
    Act on the latest pipeline batch. Returns
    {bought: [...], passed: [...], skipped: [...]} for the run summary.
    """
    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        params = await get_params(conn, force_refresh=True)
        threshold = params["confidence_threshold"]
        max_pos_pct = params["max_position_pct"]
        max_open = int(params["max_open_positions"])

        held = await get_portfolio_tickers(conn)
        open_count = await _open_position_count(conn)

        # Portfolio-level risk guards (additional to the budget-based sizing
        # below) — checked once per run, not per candidate, since they're
        # account-wide state that doesn't change mid-batch except by our own
        # buys (handled by re-checking check_position_limit per candidate).
        breaker_ok, breaker_reason = await check_daily_loss_circuit_breaker(conn)
        if not breaker_ok:
            log.warning("auto_trade: %s", breaker_reason)
            await log_activity(
                conn, event_type="NOTE", note=f"[risk] {breaker_reason}",
                payload={"guard": "daily_loss_circuit_breaker"},
            )

        candidates = await get_actionable_signals(conn, limit=20)
        # This run's batch only.
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=FRESH_WINDOW_MINUTES)
        fresh = [
            s for s in candidates
            if s.get("fired_at") and s["fired_at"] >= cutoff
        ]

        bought, passed, skipped = [], [], []
        seen_tickers: set[str] = set()

        for s in fresh:
            sid, ticker = s["id"], s["ticker"]
            conf = float(s["final_confidence"] or 0)
            price = float(s["price_at_signal"] or 0)

            if await _already_decided(conn, sid):
                skipped.append({"ticker": ticker, "signal_id": sid, "why": "already decided"})
                continue
            if ticker in seen_tickers:
                skipped.append({"ticker": ticker, "signal_id": sid, "why": "duplicate in batch"})
                continue
            seen_tickers.add(ticker)

            # Qualification rules, in order — the first failure becomes the PASS reason.
            reason = None
            if conf < threshold:
                reason = f"confidence {conf:.2f} < threshold {threshold:.2f}"
            elif ticker in held:
                reason = "already holding position"
            elif open_count >= max_open:
                reason = f"max open positions reached ({open_count}/{max_open})"
            elif not breaker_ok:
                reason = f"[risk] {breaker_reason}"
            else:
                pos_ok, pos_reason = await check_position_limit(conn)
                if not pos_ok:
                    reason = f"[risk] {pos_reason}"

            if reason is None:
                acct = await get_account(conn)  # re-read: cash moves as we buy
                budget = acct["cash_available"] * max_pos_pct
                qty = int(budget // price) if price > 0 else 0
                if qty < 1:
                    reason = (
                        f"budget ₹{budget:.0f} ({max_pos_pct*100:.0f}% of "
                        f"₹{acct['cash_available']:.0f}) can't buy 1 share @ ₹{price:.2f}"
                    )
                else:
                    order_value = qty * price
                    sector_ok, sector_reason = await check_sector_concentration(conn, ticker, order_value)
                    if not sector_ok:
                        reason = f"[risk] {sector_reason}"

            if reason:
                try:
                    await record_decision(
                        conn, action="PASS", ticker=ticker, signal_id=sid,
                        rationale=f"[AUTO] {reason}",
                    )
                    await log_activity(
                        conn, event_type="AUTO_PASS", ticker=ticker, signal_id=sid,
                        note=reason, payload={"confidence": conf, "price": price},
                    )
                    passed.append({"ticker": ticker, "signal_id": sid, "why": reason})
                except Exception as e:
                    log.warning("AUTO_PASS record failed %s: %s", ticker, e)
                continue

            # Execute the paper BUY.
            try:
                rationale = (
                    f"[AUTO] conf={conf:.2f} tf={s.get('timeframe')} "
                    f"target=₹{s.get('target_price')} eta={s.get('target_eta_days')}d"
                )
                result = await record_decision(
                    conn, action="BUY", ticker=ticker, signal_id=sid,
                    quantity=qty, price=price, rationale=rationale,
                )
                open_count += 1
                held.add(ticker)
                await log_activity(
                    conn, event_type="AUTO_BUY", ticker=ticker, signal_id=sid,
                    note=f"bought {qty} @ ₹{price:.2f} ({conf*100:.0f}% conf)",
                    payload={"quantity": qty, "price": price, "confidence": conf,
                             "target": float(s["target_price"]) if s.get("target_price") else None,
                             "stop": float(s["stop_loss"]) if s.get("stop_loss") else None,
                             "eta_days": float(s["target_eta_days"]) if s.get("target_eta_days") else None,
                             "cash_after": result["cash_after"], "mode": result["mode"]},
                )
                bought.append({"ticker": ticker, "signal_id": sid, "qty": qty, "price": price})
                log.info("AUTO BUY %s x%d @ ₹%.2f (conf %.2f, cash after ₹%.2f)",
                         ticker, qty, price, conf, result["cash_after"])
            except ValueError as e:
                # Cash raced below requirement — degrade to PASS, never crash the loop.
                try:
                    await record_decision(
                        conn, action="PASS", ticker=ticker, signal_id=sid,
                        rationale=f"[AUTO] buy rejected: {e}",
                    )
                except Exception:
                    pass
                passed.append({"ticker": ticker, "signal_id": sid, "why": str(e)})
            except asyncpg.UniqueViolationError:
                skipped.append({"ticker": ticker, "signal_id": sid, "why": "idempotency index"})

        summary = {"bought": bought, "passed": passed, "skipped": skipped}
        log.info("auto_trade: %d bought, %d passed, %d skipped",
                 len(bought), len(passed), len(skipped))
        return summary
    finally:
        if own_conn:
            await conn.close()


async def auto_exit(reviews: list[dict], conn=None) -> dict:
    """
    Turn position-monitor EXIT verdicts into paper SELLs. Idempotent by
    construction: a sold ticker leaves the portfolio, so the next review cycle
    has nothing to exit. Returns {sold: [...], errors: [...]}.
    """
    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        # Real holdings tracked here are watch-only: the brain reviews and alerts
        # on them, but must never paper-sell them out from under itself.
        watch_only = {
            row["ticker"] for row in await conn.fetch(
                "SELECT ticker FROM portfolio WHERE active = TRUE AND watch_only = TRUE"
            )
        }
        sold, errors = [], []
        for r in reviews or []:
            if (r.get("verdict") or "").upper() != "EXIT":
                continue
            ticker = r["ticker"]
            if ticker in watch_only:
                continue  # alert only; never auto-close a real position
            price = r.get("current_price")
            if not price:
                price = await conn.fetchval(
                    "SELECT close FROM ohlcv_daily WHERE ticker = $1 ORDER BY time DESC LIMIT 1",
                    ticker,
                )
            if not price:
                errors.append({"ticker": ticker, "why": "no price available"})
                continue
            try:
                status = r.get("status", "exit")
                rationale = f"[AUTO] {status}: {(r.get('reasoning') or '')[:200]}"
                result = await record_decision(
                    conn, action="SELL", ticker=ticker,
                    quantity=0,  # 0 → full close
                    price=float(price), rationale=rationale,
                )
                await log_activity(
                    conn, event_type="AUTO_SELL", ticker=ticker,
                    note=f"exited @ ₹{float(price):.2f} ({status})",
                    payload={"price": float(price), "status": status,
                             "progress_pct": r.get("progress_pct"),
                             "cash_after": result["cash_after"]},
                )
                sold.append({"ticker": ticker, "price": float(price), "status": status})
                log.info("AUTO SELL %s @ ₹%.2f (%s)", ticker, float(price), status)
            except ValueError as e:
                # Position already gone (e.g. sold earlier this cycle) — fine.
                log.info("auto_exit skip %s: %s", ticker, e)
            except Exception as e:
                errors.append({"ticker": ticker, "why": str(e)})
                log.warning("auto_exit failed %s: %s", ticker, e)

        if sold or errors:
            log.info("auto_exit: %d sold, %d errors", len(sold), len(errors))
        return {"sold": sold, "errors": errors}
    finally:
        if own_conn:
            await conn.close()
