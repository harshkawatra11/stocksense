"""
Fast intraday stop/target monitor.

StockSense is being rebuilt around intraday (MIS-style) trading, where a stop
breached at 10:31am can't wait for the next 30-min scheduler cycle
(task_position_review in scheduler/market_runner.py) to be noticed — that's
"exited up to 30 minutes late" territory. This module is a standalone asyncio
background loop (started from backend/main.py's lifespan, same pattern as
backend/services/quote_cache.py's start_quote_cache_feed) that polls the
in-memory quote_cache every few seconds for each held position's live LTP and
exits immediately on a stop/target breach.

It does NOT duplicate position_monitor.py's re-analysis logic: both the fast
loop here and the slow 30-min cycle route through the same primitive,
intelligence.position_monitor.check_stop_target_breach, and the same exit
path, intelligence.auto_trader.auto_exit. This loop only owns the "did we
breach right now" fast path; the slow loop still does the richer
re-forecast/progress-vs-ETA analysis in review_position.

Paper-only: auto_exit records a paper SELL via trading_account.record_decision.
No real order is placed (order execution doesn't exist yet).
"""
from __future__ import annotations

import asyncio
import logging

import asyncpg

from config import settings
from backend.services import quote_cache
from intelligence.position_monitor import check_stop_target_breach
from intelligence.auto_trader import auto_exit
from intelligence.activity import log_activity

log = logging.getLogger(__name__)

POLL_SECONDS = 7


async def _fallback_ltp(ticker: str) -> float | None:
    """
    Slower last-resort LTP lookup for a position the live WS feed has no
    tick for (not subscribed yet, or the feed is down) — a REST round trip
    beats leaving the position completely unmonitored for a whole 30-min
    position_review cycle. Falls back again to ohlcv_daily's latest close
    if even the REST call fails (stale, but still a number — better than
    silently skipping enforcement entirely).
    """
    try:
        from data.pipeline.upstox_client import resolve_instrument_key, get_ltp_rest
        key = await resolve_instrument_key(ticker)
        if key is not None:
            ltp = await get_ltp_rest(key)
            if ltp is not None:
                return ltp
    except Exception:
        log.exception("intraday_stops: REST LTP fallback failed for %s", ticker)

    try:
        conn2 = await asyncpg.connect(settings.DATABASE_DSN)
        try:
            row = await conn2.fetchrow(
                "SELECT close FROM ohlcv_daily WHERE ticker = $1 ORDER BY time DESC LIMIT 1",
                ticker,
            )
        finally:
            await conn2.close()
        if row and row["close"] is not None:
            return float(row["close"])
    except Exception:
        log.exception("intraday_stops: ohlcv_daily fallback close lookup failed for %s", ticker)

    return None


async def _check_once(conn) -> list[dict]:
    """One breach-detection pass over all active positions. A missing live
    quote is NOT silently swallowed anymore — it's a real coverage gap (the
    feed isn't subscribed to this ticker yet, or is down), so it's logged at
    WARNING and we fall back to a slower REST/EOD-close lookup for just that
    position rather than skipping enforcement outright."""
    positions = await conn.fetch(
        "SELECT id, ticker, quantity, avg_price, buy_date FROM portfolio WHERE active = TRUE"
    )
    breaches: list[dict] = []
    for p in positions:
        pos = dict(p)
        ticker = pos["ticker"]
        quote = quote_cache.get(ticker)
        ltp = quote.get("ltp") if quote else None
        if ltp is None:
            log.warning(
                "intraday_stops: no live WS quote for %s — falling back to REST/EOD-close "
                "lookup for this cycle (fast-path coverage gap)", ticker,
            )
            ltp = await _fallback_ltp(ticker)
            if ltp is None:
                log.warning("intraday_stops: no price available for %s from any source — skipping this cycle", ticker)
                continue
        try:
            hit = await check_stop_target_breach(conn, pos, float(ltp))
        except Exception as e:
            log.warning("intraday_stops: breach check failed for %s: %s", ticker, e)
            continue
        if hit:
            breaches.append(hit)
    return breaches


async def _run_once() -> None:
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        breaches = await _check_once(conn)
        if not breaches:
            return

        for b in breaches:
            # Distinguishable from the slow cycle's REANALYZED events via the
            # note prefix + payload.source, without needing a new event_type.
            await log_activity(
                conn, event_type="REANALYZED", ticker=b["ticker"], signal_id=b.get("signal_id"),
                note=f"fast intraday stop breach: {b['status']} -> {b['verdict']}",
                payload={
                    "progress_pct": b["progress_pct"], "current": b["current_price"],
                    "verdict": b["verdict"], "source": "fast_intraday_loop",
                },
            )

        exit_summary = await auto_exit(breaches, conn=conn)
        if exit_summary["sold"]:
            log.info(
                "intraday_stops: fast-exited %d position(s): %s",
                len(exit_summary["sold"]), [s["ticker"] for s in exit_summary["sold"]],
            )
        if exit_summary["errors"]:
            log.warning("intraday_stops: %d fast-exit error(s): %s",
                        len(exit_summary["errors"]), exit_summary["errors"])
    finally:
        await conn.close()


async def start_intraday_stop_monitor(poll_seconds: int = POLL_SECONDS) -> None:
    """
    FastAPI lifespan-compatible background loop. Runs forever, polling every
    `poll_seconds`. Never lets a single failed cycle (DB hiccup, quote_cache
    not started yet, etc.) kill the loop — logs and retries on the next tick.
    """
    log.info("intraday_stops: fast stop/target monitor starting (poll every %ss)", poll_seconds)
    while True:
        try:
            await _run_once()
        except Exception:
            log.exception("intraday_stops: cycle failed — will retry next poll")
        await asyncio.sleep(poll_seconds)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(start_intraday_stop_monitor())
