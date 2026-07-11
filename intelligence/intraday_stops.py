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


async def _check_once(conn) -> list[dict]:
    """One breach-detection pass over all active positions. Missing/stale
    quotes are skipped quietly (debug log only) — the feed may not be
    subscribed to every ticker yet, and that's expected, not an error."""
    positions = await conn.fetch(
        "SELECT id, ticker, quantity, avg_price, buy_date FROM portfolio WHERE active = TRUE"
    )
    breaches: list[dict] = []
    for p in positions:
        pos = dict(p)
        ticker = pos["ticker"]
        quote = quote_cache.get(ticker)
        if not quote or quote.get("ltp") is None:
            log.debug("intraday_stops: no live quote yet for %s — skipping this cycle", ticker)
            continue
        try:
            hit = await check_stop_target_breach(conn, pos, float(quote["ltp"]))
        except Exception as e:
            log.debug("intraday_stops: breach check failed for %s: %s", ticker, e)
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
