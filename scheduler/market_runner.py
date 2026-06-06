"""
StockSense market scheduler.
Runs periodic data pipeline tasks: incremental OHLCV + F&O updates,
signal pipeline, and EOD review.

Intended to run as: python -m scheduler.market_runner
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, time as dtime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Task functions                                                        #
# ------------------------------------------------------------------ #

async def task_incremental_ohlcv():
    """
    Pull latest daily OHLCV. Primary source is Angel One getCandleData (works on
    this network); NSE Bhavcopy is the fallback (often network-blocked here).
    """
    log.info("Starting incremental OHLCV update (Angel One)...")
    try:
        from data.pipeline.fetch_angel_daily import run_angel_backfill
        summary = await run_angel_backfill(days_back=5)
        log.info("Angel One OHLCV update complete: %s", summary)
        if summary.get("updated", 0) > 0:
            return
        log.warning("Angel One updated 0 tickers — falling back to NSE Bhavcopy.")
    except Exception as e:
        log.error(f"Angel One OHLCV update failed: {e} — falling back to NSE Bhavcopy.", exc_info=True)

    try:
        from data.pipeline.fetch_historical import run_incremental
        await run_incremental(days_back=3)
        log.info("NSE Bhavcopy fallback update complete.")
    except Exception as e:
        log.error(f"NSE Bhavcopy fallback failed: {e}", exc_info=True)


async def task_incremental_fo():
    """Pull latest F&O data from NSE Bhavcopy."""
    log.info("Starting incremental F&O update...")
    try:
        from data.pipeline.fetch_f_and_o import run_fo_incremental
        await run_fo_incremental(days_back=3)
        log.info("Incremental F&O update complete.")
    except Exception as e:
        log.error(f"Incremental F&O update failed: {e}", exc_info=True)


async def task_signal_pipeline():
    """Run the live multi-timeframe intelligence pipeline on the latest data."""
    log.info("Starting multi-timeframe signal pipeline...")
    try:
        from intelligence.signal_pipeline import run_pipeline_multi
        sigs = await run_pipeline_multi(save=True)
        log.info("Signal pipeline complete: %d BUY signals.", len(sigs))
    except Exception as e:
        log.error(f"Signal pipeline failed: {e}", exc_info=True)


async def task_position_review():
    """Re-analyze held positions vs their original predictions (HOLD/EXIT verdicts)."""
    log.info("Starting position re-analysis...")
    try:
        from intelligence.position_monitor import review_all_positions
        reviews = await review_all_positions()
        log.info("Position re-analysis complete: %d positions reviewed.", len(reviews))
    except Exception as e:
        log.error(f"Position re-analysis failed: {e}", exc_info=True)


async def task_eod_review():
    """Run end-of-day Claude review."""
    log.info("Starting EOD review...")
    try:
        from intelligence.eod_review import run_eod_review
        await run_eod_review()
        log.info("EOD review complete.")
    except Exception as e:
        log.error(f"EOD review failed: {e}", exc_info=True)


async def task_refresh_weights():
    """Refresh combine.py model weights from rolling DB accuracy."""
    try:
        from models.kronos.combine import refresh_weights_from_db
        await refresh_weights_from_db()
    except Exception as e:
        log.warning(f"Weight refresh failed: {e}")


async def task_accuracy_tracker():
    """Compute rolling 7-day accuracy + dynamic weight adjustment."""
    log.info("Starting accuracy tracker...")
    try:
        from intelligence.accuracy_tracker import run_accuracy_tracker
        await run_accuracy_tracker()
        log.info("Accuracy tracker complete.")
    except Exception as e:
        log.error(f"Accuracy tracker failed: {e}", exc_info=True)


async def task_weekend_review():
    """Saturday deep review via Claude Opus."""
    log.info("Starting weekend review...")
    try:
        from scheduler.weekend_job import run_weekend_review
        await run_weekend_review()
        log.info("Weekend review complete.")
    except Exception as e:
        log.error(f"Weekend review failed: {e}", exc_info=True)


async def task_ticker_sync():
    """Sync NSE equity ticker list to DB."""
    log.info("Syncing NSE ticker list...")
    try:
        import asyncpg
        from data.pipeline.nse_ticker_loader import load_from_nse_csv, sync_to_db
        tickers = load_from_nse_csv()
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        count = await sync_to_db(conn, tickers)
        await conn.close()
        log.info(f"Ticker sync complete: {count} tickers.")
    except Exception as e:
        log.error(f"Ticker sync failed: {e}", exc_info=True)


# ------------------------------------------------------------------ #
# Scheduler setup                                                       #
# ------------------------------------------------------------------ #

def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

    # Ticker sync: daily at 8:00 AM IST
    scheduler.add_job(
        task_ticker_sync,
        CronTrigger(hour=8, minute=0),
        id="ticker_sync",
        name="NSE ticker list sync",
        replace_existing=True,
    )

    # OHLCV incremental: Mon-Fri at 6:30 PM IST (after NSE closes at 3:30 PM)
    scheduler.add_job(
        task_incremental_ohlcv,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=30),
        id="incremental_ohlcv",
        name="Incremental OHLCV update",
        replace_existing=True,
    )

    # F&O incremental: Mon-Fri at 6:45 PM IST
    scheduler.add_job(
        task_incremental_fo,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=45),
        id="incremental_fo",
        name="Incremental F&O update",
        replace_existing=True,
    )

    # Signal pipeline: every 30 min during market hours (9:15-15:45 IST, Mon-Fri)
    scheduler.add_job(
        task_signal_pipeline,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="15,45",
            timezone="Asia/Kolkata",
        ),
        id="signal_pipeline",
        name="Signal pipeline",
        replace_existing=True,
    )

    # Position re-analysis: every 30 min during market hours, offset 10 min from the
    # pipeline so Kronos (signals) and Kronos (re-analysis) don't fight for the 4GB GPU.
    scheduler.add_job(
        task_position_review,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="25,55",
            timezone="Asia/Kolkata",
        ),
        id="position_review",
        name="Position re-analysis",
        replace_existing=True,
    )

    # EOD review: Mon-Fri at 3:45 PM IST (15 min after market close)
    scheduler.add_job(
        task_eod_review,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=45),
        id="eod_review",
        name="EOD Claude review",
        replace_existing=True,
    )

    # Refresh combine weights every hour during market hours
    scheduler.add_job(
        task_refresh_weights,
        CronTrigger(day_of_week="mon-fri", hour="9-16", minute=5),
        id="refresh_weights",
        name="Refresh model combine weights",
        replace_existing=True,
    )

    # Accuracy tracker: Mon-Fri at 8:00 PM IST (after EOD review)
    scheduler.add_job(
        task_accuracy_tracker,
        CronTrigger(day_of_week="mon-fri", hour=20, minute=0),
        id="accuracy_tracker",
        name="Rolling accuracy + weight adjustment",
        replace_existing=True,
    )

    # Weekend deep review: Saturday 9:00 AM IST
    scheduler.add_job(
        task_weekend_review,
        CronTrigger(day_of_week="sat", hour=9, minute=0),
        id="weekend_review",
        name="Weekend Claude Opus deep review",
        replace_existing=True,
    )

    return scheduler


# ------------------------------------------------------------------ #
# Main entry point                                                      #
# ------------------------------------------------------------------ #

async def main():
    log.info("StockSense market runner starting...")
    scheduler = build_scheduler()
    scheduler.start()
    log.info(f"Scheduler started. Jobs: {[j.name for j in scheduler.get_jobs()]}")

    # Graceful shutdown. loop.add_signal_handler is not supported on Windows'
    # ProactorEventLoop, so fall back to signal.signal there.
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(signame: str):
        log.info(f"Received {signame}, shutting down...")
        scheduler.shutdown(wait=False)
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown, sig.name)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: _shutdown(signal.Signals(s).name))

    await stop_event.wait()
    log.info("Market runner stopped.")


if __name__ == "__main__":
    asyncio.run(main())
