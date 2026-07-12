"""
StockSense market scheduler — the always-on autonomous brain.

If this process is running, the brain is running: data updates itself, signals
fire, paper trades execute, positions exit, parameters calibrate, the model
retrains — with no buttons anywhere. Every job execution is recorded in the
job_runs table (the heartbeat the UI reads) and in logs/<date>/app.log.

Intended to run as: python -m scheduler.market_runner
"""
from __future__ import annotations

import asyncio
import functools
import logging
import signal
import sys
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path

import asyncpg
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Daily file logging — same files the Logs panel tails                 #
# ------------------------------------------------------------------ #
_file_handler: logging.FileHandler | None = None
_file_handler_day: str | None = None


def _ensure_daily_log_handler():
    """Attach (and roll) a logs/<date>/app.log handler on the root logger."""
    global _file_handler, _file_handler_day
    today = date.today().isoformat()
    if _file_handler_day == today and _file_handler is not None:
        return
    root = logging.getLogger()
    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()
    log_dir = Path("logs") / today
    log_dir.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(_file_handler)
    _file_handler_day = today


# ------------------------------------------------------------------ #
# Job heartbeat — every run lands in job_runs                          #
# ------------------------------------------------------------------ #

def instrumented(job_id: str):
    """Wrap a task: record start/finish/status/summary in job_runs."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            _ensure_daily_log_handler()
            run_id = None
            try:
                conn = await asyncpg.connect(settings.DATABASE_DSN)
                try:
                    run_id = await conn.fetchval(
                        "INSERT INTO job_runs (job_id, started_at) VALUES ($1, NOW()) RETURNING id",
                        job_id,
                    )
                finally:
                    await conn.close()
            except Exception as e:
                log.warning("job_runs start insert failed for %s: %s", job_id, e)

            status, summary, error = "ok", "", None
            try:
                result = await fn(*args, **kwargs)
                if isinstance(result, str):
                    summary = result[:500]
                elif isinstance(result, dict):
                    summary = str(result)[:500]
            except Exception as e:
                status, error = "error", str(e)[:1000]
                log.error("Job %s failed: %s", job_id, e, exc_info=True)
            finally:
                if run_id is not None:
                    try:
                        conn = await asyncpg.connect(settings.DATABASE_DSN)
                        try:
                            await conn.execute(
                                "UPDATE job_runs SET finished_at = NOW(), status = $2, "
                                "summary = $3, error = $4 WHERE id = $1",
                                run_id, status, summary, error,
                            )
                        finally:
                            await conn.close()
                    except Exception as e:
                        log.warning("job_runs finish update failed for %s: %s", job_id, e)
        return wrapper
    return deco


# ------------------------------------------------------------------ #
# Task functions                                                        #
# ------------------------------------------------------------------ #

@instrumented("incremental_ohlcv")
async def task_incremental_ohlcv(days_back: int = 5) -> str:
    """
    Pull latest daily OHLCV. Source order:
      1. Groww (TOTP auth — works on rotating home IPs; needs creds in .env)
      2. NSE Bhavcopy new format (2024-07-08+, reliable on this network)
      3. NSE Bhavcopy old format incremental
      4. Angel One (last resort — IP-bound, usually blocked here)
    """
    try:
        from data.pipeline.fetch_groww import run_groww_daily_backfill, creds_present
        if creds_present():
            summary = await run_groww_daily_backfill(days_back=days_back)
            log.info("Groww OHLCV update: %s", summary)
            if summary.get("updated", 0) > 0:
                return f"groww: {summary}"
            log.warning("Groww updated 0 tickers — falling back to NSE Bhavcopy.")
        else:
            log.info("Groww creds absent — using NSE Bhavcopy.")
    except Exception as e:
        log.error("Groww OHLCV update failed: %s — falling back.", e)

    try:
        from datetime import timedelta
        from data.pipeline.fetch_historical_new import run as run_new_bhavcopy
        end = date.today()
        start = end - timedelta(days=days_back)
        await run_new_bhavcopy(start=start, end=end, resume=False)
        return "bhavcopy_new ok"
    except Exception as e:
        log.error("New-format Bhavcopy failed: %s — falling back.", e)

    try:
        from data.pipeline.fetch_historical import run_incremental
        await run_incremental(days_back=days_back)
        return "bhavcopy_old ok"
    except Exception as e:
        log.error("Old-format Bhavcopy failed: %s — trying Angel One.", e)

    try:
        from data.pipeline.fetch_angel_daily import run_angel_backfill
        summary = await run_angel_backfill(days_back=days_back)
        return f"angel: {summary}"
    except Exception as e:
        log.error("All OHLCV sources failed: %s", e)
        raise


@instrumented("data_freshness")
async def task_data_freshness() -> str:
    """Pre-market check (8:45 IST): if daily data is stale, backfill deeper."""
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        latest = await conn.fetchval("SELECT MAX(time) FROM ohlcv_daily")
    finally:
        await conn.close()
    if latest is None:
        return "no data at all — skipping (seed required)"
    age_days = (datetime.now(timezone.utc) - latest).days
    if age_days <= 1:
        return f"fresh (latest {latest.date()}, age {age_days}d)"
    log.warning("Daily data stale (%sd old) — backfilling %s days.", age_days, age_days + 3)
    await task_incremental_ohlcv.__wrapped__(days_back=age_days + 3)
    return f"backfilled {age_days + 3} days (was {age_days}d stale)"


@instrumented("groww_intraday")
async def task_groww_intraday() -> str:
    """Live quote snapshot every 5 min during market hours (no-op without creds).

    RETIRED from the scheduler (see build_scheduler() below) — this wrote fake
    flat-OHLC "1-minute" candles into ohlcv_1min from a 5-minute-stale LTP
    snapshot. Function left in place, unregistered, in case it's needed for
    manual/ad-hoc use. See WHAT_TO_DO_NEXT.txt Section 2.8.
    """
    from data.pipeline.fetch_groww import run_groww_intraday_snapshot, creds_present
    if not creds_present():
        return "skipped: no groww creds"
    result = await run_groww_intraday_snapshot()
    return str(result)


@instrumented("upstox_bhavcopy_reconciliation")
async def task_upstox_bhavcopy_reconciliation() -> str:
    """
    Nightly sanity check (18:35 IST, right after incremental_ohlcv at 18:30):
    compares each ticker's Upstox-sourced daily close against the NSE Bhavcopy
    close already stored in ohlcv_daily for the same date, and logs a warning
    (via intelligence.activity.log_activity) if they differ by more than 0.1%.

    Additive and non-blocking: a mismatch is a warning, not a hard failure, and
    this job never writes/overwrites ohlcv_daily — it only compares and logs.

    Upstox is being wired in by a separate workstream (data/pipeline/upstox_client.py).
    Until that lands, this defensively no-ops (logs + returns) rather than failing,
    same pattern used elsewhere in this codebase for optional dependencies.
    """
    try:
        from data.pipeline.upstox_client import (
            creds_present, resolve_instrument_keys, get_historical_candles,
        )
    except (ImportError, AttributeError) as e:
        log.info("upstox_client not available/incomplete yet (%s) — skipping Bhavcopy reconciliation.", e)
        return f"skipped: upstox_client not available ({e})"

    if not creds_present():
        return "skipped: no upstox creds"

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        today = date.today()
        rows = await conn.fetch(
            "SELECT ticker, close FROM ohlcv_daily WHERE time::date = $1",
            today,
        )
        if not rows:
            return f"no ohlcv_daily rows for {today} yet — skipping"
        # Cap the check to a bounded sample so a large universe doesn't hammer the
        # Upstox API nightly; this is a sanity spot-check, not exhaustive audit.
        bhavcopy_closes = {r["ticker"]: float(r["close"]) for r in rows if r["close"] is not None}
        sample_tickers = list(bhavcopy_closes.keys())[:200]

        try:
            instrument_keys = await resolve_instrument_keys(sample_tickers)
            upstox_closes: dict[str, float] = {}
            iso_today = today.isoformat()
            for ticker in sample_tickers:
                inst_key = instrument_keys.get(ticker)
                if not inst_key:
                    continue
                candles = await get_historical_candles(
                    inst_key, unit="days", interval="1", to_date=iso_today,
                )
                if candles:
                    # Newest candle isn't guaranteed to be last — pick by timestamp.
                    latest = max(candles, key=lambda c: c.get("timestamp", ""))
                    close = latest.get("close")
                    if close is not None:
                        upstox_closes[ticker] = float(close)
        except Exception as e:
            log.warning("Upstox daily-close fetch failed during reconciliation: %s", e)
            return f"skipped: upstox fetch failed ({e})"

        mismatches = []
        for ticker, bhav_close in bhavcopy_closes.items():
            up_close = upstox_closes.get(ticker)
            if up_close is None or bhav_close == 0:
                continue
            pct_diff = abs(up_close - bhav_close) / bhav_close * 100
            if pct_diff > 0.1:
                mismatches.append((ticker, bhav_close, up_close, pct_diff))

        if mismatches:
            from intelligence.activity import log_activity
            summary_note = (
                f"{len(mismatches)} ticker(s) mismatched >0.1%% between Upstox and "
                f"Bhavcopy closes on {today}: " +
                ", ".join(f"{t} (bhav={b:.2f} upstox={u:.2f} diff={d:.2f}%)"
                          for t, b, u, d in mismatches[:10])
            )
            log.warning(summary_note)
            await log_activity(
                conn,
                event_type="NOTE",
                note=summary_note[:500],
                payload={"mismatches": [
                    {"ticker": t, "bhavcopy_close": b, "upstox_close": u, "pct_diff": d}
                    for t, b, u, d in mismatches
                ]},
            )
        return f"{len(mismatches)} mismatch(es) out of {len(upstox_closes)} tickers checked (sampled {len(sample_tickers)})"
    finally:
        await conn.close()


@instrumented("incremental_fo")
async def task_incremental_fo() -> str:
    """Pull latest F&O data from NSE Bhavcopy."""
    from data.pipeline.fetch_f_and_o import run_fo_incremental
    await run_fo_incremental(days_back=3)
    return "ok"


@instrumented("signal_pipeline")
async def task_signal_pipeline() -> str:
    """
    The core loop: run the multi-timeframe pipeline, then let the brain act on
    its own signals. No human in the loop — every decision is logged.
    """
    from intelligence.signal_pipeline import run_pipeline_multi
    from intelligence.auto_trader import auto_trade

    sigs = await run_pipeline_multi(save=True)
    log.info("Signal pipeline complete: %d BUY signals.", len(sigs))

    trade_summary = await auto_trade()
    return (f"{len(sigs)} signals; auto: {len(trade_summary['bought'])} bought, "
            f"{len(trade_summary['passed'])} passed")


@instrumented("position_review")
async def task_position_review() -> str:
    """Re-analyze held positions, then act on EXIT verdicts automatically."""
    from intelligence.position_monitor import review_all_positions
    from intelligence.auto_trader import auto_exit

    reviews = await review_all_positions()
    log.info("Position re-analysis complete: %d positions reviewed.", len(reviews))

    exit_summary = await auto_exit(reviews)
    return f"{len(reviews)} reviewed; {len(exit_summary['sold'])} auto-exited"


@instrumented("eod_review")
async def task_eod_review() -> str:
    """Run end-of-day Claude review."""
    from intelligence.eod_review import run_eod_review
    await run_eod_review()
    return "ok"


@instrumented("calibration")
async def task_calibration() -> str:
    """Nightly self-calibration: Claude proposes bounded brain-param changes."""
    from intelligence.calibration import run_calibration
    result = await run_calibration()
    return f"{len(result['applied'])} change(s): {result['summary'][:200]}"


@instrumented("refresh_weights")
async def task_refresh_weights() -> str:
    """Refresh combine.py model weights from rolling DB accuracy."""
    from models.ml.combine import refresh_weights_from_db
    await refresh_weights_from_db()
    return "ok"


@instrumented("accuracy_tracker")
async def task_accuracy_tracker() -> str:
    """Rolling 7-day accuracy + dynamic weights + auto-retrain check."""
    from intelligence.accuracy_tracker import run_accuracy_tracker
    await run_accuracy_tracker()
    return "ok"


@instrumented("weekend_review")
async def task_weekend_review() -> str:
    """Saturday deep review via Claude Opus."""
    from scheduler.weekend_job import run_weekend_review
    await run_weekend_review()
    return "ok"


@instrumented("expire_confirmations")
async def task_expire_confirmations() -> str:
    """
    Flip stale PENDING rows in pending_trade_confirmations to EXPIRED. A row
    left PENDING for a long time quotes a price/target that's drifted from
    reality — approving it later would place an order at a stale limit
    price the human never actually reviewed. Self-gated on
    LIVE_CONFIRMATION_ENABLED (same pattern as queue_fresh_signals() in
    intelligence/live_confirmation.py) so this job is a harmless no-op
    unless the human-confirmation feature is actually turned on; registered
    unconditionally below like every other job, per this scheduler's
    "no buttons anywhere, gating lives in config" convention.
    """
    if not settings.LIVE_CONFIRMATION_ENABLED:
        return "LIVE_CONFIRMATION_ENABLED is false — nothing to expire"

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        rows = await conn.fetch(
            """
            UPDATE pending_trade_confirmations
            SET status = 'EXPIRED', resolved_at = NOW()
            WHERE status = 'PENDING'
              AND created_at < NOW() - ($1 * INTERVAL '1 minute')
            RETURNING id, ticker
            """,
            settings.CONFIRMATION_EXPIRY_MINUTES,
        )
        if rows:
            log.info("Expired %d stale confirmation(s): %s",
                      len(rows), ", ".join(r["ticker"] for r in rows))
        return f"{len(rows)} confirmation(s) expired"
    finally:
        await conn.close()


@instrumented("reconcile_sandbox_orders")
async def task_reconcile_sandbox_orders() -> str:
    """
    Poll Upstox for the real exchange-side status of every confirmation row
    that has an order_id but is still reporting execution_status='PLACED'.
    place_sandbox_order() only confirms Upstox *accepted* the order for
    processing — not that it filled. Without this job, a rejected-after-
    acceptance order would sit forever showing the misleadingly final-
    sounding "PLACED" in the UI. Self-gated on LIVE_CONFIRMATION_ENABLED,
    same pattern as task_expire_confirmations above — this whole feature
    stays inert until the human opts in.
    """
    if not settings.LIVE_CONFIRMATION_ENABLED:
        return "LIVE_CONFIRMATION_ENABLED is false — nothing to reconcile"

    from data.pipeline.upstox_orders import get_order_status

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        open_rows = await conn.fetch(
            """
            SELECT id, order_id, ticker FROM pending_trade_confirmations
            WHERE order_id IS NOT NULL AND execution_status = 'PLACED'
            """
        )
        updated = 0
        for r in open_rows:
            result = await get_order_status(r["order_id"])
            if result["status"] == "UNKNOWN":
                # Transient fetch failure — leave as PLACED, retry next run.
                continue
            await conn.execute(
                """
                UPDATE pending_trade_confirmations
                SET execution_status = $2, execution_detail = $3
                WHERE id = $1
                """,
                r["id"], result["status"], result.get("detail"),
            )
            updated += 1
            log.info("Reconciled order for %s (order_id=%s): %s",
                      r["ticker"], r["order_id"], result["status"])
        return f"{updated}/{len(open_rows)} order(s) reconciled"
    finally:
        await conn.close()


@instrumented("ticker_sync")
async def task_ticker_sync() -> str:
    """Sync NSE equity ticker list to DB."""
    from data.pipeline.nse_ticker_loader import load_from_nse_csv, sync_to_db
    tickers = load_from_nse_csv()
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        count = await sync_to_db(conn, tickers)
    finally:
        await conn.close()
    return f"{count} tickers"


# ------------------------------------------------------------------ #
# Scheduler setup                                                       #
# ------------------------------------------------------------------ #

def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    common = dict(replace_existing=True, coalesce=True,
                  max_instances=1, misfire_grace_time=300)

    # Ticker sync: daily at 8:00 AM IST
    scheduler.add_job(
        task_ticker_sync,
        CronTrigger(hour=8, minute=0),
        id="ticker_sync", name="NSE ticker list sync", **common,
    )

    # Pre-market data freshness check: 8:45 IST Mon-Fri
    scheduler.add_job(
        task_data_freshness,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=45),
        id="data_freshness", name="Pre-market data freshness check", **common,
    )

    # Groww intraday snapshot job: RETIRED (see WHAT_TO_DO_NEXT.txt Section 2.8 —
    # "the fake-candle problem"). This job wrote a single 5-minute-stale LTP into
    # ohlcv_1min as a flat OHLC "1-minute candle" (open=high=low=close, zero volume),
    # which looked like granular intraday data but wasn't. Upstox is becoming the
    # primary live feed (data/pipeline/upstox_client.py / upstox_feed.py); Groww's
    # code is left intact and unused in fetch_groww.py in case of resubscription
    # (archive, don't delete — WHAT_TO_DO_NEXT.txt Section 5). Do not re-enable
    # this job without also fixing the flat-candle issue.
    # scheduler.add_job(
    #     task_groww_intraday,
    #     CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5"),
    #     id="groww_intraday", name="Groww live quote snapshot", **common,
    # )

    # OHLCV incremental: Mon-Fri at 6:30 PM IST (after NSE closes at 3:30 PM)
    scheduler.add_job(
        task_incremental_ohlcv,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=30),
        id="incremental_ohlcv", name="Incremental OHLCV update", **common,
    )

    # Nightly Upstox-vs-Bhavcopy close reconciliation: Mon-Fri at 6:35 PM IST,
    # right after incremental_ohlcv (6:30 PM). No-ops until Upstox is wired in
    # (see task_upstox_bhavcopy_reconciliation docstring).
    scheduler.add_job(
        task_upstox_bhavcopy_reconciliation,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=35),
        id="upstox_bhavcopy_reconciliation", name="Upstox vs Bhavcopy close reconciliation", **common,
    )

    # F&O incremental: Mon-Fri at 6:45 PM IST
    scheduler.add_job(
        task_incremental_fo,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=45),
        id="incremental_fo", name="Incremental F&O update", **common,
    )

    # Signal pipeline + autonomous trading: every 30 min during market hours
    scheduler.add_job(
        task_signal_pipeline,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="15,45"),
        id="signal_pipeline", name="Signal pipeline + auto-trade", **common,
    )

    # Position re-analysis + autonomous exits: offset 10 min from the pipeline
    # so Kronos (signals) and Kronos (re-analysis) don't fight for the 4GB GPU.
    scheduler.add_job(
        task_position_review,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="25,55"),
        id="position_review", name="Position re-analysis + auto-exit", **common,
    )

    # EOD review: Mon-Fri at 3:45 PM IST (15 min after market close)
    scheduler.add_job(
        task_eod_review,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=45),
        id="eod_review", name="EOD Claude review", **common,
    )

    # Nightly calibration: RETIRED (see WHAT_TO_DO_NEXT.txt Section 2.7 / 4 Phase 2).
    # A single day's resolved outcomes was too noisy a basis for parameter changes.
    # Calibration now runs weekly, from scheduler/weekend_job.py's Saturday flow
    # (task_weekend_review below), requiring >= MIN_RESOLVED_OUTCOMES resolved
    # signals over the trailing week and EWMA-smoothing + rollback (see
    # intelligence/calibration.py). task_calibration is left intact and unregistered
    # in case of manual/ad-hoc use (archive, don't delete — same convention as the
    # Groww intraday snapshot retirement above).
    # scheduler.add_job(
    #     task_calibration,
    #     CronTrigger(day_of_week="mon-fri", hour=16, minute=15),
    #     id="calibration", name="Brain parameter calibration", **common,
    # )

    # Refresh combine weights every hour during market hours
    scheduler.add_job(
        task_refresh_weights,
        CronTrigger(day_of_week="mon-fri", hour="9-16", minute=5),
        id="refresh_weights", name="Refresh model combine weights", **common,
    )

    # Accuracy tracker + auto-retrain check: Mon-Fri at 8:00 PM IST
    scheduler.add_job(
        task_accuracy_tracker,
        CronTrigger(day_of_week="mon-fri", hour=20, minute=0),
        id="accuracy_tracker", name="Rolling accuracy + auto-retrain", **common,
    )

    # Weekend deep review: Saturday 9:00 AM IST
    scheduler.add_job(
        task_weekend_review,
        CronTrigger(day_of_week="sat", hour=9, minute=0),
        id="weekend_review", name="Weekend Claude Opus deep review", **common,
    )

    # Human-confirmed sandbox order hardening — both self-gate on
    # LIVE_CONFIRMATION_ENABLED (see task docstrings) and are cheap no-ops
    # otherwise, so they're registered unconditionally like every other job.
    # Expire stale PENDING confirmations every 10 min, all day.
    scheduler.add_job(
        task_expire_confirmations,
        CronTrigger(day_of_week="mon-fri", hour="9-18", minute="*/10"),
        id="expire_confirmations", name="Expire stale trade confirmations", **common,
    )
    # Reconcile PLACED sandbox orders against Upstox's real order status
    # every 5 min during market hours (irrelevant, but cheap, outside them).
    scheduler.add_job(
        task_reconcile_sandbox_orders,
        CronTrigger(day_of_week="mon-fri", hour="9-18", minute="*/5"),
        id="reconcile_sandbox_orders", name="Reconcile sandbox order status", **common,
    )

    return scheduler


# ------------------------------------------------------------------ #
# Main entry point                                                      #
# ------------------------------------------------------------------ #

async def main():
    _ensure_daily_log_handler()
    log.info("StockSense market runner starting (autonomous mode — no switches)...")
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
