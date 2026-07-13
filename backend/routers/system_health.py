"""
System health API — the Stage 0 truth layer, surfaced.

Answers "what is actually running right now?" for each pipeline component
(kronos, lightgbm, llm_synthesis, macro), plus overall data freshness, the
Angel One circuit-breaker state, and LightGBM model age. This is the
read-only counterpart to the per-signal components_json: it reflects
CURRENT component health rather than what was true at the moment a
particular signal fired.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import asyncpg
from fastapi import APIRouter

from config import settings
from intelligence.data_freshness import get_data_freshness
from intelligence.signal_pipeline import get_component_statuses

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


def _lightgbm_model_age() -> dict:
    """Age of models/ml/saved/lgbm_latest.pkl, independent of the pipeline's cached model."""
    try:
        from models.ml.predict import MODEL_SAVE_DIR
        path = os.path.join(MODEL_SAVE_DIR, "lgbm_latest.pkl")
        if not os.path.exists(path):
            return {"exists": False, "mtime": None, "age_hours": None}
        mtime = os.path.getmtime(path)
        age_hours = round((time.time() - mtime) / 3600, 1)
        return {
            "exists": True,
            "mtime": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            "age_hours": age_hours,
        }
    except Exception as e:  # noqa: BLE001
        return {"exists": False, "mtime": None, "age_hours": None, "error": str(e)[:160]}


def _angel_one_circuit_breaker() -> dict:
    """Reads market_data.py's module-level circuit breaker state, if importable."""
    try:
        from backend.routers import market_data as md
        fail_ts = getattr(md, "_fail_ts", 0.0)
        backoff = getattr(md, "_BACKOFF_SECS", 300)
        last_error = getattr(md, "_last_fetch_error", None)
        tripped = bool(fail_ts) and (time.monotonic() - fail_ts) < backoff
        retry_in = max(0, round(backoff - (time.monotonic() - fail_ts), 1)) if tripped else 0
        return {
            "tripped": tripped,
            "last_error": last_error,
            "retry_in_seconds": retry_in,
        }
    except Exception as e:  # noqa: BLE001
        return {"tripped": None, "last_error": str(e)[:160], "retry_in_seconds": None}


def _quote_feed_health() -> dict:
    """
    Is the Upstox live WS feed actually connected and ticking? A dead feed
    used to silently disable ALL fast intraday stop enforcement with no
    user-visible signal — this component exists so that shows up here
    instead of only as a debug log line nobody reads.
      - ok: connected and a tick landed within the last 60s
      - degraded: connected but stale (60s-5min since last tick), or briefly
        disconnected/reconnecting
      - unavailable: no tick in 5+ min, or never connected
    """
    try:
        from backend.services import quote_cache
        status = quote_cache.feed_status()
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": f"quote_cache import/status failed: {str(e)[:160]}"}

    age = status["last_tick_age_seconds"]
    connected = status["connected"]
    n = status["subscribed_count"]

    if not connected:
        detail = f"feed not connected ({n} keys subscribed when last up)"
        if status["last_connect_error"]:
            detail += f" — {status['last_connect_error']}"
        return {"status": "unavailable", "detail": detail}

    if age is None:
        return {"status": "degraded", "detail": f"connected ({n} keys) but no tick received yet"}

    detail = f"connected, {n} keys subscribed, last tick {age:.0f}s ago"
    if age <= 60:
        return {"status": "ok", "detail": detail}
    if age <= 300:
        return {"status": "degraded", "detail": detail + " — stale, fast stops may be lagging"}
    return {"status": "unavailable", "detail": detail + " — feed likely dead, fast stops NOT protecting open positions"}


async def _scheduler_heartbeat(conn) -> dict:
    """
    Is scheduler/market_runner.py actually alive? Every job it runs inserts a
    job_runs row, so MAX(started_at) is a de-facto heartbeat. Thresholds are
    deliberately simple (weekends/holidays make "stale" fuzzy):
      - during IST market hours (Mon-Fri, 9:00-18:59) jobs fire at least every
        30 min, so > 2 hours old = degraded;
      - otherwise (nights/weekends) the daily 8:00 ticker_sync still fires, so
        > 26 hours old = unavailable regardless of holidays.
    The scheduler being down was invisible for weeks (mid-June incident) —
    this component exists so that never happens silently again.
    """
    try:
        last = await conn.fetchval("SELECT MAX(started_at) FROM job_runs")
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": f"job_runs query failed: {str(e)[:160]}"}
    if last is None:
        return {"status": "unavailable", "detail": "no job_runs rows at all — scheduler has never run"}

    now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age_hours = (now - last).total_seconds() / 3600

    # Market hours in IST (UTC+5:30), Mon-Fri 9:00-18:59 — covers the trading
    # cadence plus the evening data jobs.
    ist = now + timedelta(hours=5, minutes=30)
    market_hours = ist.weekday() < 5 and 9 <= ist.hour < 19
    threshold = 2 if market_hours else 26

    detail = f"last job_runs heartbeat {age_hours:.1f}h ago (threshold {threshold}h)"
    if age_hours <= threshold:
        return {"status": "ok", "detail": detail}
    if age_hours <= 26:
        return {"status": "degraded", "detail": detail + " — scheduler may be down or asleep"}
    return {"status": "unavailable",
            "detail": detail + " — scheduler process appears DOWN (check start.ps1 / market_runner window)"}


@router.get("/health")
async def system_health():
    """
    One-shot rollup of every truth-layer signal in the app:
      - components: {kronos, lightgbm, llm_synthesis, macro} status per the shared contract
      - data_freshness: is the pipeline running on live intraday or stale EOD data
      - angel_one: circuit-breaker state for the live index feed
      - lightgbm_model: on-disk model file age (independent of in-memory cache)
    """
    components = get_component_statuses()

    conn = await asyncpg.connect(DB_DSN)
    try:
        freshness = await get_data_freshness(conn)
        # Scheduler heartbeat rides in the components dict so the frontend
        # SystemHealthBar (which iterates Object.entries(components)) renders
        # it without special-casing.
        components["scheduler"] = await _scheduler_heartbeat(conn)
        components["quote_feed"] = _quote_feed_health()
    finally:
        await conn.close()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "data_freshness": freshness,
        "angel_one": _angel_one_circuit_breaker(),
        "lightgbm_model": _lightgbm_model_age(),
    }
