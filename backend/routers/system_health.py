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
from datetime import datetime, timezone

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
    finally:
        await conn.close()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "data_freshness": freshness,
        "angel_one": _angel_one_circuit_breaker(),
        "lightgbm_model": _lightgbm_model_age(),
    }
