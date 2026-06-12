"""
Brain API — observability for the autonomous loop.

The brain has no controls, only windows: current adaptive parameters and their
change history, the scheduler heartbeat (job_runs), the paper equity curve,
and an at-a-glance status rollup. Everything here is read-only by design.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, Query

from config import settings
from intelligence.data_freshness import get_data_freshness
from intelligence.trading_mode import get_trading_mode

router = APIRouter()
DB_DSN = settings.DATABASE_DSN

# Expected cadence per job — lets the UI flag a job that's gone quiet.
JOB_CADENCE = {
    "ticker_sync": "daily 08:00",
    "data_freshness": "Mon–Fri 08:45",
    "groww_intraday": "every 5 min (market hours)",
    "incremental_ohlcv": "Mon–Fri 18:30",
    "incremental_fo": "Mon–Fri 18:45",
    "signal_pipeline": "every 30 min (market hours)",
    "position_review": "every 30 min (market hours)",
    "eod_review": "Mon–Fri 15:45",
    "calibration": "Mon–Fri 16:15",
    "refresh_weights": "hourly (market hours)",
    "accuracy_tracker": "Mon–Fri 20:00",
    "weekend_review": "Sat 09:00",
}


@router.get("/status")
async def brain_status():
    """One-call rollup: params, heartbeat, mode, freshness, model, feed."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        params = [dict(r) for r in await conn.fetch(
            "SELECT param_name, value, min_value, max_value, updated_at, updated_by, reason "
            "FROM brain_params ORDER BY param_name"
        )]

        # Latest run per job.
        job_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (job_id) job_id, started_at, finished_at, status, summary, error
            FROM job_runs ORDER BY job_id, started_at DESC
            """
        )
        jobs = []
        for r in job_rows:
            d = dict(r)
            d["cadence"] = JOB_CADENCE.get(d["job_id"], "")
            jobs.append(d)

        mode = await get_trading_mode(conn)
        freshness = await get_data_freshness(conn)

        acct = await conn.fetchrow(
            "SELECT cash_available, cash_reserve, updated_at FROM account ORDER BY id DESC LIMIT 1"
        )
        holdings = await conn.fetchrow(
            """
            SELECT COUNT(*) AS positions,
                   COALESCE(SUM(p.quantity * (
                       SELECT close FROM ohlcv_daily o WHERE o.ticker = p.ticker
                       ORDER BY time DESC LIMIT 1
                   )), 0) AS market_value
            FROM portfolio p WHERE p.active = TRUE
            """
        )

        today = await conn.fetchrow(
            """
            SELECT COUNT(*) FILTER (WHERE action = 'BUY')  AS buys,
                   COUNT(*) FILTER (WHERE action = 'SELL') AS sells,
                   COUNT(*) FILTER (WHERE action = 'PASS') AS passes,
                   COALESCE(SUM(pnl) FILTER (WHERE action = 'SELL'), 0) AS realized_pnl
            FROM decisions
            WHERE decided_at::date = CURRENT_DATE AND rationale LIKE '[AUTO]%'
            """
        )

        # Model artifact freshness.
        model_path = os.path.join(str(settings.MODELS_DIR), "ml", "saved", "lgbm_latest.pkl")
        model_mtime = None
        if os.path.exists(model_path):
            model_mtime = datetime.fromtimestamp(
                os.path.getmtime(model_path), tz=timezone.utc
            ).isoformat()

        from data.pipeline.fetch_groww import creds_present
        cash = float(acct["cash_available"]) if acct else 0.0
        mv = float(holdings["market_value"]) if holdings else 0.0

        return {
            "autonomous": True,  # always — there is no switch
            "mode": mode,
            "data": freshness,
            "params": params,
            "jobs": jobs,
            "account": {
                "cash_available": cash,
                "cash_reserve": float(acct["cash_reserve"]) if acct else 0.0,
                "positions": int(holdings["positions"]) if holdings else 0,
                "market_value": round(mv, 2),
                "equity": round(cash + mv, 2),
            },
            "today": dict(today) if today else {},
            "model": {"lgbm_latest_mtime": model_mtime},
            "groww_feed": {"configured": creds_present()},
        }
    finally:
        await conn.close()


@router.get("/params/history")
async def param_history(limit: int = Query(100, le=500)):
    """Every parameter change the brain has made to itself, newest first."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            """
            SELECT param_name, old_value, new_value, changed_by, reason, changed_at
            FROM brain_param_history ORDER BY changed_at DESC LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@router.get("/jobs")
async def job_history(limit: int = Query(100, le=500), job_id: str | None = None):
    """Scheduler heartbeat history."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        if job_id:
            rows = await conn.fetch(
                "SELECT job_id, started_at, finished_at, status, summary, error "
                "FROM job_runs WHERE job_id = $2 ORDER BY started_at DESC LIMIT $1",
                limit, job_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT job_id, started_at, finished_at, status, summary, error "
                "FROM job_runs ORDER BY started_at DESC LIMIT $1",
                limit,
            )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@router.get("/equity")
async def equity_curve(days: int = Query(90, le=730)):
    """
    Daily paper equity curve: cash (last ledger row per day, carried forward)
    + mark-to-market of whatever was held at each day's close.
    """
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            """
            WITH days AS (
                SELECT generate_series(
                    GREATEST(
                        (SELECT MIN(decided_at)::date FROM decisions),
                        CURRENT_DATE - $1::int
                    ),
                    CURRENT_DATE, '1 day'
                )::date AS day
            ),
            cash_by_day AS (
                -- last known cash on or before each day
                SELECT d.day,
                       (SELECT a.cash_available FROM account a
                        WHERE a.updated_at::date <= d.day
                        ORDER BY a.id DESC LIMIT 1) AS cash
                FROM days d
            ),
            -- net shares held per ticker as of each day (BUYs minus SELLs)
            pos_by_day AS (
                SELECT d.day, dec.ticker,
                       SUM(CASE WHEN dec.action = 'BUY' THEN dec.quantity
                                WHEN dec.action = 'SELL' THEN -dec.quantity
                                ELSE 0 END) AS qty
                FROM days d
                JOIN decisions dec
                  ON dec.decided_at::date <= d.day
                 AND dec.action IN ('BUY','SELL')
                GROUP BY d.day, dec.ticker
                HAVING SUM(CASE WHEN dec.action = 'BUY' THEN dec.quantity
                                WHEN dec.action = 'SELL' THEN -dec.quantity
                                ELSE 0 END) > 0
            ),
            mv_by_day AS (
                SELECT p.day,
                       SUM(p.qty * (
                           SELECT o.close FROM ohlcv_daily o
                           WHERE o.ticker = p.ticker AND o.time::date <= p.day
                           ORDER BY o.time DESC LIMIT 1
                       )) AS market_value
                FROM pos_by_day p
                GROUP BY p.day
            )
            SELECT c.day,
                   COALESCE(c.cash, 0)                          AS cash,
                   COALESCE(m.market_value, 0)                  AS market_value,
                   COALESCE(c.cash, 0) + COALESCE(m.market_value, 0) AS equity
            FROM cash_by_day c
            LEFT JOIN mv_by_day m ON m.day = c.day
            ORDER BY c.day
            """,
            days,
        )
        return [
            {
                "date": r["day"].isoformat(),
                "cash": float(r["cash"]),
                "market_value": float(r["market_value"]),
                "equity": float(r["equity"]),
            }
            for r in rows
        ]
    finally:
        await conn.close()
