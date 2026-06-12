"""
Nightly calibration — the brain adjusting its own knobs.

Runs after the EOD review (16:15 IST). Gathers the day's resolved signal
outcomes, auto-trade results, and learnings; asks Claude Opus (via the CLI)
for bounded parameter adjustments; applies them through brain_params.set_param
(clamped, audited). Unparseable or empty responses change nothing — the system
can only drift within its hard bounds, and every move is logged.
"""
from __future__ import annotations

import logging

import asyncpg

from config import settings
from intelligence.activity import log_activity
from intelligence.brain_params import set_param
from intelligence.claude_cli import calibration_call

log = logging.getLogger(__name__)


async def _gather_day_stats(conn) -> dict:
    """Today's resolved outcomes + auto-trade ledger, the evidence calibration runs on."""
    sig = await conn.fetchrow(
        """
        SELECT COUNT(*)                                            AS fired,
               COUNT(*) FILTER (WHERE status = 'hit_target')       AS hit_target,
               COUNT(*) FILTER (WHERE status = 'hit_stop')         AS hit_stop,
               COUNT(*) FILTER (WHERE status = 'expired')          AS expired,
               AVG(final_confidence)                               AS avg_confidence
        FROM signals WHERE fired_at::date = CURRENT_DATE
        """
    )
    dec = await conn.fetchrow(
        """
        SELECT COUNT(*) FILTER (WHERE action = 'BUY')              AS buys,
               COUNT(*) FILTER (WHERE action = 'SELL')             AS sells,
               COUNT(*) FILTER (WHERE action = 'PASS')             AS passes,
               COALESCE(SUM(pnl) FILTER (WHERE action = 'SELL'), 0) AS realized_pnl,
               COUNT(*) FILTER (WHERE action = 'SELL' AND outcome = 'win')  AS wins,
               COUNT(*) FILTER (WHERE action = 'SELL' AND outcome = 'loss') AS losses
        FROM decisions WHERE decided_at::date = CURRENT_DATE
        """
    )
    # Resolved over the last 7 days for a less noisy win-rate baseline.
    week = await conn.fetchrow(
        """
        SELECT COUNT(*)                                            AS resolved,
               COUNT(*) FILTER (WHERE outcome = 'win')             AS wins,
               COALESCE(SUM(pnl), 0)                               AS pnl
        FROM decisions
        WHERE action = 'SELL' AND resolved_at >= NOW() - INTERVAL '7 days'
        """
    )
    acct = await conn.fetchrow(
        "SELECT cash_available FROM account ORDER BY id DESC LIMIT 1"
    )
    return {
        "signals_today": dict(sig) if sig else {},
        "decisions_today": dict(dec) if dec else {},
        "week_trades": dict(week) if week else {},
        "cash_available": float(acct["cash_available"]) if acct else 0.0,
    }


async def run_calibration() -> dict:
    """Full calibration cycle. Returns {applied: [...], summary: str}."""
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        day_stats = await _gather_day_stats(conn)
        params = [dict(r) for r in await conn.fetch(
            "SELECT param_name, value, min_value, max_value FROM brain_params ORDER BY param_name"
        )]
        learnings = [dict(r) for r in await conn.fetch(
            "SELECT title, body FROM learnings WHERE learning_date = CURRENT_DATE LIMIT 20"
        )]

        log.info("Calibration: %s", day_stats)
        result = calibration_call(day_stats, params, learnings)
        summary = result.get("summary", "")
        changes = result.get("changes", [])

        known = {p["param_name"] for p in params}
        applied = []
        for ch in changes:
            name = ch.get("param")
            value = ch.get("new_value")
            reason = ch.get("reason", "")
            if name not in known or not isinstance(value, (int, float)):
                log.warning("Calibration rejected change %s — unknown param or bad value", ch)
                continue
            try:
                applied.append(await set_param(
                    conn, name, float(value),
                    changed_by="claude_calibration", reason=reason,
                ))
            except Exception as e:
                log.warning("Calibration apply failed for %s: %s", name, e)

        await log_activity(
            conn, event_type="NOTE",
            note=f"Calibration: {len(applied)} change(s). {summary[:300]}",
            payload={"applied": applied, "summary": summary,
                     "day_stats": {k: str(v) for k, v in day_stats.items()}},
        )
        log.info("Calibration done: %d applied. %s", len(applied), summary[:200])
        return {"applied": applied, "summary": summary}
    finally:
        await conn.close()
