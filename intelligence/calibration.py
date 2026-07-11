"""
Weekly calibration — the brain adjusting its own knobs.

Was nightly; moved to the Saturday weekend flow (scheduler/weekend_job.py) so
proposals are backed by a full trading week of resolved outcomes instead of a
single noisy day (see WHAT_TO_DO_NEXT.txt Section 2.7 / 4 Phase 2). Requires
at least MIN_RESOLVED_OUTCOMES resolved signals over the trailing week before
proposing anything; otherwise it skips and logs why.

Claude's proposed new_value is never applied verbatim — it's EWMA-blended
with the current value (CALIBRATION_ALPHA weight on the proposal) so a single
week's read can't yank a param to its bound in one move. Every change is
still clamped + audited through brain_params.set_param.

Rollback: before proposing anything new, compares last week's realized P&L to
the week before it. If it's worse AND there were claude_calibration changes
in that prior window (per brain_param_history), those specific param changes
are reverted to their pre-calibration value and the rollback is logged as an
activity event — the brain can undo its own mistakes, not just make them.
"""
from __future__ import annotations

import logging

import asyncpg

from config import settings
from intelligence.activity import log_activity
from intelligence.brain_params import set_param
from intelligence.claude_cli import calibration_call
from intelligence.eod_review import get_active_learnings

log = logging.getLogger(__name__)

MIN_RESOLVED_OUTCOMES = 15   # don't propose changes on a thin week
CALIBRATION_ALPHA = 0.3      # EWMA weight on Claude's proposed value vs current


async def _gather_week_stats(conn) -> dict:
    """Trailing 7-day resolved outcomes + auto-trade ledger — the evidence
    weekly calibration runs on."""
    sig = await conn.fetchrow(
        """
        SELECT COUNT(*)                                            AS resolved,
               COUNT(*) FILTER (WHERE status = 'hit_target')       AS hit_target,
               COUNT(*) FILTER (WHERE status = 'hit_stop')         AS hit_stop,
               COUNT(*) FILTER (WHERE status = 'expired')          AS expired,
               AVG(final_confidence)                               AS avg_confidence
        FROM signals
        WHERE resolved_at >= NOW() - INTERVAL '7 days'
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
        FROM decisions
        WHERE decided_at >= NOW() - INTERVAL '7 days'
        """
    )
    acct = await conn.fetchrow(
        "SELECT cash_available FROM account ORDER BY id DESC LIMIT 1"
    )
    return {
        "week_signals": dict(sig) if sig else {},
        "week_decisions": dict(dec) if dec else {},
        "cash_available": float(acct["cash_available"]) if acct else 0.0,
    }


async def _maybe_rollback(conn, week_decisions: dict) -> list[dict]:
    """
    Compare this week's realized P&L to the week before it (14-7 days ago).
    If it's worse and there were claude_calibration changes in that prior
    window, revert those specific params to their old_value. Returns the list
    of reverted-param dicts (same shape as set_param's return).
    """
    prev = await conn.fetchrow(
        """
        SELECT COALESCE(SUM(pnl) FILTER (WHERE action = 'SELL'), 0) AS realized_pnl
        FROM decisions
        WHERE decided_at >= NOW() - INTERVAL '14 days'
          AND decided_at <  NOW() - INTERVAL '7 days'
        """
    )
    this_pnl = float(week_decisions.get("realized_pnl") or 0)
    prev_pnl = float(prev["realized_pnl"]) if prev else 0.0

    if prev_pnl == 0.0 and this_pnl == 0.0:
        return []  # nothing traded either week — no signal to act on
    if this_pnl >= prev_pnl:
        return []  # expectancy didn't get worse — nothing to roll back

    changes = await conn.fetch(
        """
        SELECT DISTINCT ON (param_name) param_name, old_value, new_value, changed_at
        FROM brain_param_history
        WHERE changed_by = 'claude_calibration'
          AND changed_at >= NOW() - INTERVAL '14 days'
          AND changed_at <  NOW() - INTERVAL '7 days'
        ORDER BY param_name, changed_at DESC
        """
    )
    reverted = []
    for ch in changes:
        try:
            result = await set_param(
                conn, ch["param_name"], float(ch["old_value"]),
                changed_by="calibration_rollback",
                reason=(f"weekly expectancy worsened (₹{this_pnl:.2f} < ₹{prev_pnl:.2f}); "
                        f"reverting to pre-calibration value"),
            )
            reverted.append(result)
            await log_activity(
                conn, event_type="PARAM_CHANGE",
                note=f"Rollback {ch['param_name']}: {ch['new_value']} → {ch['old_value']} "
                     f"(expectancy worsened)",
                payload={"param": ch["param_name"], "reverted_to": ch["old_value"],
                         "week_pnl": this_pnl, "prev_week_pnl": prev_pnl},
            )
        except Exception as e:
            log.warning("Rollback failed for %s: %s", ch["param_name"], e)
    return reverted


async def run_calibration() -> dict:
    """
    Weekly calibration cycle (called from scheduler/weekend_job.py's Saturday
    flow). Returns {applied: [...], reverted: [...], summary: str}.
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        week_stats = await _gather_week_stats(conn)
        reverted = await _maybe_rollback(conn, week_stats["week_decisions"])

        resolved = int(week_stats["week_signals"].get("resolved") or 0)
        if resolved < MIN_RESOLVED_OUTCOMES:
            summary = (f"Skipped: only {resolved} resolved outcomes this week "
                       f"(need >= {MIN_RESOLVED_OUTCOMES}).")
            log.info("Calibration: %s", summary)
            await log_activity(
                conn, event_type="NOTE", note=f"Calibration skipped: {summary}",
                payload={"reverted": reverted,
                         "week_stats": {k: str(v) for k, v in week_stats.items()}},
            )
            return {"applied": [], "reverted": reverted, "summary": summary}

        params = [dict(r) for r in await conn.fetch(
            "SELECT param_name, value, min_value, max_value FROM brain_params ORDER BY param_name"
        )]
        learnings = await get_active_learnings(conn, limit=20)

        log.info("Weekly calibration: %s", week_stats)
        result = calibration_call(week_stats, params, learnings)
        summary = result.get("summary", "")
        changes = result.get("changes", [])

        known = {p["param_name"]: p for p in params}
        applied = []
        for ch in changes:
            name = ch.get("param")
            proposed = ch.get("new_value")
            reason = ch.get("reason", "")
            if name not in known or not isinstance(proposed, (int, float)):
                log.warning("Calibration rejected change %s — unknown param or bad value", ch)
                continue
            current = float(known[name]["value"])
            smoothed = CALIBRATION_ALPHA * float(proposed) + (1 - CALIBRATION_ALPHA) * current
            try:
                applied.append(await set_param(
                    conn, name, smoothed,
                    changed_by="claude_calibration",
                    reason=(f"{reason} (EWMA smoothed: proposed={proposed}, "
                            f"current={current}, alpha={CALIBRATION_ALPHA})"),
                ))
            except Exception as e:
                log.warning("Calibration apply failed for %s: %s", name, e)

        await log_activity(
            conn, event_type="NOTE",
            note=f"Weekly calibration: {len(applied)} change(s), {len(reverted)} rollback(s). {summary[:300]}",
            payload={"applied": applied, "reverted": reverted, "summary": summary,
                     "week_stats": {k: str(v) for k, v in week_stats.items()}},
        )
        log.info("Calibration done: %d applied, %d reverted. %s", len(applied), len(reverted), summary[:200])
        return {"applied": applied, "reverted": reverted, "summary": summary}
    finally:
        await conn.close()
