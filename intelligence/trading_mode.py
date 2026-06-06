"""
Paper → Live auto-gate.

With no accuracy history and ₹500 of capital, acting on Day-1 signals is pure
speculation. So the product stays in PAPER mode — signals are tracked (decisions,
activity, position reviews all still log), but framed as paper trades, not
execution prompts — until a real track record exists.

The gate flips itself to LIVE the day three conditions all hold:
  - the resolved-signal history spans at least PAPER_MIN_DAYS (4+ weeks), AND
  - there are at least PAPER_MIN_RESOLVED resolved signals (statistically meaningful), AND
  - rolling accuracy clears PAPER_ACCURACY_GATE.

No manual trading override — the safety gate is the whole point.

Used by:
  - GET /api/live/mode
  - record_decision()  (tags a BUY as [PAPER] while in paper mode)
"""
from __future__ import annotations

from intelligence.accuracy_tracker import compute_rolling_accuracy

PAPER_MIN_DAYS = 28          # 4+ weeks of live track record
PAPER_MIN_RESOLVED = 50      # enough resolved signals to be meaningful
PAPER_ACCURACY_GATE = 0.55   # rolling accuracy must clear this to go LIVE


async def get_trading_mode(conn) -> dict:
    """
    Returns the current mode and the progress toward unlocking LIVE:
        {
          "mode": "PAPER" | "LIVE",
          "reason": "12/28 days of history, 8/50 resolved, accuracy 48% (need 55%)",
          "resolved_count": 8,
          "span_days": 12,
          "rolling_accuracy": 0.48,
          "gate": {"min_days": 28, "min_resolved": 50, "min_accuracy": 0.55},
        }
    """
    # How long has the resolved-signal history spanned, and how many are resolved?
    span_row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS resolved_count,
            MIN(fired_at) AS first_fired,
            MAX(fired_at) AS last_fired
        FROM signals
        WHERE status != 'active'
        """
    )
    resolved_count = int(span_row["resolved_count"]) if span_row else 0
    span_days = 0
    if span_row and span_row["first_fired"] and span_row["last_fired"]:
        span_days = (span_row["last_fired"] - span_row["first_fired"]).days

    # Rolling accuracy over the full track-record window (use combined where present).
    accs = await compute_rolling_accuracy(conn, days=max(PAPER_MIN_DAYS, span_days or 1))
    rolling_accuracy = None
    if accs:
        combined = accs.get("combined")
        vals = [v for v in accs.values() if v is not None]
        rolling_accuracy = combined if combined is not None else (
            round(sum(vals) / len(vals), 4) if vals else None
        )

    meets_days = span_days >= PAPER_MIN_DAYS
    meets_count = resolved_count >= PAPER_MIN_RESOLVED
    meets_acc = rolling_accuracy is not None and rolling_accuracy >= PAPER_ACCURACY_GATE

    mode = "LIVE" if (meets_days and meets_count and meets_acc) else "PAPER"

    acc_pct = f"{rolling_accuracy * 100:.0f}%" if rolling_accuracy is not None else "n/a"
    reason = (
        f"{span_days}/{PAPER_MIN_DAYS} days of history, "
        f"{resolved_count}/{PAPER_MIN_RESOLVED} resolved, "
        f"accuracy {acc_pct} (need {PAPER_ACCURACY_GATE * 100:.0f}%)"
    )

    return {
        "mode": mode,
        "reason": reason,
        "resolved_count": resolved_count,
        "span_days": span_days,
        "rolling_accuracy": rolling_accuracy,
        "gate": {
            "min_days": PAPER_MIN_DAYS,
            "min_resolved": PAPER_MIN_RESOLVED,
            "min_accuracy": PAPER_ACCURACY_GATE,
        },
    }
