"""
Paper → Live auto-gate.

With no accuracy history and ₹500 of capital, acting on Day-1 signals is pure
speculation. So the product stays in PAPER mode — signals are tracked (decisions,
activity, position reviews all still log), but framed as paper trades, not
execution prompts — until a real track record exists.

The gate flips itself to LIVE the day three conditions all hold:
  - the resolved-signal history spans at least PAPER_MIN_DAYS (4+ weeks), AND
  - there are at least PAPER_MIN_RESOLVED resolved signals (statistically meaningful), AND
  - net (cost-adjusted) expectancy over the trailing PAPER_EXPECTANCY_WINDOW
    EXECUTED trades is > PAPER_MIN_NET_EXPECTANCY_BPS.

The old gate compared raw win rate to PAPER_ACCURACY_GATE (0.55) — passable by
a long-bias random walk in an up market, and blind to transaction costs. The
new gate requires the strategy to be profitable net of a conservative 0.30%
round-trip cost (see intelligence/costs.py), and only counts trades that were
actually EXECUTED (decisions.action='BUY' with quantity>0, i.e. real capital
was committed) rather than every signal fired — a signal that was suggested
but never bought shouldn't count toward "can we trust this with real money".

No manual trading override — the safety gate is the whole point.

Epoch filter: every query in this module (span/resolved count, and the executed-
trade expectancy) only considers signals/decisions at/after the current funding
epoch's start (intelligence.trading_account.get_current_epoch_start) — the old
₹500-era ledger, where nearly every BUY resolved to 0 affordable shares, never
counts toward the LIVE gate.

Used by:
  - GET /api/live/mode
  - record_decision()  (tags a BUY as [PAPER] while in paper mode)
"""
from __future__ import annotations

from intelligence.accuracy_tracker import (
    compute_rolling_accuracy,
    classify_and_return,
    RESOLVED_STATUSES,
)
from intelligence.costs import net_return
from intelligence.trading_account import get_current_epoch_start

PAPER_MIN_DAYS = 28                    # 4+ weeks of live track record
PAPER_MIN_RESOLVED = 50                # enough resolved signals to be meaningful
PAPER_EXPECTANCY_WINDOW = 60           # trailing N executed trades used for the expectancy gate
PAPER_MIN_NET_EXPECTANCY_BPS = 0.0     # net expectancy must be > this (cost-adjusted, bps) to go LIVE

# Kept for backward compatibility / display only — no longer used to gate LIVE mode.
PAPER_ACCURACY_GATE = 0.55


async def compute_executed_net_expectancy(conn, n: int = PAPER_EXPECTANCY_WINDOW) -> dict:
    """
    Net (cost-adjusted) expectancy over the trailing `n` EXECUTED trades, i.e.
    signals that were actually bought (decisions.action='BUY' AND quantity>0),
    joined back to the signal's resolved outcome. Signals that were only
    suggested/passed on don't count — this measures whether real capital
    commitments would have been profitable net of costs.

    Returns:
        {
          "n": <int, actual sample size used>,
          "net_win_rate": <float 0..1 | None>,
          "net_expectancy_bps": <float | None>,
        }
    """
    epoch_start = await get_current_epoch_start(conn)
    rows = await conn.fetch(
        """
        SELECT s.signal_type, s.status, s.price_at_signal, s.target_price,
               s.stop_loss, s.actual_close
        FROM decisions d
        JOIN signals s ON s.id = d.signal_id
        WHERE d.action = 'BUY'
          AND d.quantity > 0
          AND s.status = ANY($1::text[])
          AND d.decided_at >= $3
        ORDER BY d.decided_at DESC
        LIMIT $2
        """,
        list(RESOLVED_STATUSES), n, epoch_start,
    )

    net_returns: list[float] = []
    net_wins = 0
    for row in rows:
        outcome_ret = classify_and_return(
            row["signal_type"], row["status"], float(row["price_at_signal"]),
            float(row["target_price"]) if row["target_price"] is not None else None,
            float(row["stop_loss"]) if row["stop_loss"] is not None else None,
            float(row["actual_close"]) if row["actual_close"] is not None else None,
        )
        if outcome_ret is None:
            continue
        _, gross = outcome_ret
        net = net_return(gross)
        net_returns.append(net)
        if net > 0:
            net_wins += 1

    sample = len(net_returns)
    if sample == 0:
        return {"n": 0, "net_win_rate": None, "net_expectancy_bps": None}

    return {
        "n": sample,
        "net_win_rate": round(net_wins / sample, 4),
        "net_expectancy_bps": round((sum(net_returns) / sample) * 10000, 2),
    }


async def get_trading_mode(conn) -> dict:
    """
    Returns the current mode and the progress toward unlocking LIVE:
        {
          "mode": "PAPER" | "LIVE",
          "reason": "12/28 days of history, 8/50 resolved, net expectancy -4.2bps over 8 executed (need >0bps)",
          "resolved_count": 8,
          "span_days": 12,
          "rolling_accuracy": 0.48,               # raw win rate, kept for display/back-compat — NOT the gate
          "net_expectancy_bps": -4.2,              # cost-adjusted expectancy over trailing executed trades — the gate
          "net_win_rate": 0.375,
          "executed_trade_count": 8,
          "gate": {
              "min_days": 28, "min_resolved": 50,
              "min_accuracy": 0.55,                 # legacy, unused for gating — kept for back-compat
              "expectancy_window": 60, "min_net_expectancy_bps": 0.0,
          },
        }
    """
    epoch_start = await get_current_epoch_start(conn)

    # How long has the resolved-signal history spanned, and how many are resolved?
    # Epoch-filtered: pre-reset (₹500-era) signals never count toward the gate.
    span_row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS resolved_count,
            MIN(fired_at) AS first_fired,
            MAX(fired_at) AS last_fired
        FROM signals
        WHERE status != 'active'
          AND fired_at >= $1
        """,
        epoch_start,
    )
    resolved_count = int(span_row["resolved_count"]) if span_row else 0
    span_days = 0
    if span_row and span_row["first_fired"] and span_row["last_fired"]:
        span_days = (span_row["last_fired"] - span_row["first_fired"]).days

    # Raw rolling accuracy over the full track-record window — kept for display/back-compat only.
    accs = await compute_rolling_accuracy(conn, days=max(PAPER_MIN_DAYS, span_days or 1))
    rolling_accuracy = None
    if accs:
        combined = accs.get("combined")
        vals = [v for v in accs.values() if v is not None]
        rolling_accuracy = combined if combined is not None else (
            round(sum(vals) / len(vals), 4) if vals else None
        )

    # The actual gate: net (cost-adjusted) expectancy over trailing EXECUTED trades.
    expectancy = await compute_executed_net_expectancy(conn, n=PAPER_EXPECTANCY_WINDOW)
    net_expectancy_bps = expectancy["net_expectancy_bps"]
    net_win_rate = expectancy["net_win_rate"]
    executed_trade_count = expectancy["n"]

    meets_days = span_days >= PAPER_MIN_DAYS
    meets_count = resolved_count >= PAPER_MIN_RESOLVED
    meets_expectancy = (
        net_expectancy_bps is not None and net_expectancy_bps > PAPER_MIN_NET_EXPECTANCY_BPS
    )

    mode = "LIVE" if (meets_days and meets_count and meets_expectancy) else "PAPER"

    exp_str = f"{net_expectancy_bps:.1f}bps" if net_expectancy_bps is not None else "n/a"
    reason = (
        f"{span_days}/{PAPER_MIN_DAYS} days of history, "
        f"{resolved_count}/{PAPER_MIN_RESOLVED} resolved, "
        f"net expectancy {exp_str} over {executed_trade_count} executed "
        f"(need >{PAPER_MIN_NET_EXPECTANCY_BPS:.0f}bps)"
    )

    return {
        "mode": mode,
        "reason": reason,
        "resolved_count": resolved_count,
        "span_days": span_days,
        "rolling_accuracy": rolling_accuracy,
        "net_expectancy_bps": net_expectancy_bps,
        "net_win_rate": net_win_rate,
        "executed_trade_count": executed_trade_count,
        "gate": {
            "min_days": PAPER_MIN_DAYS,
            "min_resolved": PAPER_MIN_RESOLVED,
            "min_accuracy": PAPER_ACCURACY_GATE,
            "expectancy_window": PAPER_EXPECTANCY_WINDOW,
            "min_net_expectancy_bps": PAPER_MIN_NET_EXPECTANCY_BPS,
        },
    }
