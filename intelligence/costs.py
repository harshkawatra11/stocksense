"""
Transaction cost model — shared, minimal.

A single conservative round-trip cost estimate (brokerage + STT + slippage)
used to turn gross returns into honest, cost-adjusted (net) returns anywhere
accuracy/expectancy is computed. This is intentionally NOT a full cost model
(per-broker fee schedules, STT slabs, impact cost by liquidity, etc.) — that
is a later-stage job (see WHAT_TO_DO_NEXT.txt Phase 2). For now, one constant
unblocks Phase 0's honest-accuracy requirement.
"""
from __future__ import annotations

# Conservative round-trip cost estimate, expressed as a fraction (0.0030 = 0.30%).
# Covers brokerage + STT + exchange/DP charges + expected slippage on entry+exit.
ROUND_TRIP_COST_PCT = 0.0030


def net_return(gross_return_pct: float) -> float:
    """
    Subtract the round-trip transaction cost from a gross return.

    `gross_return_pct` is a fractional return (e.g. 0.012 for +1.2%), not a
    percentage-points integer. Returns the cost-adjusted (net) fractional
    return.
    """
    return gross_return_pct - ROUND_TRIP_COST_PCT
