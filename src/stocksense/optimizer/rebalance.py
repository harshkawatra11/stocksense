"""
Cost-aware rebalance recommendation. Reuses portfolio/construct.py's
existing no-trade-band and turnover-budget primitives rather than
re-deriving them; this module's only new contribution is turning a
target-weight change into an EXPLICIT, itemized recommendation (exit /
trim / hold / add) with the round-trip cost that specific move would
incur (execution.cost_model.compute_charges) attached to each line.

Explicit boundary, stated once because it matters: this module
RECOMMENDS. It never places an order. That is a deliberate scope
decision from the plan, not an oversight -- live execution needs its
own risk review, separate from anything built here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocksense.execution.cost_model import compute_charges
from stocksense.portfolio.construct import apply_no_trade_band, one_way_turnover


@dataclass(frozen=True)
class RebalanceAction:
    symbol: str
    action: str  # 'exit' | 'trim' | 'hold' | 'add' | 'enter'
    current_weight: float
    target_weight: float
    weight_delta: float
    estimated_cost_inr: float
    portfolio_value_inr: float


def _classify(current: float, target: float, band: float) -> str:
    delta = target - current
    if abs(delta) < band:
        return "hold"
    if target <= 1e-9:
        return "exit"
    if current <= 1e-9:
        return "enter"
    return "add" if delta > 0 else "trim"


def recommend_rebalance(
    target_weights: pd.Series, current_weights: pd.Series, portfolio_value_inr: float,
    band: float = 0.02, segment: str = "equity_delivery",
) -> list[RebalanceAction]:
    """Compares target vs current weights and produces one itemized
    action per symbol, each carrying the actual rupee cost that specific
    move would incur. Weights inside `band` of current are always
    'hold' — reusing apply_no_trade_band's exact definition of "close
    enough not to bother" so this recommendation can never disagree with
    what the backtest itself would have done in the same situation."""
    snapped_target = apply_no_trade_band(target_weights, current_weights, band=band)
    idx = snapped_target.index.union(current_weights.index)
    t = snapped_target.reindex(idx, fill_value=0.0)
    c = current_weights.reindex(idx, fill_value=0.0)

    actions = []
    for symbol in idx:
        current_w = float(c[symbol])
        target_w = float(t[symbol])
        delta = target_w - current_w
        action = _classify(current_w, target_w, band)

        trade_value = abs(delta) * portfolio_value_inr
        if action == "hold" or trade_value <= 0:
            cost = 0.0
        else:
            side = "sell" if delta < 0 else "buy"
            charges = compute_charges(segment, side, quantity=1, price=trade_value)  # quantity=1, price=trade_value: cost scales with rupee value, not share count
            cost = charges.total_charges

        actions.append(RebalanceAction(
            symbol=symbol, action=action, current_weight=current_w, target_weight=target_w,
            weight_delta=delta, estimated_cost_inr=cost, portfolio_value_inr=portfolio_value_inr,
        ))
    return actions


def filter_cost_justified(actions: list[RebalanceAction], min_benefit_inr: float) -> list[RebalanceAction]:
    """Drops any non-hold action whose estimated cost exceeds
    `min_benefit_inr` (the caller's estimate of what the move is worth,
    e.g. from expected alpha * trade value) -- the property named in the
    plan: never recommend a trade whose expected benefit is smaller than
    its round-trip cost."""
    return [a for a in actions if a.action == "hold" or a.estimated_cost_inr <= min_benefit_inr]


def summarize_actions(actions: list[RebalanceAction]) -> dict:
    non_hold = [a for a in actions if a.action != "hold"]
    return {
        "n_actions": len(non_hold),
        "n_exit": sum(1 for a in non_hold if a.action == "exit"),
        "n_enter": sum(1 for a in non_hold if a.action == "enter"),
        "n_trim": sum(1 for a in non_hold if a.action == "trim"),
        "n_add": sum(1 for a in non_hold if a.action == "add"),
        "total_estimated_cost_inr": sum(a.estimated_cost_inr for a in actions),
        "one_way_turnover": sum(abs(a.weight_delta) for a in actions) / 2.0,
    }
