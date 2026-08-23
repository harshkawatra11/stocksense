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
from stocksense.portfolio.construct import apply_no_trade_band, one_way_turnover, target_weights_top_n


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


@dataclass(frozen=True)
class TodaysActions:
    is_rebalance_point: bool  # True if today's own prediction run IS a rebalance point
    as_of_date: str  # the date the returned actions were actually computed as of (today's, or the last rebalance point's)
    actions: list[RebalanceAction]
    next_rebalance_in_trading_days: int  # 0 when is_rebalance_point is True


def _weights_at(preds: pd.DataFrame, as_of_date, top_n: int) -> pd.Series:
    scores = preds[preds["as_of_date"] == as_of_date].set_index("symbol")["score"]
    return target_weights_top_n(scores, top_n)


def recommend_todays_actions(
    store, model_id: str, horizon_bars: int, top_n: int, band: float = 0.02, segment: str = "equity_delivery",
) -> TodaysActions | None:
    """Phase H4: the fix for a real bug -- record_predictions computes a
    fresh top-N ranking every day the reconcile loop runs (correct: more
    graded ledger rows is strictly better), but every PASS in
    research/verdict_bhavcopy_rerun.md was earned rebalancing every
    `horizon_bars` TRADING DAYS, not daily. Showing "today's top-N" as
    an actionable move list every morning would generate real buy/sell
    churn nothing in the backtest ever paid for or validated -- the
    exact mechanism (docs/STATUS.md's Phase E4 entry, the user's own
    real account: gross +Rs2,492, net -Rs1,293 over 441 round trips)
    that turns a genuinely positive selection into a net-negative one.

    Derives "current holdings" from the last actual REBALANCE POINT
    (Rebalances at index 0, then every subsequent date that is >=
    `horizon_bars` recorded trading days after the previous rebalance
    point -- using the predictions ledger's OWN distinct as_of_dates as
    the trading calendar, since record_predictions only ever writes a
    row for `feats["date"].max()`, a real candle date, so each entry
    already IS one observed trading day; no separate calendar lookup
    needed), never from yesterday's prediction.

    Returns None if nothing has been recorded for this model yet (the
    caller -- /api/brief -- already handles "no_predictions" before
    reaching here; this is a graceful non-crash for a direct caller).
    """
    preds = store.read_predictions()
    preds = preds[preds["model_version"] == model_id]
    if preds.empty:
        return None

    dates = sorted(preds["as_of_date"].unique())

    rebalance_points = [dates[0]]
    last_idx = 0
    for i, d in enumerate(dates[1:], start=1):
        if i - last_idx >= horizon_bars:
            rebalance_points.append(d)
            last_idx = i

    latest_date = dates[-1]
    # The most recent rebalance point that has actually occurred (<= latest_date by construction)
    point = rebalance_points[-1]
    point_idx = rebalance_points.index(point)
    prev_point = rebalance_points[point_idx - 1] if point_idx > 0 else None

    target = _weights_at(preds, point, top_n)
    current = _weights_at(preds, prev_point, top_n) if prev_point is not None else pd.Series(dtype=float)

    # portfolio_value_inr=1.0 is deliberate, not a placeholder: equity_
    # delivery costs are exactly proportional to trade value
    # (optimizer/sizing.py), so running the cost computation at a
    # notional Rs1 keeps every estimated_cost_inr output as a FRACTION
    # of portfolio, comparable directly to alpha -- no capital figure
    # enters this calculation at all.
    actions = recommend_rebalance(target, current, portfolio_value_inr=1.0, band=band, segment=segment)

    is_rebalance_point = point == latest_date
    next_in = 0 if is_rebalance_point else horizon_bars - (dates.index(latest_date) - dates.index(point))

    return TodaysActions(
        is_rebalance_point=is_rebalance_point,
        as_of_date=str(point),
        actions=actions,
        next_rebalance_in_trading_days=next_in,
    )


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
