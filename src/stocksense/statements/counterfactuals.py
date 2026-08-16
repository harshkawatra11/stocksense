"""
Counterfactual engine — "what if you had..." replays on real fills
(docs/12-statement-forensics.md). Every scenario here is arithmetic on
historical positions, explicitly labeled as such: these are not
predictions, and changed behavior would have changed market impact and
the trader's own subsequent decisions, which this cannot model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Counterfactual:
    scenario_name: str
    actual_pnl: float
    scenario_pnl: float
    delta_pnl: float
    n_trades_affected: int
    detail: dict


def _actual_pnl(positions: pd.DataFrame) -> float:
    return float(positions["net_pnl"].sum()) if len(positions) else 0.0


def never_trade_first_15min(positions: pd.DataFrame) -> Counterfactual:
    p = positions.dropna(subset=["open_time"])
    affected = p[p["open_time"].astype(str).str[:5] <= "09:30"]
    scenario_pnl = _actual_pnl(positions) - float(affected["net_pnl"].sum())
    actual = _actual_pnl(positions)
    return Counterfactual(
        "never_trade_first_15min", actual, scenario_pnl, scenario_pnl - actual,
        int(len(affected)), {"excluded_pnl": float(affected["net_pnl"].sum())},
    )


def cap_position_size_at_median(positions: pd.DataFrame) -> Counterfactual:
    if positions.empty:
        return Counterfactual("cap_position_size_at_median", 0.0, 0.0, 0.0, 0, {})
    values = positions["quantity"] * positions["entry_price"]
    median_value = values.median()
    scale = np.minimum(1.0, median_value / values.replace(0, np.nan))
    scaled_gross = positions["gross_pnl"] * scale.fillna(1.0)
    # charges scale roughly linearly with position size too
    scaled_charges = positions["charges"] * scale.fillna(1.0)
    scenario_pnl = float((scaled_gross - scaled_charges).sum())
    actual = _actual_pnl(positions)
    n_affected = int((scale < 1.0).sum())
    return Counterfactual(
        "cap_position_size_at_median", actual, scenario_pnl, scenario_pnl - actual, n_affected,
        {"median_position_value": float(median_value)},
    )


def hard_stop_loss(positions: pd.DataFrame, stop_pct: float = 0.02) -> Counterfactual:
    """Cap every losing position's loss at stop_pct of entry value —
    optimistic (assumes perfect fill at the stop level), so this is an
    upper bound on what a real stop-loss discipline would have achieved."""
    p = positions.copy()
    max_loss = -stop_pct * p["entry_price"] * p["quantity"]
    capped_gross = np.maximum(p["gross_pnl"], max_loss)
    scenario_pnl = float((capped_gross - p["charges"]).sum())
    actual = _actual_pnl(positions)
    n_affected = int((p["gross_pnl"] < max_loss).sum())
    return Counterfactual(
        f"hard_stop_loss_{int(stop_pct * 100)}pct", actual, scenario_pnl, scenario_pnl - actual, n_affected, {},
    )


def stop_after_two_losses_per_day(positions: pd.DataFrame) -> Counterfactual:
    p = positions.dropna(subset=["open_date"]).sort_values(["open_date", "open_time"])
    keep_mask = []
    for date, group in p.groupby("open_date"):
        losses_so_far = 0
        for pnl in group["net_pnl"]:
            if losses_so_far >= 2:
                keep_mask.append(False)
            else:
                keep_mask.append(True)
                if pnl < 0:
                    losses_so_far += 1
    p = p.assign(_keep=keep_mask)
    excluded = p[~p["_keep"]]
    scenario_pnl = _actual_pnl(positions) - float(excluded["net_pnl"].sum())
    actual = _actual_pnl(positions)
    return Counterfactual(
        "stop_after_two_losses_per_day", actual, scenario_pnl, scenario_pnl - actual, int(len(excluded)), {},
    )


def remove_worst_trade(positions: pd.DataFrame) -> Counterfactual:
    if positions.empty:
        return Counterfactual("remove_worst_trade", 0.0, 0.0, 0.0, 0, {})
    worst_idx = positions["net_pnl"].idxmin()
    worst_pnl = float(positions.loc[worst_idx, "net_pnl"])
    actual = _actual_pnl(positions)
    scenario_pnl = actual - worst_pnl
    return Counterfactual(
        "remove_worst_trade", actual, scenario_pnl, scenario_pnl - actual, 1,
        {"worst_trade_symbol": str(positions.loc[worst_idx, "symbol"]), "worst_trade_pnl": worst_pnl},
    )


def remove_worst_5pct(positions: pd.DataFrame) -> Counterfactual:
    n = len(positions)
    if n == 0:
        return Counterfactual("remove_worst_5pct", 0.0, 0.0, 0.0, 0, {})
    n_worst = max(1, int(0.05 * n))
    worst = positions.nsmallest(n_worst, "net_pnl")
    actual = _actual_pnl(positions)
    scenario_pnl = actual - float(worst["net_pnl"].sum())
    return Counterfactual(
        "remove_worst_5pct", actual, scenario_pnl, scenario_pnl - actual, n_worst, {},
    )


def zero_brokerage(positions: pd.DataFrame) -> Counterfactual:
    """Isolates pure cost impact: what if only STT/exchange/GST applied
    (statutory, unavoidable) and brokerage were zero? Requires per-trade
    charge breakdown; approximated here from total charges since
    positions only carries the aggregate — refined once trade_charges is
    joined in by the caller."""
    actual = _actual_pnl(positions)
    # Conservative approximation: brokerage is typically the smallest
    # component for a discount broker (~₹40 of ₹82.68 round-trip per the
    # verified cost table) — this counterfactual is most meaningful when
    # the caller has real per-trade brokerage figures from trade_charges;
    # documented as an approximation, not a precise replay.
    approx_brokerage_share = 0.35
    charges = positions["charges"].sum()
    scenario_pnl = actual + charges * approx_brokerage_share
    return Counterfactual(
        "zero_brokerage_approx", actual, float(scenario_pnl), float(scenario_pnl - actual), len(positions),
        {"note": "approximation using aggregate charges; see trade_charges for exact figures"},
    )


ALL_COUNTERFACTUALS = [
    never_trade_first_15min,
    cap_position_size_at_median,
    hard_stop_loss,
    stop_after_two_losses_per_day,
    remove_worst_trade,
    remove_worst_5pct,
    zero_brokerage,
]


def run_all(positions: pd.DataFrame) -> list[Counterfactual]:
    return [fn(positions) for fn in ALL_COUNTERFACTUALS]
