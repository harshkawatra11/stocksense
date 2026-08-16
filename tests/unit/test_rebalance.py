"""Rebalance recommendation tests. The property the plan calls out
explicitly: the optimizer must never recommend a trade whose cost
exceeds its stated benefit -- filter_cost_justified is the enforcement
point, tested as a property, not just an example."""

from __future__ import annotations

import pandas as pd

from stocksense.optimizer.rebalance import filter_cost_justified, recommend_rebalance, summarize_actions


def test_new_position_classified_as_enter() -> None:
    target = pd.Series({"NEWCO": 0.10})
    current = pd.Series({}, dtype=float)
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000)
    action = next(a for a in actions if a.symbol == "NEWCO")
    assert action.action == "enter"
    assert action.estimated_cost_inr > 0


def test_full_exit_classified_as_exit() -> None:
    target = pd.Series({"OLDCO": 0.0})
    current = pd.Series({"OLDCO": 0.10})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000)
    action = next(a for a in actions if a.symbol == "OLDCO")
    assert action.action == "exit"


def test_small_change_within_band_is_hold() -> None:
    target = pd.Series({"AAA": 0.101})
    current = pd.Series({"AAA": 0.100})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000, band=0.02)
    action = next(a for a in actions if a.symbol == "AAA")
    assert action.action == "hold"
    assert action.estimated_cost_inr == 0.0


def test_large_increase_classified_as_add() -> None:
    target = pd.Series({"AAA": 0.20})
    current = pd.Series({"AAA": 0.05})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000, band=0.02)
    action = next(a for a in actions if a.symbol == "AAA")
    assert action.action == "add"
    assert action.estimated_cost_inr > 0


def test_large_decrease_classified_as_trim() -> None:
    target = pd.Series({"AAA": 0.05})
    current = pd.Series({"AAA": 0.20})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000, band=0.02)
    action = next(a for a in actions if a.symbol == "AAA")
    assert action.action == "trim"


def test_cost_scales_with_portfolio_value() -> None:
    target = pd.Series({"AAA": 0.10})
    current = pd.Series({}, dtype=float)
    small = recommend_rebalance(target, current, portfolio_value_inr=100_000)
    large = recommend_rebalance(target, current, portfolio_value_inr=10_000_000)
    small_cost = next(a for a in small if a.symbol == "AAA").estimated_cost_inr
    large_cost = next(a for a in large if a.symbol == "AAA").estimated_cost_inr
    assert large_cost > small_cost


def test_filter_cost_justified_drops_expensive_trades_relative_to_benefit() -> None:
    target = pd.Series({"AAA": 0.10})
    current = pd.Series({}, dtype=float)
    actions = recommend_rebalance(target, current, portfolio_value_inr=10_000_000)  # large trade -> real cost
    action = next(a for a in actions if a.symbol == "AAA")
    assert action.estimated_cost_inr > 0

    # benefit smaller than the actual cost -> filtered out
    filtered = filter_cost_justified(actions, min_benefit_inr=action.estimated_cost_inr / 2)
    assert not any(a.symbol == "AAA" for a in filtered)

    # benefit larger than the actual cost -> kept
    kept = filter_cost_justified(actions, min_benefit_inr=action.estimated_cost_inr * 2)
    assert any(a.symbol == "AAA" for a in kept)


def test_filter_cost_justified_never_drops_holds() -> None:
    target = pd.Series({"AAA": 0.101})
    current = pd.Series({"AAA": 0.100})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000, band=0.02)
    filtered = filter_cost_justified(actions, min_benefit_inr=0.0)
    assert len(filtered) == len(actions)  # hold is always kept regardless of "benefit"


def test_summarize_actions_counts_by_type() -> None:
    target = pd.Series({"A": 0.0, "B": 0.20, "C": 0.10})
    current = pd.Series({"A": 0.10, "B": 0.05, "C": 0.10})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000, band=0.02)
    summary = summarize_actions(actions)
    assert summary["n_exit"] == 1  # A
    assert summary["n_add"] == 1  # B
    assert summary["total_estimated_cost_inr"] > 0


def test_summarize_actions_all_hold_has_zero_cost() -> None:
    target = pd.Series({"A": 0.10})
    current = pd.Series({"A": 0.10})
    actions = recommend_rebalance(target, current, portfolio_value_inr=1_000_000)
    summary = summarize_actions(actions)
    assert summary["n_actions"] == 0
    assert summary["total_estimated_cost_inr"] == 0.0
