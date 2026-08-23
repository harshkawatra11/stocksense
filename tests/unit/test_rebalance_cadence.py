"""Phase H4: recommend_todays_actions -- the fix for showing "today's
top-N" as an actionable move list every morning, which would generate
real churn nothing in the backtest ever paid for (every G1 PASS was
earned rebalancing every `horizon_bars` TRADING DAYS, not daily; the
user's own real account is the concrete cost of getting this wrong:
gross +Rs2,492, net -Rs1,293 over 441 round trips). These tests build a
synthetic predictions ledger spanning multiple rebalance windows and
check the rebalance-point math directly."""

from __future__ import annotations

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.optimizer.rebalance import recommend_todays_actions


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _write_predictions_for_date(store, model_id, as_of_date, horizon_bars, symbol_scores: dict) -> None:
    rows = [
        {
            "run_id": f"run-{as_of_date}", "symbol": sym, "as_of_date": as_of_date,
            "horizon_bars": horizon_bars, "score": score, "rank": rank,
            "model_version": model_id, "horizon_type": "short",
            "predicted_return": score, "confidence": None, "feature_snapshot_hash": "h",
        }
        for rank, (sym, score) in enumerate(sorted(symbol_scores.items(), key=lambda kv: -kv[1]), start=1)
    ]
    store.write_predictions(pd.DataFrame(rows))


def _seed_daily_scores(store, model_id, horizon_bars, n_days, symbols, seed=0):
    """One prediction row per symbol per trading day, scores drifting
    slightly so rank order can plausibly change day to day -- exactly
    the daily-reconcile pattern that motivates this whole fix."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    for d in dates:
        scores = {s: float(rng.normal(0, 0.02)) for s in symbols}
        _write_predictions_for_date(store, model_id, d.date(), horizon_bars, scores)
    return dates


def test_no_predictions_returns_none(tmp_store) -> None:
    result = recommend_todays_actions(tmp_store, "m1", horizon_bars=10, top_n=3)
    assert result is None


def test_first_ever_prediction_is_a_rebalance_point_all_enters(tmp_store) -> None:
    _write_predictions_for_date(tmp_store, "m1", pd.Timestamp("2026-01-05").date(), 10, {"A": 0.05, "B": 0.03, "C": 0.01})
    result = recommend_todays_actions(tmp_store, "m1", horizon_bars=10, top_n=2)

    assert result.is_rebalance_point is True
    assert result.next_rebalance_in_trading_days == 0
    non_hold = [a for a in result.actions if a.action != "hold"]
    assert all(a.action == "enter" for a in non_hold)
    assert {a.symbol for a in non_hold} == {"A", "B"}  # top_n=2


def test_intermediate_days_are_not_rebalance_points(tmp_store) -> None:
    """Days between rebalance points must not be treated as actionable
    -- this is the exact property that prevents daily churn."""
    dates = _seed_daily_scores(tmp_store, "m1", horizon_bars=10, n_days=5, symbols=["A", "B", "C", "D"])
    result = recommend_todays_actions(tmp_store, "m1", horizon_bars=10, top_n=2)

    assert result.is_rebalance_point is False
    assert result.next_rebalance_in_trading_days == 10 - 4  # day index 4, last rebalance at index 0
    assert pd.Timestamp(result.as_of_date).date() == dates[0].date()


def test_rebalance_point_recurs_every_horizon_trading_days(tmp_store) -> None:
    dates = _seed_daily_scores(tmp_store, "m1", horizon_bars=10, n_days=11, symbols=["A", "B", "C", "D"])
    # day index 10 (the 11th recorded day) is exactly horizon_bars=10 after day index 0
    result = recommend_todays_actions(tmp_store, "m1", horizon_bars=10, top_n=2)

    assert result.is_rebalance_point is True
    assert result.next_rebalance_in_trading_days == 0
    assert pd.Timestamp(result.as_of_date).date() == dates[10].date()


def test_actions_at_a_later_rebalance_point_compare_against_the_previous_one(tmp_store) -> None:
    """The core property: at the SECOND rebalance point, 'current' must
    be the FIRST rebalance point's weights, not day-to-day noise from
    the days in between."""
    # Build an explicit, controlled scenario instead of random scores:
    # day 0 (rebalance point 1): top-2 = A, B
    # day 10 (rebalance point 2): top-2 = C, D -- total turnover, both prior names exit
    d0 = pd.Timestamp("2026-01-05").date()
    _write_predictions_for_date(tmp_store, "m1", d0, 10, {"A": 0.05, "B": 0.03, "C": 0.01, "D": 0.00})
    trading_dates = pd.bdate_range("2026-01-05", periods=15)
    for i, d in enumerate(trading_dates[1:11], start=1):  # days 1..10
        if i < 10:
            _write_predictions_for_date(tmp_store, "m1", d.date(), 10, {"A": 0.05, "B": 0.03, "C": 0.01, "D": 0.00})
        else:  # day index 10 -- the second rebalance point, ranks flip
            _write_predictions_for_date(tmp_store, "m1", d.date(), 10, {"A": 0.00, "B": 0.01, "C": 0.05, "D": 0.03})

    result = recommend_todays_actions(tmp_store, "m1", horizon_bars=10, top_n=2)
    assert result.is_rebalance_point is True

    by_symbol = {a.symbol: a.action for a in result.actions}
    assert by_symbol["C"] == "enter"
    assert by_symbol["D"] == "enter"
    assert by_symbol["A"] == "exit"
    assert by_symbol["B"] == "exit"


def test_costs_are_capital_agnostic_fractions_not_rupee_figures(tmp_store) -> None:
    """portfolio_value_inr=1.0 is deliberate -- estimated_cost_inr must
    come out as a small FRACTION (comparable to alpha), never a rupee
    figure implying a real ₹1 portfolio or any other assumed capital."""
    _write_predictions_for_date(tmp_store, "m1", pd.Timestamp("2026-01-05").date(), 10, {"A": 0.05, "B": 0.03})
    result = recommend_todays_actions(tmp_store, "m1", horizon_bars=10, top_n=2)

    non_hold = [a for a in result.actions if a.action != "hold"]
    assert len(non_hold) > 0
    for a in non_hold:
        assert 0 < a.estimated_cost_inr < 0.01  # a few bps of a unit portfolio, not rupees
        assert a.portfolio_value_inr == 1.0
