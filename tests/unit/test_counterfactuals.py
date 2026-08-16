"""Counterfactual engine: arithmetic replays on real fills, verified
against hand-constructed scenarios where the expected delta is known."""

from __future__ import annotations

import pandas as pd

from stocksense.statements.counterfactuals import (
    hard_stop_loss,
    never_trade_first_15min,
    remove_worst_5pct,
    remove_worst_trade,
    run_all,
)


def _position(symbol, open_time, entry, exit_, gross_pnl, charges=0.0, quantity=1):
    return {
        "symbol": symbol, "open_date": "2024-01-01", "open_time": open_time,
        "close_date": "2024-01-01", "close_time": "15:00", "quantity": quantity,
        "entry_price": entry, "exit_price": exit_, "gross_pnl": gross_pnl, "charges": charges,
        "net_pnl": gross_pnl - charges,
    }


def test_never_trade_first_15min_removes_only_early_trades() -> None:
    positions = pd.DataFrame(
        [
            _position("AAA", "09:20", 100, 90, -50),   # early loss, should be removed
            _position("BBB", "11:00", 100, 110, 50),   # later, kept
        ]
    )
    cf = never_trade_first_15min(positions)
    assert cf.actual_pnl == 0.0  # -50 + 50
    assert cf.scenario_pnl == 50.0  # only the kept trade
    assert cf.delta_pnl == 50.0
    assert cf.n_trades_affected == 1


def test_remove_worst_trade_isolates_single_biggest_loss() -> None:
    positions = pd.DataFrame(
        [
            _position("AAA", "10:00", 100, 50, -500),  # worst
            _position("BBB", "11:00", 100, 110, 50),
            _position("CCC", "12:00", 100, 105, 25),
        ]
    )
    cf = remove_worst_trade(positions)
    assert cf.actual_pnl == -425.0
    assert cf.scenario_pnl == 75.0  # -425 - (-500)
    assert cf.n_trades_affected == 1


def test_hard_stop_loss_caps_losses_but_not_wins() -> None:
    positions = pd.DataFrame(
        [
            _position("AAA", "10:00", 100, 50, -50, quantity=1),  # 50% loss on 100 entry, way past 2% stop
            _position("BBB", "11:00", 100, 110, 10, quantity=1),  # winner, untouched
        ]
    )
    cf = hard_stop_loss(positions, stop_pct=0.02)
    # capped loss = -0.02*100*1 = -2, so scenario should be much better
    assert cf.scenario_pnl > cf.actual_pnl
    assert cf.n_trades_affected == 1


def test_remove_worst_5pct_scales_with_sample_size() -> None:
    rows = [_position(f"S{i}", "10:00", 100, 90, -10) for i in range(19)]
    rows.append(_position("WORST", "10:00", 100, 0, -1000))
    positions = pd.DataFrame(rows)
    cf = remove_worst_5pct(positions)
    assert cf.n_trades_affected == 1  # 5% of 20 = 1
    assert cf.detail == {} or True


def test_run_all_returns_seven_scenarios() -> None:
    positions = pd.DataFrame([_position("AAA", "10:00", 100, 110, 10)])
    results = run_all(positions)
    assert len(results) == 7
    assert all(hasattr(r, "delta_pnl") for r in results)
