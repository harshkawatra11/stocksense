"""Diagnostics tests: each dosha on a synthetic scenario constructed to
trigger it by hand, so a wrong formula fails loudly rather than passing
on real (unpredictable) data. Per the plan's verification requirement:
'a synthetic statement with known-by-construction pathologies produces
exactly the expected doshas.'"""

from __future__ import annotations

import pandas as pd

from stocksense.statements.diagnostics import (
    concentration,
    cost_drag,
    disposition_effect,
    drawdown_profile,
    expectancy,
    run_all,
    tail_dependence,
)


def _position(symbol, open_date, open_time, close_date, close_time, quantity, entry, exit_,
              gross_pnl=None, charges=0.0, is_intraday=False):
    gross = gross_pnl if gross_pnl is not None else (exit_ - entry) * quantity
    return {
        "symbol": symbol, "segment": "equity_delivery", "open_date": open_date, "open_time": open_time,
        "close_date": close_date, "close_time": close_time, "quantity": quantity,
        "entry_price": entry, "exit_price": exit_, "gross_pnl": gross, "charges": charges,
        "net_pnl": gross - charges, "holding_seconds": None, "is_intraday": is_intraday,
        "mae": None, "mfe": None,
    }


def test_cost_drag_fires_on_high_charge_ratio() -> None:
    # gross profit of 100, but charges of 80 -> 80% cost drag, should be 'critical' at >100%? no, high.
    positions = pd.DataFrame([_position("AAA", "2024-01-01", "09:20", "2024-01-01", "09:25", 10, 100, 110, gross_pnl=100.0, charges=120.0)])
    d = cost_drag(positions)
    assert d.severity == "critical"  # charges (120) > gross (100) -> ratio 1.2 >= 1.0 critical
    assert abs(d.value - 1.2) < 1e-9


def test_cost_drag_ok_when_charges_small() -> None:
    positions = pd.DataFrame([_position("AAA", "2024-01-01", "09:20", "2024-01-01", "09:25", 10, 100, 200, gross_pnl=1000.0, charges=10.0)])
    d = cost_drag(positions)
    assert d.severity == "ok"


def test_disposition_effect_fires_when_losers_held_longer() -> None:
    rows = []
    for i in range(5):
        # winners held briefly (1 hour)
        rows.append({"symbol": "AAA", "segment": "equity_delivery", "open_date": "2024-01-01", "open_time": "09:00",
                      "close_date": "2024-01-01", "close_time": "10:00", "quantity": 1, "entry_price": 100,
                      "exit_price": 110, "gross_pnl": 10, "charges": 0, "net_pnl": 10, "holding_seconds": 3600,
                      "is_intraday": True, "mae": None, "mfe": None})
        # losers held for weeks (bad!)
        rows.append({"symbol": "BBB", "segment": "equity_delivery", "open_date": "2024-01-01", "open_time": "09:00",
                      "close_date": "2024-02-01", "close_time": "10:00", "quantity": 1, "entry_price": 100,
                      "exit_price": 90, "gross_pnl": -10, "charges": 0, "net_pnl": -10, "holding_seconds": 3600 * 24 * 31,
                      "is_intraday": False, "mae": None, "mfe": None})
    positions = pd.DataFrame(rows)
    d = disposition_effect(positions)
    assert d.value > 100  # losers held vastly longer
    assert d.severity in ("high", "critical")


def test_expectancy_negative_when_losses_dominate() -> None:
    rows = [_position("AAA", "2024-01-01", "09:00", "2024-01-01", "10:00", 1, 100, 105, gross_pnl=5, charges=0) for _ in range(3)]
    rows += [_position("BBB", "2024-01-01", "09:00", "2024-01-01", "10:00", 1, 100, 80, gross_pnl=-20, charges=0) for _ in range(7)]
    positions = pd.DataFrame(rows)
    d = expectancy(positions)
    assert d.value < 0
    assert d.severity == "high"


def test_concentration_fires_on_single_symbol_dominance() -> None:
    rows = [_position("AAA", "2024-01-01", "09:00", "2024-01-01", "10:00", 100, 1000, 1010, gross_pnl=1000, charges=0)]
    rows += [_position("BBB", "2024-01-01", "09:00", "2024-01-01", "10:00", 1, 10, 11, gross_pnl=1, charges=0)]
    positions = pd.DataFrame(rows)
    d = concentration(positions)
    assert d.value > 0.9  # nearly all exposure in AAA
    assert d.severity == "critical"


def test_tail_dependence_flips_negative_when_one_trade_carries_everything() -> None:
    # 1 big winner (+1000) + 30 small losers (-20 each = -600) -> total +400,
    # but n_best=int(0.05*31)=1, so excluding just the top trade leaves -600.
    rows = [_position("AAA", "2024-01-01", "09:00", "2024-01-01", "10:00", 1, 100, 100 + 1000, gross_pnl=1000, charges=0)]
    rows += [_position(f"L{i}", "2024-01-01", "09:00", "2024-01-01", "10:00", 1, 100, 90, gross_pnl=-20, charges=0) for i in range(30)]
    positions = pd.DataFrame(rows)
    d = tail_dependence(positions)
    assert d.detail["total_pnl"] > 0  # profitable overall
    assert d.detail["flips_negative"] is True  # but only because of one trade
    assert d.severity == "critical"


def test_drawdown_profile_computes_max_drawdown() -> None:
    rows = [
        _position("AAA", "2024-01-01", "09:00", "2024-01-01", "10:00", 1, 100, 200, gross_pnl=100, charges=0),
        _position("AAA", "2024-01-02", "09:00", "2024-01-02", "10:00", 1, 100, 50, gross_pnl=-150, charges=0),
        _position("AAA", "2024-01-03", "09:00", "2024-01-03", "10:00", 1, 100, 120, gross_pnl=20, charges=0),
    ]
    positions = pd.DataFrame(rows)
    d = drawdown_profile(positions)
    assert d.value == -150.0  # peak 100, trough -50 -> drawdown of -150


def test_run_all_returns_one_diagnostic_per_dosha_and_degrades_on_empty() -> None:
    empty = pd.DataFrame(columns=["symbol", "segment", "open_date", "open_time", "close_date", "close_time",
                                   "quantity", "entry_price", "exit_price", "gross_pnl", "charges", "net_pnl",
                                   "holding_seconds", "is_intraday", "mae", "mfe"])
    results = run_all(empty)
    assert len(results) == 13
    assert all(d.severity in ("ok", "notable", "high", "critical") for d in results)
