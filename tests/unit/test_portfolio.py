from __future__ import annotations

import pandas as pd

from stocksense.portfolio.construct import (
    apply_no_trade_band,
    enforce_turnover_budget,
    one_way_turnover,
    target_weights_top_n,
)


def test_target_weights_top_n_equal_weights() -> None:
    scores = pd.Series({"A": 5.0, "B": 3.0, "C": 4.0, "D": 1.0, "E": 2.0})
    w = target_weights_top_n(scores, top_n=3)
    assert set(w[w > 0].index) == {"A", "C", "B"}
    assert abs(w.sum() - 1.0) < 1e-9
    assert all(abs(v - 1 / 3) < 1e-9 for v in w[w > 0])


def test_target_weights_handles_nan_scores() -> None:
    scores = pd.Series({"A": 5.0, "B": float("nan"), "C": 4.0})
    w = target_weights_top_n(scores, top_n=2)
    assert set(w[w > 0].index) == {"A", "C"}


def test_no_trade_band_holds_small_changes() -> None:
    current = pd.Series({"A": 0.20, "B": 0.20})
    target = pd.Series({"A": 0.21, "B": 0.05})  # A: +0.01 (small), B: -0.15 (large)
    out = apply_no_trade_band(target, current, band=0.02)
    assert abs(out["A"] - current["A"]) < 1e-9  # held, change too small
    assert abs(out["B"] - target["B"]) < 1e-9  # traded, change exceeds band


def test_no_trade_band_produces_zero_turnover_when_target_equals_current() -> None:
    current = pd.Series({"A": 0.5, "B": 0.5})
    out = apply_no_trade_band(current.copy(), current, band=0.02)
    assert one_way_turnover(out, current) == 0.0


def test_one_way_turnover_full_replacement() -> None:
    current = pd.Series({"A": 1.0})
    target = pd.Series({"B": 1.0})
    # sum(|target - current|) / 2 = (|0-1| + |1-0|) / 2 = 1.0
    assert abs(one_way_turnover(target, current) - 1.0) < 1e-9


def test_turnover_budget_scales_toward_target() -> None:
    current = pd.Series({"A": 1.0})
    target = pd.Series({"B": 1.0})  # full replacement, turnover = 1.0
    out = enforce_turnover_budget(target, current, max_turnover=0.3)
    realized = one_way_turnover(out, current)
    assert abs(realized - 0.3) < 1e-6


def test_turnover_budget_noop_when_under_budget() -> None:
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.55, "B": 0.45})  # turnover = 0.05
    out = enforce_turnover_budget(target, current, max_turnover=0.5)
    pd.testing.assert_series_equal(out.sort_index(), target.sort_index(), check_names=False)
