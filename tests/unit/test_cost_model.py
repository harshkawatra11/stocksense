from __future__ import annotations

from stocksense.execution.cost_model import (
    apply_turnover_cost,
    indian_delivery_cost_bps,
    realistic_round_trip_bps,
)


def test_cost_breakdown_totals() -> None:
    b = indian_delivery_cost_bps(brokerage_bps=0.0, slippage_bps=5.0)
    # hand-computed: 10 (STT) + 0.3 (exchange) + 0.01 (SEBI) + 1.5 (stamp)
    #   + 0.18*(0 + 0.3) (GST on brokerage+exchange) + 5 (slippage)
    expected = 10.0 + 0.3 + 0.01 + 1.5 + 0.18 * 0.3 + 5.0
    assert abs(b.total_bps - expected) < 1e-9


def test_realistic_round_trip_excludes_stamp_duty_on_sell_leg() -> None:
    rt = realistic_round_trip_bps(slippage_bps=5.0)
    buy_leg = indian_delivery_cost_bps(slippage_bps=5.0).total_bps
    sell_leg = indian_delivery_cost_bps(slippage_bps=5.0).total_bps - 1.5  # no stamp duty on sell
    assert abs(rt - (buy_leg + sell_leg)) < 1e-9
    # sanity: round trip should be roughly double one leg, minus stamp duty savings
    assert rt < 2 * buy_leg


def test_turnover_cost_scales_linearly() -> None:
    cost_full = apply_turnover_cost(turnover_fraction=1.0, round_trip_cost_bps=25.0)
    cost_half = apply_turnover_cost(turnover_fraction=0.5, round_trip_cost_bps=25.0)
    assert abs(cost_full - 2 * cost_half) < 1e-12
    assert abs(cost_full - 0.0025) < 1e-12  # 25 bps == 0.25% == 0.0025 as a fraction


def test_zero_turnover_zero_cost() -> None:
    assert apply_turnover_cost(0.0, 25.0) == 0.0
