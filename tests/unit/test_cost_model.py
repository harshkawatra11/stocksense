from __future__ import annotations

from stocksense.execution.cost_model import (
    apply_turnover_cost,
    compute_charges,
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


# ---- compute_charges: hand-computed against a Rs100,000 position,
# matching research/phase0_verdict.md's intraday-cost-correction table ----

def test_intraday_round_trip_matches_hand_computation() -> None:
    buy = compute_charges("equity_intraday", "buy", quantity=1000, price=100.0)
    sell = compute_charges("equity_intraday", "sell", quantity=1000, price=100.0)
    # hand-computed: brokerage 20+20=40, STT 0.025%*100000=25 (sell only),
    # exchange 0.00307%*100000*2=6.14, SEBI 10/cr*100000*2=0.20,
    # stamp 0.003%*100000=3 (buy only), GST 18%*(40+6.14+0.20)=8.3448
    assert abs(buy.brokerage + sell.brokerage - 40.0) < 0.01
    assert abs(buy.stt + sell.stt - 25.0) < 0.01
    assert abs(buy.stamp_duty + sell.stamp_duty - 3.0) < 0.01
    total = buy.total_charges + sell.total_charges
    assert abs(total - 82.68) < 0.05


def test_delivery_round_trip_matches_hand_computation() -> None:
    buy = compute_charges("equity_delivery", "buy", quantity=1000, price=100.0)
    sell = compute_charges("equity_delivery", "sell", quantity=1000, price=100.0)
    # hand-computed: brokerage 0, STT 0.1%*100000*2=200 (both legs),
    # stamp 0.015%*100000=15 (buy only)
    assert buy.brokerage == 0.0 and sell.brokerage == 0.0
    assert abs(buy.stt + sell.stt - 200.0) < 0.01
    assert abs(buy.stamp_duty + sell.stamp_duty - 15.0) < 0.01
    total = buy.total_charges + sell.total_charges
    assert abs(total - 222.48) < 0.05


def test_intraday_cheaper_than_delivery() -> None:
    """The retracted claim this project got wrong once: intraday MIS STT
    is 0.025% sell-side only, so intraday is CHEAPER than delivery's 0.1%
    both-legs STT, not more expensive. See research/phase0_verdict.md."""
    i_buy = compute_charges("equity_intraday", "buy", quantity=1000, price=100.0)
    i_sell = compute_charges("equity_intraday", "sell", quantity=1000, price=100.0)
    d_buy = compute_charges("equity_delivery", "buy", quantity=1000, price=100.0)
    d_sell = compute_charges("equity_delivery", "sell", quantity=1000, price=100.0)
    assert (i_buy.total_charges + i_sell.total_charges) < (d_buy.total_charges + d_sell.total_charges)


def test_stt_only_on_sell_leg_for_intraday_and_fno() -> None:
    for segment in ("equity_intraday", "fno_futures", "fno_options"):
        buy = compute_charges(segment, "buy", quantity=100, price=1000.0)
        assert buy.stt == 0.0, f"{segment} buy leg should have zero STT"


def test_stamp_duty_only_on_buy_leg() -> None:
    for segment in ("equity_delivery", "equity_intraday"):
        sell = compute_charges(segment, "sell", quantity=100, price=1000.0)
        assert sell.stamp_duty == 0.0, f"{segment} sell leg should have zero stamp duty"


def test_unknown_segment_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        compute_charges("bogus_segment", "buy", quantity=100, price=1000.0)
