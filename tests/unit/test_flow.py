"""Order-flow features: OFI, Lee-Ready trade-sign classification, VPIN.

These are features nothing in the six failed intraday attempts ever had --
all six ranked cross-sections on price-derived technicals. Order flow is a
structurally different information source, which is why it's worth one
honest, separately-tested attempt.
"""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.microstructure.flow import (
    classify_trade_side,
    order_flow_imbalance,
    vpin,
)


def test_classify_trade_side_above_mid_is_buy():
    assert classify_trade_side(trade_price=100.5, mid_price=100.0) == 1


def test_classify_trade_side_below_mid_is_sell():
    assert classify_trade_side(trade_price=99.5, mid_price=100.0) == -1


def test_classify_trade_side_at_mid_uses_tick_test_uptick_is_buy():
    # Lee-Ready: a trade exactly at the mid falls back to the tick test --
    # compare against the previous trade price.
    assert classify_trade_side(trade_price=100.0, mid_price=100.0, prev_price=99.9) == 1


def test_classify_trade_side_at_mid_uses_tick_test_downtick_is_sell():
    assert classify_trade_side(trade_price=100.0, mid_price=100.0, prev_price=100.1) == -1


def test_order_flow_imbalance_is_sign_correct_on_pure_buy_pressure():
    # a constructed sequence that is entirely bid-side additions / ask-side
    # cancellations should read as strongly positive (buy pressure).
    bid_deltas = np.array([50.0, 30.0, 20.0])
    ask_deltas = np.array([-10.0, -5.0, -8.0])

    ofi = order_flow_imbalance(bid_deltas, ask_deltas)

    assert ofi > 0


def test_order_flow_imbalance_is_sign_correct_on_pure_sell_pressure():
    bid_deltas = np.array([-50.0, -30.0, -20.0])
    ask_deltas = np.array([10.0, 5.0, 8.0])

    ofi = order_flow_imbalance(bid_deltas, ask_deltas)

    assert ofi < 0


def test_order_flow_imbalance_is_zero_when_balanced():
    bid_deltas = np.array([10.0, -10.0])
    ask_deltas = np.array([10.0, -10.0])

    assert order_flow_imbalance(bid_deltas, ask_deltas) == pytest.approx(0.0)


def test_vpin_is_high_when_all_volume_buckets_are_one_sided():
    buy_volume = np.array([100.0, 100.0, 100.0])
    sell_volume = np.array([0.0, 0.0, 0.0])

    assert vpin(buy_volume, sell_volume) == pytest.approx(1.0)


def test_vpin_is_low_when_volume_buckets_are_balanced():
    buy_volume = np.array([50.0, 50.0, 50.0])
    sell_volume = np.array([50.0, 50.0, 50.0])

    assert vpin(buy_volume, sell_volume) == pytest.approx(0.0)


def test_vpin_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        vpin(np.array([1.0, 2.0]), np.array([1.0]))
