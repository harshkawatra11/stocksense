"""Spread and impact estimators, checked against synthetic series with a
planted answer -- the only way to know an estimator recovers the truth."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.microstructure.spread import (
    amihud_illiquidity,
    effective_spread_bps,
    kyle_lambda,
    quoted_spread,
    roll_spread,
)


def test_quoted_spread_is_ask_minus_bid():
    assert quoted_spread(bid=99.5, ask=100.5) == 1.0


def test_quoted_spread_rejects_crossed_book():
    with pytest.raises(ValueError):
        quoted_spread(bid=100.5, ask=99.5)


def test_effective_spread_bps_buy_at_ask():
    # trade at 100.5 against a mid of 100.0 -> paid half the quoted spread
    # over mid, doubled and expressed in bps of mid.
    result = effective_spread_bps(trade_price=100.5, mid_price=100.0, side="buy")
    assert result == 100.0  # 2 * 0.5 / 100.0 * 10_000


def test_effective_spread_bps_sell_at_bid():
    result = effective_spread_bps(trade_price=99.5, mid_price=100.0, side="sell")
    assert result == 100.0


def test_roll_spread_recovers_a_planted_bid_ask_bounce():
    # Pure Roll bounce, no fundamental noise: eta_t is iid +-planted/2 around
    # a constant efficient price, exactly Roll's (1984) model, whose formula
    # cov(dp_t, dp_t-1) = -s^2/4 assumes IID bounce, not deterministic
    # alternation (which would make dp_t, dp_t-1 perfectly anti-correlated
    # at magnitude s instead of s/2, and recover 2x the planted spread).
    rng = np.random.default_rng(1)
    planted_spread = 0.20
    half = planted_spread / 2
    eta = rng.choice([half, -half], size=200_000)
    prices = 100.0 + eta

    recovered = roll_spread(prices)

    assert recovered == pytest.approx(planted_spread, rel=1e-2)


def test_roll_spread_is_nan_when_autocovariance_is_nonnegative():
    # A monotone trend has positive serial covariance in price changes --
    # Roll's model is inapplicable and must say so, not return a fake number.
    prices = np.arange(100.0, 110.0, 1.0)
    assert np.isnan(roll_spread(prices))


def test_amihud_illiquidity_scales_with_price_impact_per_rupee_traded():
    # Two days with identical |return| but day 2 has 10x the turnover ->
    # day 2 is 10x more liquid, so illiquidity should be 10x smaller.
    returns = np.array([0.01, 0.01])
    dollar_volume = np.array([1_000_000.0, 10_000_000.0])

    illiq = amihud_illiquidity(returns, dollar_volume)

    assert illiq[0] == pytest.approx(illiq[1] * 10)


def test_kyle_lambda_recovers_a_planted_linear_impact_slope():
    rng = np.random.default_rng(0)
    signed_volume = rng.normal(0, 1000, size=2000)
    planted_lambda = 0.0005
    price_changes = planted_lambda * signed_volume  # noiseless, exact slope

    recovered = kyle_lambda(price_changes, signed_volume)

    assert recovered == pytest.approx(planted_lambda, rel=1e-6)

