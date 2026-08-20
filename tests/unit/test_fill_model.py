"""Execution realism tests (Phase E3). Pure functions, no mocking needed
-- what's under test is that each of the four rejection reasons fires on
its own trigger (and only its own trigger), that the fill price always
moves against the trader (never in their favor), and that simulate_fill
applies all four checks in the documented order (cheapest/most-
disqualifying first) so a caller can trust the FIRST rejection reason
returned, not have to re-derive it."""

from __future__ import annotations

import pytest

from stocksense.execution.fill_model import (
    CIRCUIT_LOCKED,
    DEFAULT_LEVERAGE,
    LEVERAGE_UNAVAILABLE,
    NO_NEXT_BAR,
    PARTICIPATION_EXCEEDED,
    check_participation,
    compute_fill_price,
    get_mis_leverage,
    is_circuit_locked,
    simulate_fill,
)


# ---- is_circuit_locked ----

def test_circuit_locked_when_high_equals_low() -> None:
    assert is_circuit_locked(100.0, 100.0) is True


def test_not_circuit_locked_with_real_intrabar_range() -> None:
    assert is_circuit_locked(101.5, 99.8) is False


# ---- get_mis_leverage ----

def test_unlisted_symbol_defaults_to_no_leverage() -> None:
    assert get_mis_leverage("RANDOMCO", leverage_table=None) == DEFAULT_LEVERAGE
    assert DEFAULT_LEVERAGE == 1.0  # the conservative direction, never assumed 5x


def test_listed_symbol_uses_table_value() -> None:
    table = {"RELIANCE": 5.0}
    assert get_mis_leverage("RELIANCE", leverage_table=table) == 5.0
    assert get_mis_leverage("UNLISTEDCO", leverage_table=table) == DEFAULT_LEVERAGE


# ---- check_participation ----

def test_participation_within_cap_passes() -> None:
    assert check_participation(order_qty=500, bar_volume=10_000, max_participation_pct=0.1) is True


def test_participation_exceeding_cap_fails() -> None:
    assert check_participation(order_qty=2_000, bar_volume=10_000, max_participation_pct=0.1) is False


def test_participation_at_exact_boundary_passes() -> None:
    assert check_participation(order_qty=1_000, bar_volume=10_000, max_participation_pct=0.1) is True


def test_zero_bar_volume_never_passes() -> None:
    assert check_participation(order_qty=1, bar_volume=0, max_participation_pct=0.1) is False


def test_negative_bar_volume_never_passes() -> None:
    assert check_participation(order_qty=1, bar_volume=-5, max_participation_pct=0.1) is False


# ---- compute_fill_price: spread always costs the trader ----

def test_buy_fills_above_next_bar_open() -> None:
    price = compute_fill_price(next_bar_open=100.0, direction="buy", half_spread_bps=2.5)
    assert price > 100.0
    assert price == pytest.approx(100.0 * 1.00025)


def test_sell_fills_below_next_bar_open() -> None:
    price = compute_fill_price(next_bar_open=100.0, direction="sell", half_spread_bps=2.5)
    assert price < 100.0
    assert price == pytest.approx(100.0 * 0.99975)


def test_larger_spread_costs_more_in_both_directions() -> None:
    buy_tight = compute_fill_price(100.0, "buy", half_spread_bps=1.0)
    buy_wide = compute_fill_price(100.0, "buy", half_spread_bps=10.0)
    assert buy_wide > buy_tight

    sell_tight = compute_fill_price(100.0, "sell", half_spread_bps=1.0)
    sell_wide = compute_fill_price(100.0, "sell", half_spread_bps=10.0)
    assert sell_wide < sell_tight


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError):
        compute_fill_price(100.0, direction="hold")


# ---- simulate_fill: the combined entry point, ordering, rejection reasons ----

def _good_bar(**overrides):
    defaults = dict(
        symbol="RELIANCE", direction="buy", order_qty=100,
        next_bar_open=2500.0, next_bar_high=2510.0, next_bar_low=2495.0, next_bar_volume=50_000,
    )
    defaults.update(overrides)
    return defaults


def test_simulate_fill_succeeds_on_a_clean_bar() -> None:
    result = simulate_fill(**_good_bar())
    assert result.filled is True
    assert result.rejection_reason is None
    assert result.fill_price == pytest.approx(2500.0 * 1.00025)
    assert result.leverage_applied == DEFAULT_LEVERAGE


def test_simulate_fill_rejects_when_no_next_bar() -> None:
    result = simulate_fill(**_good_bar(next_bar_open=None))
    assert result.filled is False
    assert result.rejection_reason == NO_NEXT_BAR
    assert result.fill_price is None


def test_simulate_fill_rejects_circuit_locked_bar() -> None:
    result = simulate_fill(**_good_bar(next_bar_high=2500.0, next_bar_low=2500.0))
    assert result.filled is False
    assert result.rejection_reason == CIRCUIT_LOCKED


def test_simulate_fill_rejects_unavailable_leverage() -> None:
    result = simulate_fill(**_good_bar(leverage_table={"RELIANCE": 0.0}))
    assert result.filled is False
    assert result.rejection_reason == LEVERAGE_UNAVAILABLE
    assert result.leverage_applied == 0.0


def test_simulate_fill_rejects_when_participation_exceeded() -> None:
    result = simulate_fill(**_good_bar(order_qty=10_000, next_bar_volume=50_000, max_participation_pct=0.1))
    assert result.filled is False
    assert result.rejection_reason == PARTICIPATION_EXCEEDED


def test_simulate_fill_checks_circuit_lock_before_leverage() -> None:
    """A bar that is BOTH circuit-locked AND has unavailable leverage
    must report circuit_locked -- the documented cheapest/most-
    disqualifying-first order, so a caller aggregating rejection reasons
    isn't misled about which check actually would have blocked the
    order in a cheaper universe."""
    result = simulate_fill(**_good_bar(
        next_bar_high=2500.0, next_bar_low=2500.0, leverage_table={"RELIANCE": 0.0},
    ))
    assert result.rejection_reason == CIRCUIT_LOCKED


def test_simulate_fill_checks_leverage_before_participation() -> None:
    result = simulate_fill(**_good_bar(
        leverage_table={"RELIANCE": 0.0}, order_qty=10_000, next_bar_volume=50_000,
    ))
    assert result.rejection_reason == LEVERAGE_UNAVAILABLE


def test_simulate_fill_sell_direction_fills_below_open() -> None:
    result = simulate_fill(**_good_bar(direction="sell"))
    assert result.filled is True
    assert result.fill_price < 2500.0
