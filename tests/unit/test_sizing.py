"""Phase G3: capital-agnostic cost/sizing tests. The load-bearing claim
this backs (docs/STATUS.md's Phase G design invariant): equity_delivery
costs are ticket-size invariant so no capital figure is needed to know
the strategy's cost economics, while equity_intraday is NOT -- derived
from execution.cost_model.compute_charges directly, not hardcoded, so
it can never silently drift out of sync with that module."""

from __future__ import annotations

import pytest

from stocksense.optimizer.sizing import (
    cost_bps_is_ticket_size_invariant,
    min_capital_for_full_positions,
    round_trip_cost_bps,
)


def test_equity_delivery_cost_bps_is_ticket_size_invariant() -> None:
    assert cost_bps_is_ticket_size_invariant("equity_delivery")


def test_equity_intraday_cost_bps_is_not_ticket_size_invariant() -> None:
    """The contrasting case -- proves the invariance test itself is
    capable of detecting a real dependency, not just always returning
    True."""
    assert not cost_bps_is_ticket_size_invariant("equity_intraday")


def test_round_trip_cost_bps_matches_zerodha_verified_delivery_figure() -> None:
    """docs/STATUS.md's own verified figure: ~22.2bps round-trip on a
    ₹100k delivery position."""
    bps = round_trip_cost_bps("equity_delivery", price=1000.0, quantity=100.0)
    assert bps == pytest.approx(22.2481, abs=0.01)


def test_round_trip_cost_bps_delivery_same_at_any_size() -> None:
    small = round_trip_cost_bps("equity_delivery", price=100.0, quantity=10.0)
    large = round_trip_cost_bps("equity_delivery", price=100.0, quantity=100_000.0)
    assert small == pytest.approx(large, abs=1e-6)


def test_round_trip_cost_bps_intraday_shrinks_with_size() -> None:
    small = round_trip_cost_bps("equity_intraday", price=100.0, quantity=10.0)
    large = round_trip_cost_bps("equity_intraday", price=100.0, quantity=10_000.0)
    assert large < small


def test_min_capital_for_full_positions_binding_constraint() -> None:
    # A: price 3000, weight 30% -> needs 10,000 to hold 1 share
    # B: price 5, weight 2% -> needs 250 to hold 1 share
    # C: price 500, weight 5% -> needs 10,000 to hold 1 share (ties A)
    prices = {"A": 3000.0, "B": 5.0, "C": 500.0}
    weights = {"A": 0.30, "B": 0.02, "C": 0.05}
    result = min_capital_for_full_positions(prices, weights)
    assert result == pytest.approx(10_000.0)


def test_min_capital_for_full_positions_cheap_stock_can_bind_harder() -> None:
    """The property the docstring calls out explicitly: a cheap stock at
    a tiny weight can require MORE capital than an expensive stock at a
    large weight."""
    prices = {"EXPENSIVE": 3000.0, "CHEAP_BUT_TINY_WEIGHT": 50.0}
    weights = {"EXPENSIVE": 0.50, "CHEAP_BUT_TINY_WEIGHT": 0.01}
    # EXPENSIVE needs 3000/0.50 = 6000; CHEAP needs 50/0.01 = 5000
    result = min_capital_for_full_positions(prices, weights)
    assert result == pytest.approx(6000.0)

    # now flip so the cheap stock binds harder
    weights2 = {"EXPENSIVE": 0.50, "CHEAP_BUT_TINY_WEIGHT": 0.001}
    result2 = min_capital_for_full_positions(prices, weights2)
    assert result2 == pytest.approx(50.0 / 0.001)
    assert result2 > result


def test_min_capital_for_full_positions_ignores_zero_weight() -> None:
    prices = {"A": 100.0, "B": 100000.0}
    weights = {"A": 1.0, "B": 0.0}
    result = min_capital_for_full_positions(prices, weights)
    assert result == pytest.approx(100.0)


def test_min_capital_for_full_positions_empty_returns_zero() -> None:
    assert min_capital_for_full_positions({}, {}) == 0.0
