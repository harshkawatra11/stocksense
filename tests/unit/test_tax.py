"""Tax module tests. The property that matters most: the Rs 1.25L LTCG
exemption applies ONCE across total LTCG gain for the FY, not per
position -- applying it per-trade would understate tax owed for anyone
with multiple LTCG positions, which is the normal case, not an edge
case."""

from __future__ import annotations

import pandas as pd

from stocksense.optimizer.tax import (
    LTCG_EXEMPTION_PER_FY,
    LTCG_RATE,
    STCG_RATE,
    classify_gain,
    classify_positions,
    compute_tax_liability,
    days_to_ltcg,
    harvest_candidates,
)


def _position(symbol, open_date, close_date, net_pnl, segment="equity_delivery"):
    return {"symbol": symbol, "segment": segment, "open_date": open_date, "close_date": close_date, "net_pnl": net_pnl}


def test_classify_gain_short_holding_is_stcg() -> None:
    assert classify_gain("2024-01-01", "2024-06-01") == "STCG"


def test_classify_gain_long_holding_is_ltcg() -> None:
    assert classify_gain("2023-01-01", "2024-06-01") == "LTCG"


def test_classify_gain_exactly_365_days_is_ltcg() -> None:
    assert classify_gain("2023-01-01", "2024-01-01") == "LTCG"  # 366 days (2024 is a leap year) >= 365


def test_classify_positions_excludes_intraday_and_fno() -> None:
    positions = pd.DataFrame(
        [
            _position("A", "2023-01-01", "2024-06-01", 1000, segment="equity_delivery"),
            _position("B", "2024-01-01", "2024-01-01", 500, segment="equity_intraday"),
            _position("C", "2024-01-01", "2024-02-01", 300, segment="fno_futures"),
        ]
    )
    classified = classify_positions(positions)
    assert len(classified) == 1
    assert classified.iloc[0]["symbol"] == "A"


def test_exemption_applies_once_across_multiple_ltcg_positions() -> None:
    """The load-bearing property: two LTCG positions each gaining
    Rs 100,000 (Rs 200,000 total) must NOT each get their own Rs 125,000
    exemption -- only Rs 125,000 total is exempt, leaving Rs 75,000
    taxable, not zero."""
    positions = pd.DataFrame(
        [
            _position("A", "2023-01-01", "2024-06-01", 100_000),
            _position("B", "2023-02-01", "2024-06-01", 100_000),
        ]
    )
    summary = compute_tax_liability(positions)
    assert summary.total_ltcg == 200_000
    assert summary.ltcg_exemption_used == LTCG_EXEMPTION_PER_FY
    assert summary.taxable_ltcg == 200_000 - LTCG_EXEMPTION_PER_FY  # 75,000
    assert abs(summary.ltcg_tax - (75_000 * LTCG_RATE)) < 1e-6


def test_ltcg_below_exemption_owes_zero_ltcg_tax() -> None:
    positions = pd.DataFrame([_position("A", "2023-01-01", "2024-06-01", 50_000)])
    summary = compute_tax_liability(positions)
    assert summary.taxable_ltcg == 0.0
    assert summary.ltcg_tax == 0.0


def test_stcg_taxed_at_flat_rate_no_exemption() -> None:
    positions = pd.DataFrame([_position("A", "2024-01-01", "2024-06-01", 50_000)])
    summary = compute_tax_liability(positions)
    assert abs(summary.stcg_tax - (50_000 * STCG_RATE)) < 1e-6


def test_losses_do_not_generate_negative_tax() -> None:
    positions = pd.DataFrame([_position("A", "2024-01-01", "2024-06-01", -50_000)])
    summary = compute_tax_liability(positions)
    assert summary.total_stcg == 0.0  # loss excluded from taxable gain sum
    assert summary.total_tax == 0.0


def test_exemption_already_used_elsewhere_reduces_remaining() -> None:
    positions = pd.DataFrame([_position("A", "2023-01-01", "2024-06-01", 100_000)])
    summary = compute_tax_liability(positions, ltcg_exemption_used_this_fy=125_000)
    assert summary.ltcg_exemption_used == 0.0  # exemption already fully consumed
    assert summary.taxable_ltcg == 100_000


def test_cess_applied_on_top_of_total_tax() -> None:
    positions = pd.DataFrame([_position("A", "2024-01-01", "2024-06-01", 100_000)])
    summary = compute_tax_liability(positions)
    expected_base_tax = 100_000 * STCG_RATE
    assert abs(summary.cess - expected_base_tax * 0.04) < 1e-6
    assert abs(summary.total_tax - (expected_base_tax + summary.cess)) < 1e-6


def test_days_to_ltcg_countdown() -> None:
    days = days_to_ltcg("2024-01-01", as_of_date="2024-06-01")
    assert days > 0
    assert days < 365


def test_days_to_ltcg_negative_once_crossed() -> None:
    days = days_to_ltcg("2023-01-01", as_of_date="2024-06-01")
    assert days < 0


def test_harvest_candidates_computes_unrealized_pnl_and_sorts_worst_first() -> None:
    open_positions = pd.DataFrame(
        [
            {"symbol": "WINNER", "open_date": "2024-01-01", "quantity": 10, "entry_price": 100, "current_price": 150},
            {"symbol": "LOSER", "open_date": "2024-01-01", "quantity": 10, "entry_price": 100, "current_price": 60},
        ]
    )
    result = harvest_candidates(open_positions, as_of_date="2024-06-01")
    assert result.iloc[0]["symbol"] == "LOSER"  # worst unrealized P&L first
    assert result.iloc[0]["unrealized_pnl"] == -400
    assert result.iloc[1]["unrealized_pnl"] == 500


def test_harvest_candidates_flags_near_ltcg_threshold() -> None:
    open_positions = pd.DataFrame(
        [{"symbol": "NEARLTCG", "open_date": "2023-07-01", "quantity": 10, "entry_price": 100, "current_price": 90}]
    )
    result = harvest_candidates(open_positions, as_of_date="2024-06-25")
    row = result.iloc[0]
    assert row["gain_type_if_sold_today"] == "STCG"
    assert 0 < row["days_to_ltcg"] < 30
