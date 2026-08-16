"""Risk layer tests (audit finding MED-8). Position and sector caps must
actually hold after redistribution — not just reduce the offending
weight while creating a new violation elsewhere, which is the failure
mode naive redistribution logic falls into."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.optimizer.risk_layer import (
    RiskLimits,
    apply_position_cap,
    apply_risk_limits,
    apply_sector_cap,
    correlation_cluster_weight,
    volatility_scale_factor,
)


def test_position_cap_caps_the_offending_weight() -> None:
    weights = pd.Series({"A": 0.40, "B": 0.20, "C": 0.20, "D": 0.20})
    capped = apply_position_cap(weights, max_weight=0.15)
    assert capped.max() <= 0.15 + 1e-9


def test_position_cap_preserves_total_when_redistribution_possible() -> None:
    # only A exceeds the 0.15 cap; the rest sit at 0.10 each (comfortably
    # under, with plenty of combined headroom) and can absorb A's excess
    weights = pd.Series({"A": 0.30, "B": 0.10, "C": 0.10, "D": 0.10, "E": 0.10, "F": 0.10, "G": 0.10, "H": 0.10})
    capped = apply_position_cap(weights, max_weight=0.15)
    assert abs(capped.sum() - weights.sum()) < 1e-6


def test_position_cap_no_op_when_all_already_under() -> None:
    weights = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10})
    capped = apply_position_cap(weights, max_weight=0.15)
    pd.testing.assert_series_equal(capped, weights, check_exact=False)


def test_position_cap_handles_uniform_over_cap_gracefully() -> None:
    """Every name over cap and nothing to redistribute into -- must not
    infinite-loop or raise; excess weight is simply not fully deployed."""
    weights = pd.Series({"A": 0.30, "B": 0.30, "C": 0.30, "D": 0.10})
    capped = apply_position_cap(weights, max_weight=0.15)
    assert capped.max() <= 0.15 + 1e-9


def test_sector_cap_holds_after_redistribution() -> None:
    weights = pd.Series({"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.20})
    sector_map = {"A": "IT", "B": "IT", "C": "BANK", "D": "PHARMA"}
    capped = apply_sector_cap(weights, sector_map, max_sector_weight=0.35)

    sectors = pd.Series(sector_map)
    sector_totals = capped.groupby(sectors).sum()
    assert sector_totals.max() <= 0.35 + 1e-6


def test_sector_cap_preserves_total_weight() -> None:
    weights = pd.Series({"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.20})
    sector_map = {"A": "IT", "B": "IT", "C": "BANK", "D": "PHARMA"}
    capped = apply_sector_cap(weights, sector_map, max_sector_weight=0.35)
    assert abs(capped.sum() - weights.sum()) < 1e-4


def test_sector_cap_no_op_when_no_sector_exceeds() -> None:
    weights = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.20})
    sector_map = {"A": "IT", "B": "BANK", "C": "PHARMA", "D": "AUTO", "E": "FMCG"}
    capped = apply_sector_cap(weights, sector_map, max_sector_weight=0.35)
    pd.testing.assert_series_equal(capped.sort_index(), weights.sort_index(), check_exact=False)


def test_volatility_scale_factor_shrinks_when_realized_vol_too_high() -> None:
    rng = np.random.default_rng(1)
    high_vol_returns = pd.Series(rng.normal(0, 0.05, 100))  # ~79% annualized vol
    scale = volatility_scale_factor(high_vol_returns, target_annual_vol=0.15)
    assert scale < 1.0


def test_volatility_scale_factor_grows_when_realized_vol_too_low() -> None:
    rng = np.random.default_rng(2)
    low_vol_returns = pd.Series(rng.normal(0, 0.002, 100))  # ~3% annualized vol
    scale = volatility_scale_factor(low_vol_returns, target_annual_vol=0.15)
    assert scale > 1.0


def test_volatility_scale_factor_neutral_on_insufficient_history() -> None:
    short_returns = pd.Series([0.01, -0.01, 0.02])
    scale = volatility_scale_factor(short_returns, target_annual_vol=0.15)
    assert scale == 1.0


def test_correlation_cluster_weight_groups_correlated_names() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 100)
    returns = pd.DataFrame(
        {
            "A": base + rng.normal(0, 0.001, 100),  # near-identical to base
            "B": base + rng.normal(0, 0.001, 100),  # also near-identical -> A,B correlated
            "C": rng.normal(0, 0.01, 100),           # independent
        }
    )
    weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    cluster = correlation_cluster_weight(weights, returns, corr_threshold=0.7)
    assert cluster["A"] >= 0.5  # A's cluster includes both A and B
    assert cluster["C"] <= 0.4 + 1e-9  # C is not correlated with anything, cluster = itself


def test_apply_risk_limits_composes_position_and_sector_caps() -> None:
    weights = pd.Series({"A": 0.50, "B": 0.20, "C": 0.20, "D": 0.10})
    sector_map = {"A": "IT", "B": "IT", "C": "BANK", "D": "PHARMA"}
    limits = RiskLimits(max_position_weight=0.15, max_sector_weight=0.35)

    result = apply_risk_limits(weights, limits, sector_map=sector_map)
    assert result.max() <= 0.15 + 1e-6

    sectors = pd.Series(sector_map)
    sector_totals = result.groupby(sectors).sum()
    assert sector_totals.max() <= 0.35 + 1e-4


def test_apply_risk_limits_applies_volatility_scaling_last() -> None:
    weights = pd.Series({"A": 0.10, "B": 0.10})
    rng = np.random.default_rng(4)
    high_vol_returns = pd.Series(rng.normal(0, 0.05, 100))
    limits = RiskLimits(max_position_weight=0.15, max_sector_weight=0.35, target_volatility=0.10)

    result = apply_risk_limits(weights, limits, portfolio_daily_returns=high_vol_returns)
    assert result.sum() < weights.sum()  # scaled down due to excess vol
