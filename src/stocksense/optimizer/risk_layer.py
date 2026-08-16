"""
Portfolio risk layer (audit finding MED-8): the target-weight
construction in portfolio/construct.py is equal-weight top-N with
nothing preventing a book from sitting 60% in one sector. This module
adds constraints on TOP of a target_weights_top_n() output — it never
generates weights itself, only trims them, so it composes with the
existing pipeline rather than replacing it.

All functions here are pure position-sizing arithmetic. No LLM
involvement, no I/O — this is exactly the kind of thing that must never
be narrated instead of computed (agent/claude_cli.py's compute/narrate
rule), because a sector cap either held or it didn't; there's nothing
to explain that a number doesn't already say.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.15   # no single name above 15% of the book
    max_sector_weight: float = 0.35      # no single sector above 35%
    target_volatility: float | None = None  # annualized; None disables vol targeting
    max_correlation_cluster_weight: float = 0.50  # no cluster of >0.7-correlated names above 50%


def apply_position_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    """Caps each individual weight at `max_weight` and redistributes the
    excess proportionally across the uncapped names, iterating until no
    weight exceeds the cap (a single pass can leave a redistributed name
    back over the cap if it was already close to it)."""
    w = weights.copy().astype(float)
    for _ in range(len(w) + 1):  # bounded: each iteration caps at least one more name or terminates
        over = w[w > max_weight]
        if over.empty:
            break
        excess = (over - max_weight).sum()
        w[over.index] = max_weight
        under = w[w < max_weight]
        under_total = under.sum()
        if under_total <= 1e-12:
            break  # nothing left to redistribute into; excess weight is simply dropped (holds less than target sum)
        w[under.index] = under + (under / under_total) * excess
    return w


def apply_sector_cap(weights: pd.Series, sector_map: dict[str, str], max_sector_weight: float) -> pd.Series:
    """Caps aggregate weight per sector, redistributing excess
    proportionally across names OUTSIDE the capped sector(s). Symbols
    absent from `sector_map` are treated as their own singleton sector
    (never capped by this function alone, since a sector of one name is
    already bounded by apply_position_cap)."""
    w = weights.copy().astype(float)
    sectors = pd.Series({sym: sector_map.get(sym, sym) for sym in w.index})

    for _ in range(sectors.nunique() + 1):
        sector_totals = w.groupby(sectors).sum()
        over_sectors = sector_totals[sector_totals > max_sector_weight]
        if over_sectors.empty:
            break
        for sector in over_sectors.index:
            members = sectors[sectors == sector].index
            sector_total = w[members].sum()
            scale = max_sector_weight / sector_total
            excess = sector_total - max_sector_weight
            w[members] = w[members] * scale

            outside = w.index.difference(members)
            outside_total = w[outside].sum()
            if outside_total > 1e-12:
                w[outside] = w[outside] + (w[outside] / outside_total) * excess
    return w


def volatility_scale_factor(portfolio_daily_returns: pd.Series, target_annual_vol: float,
                             trading_days_per_year: int = 252) -> float:
    """Scalar to apply to the whole book's weights (not per-name) so
    realized volatility tracks `target_annual_vol`. Returns 1.0
    (no scaling) if there isn't enough history to estimate volatility
    reliably — an unreliable estimate should not silently lever or
    delever the book."""
    if len(portfolio_daily_returns) < 20:
        return 1.0
    realized_annual_vol = portfolio_daily_returns.std() * np.sqrt(trading_days_per_year)
    if realized_annual_vol <= 1e-9:
        return 1.0
    return float(target_annual_vol / realized_annual_vol)


def correlation_cluster_weight(weights: pd.Series, returns: pd.DataFrame, corr_threshold: float = 0.7) -> pd.Series:
    """For each held name, the total weight of the cluster it belongs to
    (itself plus every other held name correlated above `corr_threshold`
    to it, computed from `returns` — a wide symbol-columns daily-return
    frame). This is a diagnostic, not itself a constraint: a name in a
    50%-weighted correlated cluster isn't necessarily wrong, but it
    should be visible before it's decided to be fine."""
    held = weights[weights > 0].index
    common = [s for s in held if s in returns.columns]
    if len(common) < 2:
        return pd.Series(weights[held].values, index=held)  # nothing to cluster against

    corr = returns[common].corr()
    cluster_weight = {}
    for sym in held:
        if sym not in common:
            cluster_weight[sym] = float(weights[sym])
            continue
        correlated = corr.index[(corr[sym] > corr_threshold)]
        cluster_weight[sym] = float(weights.reindex(correlated, fill_value=0.0).sum())
    return pd.Series(cluster_weight)


def apply_risk_limits(weights: pd.Series, limits: RiskLimits, sector_map: dict[str, str] | None = None,
                       portfolio_daily_returns: pd.Series | None = None) -> pd.Series:
    """Applies position cap, then sector cap, then (if configured)
    portfolio-level volatility scaling, in that order. Order matters:
    volatility scaling is a uniform scalar applied last so it can't be
    partially undone by the redistribution steps above it."""
    w = apply_position_cap(weights, limits.max_position_weight)
    if sector_map:
        w = apply_sector_cap(w, sector_map, limits.max_sector_weight)
    if limits.target_volatility is not None and portfolio_daily_returns is not None:
        scale = volatility_scale_factor(portfolio_daily_returns, limits.target_volatility)
        w = w * scale
    return w
