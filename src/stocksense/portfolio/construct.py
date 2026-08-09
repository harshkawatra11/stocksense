"""
Target-weight portfolio construction from cross-sectional scores.

The core idea (the build plan's "structural change"): the decision is not
"which stocks to buy" but "what should target weights be, given current
holdings and what changing them costs." This makes turnover a controlled
input rather than an emergent side effect (v1's failure mode), and makes
"do nothing today" a reachable output rather than something that has to
be special-cased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_weights_top_n(scores: pd.Series, top_n: int, symbols: list[str] | None = None) -> pd.Series:
    """Equal-weight the top-N ranked symbols by score on a single date.

    `scores` is indexed by symbol. Ties beyond top_n are broken by score
    order deterministically (pandas nlargest is stable on the sort it
    performs). Returns a weight series over the union of `symbols` (if
    given) and the top-N, so callers can align against a full universe
    with zeros elsewhere.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    chosen = ranked.head(top_n).index
    weights = pd.Series(0.0, index=(symbols if symbols is not None else scores.index))
    if len(chosen) == 0:
        return weights
    weights.loc[chosen] = 1.0 / len(chosen)
    return weights


def apply_no_trade_band(
    target: pd.Series, current: pd.Series, band: float
) -> pd.Series:
    """Snap target weights back to current weights wherever the proposed
    change is smaller than `band`. This is what makes "no trade today" an
    emergent, budget-respecting outcome instead of a special case: if the
    new target is close enough to what is already held, don't pay the
    round trip to get there.
    """
    idx = target.index.union(current.index)
    t = target.reindex(idx, fill_value=0.0)
    c = current.reindex(idx, fill_value=0.0)
    delta = t - c
    out = c.where(delta.abs() < band, t)
    return out


def one_way_turnover(target: pd.Series, current: pd.Series) -> float:
    """sum(|target - current|) / 2 — the standard one-way turnover
    definition consumed by execution.cost_model.apply_turnover_cost."""
    idx = target.index.union(current.index)
    t = target.reindex(idx, fill_value=0.0)
    c = current.reindex(idx, fill_value=0.0)
    return float((t - c).abs().sum() / 2.0)


def enforce_turnover_budget(
    target: pd.Series, current: pd.Series, max_turnover: float
) -> pd.Series:
    """If the proposed rebalance exceeds `max_turnover` (one-way), scale
    the move toward the target proportionally so realized turnover equals
    the budget exactly, prioritizing nothing — a uniform partial step
    toward the target. This keeps turnover a hard, budgeted constraint
    rather than something the sweep discovers after the fact.
    """
    realized = one_way_turnover(target, current)
    if realized <= max_turnover or realized == 0:
        return target
    idx = target.index.union(current.index)
    t = target.reindex(idx, fill_value=0.0)
    c = current.reindex(idx, fill_value=0.0)
    scale = max_turnover / realized
    return c + (t - c) * scale
