"""
Phase G3: the one place capital legitimately re-enters a system that is
otherwise deliberately capital-agnostic (no account size lives in
config, a model, the gate, or the prediction ledger -- see
docs/STATUS.md's Phase G entry). Indian equity charges are not
uniformly proportional to trade value, so a strategy's viability CAN
depend on ticket size -- but the direction of that dependency is
segment-specific and should be derived from execution.cost_model, never
assumed.

Measured directly (see the accompanying tests): `equity_delivery` costs
are EXACTLY proportional to trade value at every ticket size (zero
brokerage at discount brokers; STT/exchange/SEBI/stamp/GST all scale
with value) -- cost in bps is IDENTICAL whether the position is ₹1,000
or ₹10,00,000. `equity_intraday` is NOT: the ₹20-flat-or-0.03%-
whichever-lower brokerage floor makes small intraday tickets bear more
cost in bps than large ones (confirmed live in research/verdict_
intraday.md's own real-account evidence, and the reason the retired
intraday track needed a "break-even ticket size" concept at all).

Phase G's daily/monthly track (h=10, ~10-trading-day holds) is a
`equity_delivery` product, not intraday -- so the honest answer for
"how much capital do I need" is NOT a break-even ticket size (there
isn't one; costs don't depend on ticket size for this segment). The
capital question that DOES apply is whole-share divisibility:
`min_capital_for_full_positions` below.
"""

from __future__ import annotations

from stocksense.execution.cost_model import compute_charges


def cost_bps_is_ticket_size_invariant(segment: str, prices: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0),
                                       quantities: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
                                       tolerance_bps: float = 1e-6) -> bool:
    """Derives, rather than assumes, whether `segment`'s round-trip cost
    in bps is the same across ticket sizes -- sweeps compute_charges over
    a grid of (price, quantity) combinations spanning several orders of
    magnitude of trade value and checks the round-trip bps figure stays
    within `tolerance_bps` of the first one. This is what backs the
    module docstring's claim above; the claim is checked here rather than
    hardcoded, so it can never silently drift out of sync with a future
    change to cost_model.compute_charges."""
    bps_values = []
    for price in prices:
        for qty in quantities:
            value = price * qty
            buy = compute_charges(segment, "buy", qty, price)
            sell = compute_charges(segment, "sell", qty, price)
            bps_values.append((buy.total_charges + sell.total_charges) / value * 10_000)

    return (max(bps_values) - min(bps_values)) <= tolerance_bps


def round_trip_cost_bps(segment: str, price: float = 1000.0, quantity: float = 100.0) -> float:
    """The round-trip cost in bps for `segment`, computed at one
    representative ticket size. For a ticket-size-invariant segment (see
    above), the specific price/quantity chosen doesn't matter -- this
    single number IS the answer for any capital size. For a ticket-size-
    DEPENDENT segment (intraday), this is only valid at the specific
    ticket size passed in; callers needing intraday economics at a
    specific size should call compute_charges directly rather than treat
    this as size-agnostic."""
    value = price * quantity
    buy = compute_charges(segment, "buy", quantity, price)
    sell = compute_charges(segment, "sell", quantity, price)
    return (buy.total_charges + sell.total_charges) / value * 10_000


def min_capital_for_full_positions(prices: dict[str, float], weights: dict[str, float]) -> float:
    """The capital question that actually applies to a whole-share
    equity portfolio: the minimum capital such that EVERY target
    position can hold at least one whole share. Below this figure, at
    least one target position rounds to zero shares and the portfolio
    silently drops a name the model actually picked -- a real
    constraint, distinct from (and unrelated to) transaction costs.

    For symbol i at target weight w_i and price p_i, holding >= 1 share
    requires capital >= p_i / w_i. The binding constraint is the MAXIMUM
    of that ratio across the portfolio -- not simply the most expensive
    share outright, since a modestly-priced stock at a small target
    weight can require more capital to hold even one share of than an
    expensive stock at a large weight.

    Symbols with a zero or missing weight are ignored (not a target
    position at all, so they impose no constraint). Raises no error on
    an empty portfolio -- returns 0.0, meaning no capital is needed to
    hold nothing.
    """
    ratios = [prices[s] / weights[s] for s in weights if s in prices and weights[s] > 0]
    return max(ratios) if ratios else 0.0
