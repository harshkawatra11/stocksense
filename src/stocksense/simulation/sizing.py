"""Position sizing, and the tradeable price band.

## The share-count fallacy, settled numerically

A recurring and expensive intuition: "with 25,000 rupees, buy the 88-rupee stock
rather than the 1,000-rupee stock -- you get 284 shares instead of 25, so a
1-rupee move pays 284 rupees instead of 25."

It does not survive the arithmetic, because P&L = capital x return and the share
count cancels:

    25,000 in an   88 stock, +1%  ->  284 sh x 0.88 = 250 rupees
    25,000 in a  1000 stock, +1%  ->   25 sh x 10.00 = 250 rupees

Identical. The intuition compares a 1-rupee move on both, but 1 rupee is 1.14% of
88 and 0.10% of 1,000 -- a large move against a small one, not a cheap stock
against an expensive one. Share count is an accounting artifact, never a source
of edge.

## What price level DOES change, and it pulls both ways

**Tick drag favours EXPENSIVE stocks.** NSE's tick is 0.05 rupees. As a fraction
of price that is 5.7 bps on an 88-rupee stock and 0.5 bps on a 1,000-rupee stock.
Crossing the spread on entry and exit therefore costs ~11 bps on the cheap name
versus ~1 bp on the dear one -- so break-even is ~19 bps against ~9 bps once the
verified 8.3 bps of charges are added. Cheap stocks are *harder*, not easier.

**Divisibility favours CHEAP stocks.** Shares are integral. At 87,500 rupees of
exposure a 14,850-rupee stock buys 5 shares and strands 13,000 rupees, and a
40,000-rupee stock cannot be sized at all. That is a real constraint at this
account size.

Between the two there is a band, and `tradeable_price_band` computes it from the
account rather than asserting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NSE_TICK_INR = 0.05


@dataclass(frozen=True)
class PriceBand:
    """The price range in which a given account can trade efficiently."""

    min_price_inr: float
    max_price_inr: float
    exposure_inr: float
    min_shares: int
    max_tick_bps: float

    def contains(self, price: float) -> bool:
        return self.min_price_inr <= price <= self.max_price_inr


def tick_drag_bps(price: float, tick: float = NSE_TICK_INR) -> float:
    """One tick as a fraction of price, in basis points.

    This is the FLOOR on spread cost: no trade can be cheaper than the minimum
    price increment, so it is a hard tax that scales inversely with price.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    return float(tick / price * 10_000)


def tradeable_price_band(
    equity_inr: float,
    leverage: float = 5.0,
    max_positions: int = 2,
    max_tick_bps: float = 2.0,
    min_shares: int = 20,
) -> PriceBand:
    """The price range this account can trade without structural handicap.

    Lower bound -- tick drag. Reject names where a single tick exceeds
    `max_tick_bps` of price, because that cost is unavoidable and compounds on
    every entry and exit. At the 2 bps default the floor is 250 rupees.

    Upper bound -- divisibility. Require at least `min_shares` per position so
    that the intended size is achievable and stranded capital stays under ~5%.
    Fewer shares also makes position sizing coarse: with 5 shares the smallest
    change you can make is a 20% change in exposure.

    Args:
        equity_inr: account equity.
        leverage: MIS multiple. Note the plan's warning that Monte Carlo sizing
            will likely argue for less than the full 5x on 1-2 concentrated names.
        max_positions: capital is split across at most this many names.
        max_tick_bps: the tick-drag ceiling that sets the lower price bound.
        min_shares: the divisibility floor that sets the upper price bound.
    """
    if equity_inr <= 0 or leverage <= 0 or max_positions < 1:
        raise ValueError("equity, leverage and max_positions must be positive")

    exposure = equity_inr * leverage
    per_position = exposure / max_positions

    return PriceBand(
        min_price_inr=NSE_TICK_INR / (max_tick_bps / 10_000),
        max_price_inr=per_position / min_shares,
        exposure_inr=exposure,
        min_shares=min_shares,
        max_tick_bps=max_tick_bps,
    )


def whole_share_quantity(capital_inr: float, price: float) -> tuple[int, float]:
    """(shares, stranded capital). Shares are integral -- always have been.

    Ignoring this is how a backtest quietly reports a return the account could
    never have achieved, because it sized a fractional position.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    qty = int(math.floor(capital_inr / price))
    return qty, float(capital_inr - qty * price)


def capital_efficiency(capital_inr: float, price: float) -> float:
    """Fraction of capital actually deployed after whole-share rounding."""
    qty, stranded = whole_share_quantity(capital_inr, price)
    if capital_inr <= 0:
        return 0.0
    return float((capital_inr - stranded) / capital_inr)


def breakeven_bps(price: float, charges_bps: float = 8.3, spread_ticks: float = 2.0) -> float:
    """Gross move required just to break even, in basis points.

    `charges_bps` defaults to the 8.3 bps MIS round trip verified against a real
    charge sheet. `spread_ticks` is how many ticks the round trip crosses --
    2.0 assumes paying the spread on both entry and exit, which is what a market
    order does and therefore the honest default.

    Demonstrates the point above: at 88 rupees this returns ~19.7 bps, at 1,000
    rupees ~9.3 bps. The cheap stock must move more than twice as far, in
    percentage terms, before it earns anything.
    """
    return float(charges_bps + spread_ticks * tick_drag_bps(price))


def fractional_kelly(
    win_prob: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
) -> float:
    """Kelly fraction of capital, scaled down.

        f* = p - (1 - p) / b

    Full Kelly maximises long-run growth but has brutal drawdowns and assumes the
    edge is known exactly. It is not -- it is estimated from a noisy backtest, and
    an overestimated edge makes full Kelly ruinous. Quarter-Kelly is the standard
    defensive choice and is the default here.

    Returns 0.0 when the edge is non-positive: no edge, no position.
    """
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError("win_prob must be in [0, 1]")
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    edge = win_prob - (1.0 - win_prob) / win_loss_ratio
    return float(max(0.0, edge) * fraction)


def probability_of_ruin(
    daily_returns: np.ndarray,
    equity_inr: float,
    ruin_threshold: float = 0.5,
    horizon_days: int = 250,
    n_paths: int = 100_000,
    seed: int = 7,
) -> float:
    """P(equity falls below `ruin_threshold` of its start) within the horizon.

    Bootstrapped from the EMPIRICAL return distribution rather than a normal, so
    fat tails survive -- which is the entire reason for running this. Sampling
    is with replacement from the observed daily returns.

    This is the number that should decide leverage on 1-2 concentrated names.
    Concentration and leverage together are exactly the configuration where the
    arithmetic of ruin bites hardest, and the honest expectation is that it
    argues for less than 5x.
    """
    if daily_returns.size == 0:
        raise ValueError("need a non-empty return series")
    if equity_inr <= 0:
        raise ValueError("equity must be positive")

    rng = np.random.default_rng(seed)
    draws = rng.choice(daily_returns, size=(n_paths, horizon_days), replace=True)
    equity_paths = equity_inr * np.cumprod(1.0 + draws, axis=1)
    hit_ruin = (equity_paths <= equity_inr * ruin_threshold).any(axis=1)
    return float(hit_ruin.mean())
