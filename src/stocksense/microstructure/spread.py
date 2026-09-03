"""Spread and impact estimators.

Intraday spread is U-shaped and per-symbol, not a flat constant -- the
previous cost model's flat 5 bps slippage assumption is exactly what these
replace. Callers are expected to bucket by symbol and time-of-day before
calling these; the functions here are the per-bucket estimators themselves.
"""

from __future__ import annotations

import numpy as np


def quoted_spread(bid: float, ask: float) -> float:
    """ask - bid. A crossed book (ask < bid) is a data error, not a value."""
    if ask < bid:
        raise ValueError("crossed book: ask must be >= bid")
    return float(ask - bid)


def effective_spread_bps(trade_price: float, mid_price: float, side: str) -> float:
    """2 * signed distance from mid, in bps of mid.

    This is what the trade actually paid over the "fair" price, as opposed to
    the quoted spread, which is what was posted and may not be what filled.
    `side` is the taker's side: "buy" pays through the ask, "sell" through the
    bid, and both should return a positive cost when priced correctly.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if mid_price <= 0:
        raise ValueError("mid_price must be positive")
    sign = 1.0 if side == "buy" else -1.0
    return float(2.0 * sign * (trade_price - mid_price) / mid_price * 10_000)


def roll_spread(prices: np.ndarray) -> float:
    """Roll's (1984) implied spread from the serial covariance of price changes.

        s = 2 * sqrt(-cov(delta_p_t, delta_p_t-1))

    Requires no quote data, only trade prices -- the whole point of the
    estimator. Roll's model predicts negative first-order autocovariance from
    bid-ask bounce; when the observed covariance is non-negative (trend
    dominates, or too little data), the estimator is inapplicable and this
    returns nan rather than a fabricated number from sqrt of a negative.
    """
    deltas = np.diff(np.asarray(prices, dtype=float))
    if deltas.size < 2:
        return float("nan")
    cov = np.cov(deltas[1:], deltas[:-1], ddof=1)[0, 1]
    if cov >= 0:
        return float("nan")
    return float(2.0 * np.sqrt(-cov))


def amihud_illiquidity(returns: np.ndarray, dollar_volume: np.ndarray) -> np.ndarray:
    """Amihud (2002): |return| per rupee traded -- price impact per unit flow.

    Elementwise, not averaged, so callers can aggregate (mean over a window,
    grouped by symbol) however the search needs it.
    """
    returns = np.asarray(returns, dtype=float)
    dollar_volume = np.asarray(dollar_volume, dtype=float)
    if np.any(dollar_volume <= 0):
        raise ValueError("dollar_volume must be positive")
    return np.abs(returns) / dollar_volume


def kyle_lambda(price_changes: np.ndarray, signed_volume: np.ndarray) -> float:
    """Kyle's (1985) lambda: price impact per unit of signed order flow.

    OLS slope of price_changes on signed_volume through the origin -- no
    intercept, because Kyle's model has no drift term independent of flow.
    """
    price_changes = np.asarray(price_changes, dtype=float)
    signed_volume = np.asarray(signed_volume, dtype=float)
    denom = float(np.sum(signed_volume**2))
    if denom == 0:
        raise ValueError("signed_volume has zero variance")
    return float(np.sum(signed_volume * price_changes) / denom)
