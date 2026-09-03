"""Order flow: trade-sign classification, order-flow imbalance, VPIN.

Structurally different from the price-derived technicals every one of the
six failed intraday cross-sectional attempts relied on. Whether it holds up
is an empirical question for the search in Q4 -- this module only supplies
honest, separately-tested features for it to test.
"""

from __future__ import annotations

import numpy as np


def classify_trade_side(trade_price: float, mid_price: float, prev_price: float | None = None) -> int:
    """Lee & Ready (1991): +1 buy-initiated, -1 sell-initiated.

    A trade above the mid quote is buyer-initiated, below is seller-initiated.
    Exactly at the mid, the quote rule can't decide, so it falls back to the
    tick test: an uptick from the previous trade is a buy, a downtick a sell.
    """
    if trade_price > mid_price:
        return 1
    if trade_price < mid_price:
        return -1
    if prev_price is None:
        raise ValueError("trade at mid requires prev_price for the tick test")
    if trade_price > prev_price:
        return 1
    if trade_price < prev_price:
        return -1
    raise ValueError("trade at mid with no price change from prev_price is unclassifiable")


def order_flow_imbalance(bid_deltas: np.ndarray, ask_deltas: np.ndarray) -> float:
    """Cont, Kukanov & Stoikov (2014) OFI, summed over the window.

    Each interval's contribution is (bid-side change) - (ask-side change):
    a bid addition or ask cancellation both push OFI positive (buy pressure);
    an ask addition or bid cancellation both push it negative.
    """
    bid_deltas = np.asarray(bid_deltas, dtype=float)
    ask_deltas = np.asarray(ask_deltas, dtype=float)
    if bid_deltas.shape != ask_deltas.shape:
        raise ValueError("bid_deltas and ask_deltas must have the same shape")
    return float(np.sum(bid_deltas - ask_deltas))


def vpin(buy_volume: np.ndarray, sell_volume: np.ndarray) -> float:
    """Easley, Lopez de Prado & O'Hara (2012) VPIN, over volume buckets.

        VPIN = mean_i(|buy_i - sell_i|) / mean_i(buy_i + sell_i)

    1.0 when every bucket is entirely one-sided (maximal order-flow
    toxicity), 0.0 when every bucket is exactly balanced.
    """
    buy_volume = np.asarray(buy_volume, dtype=float)
    sell_volume = np.asarray(sell_volume, dtype=float)
    if buy_volume.shape != sell_volume.shape:
        raise ValueError("buy_volume and sell_volume must have the same shape")
    total = buy_volume + sell_volume
    if np.sum(total) == 0:
        raise ValueError("no volume in any bucket")
    return float(np.sum(np.abs(buy_volume - sell_volume)) / np.sum(total))
