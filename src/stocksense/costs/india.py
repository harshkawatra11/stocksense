from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stocksense.core.config import Costs, ProductCosts

Side = Literal["BUY", "SELL"]
Product = Literal["delivery", "intraday"]


@dataclass(frozen=True)
class CostBreakdown:
    """Rupee charges of one executed order."""

    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    dp: float
    total: float


def _product(sched: Costs, product: Product) -> ProductCosts:
    if product not in ("delivery", "intraday"):
        raise ValueError(f"unknown product {product!r}")
    return sched.delivery if product == "delivery" else sched.intraday


def _brokerage(p: ProductCosts, value_inr: float) -> float:
    return max(p.brokerage_min_inr, min(p.brokerage_cap_inr, p.brokerage_rate * value_inr))


def trade_cost(
    side: Side,
    product: Product,
    value_inr: float,
    sched: Costs,
    first_sell_of_scrip_today: bool = True,
) -> CostBreakdown:
    """Rupee cost of one order of `value_inr` (brokerage, STT, exchange, SEBI, stamp, GST, DP)."""
    if side not in ("BUY", "SELL"):
        raise ValueError(f"unknown side {side!r}")
    p = _product(sched, product)
    brokerage = _brokerage(p, value_inr)
    stt = value_inr * (p.stt_buy if side == "BUY" else p.stt_sell)
    stamp = value_inr * p.stamp_buy if side == "BUY" else 0.0
    exchange = value_inr * p.exchange
    sebi = value_inr * p.sebi
    dp = p.dp_per_scrip_sell_inr if (side == "SELL" and first_sell_of_scrip_today) else 0.0
    gst = p.gst * (brokerage + exchange + sebi + dp)
    total = brokerage + stt + exchange + sebi + stamp + gst + dp
    return CostBreakdown(brokerage, stt, exchange, sebi, stamp, gst, dp, total)


def round_trip_cost_frac(product: Product, value_inr: float, sched: Costs) -> float:
    """(buy cost + sell cost) / value as a fraction (0.01 = 1%) for a round trip at one value."""
    buy = trade_cost("BUY", product, value_inr, sched).total
    sell = trade_cost("SELL", product, value_inr, sched).total
    return (buy + sell) / value_inr


def min_position_for_fixed_cost(product: Product, sched: Costs, max_fixed_bps: float) -> float:
    """Smallest position in rupees (1,000 steps from 5,000 to 5,00,000) whose fixed costs fit the cap."""
    p = _product(sched, product)
    for v in range(5_000, 500_001, 1_000):
        fixed = (2 * _brokerage(p, v) + p.dp_per_scrip_sell_inr) * (1 + p.gst)
        if fixed / v <= max_fixed_bps / 1e4:
            return float(v)
    return float("inf")
