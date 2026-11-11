from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

from stocksense.core.config import Costs
from stocksense.costs.india import CostBreakdown, trade_cost
from stocksense.costs.slippage import SlippageTable, slip_frac

TICK = 0.05
_BANDS = (0.02, 0.05, 0.10, 0.20)
_LOCK_TOL = 0.0025


@dataclass(frozen=True)
class DayBar:
    """One session's bar of one security; prev_base = previous close x gap_factor; returns are fractions."""

    open: float
    high: float
    low: float
    close: float
    prev_base: float
    ret_co: float
    ret: float
    flag_one_price: bool = False
    flag_locked_up: bool = False
    flag_locked_dn: bool = False


@dataclass(frozen=True)
class Order:
    """An order for a later session; `when` is the entry point (open or close of trade_date)."""

    sid: int
    side: Literal["BUY", "SELL"]
    product: Literal["delivery", "intraday"]
    when: Literal["open", "close"]
    qty: int
    trade_date: date | None = None
    priority: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    """An executed order: price in rupees, slip as a fraction, costs in rupees."""

    order: Order
    qty: int
    price: float
    ref_price: float
    slip: float
    cost: CostBreakdown

    @property
    def value(self) -> float:
        return self.qty * self.price


@dataclass(frozen=True)
class Reject:
    """An order that could not trade (reason is a short code)."""

    order: Order
    reason: str


@dataclass(frozen=True)
class FillContext:
    """Per-security, per-day inputs to the fill rules."""

    turnover_rank: int
    vol_21: float
    med_turn_20: float
    slippage: SlippageTable
    costs: Costs
    capital: float
    max_adv_participation: float = 0.01
    first_sell_of_scrip_today: bool = True


def _near_band(x: float) -> bool:
    return min(abs(x - b) for b in _BANDS) <= _LOCK_TOL


def _tick_round(price: float, side: str) -> float:
    ticks = round(price / TICK, 6)
    return (math.ceil(ticks) if side == "BUY" else math.floor(ticks)) * TICK


def try_fill(order: Order, bar: DayBar | None, ctx: FillContext) -> Fill | Reject:
    """Fill an order at the session's open or close under the T5.3 rules, or reject it with a reason."""
    if order.when not in ("open", "close"):
        raise ValueError(f"daily fills need when 'open' or 'close', got {order.when!r}")
    if bar is None:
        return Reject(order, "no_trade")
    if bar.flag_one_price:
        return Reject(order, "one_price")
    buy = order.side == "BUY"
    if order.when == "open":
        if buy and bar.open == bar.high and _near_band(bar.ret_co):
            return Reject(order, "locked_up_open")
        if not buy and bar.open == bar.low and _near_band(-bar.ret_co):
            return Reject(order, "locked_dn_open")
    else:
        if buy and bar.flag_locked_up:
            return Reject(order, "locked_up_close")
        if not buy and bar.flag_locked_dn:
            return Reject(order, "locked_dn_close")
    ref = bar.open if order.when == "open" else bar.close
    max_value = ctx.max_adv_participation * ctx.med_turn_20
    qty = min(order.qty, math.floor(max_value / ref))
    if qty <= 0:
        return Reject(order, "too_small")
    slip = slip_frac(
        ctx.turnover_rank,
        ctx.vol_21,
        qty * ref,
        ctx.med_turn_20,
        order.when,
        ctx.slippage,
        as_of=order.trade_date,
    )
    raw = ref * (1 + slip) if buy else ref * (1 - slip)
    price = _tick_round(raw, order.side)
    cost = trade_cost(order.side, order.product, qty * price, ctx.costs, ctx.first_sell_of_scrip_today)
    return Fill(order, qty, price, ref, slip, cost)
