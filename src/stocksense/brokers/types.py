from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Funds:
    """Account funds in rupees."""

    available_cash: float
    net: float
    utilised: float


@dataclass(frozen=True)
class Holding:
    symbol: str
    qty: float
    avg_price: float
    ltp: float


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    ltp: float
    product: str


@dataclass(frozen=True)
class Quote:
    """One market quote; prices in rupees, `ts` is the exchange feed time (IST)."""

    symbol: str
    ts: datetime
    ltp: float | None
    open: float | None
    high: float | None
    low: float | None
    prev_close: float | None
    volume: float | None
    bid: float | None
    ask: float | None
    bid_qty: float | None
    ask_qty: float | None
    upper_circuit: float | None
    lower_circuit: float | None
    source: str


def to_float(v: object) -> float | None:
    """Float of a number or numeric string; None for null, '' and non-numbers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
