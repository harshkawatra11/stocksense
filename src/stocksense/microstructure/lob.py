"""A price-time priority limit order book.

Replaying real depth data through this is what turns a fill assumption
("filled at mid, no slippage") into a measured one -- the previous builds'
cost model used a flat slippage constant with no queue or participation
check at all, and the account's gross-positive/net-negative gap is exactly
where that lie hides.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Fill:
    maker_id: str
    taker_id: str
    price: float
    qty: float


@dataclass
class _RestingOrder:
    order_id: str
    price: float
    qty: float
    seq: int


class OrderBook:
    """Two price-ordered ladders of time-ordered (FIFO) queues per level."""

    def __init__(self) -> None:
        self._bids: dict[float, deque[_RestingOrder]] = {}
        self._asks: dict[float, deque[_RestingOrder]] = {}
        self._locations: dict[str, tuple[Side, float]] = {}
        self._seq = itertools.count()

    def _book(self, side: Side) -> dict[float, deque[_RestingOrder]]:
        return self._bids if side is Side.BUY else self._asks

    def add_limit(self, order_id: str, side: Side, price: float, qty: float) -> list[Fill]:
        if qty <= 0:
            raise ValueError("qty must be positive")
        resting = _RestingOrder(order_id, price, qty, next(self._seq))
        book = self._book(side)
        book.setdefault(price, deque()).append(resting)
        self._locations[order_id] = (side, price)
        return []

    def cancel(self, order_id: str) -> bool:
        loc = self._locations.get(order_id)
        if loc is None:
            return False
        side, price = loc
        queue = self._book(side)[price]
        for resting in queue:
            if resting.order_id == order_id:
                queue.remove(resting)
                break
        if not queue:
            del self._book(side)[price]
        del self._locations[order_id]
        return True

    def modify(self, order_id: str, new_qty: float) -> bool:
        if new_qty <= 0:
            raise ValueError("new_qty must be positive")
        loc = self._locations.get(order_id)
        if loc is None:
            return False
        side, price = loc
        queue = self._book(side)[price]
        resting = next(r for r in queue if r.order_id == order_id)

        if new_qty <= resting.qty:
            resting.qty = new_qty
        else:
            queue.remove(resting)
            resting.qty = new_qty
            resting.seq = next(self._seq)
            queue.append(resting)
        return True

    def market_order(self, order_id: str, side: Side, qty: float) -> list[Fill]:
        if qty <= 0:
            raise ValueError("qty must be positive")
        opposite_side = Side.SELL if side is Side.BUY else Side.BUY
        opposite = self._book(opposite_side)
        remaining = qty
        fills: list[Fill] = []

        for price in sorted(opposite, reverse=(side is Side.SELL)):
            queue = opposite[price]
            while remaining > 0 and queue:
                maker = queue[0]
                traded = min(remaining, maker.qty)
                fills.append(Fill(maker.order_id, order_id, price, traded))
                maker.qty -= traded
                remaining -= traded
                if maker.qty == 0:
                    queue.popleft()
                    del self._locations[maker.order_id]
            if not queue:
                del opposite[price]
            if remaining == 0:
                break

        return fills

    def depth(self, levels: int = 5) -> "L2Snapshot":
        def side_levels(book: dict[float, deque[_RestingOrder]], reverse: bool) -> list[tuple[float, float]]:
            prices = sorted(book, reverse=reverse)[:levels]
            return [(price, sum(o.qty for o in book[price])) for price in prices]

        return L2Snapshot(
            bids=side_levels(self._bids, reverse=True),
            asks=side_levels(self._asks, reverse=False),
        )

    def queue_position(self, order_id: str) -> float:
        side, price = self._locations[order_id]
        queue = self._book(side)[price]
        ahead = 0.0
        for resting in queue:
            if resting.order_id == order_id:
                break
            ahead += resting.qty
        return ahead


@dataclass(frozen=True)
class L2Snapshot:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
