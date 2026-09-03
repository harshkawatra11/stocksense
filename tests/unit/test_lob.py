"""A hand-built book where a known sequence produces a known fill."""

from __future__ import annotations

from stocksense.microstructure.lob import OrderBook, Side


def test_market_buy_fills_resting_sells_in_price_time_priority():
    book = OrderBook()
    book.add_limit(order_id="S1", side=Side.SELL, price=100.0, qty=10)
    book.add_limit(order_id="S2", side=Side.SELL, price=100.0, qty=5)

    fills = book.market_order(order_id="B1", side=Side.BUY, qty=12)

    assert [(f.maker_id, f.taker_id, f.price, f.qty) for f in fills] == [
        ("S1", "B1", 100.0, 10.0),
        ("S2", "B1", 100.0, 2.0),
    ]


def test_market_order_climbs_price_levels_best_first():
    book = OrderBook()
    book.add_limit(order_id="S1", side=Side.SELL, price=101.0, qty=5)
    book.add_limit(order_id="S2", side=Side.SELL, price=100.0, qty=5)

    fills = book.market_order(order_id="B1", side=Side.BUY, qty=6)

    assert [(f.maker_id, f.price, f.qty) for f in fills] == [
        ("S2", 100.0, 5.0),
        ("S1", 101.0, 1.0),
    ]


def test_cancelled_order_does_not_fill():
    book = OrderBook()
    book.add_limit(order_id="S1", side=Side.SELL, price=100.0, qty=10)
    book.cancel("S1")

    fills = book.market_order(order_id="B1", side=Side.BUY, qty=10)

    assert fills == []


def test_cancel_unknown_order_returns_false():
    book = OrderBook()
    assert book.cancel("nope") is False


def test_modify_shrinking_qty_keeps_time_priority():
    book = OrderBook()
    book.add_limit(order_id="S1", side=Side.SELL, price=100.0, qty=10)
    book.add_limit(order_id="S2", side=Side.SELL, price=100.0, qty=5)
    book.modify("S1", new_qty=3)

    fills = book.market_order(order_id="B1", side=Side.BUY, qty=4)

    assert [(f.maker_id, f.qty) for f in fills] == [("S1", 3.0), ("S2", 1.0)]


def test_modify_growing_qty_loses_time_priority():
    book = OrderBook()
    book.add_limit(order_id="S1", side=Side.SELL, price=100.0, qty=5)
    book.add_limit(order_id="S2", side=Side.SELL, price=100.0, qty=5)
    book.modify("S1", new_qty=8)

    fills = book.market_order(order_id="B1", side=Side.BUY, qty=6)

    assert [(f.maker_id, f.qty) for f in fills] == [("S2", 5.0), ("S1", 1.0)]


def test_depth_aggregates_by_price_level_best_first():
    book = OrderBook()
    book.add_limit(order_id="B1", side=Side.BUY, price=99.0, qty=5)
    book.add_limit(order_id="B2", side=Side.BUY, price=100.0, qty=3)
    book.add_limit(order_id="S1", side=Side.SELL, price=101.0, qty=4)
    book.add_limit(order_id="S2", side=Side.SELL, price=101.0, qty=2)
    book.add_limit(order_id="S3", side=Side.SELL, price=102.0, qty=1)

    snapshot = book.depth(levels=2)

    assert snapshot.bids == [(100.0, 3.0), (99.0, 5.0)]
    assert snapshot.asks == [(101.0, 6.0), (102.0, 1.0)]


def test_queue_position_is_qty_ahead_at_same_price():
    book = OrderBook()
    book.add_limit(order_id="S1", side=Side.SELL, price=100.0, qty=10)
    book.add_limit(order_id="S2", side=Side.SELL, price=100.0, qty=5)
    book.add_limit(order_id="S3", side=Side.SELL, price=100.0, qty=7)

    assert book.queue_position("S1") == 0.0
    assert book.queue_position("S2") == 10.0
    assert book.queue_position("S3") == 15.0
