from __future__ import annotations

import threading
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from stocksense.brokers.types import Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.data import io
from stocksense.marketlab.store import LabStore
from stocksense.orderflow.heatmap import DepthCapture, DepthRing, heatmap

T0 = datetime(2026, 11, 12, 10, 0, tzinfo=IST)  # a Thursday


def _rows(sym, ts, bid_px=100.0, qty=10.0):
    out = []
    for lvl in range(5):
        out.append({"symbol": sym, "ts": ts, "side": "bid", "level": lvl, "price": bid_px - lvl, "qty": qty})
        out.append(
            {"symbol": sym, "ts": ts, "side": "ask", "level": lvl, "price": bid_px + 1 + lvl, "qty": qty}
        )
    return out


def test_ring_keeps_two_hours_per_symbol():
    ring = DepthRing(max_age_s=7200)
    ring.add(_rows("AAA", T0))
    ring.add(_rows("AAA", T0 + timedelta(hours=1)))
    ring.add(_rows("AAA", T0 + timedelta(hours=2, minutes=1)))  # first snapshot is now older than 2 h
    ring.add(_rows("BBB", T0 + timedelta(hours=2, minutes=1)))
    a = ring.frame("AAA")
    assert a["ts"].n_unique() == 2 and a["ts"].min() == T0 + timedelta(hours=1)
    assert ring.frame("BBB").height == 10
    assert ring.frame("ZZZ").height == 0
    assert ring.symbols() == ["AAA", "BBB"]


def test_heatmap_matrix_from_three_snapshots():
    snaps = []
    for k, (q_bid, px) in enumerate([(10.0, 100.0), (30.0, 100.0), (20.0, 101.0)]):
        snaps += _rows("AAA", T0 + timedelta(seconds=30 * k), bid_px=px, qty=q_bid)
    df = pl.DataFrame(snaps).with_columns(pl.col("level").cast(pl.Int32))
    h = heatmap(df, bin=1.0, step_s=30)
    assert len(h["times"]) == 3
    assert h["prices"] == sorted(h["prices"]) and len(h["qty"]) == len(h["times"])
    assert all(len(r) == len(h["prices"]) for r in h["qty"])
    p0 = h["prices"].index(100.0)
    assert h["qty"][0][p0] == 10.0 and h["qty"][1][p0] == 30.0
    assert h["qty"][2][p0] == 20.0  # 100 is bid level 1 of the third snapshot
    assert max(max(r) for r in h["qty"]) == 30.0


def test_heatmap_takes_max_inside_a_bucket_and_empty_input():
    snaps = _rows("AAA", T0, qty=5.0) + _rows("AAA", T0 + timedelta(seconds=10), qty=9.0)
    h = heatmap(pl.DataFrame(snaps).with_columns(pl.col("level").cast(pl.Int32)), bin=1.0, step_s=30)
    assert len(h["times"]) == 1 and np.max(h["qty"]) == 9.0
    assert heatmap(pl.DataFrame(snaps).head(0), bin=1.0) == {"times": [], "prices": [], "qty": []}


class FakeAngel:
    def __init__(self):
        self.calls = []

    def quote_full(self, tokens):
        self.calls.append(list(tokens))
        return [
            Quote(
                symbol="AAA",
                ts=T0,
                ltp=100.0,
                open=None,
                high=None,
                low=None,
                prev_close=None,
                volume=None,
                bid=100.0,
                ask=101.0,
                bid_qty=10.0,
                ask_qty=10.0,
                upper_circuit=None,
                lower_circuit=None,
                source="angel",
                depth_bid=tuple((100.0 - i, 10.0 + i) for i in range(5)),
                depth_ask=tuple((101.0 + i, 20.0 + i) for i in range(5)),
            )
        ]


def _cal():
    return TradingCalendar.from_dates([T0.date() - timedelta(days=i) for i in range(1, 30)])


def _capture(tmp_data_root, clock, store=None, angel=None):
    store = store or LabStore()
    store.tape["stocks"] = {"AAA": {"side": "bull", "rank": 1}}
    return DepthCapture(
        tmp_data_root, clock, _cal(), store, angel or FakeAngel(), lambda s: "111" if s == "AAA" else None
    )


def test_loop_is_skipped_outside_market_hours(tmp_data_root):
    clock = FixedClock(datetime(2026, 11, 12, 20, 0, tzinfo=IST))
    angel = FakeAngel()
    cap = _capture(tmp_data_root, clock, angel=angel)
    assert cap.step() == 60.0
    assert angel.calls == [] and cap.ring.symbols() == []


def test_step_in_market_hours_fills_ring_and_flushes_each_minute(tmp_data_root):
    clock = FixedClock(T0)
    angel = FakeAngel()
    cap = _capture(tmp_data_root, clock, angel=angel)
    assert cap.step() == 5.0
    assert angel.calls == [["111"]] and cap.ring.frame("AAA").height == 10
    assert io.list_partitions(tmp_data_root, "depth_snaps") == []  # not yet 60 s
    cap.flush()
    df = io.read(tmp_data_root, "depth_snaps").collect()
    assert df.height == 10 and set(df["side"]) == {"bid", "ask"}
    cap.flush()  # idempotent: same keys, same rows
    assert io.read(tmp_data_root, "depth_snaps").collect().height == 10


def test_watchlist_is_capped_and_tokenless_symbols_dropped(tmp_data_root):
    store = LabStore()
    store.tape["stocks"] = {
        f"S{i:02d}": {"side": "bull" if i % 2 else "bear", "rank": i // 2 + 1} for i in range(60)
    }
    cap = DepthCapture(tmp_data_root, FixedClock(T0), _cal(), store, FakeAngel(), lambda s: f"t{s}")
    wl = cap.watchlist()
    assert len(wl) == 20 and len(set(wl)) == 20
    cap2 = DepthCapture(tmp_data_root, FixedClock(T0), _cal(), store, FakeAngel(), lambda s: None)
    assert cap2.watchlist() == []


def test_stop_event_type():
    assert isinstance(threading.Event(), threading.Event)


def test_build_returns_none_without_angel_secrets(tmp_data_root):
    from stocksense.orderflow.heatmap import build_depth_capture

    assert build_depth_capture(tmp_data_root, FixedClock(T0), LabStore()) is None
