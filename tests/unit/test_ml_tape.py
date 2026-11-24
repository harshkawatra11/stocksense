from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.marketlab import tape as T
from stocksense.marketlab.feeds import FeedManager
from stocksense.marketlab.store import LabStore
from tests.unit.test_ml_structure_golden import same

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"
CAL = TradingCalendar.from_dates([date(2026, 11, 10), date(2026, 11, 11)])


def row(**kw: Any) -> dict[str, Any]:
    base = {"from_open": 0.0, "disp": 0.0, "pct": 0.0, "sector": "X", "sym": "S"}
    return {**base, **kw}


def test_disp_uses_normal_range_or_default_two():
    nse = {"name": "Foo", "open": 100.0, "prev": 100.0, "ltp": 102.0}
    r = T.stock_row("FOO", nse, None, {"range": 4.0, "pdh": 1, "pdl": 1, "pdc": 1}, "", 0, {}, False)
    assert r["from_open"] == 2.0 and r["disp"] == 0.5
    r2 = T.stock_row("FOO", nse, None, None, "", 0, {}, False)
    assert r2["disp"] == 1.0


def test_row_needs_price_prev_and_open():
    assert (
        T.stock_row(
            "F", {"name": "F", "open": None, "prev": 100.0, "ltp": 101.0}, None, None, "", 0, {}, False
        )
        is None
    )
    assert (
        T.stock_row(
            "F", {"name": "F", "open": 100.0, "prev": None, "ltp": 101.0}, None, None, "", 0, {}, False
        )
        is None
    )


def test_rank_bull_and_bear_order():
    rows = {
        "A": row(sym="A", from_open=1.0, disp=0.5),
        "B": row(sym="B", from_open=3.0, disp=1.5),
        "C": row(sym="C", from_open=-2.0, disp=-1.0),
        "D": row(sym="D", from_open=-0.5, disp=-0.25),
        "E": row(sym="E", from_open=0.0, disp=0.0),
    }
    T.rank_rows(rows)
    assert [(s, rows[s]["rank"], rows[s]["side"]) for s in "BA"] == [("B", 1, "bull"), ("A", 2, "bull")]
    assert [(s, rows[s]["rank"], rows[s]["side"]) for s in "CDE"] == [
        ("C", 1, "bear"),
        ("D", 2, "bear"),
        ("E", 3, "bear"),
    ]


def test_sector_chip_mean_and_counts():
    chips = T.sector_chips([row(pct=1.0), row(pct=-1.0), row(pct=3.0)])
    assert chips == [{"name": "X", "pct": 1.0, "up": 2, "down": 1, "n": 3}]


def test_liq_flags_use_nse_high_and_low():
    nse = {"name": "F", "open": 100.0, "prev": 100.0, "ltp": 101.0, "high": 105.0, "low": 99.0}
    dly = {"pdh": 104.0, "pdl": 98.0, "pdc": 100.0, "range": 2.0}
    assert T.stock_row("F", nse, None, dly, "", 0, {}, False)["liq"] == ["PDH"]
    nse2 = {**nse, "high": 101.0, "low": 97.0}
    assert T.stock_row("F", nse2, None, dly, "", 0, {}, False)["liq"] == ["PDL"]


def test_bars5_buckets_and_official_open():
    t0 = 1_767_000_000 // 300 * 300  # a five-minute boundary
    ts = [t0 + 60 * k for k in range(7)]
    close = [10.0, 11.0, 9.0, 10.5, 12.0, 8.0, 10.0]
    b = T.bars5(ts, close, open_px=9.5)
    assert list(b.ts) == [t0, t0 + 300]
    assert list(b.open) == [9.5, 8.0] and list(b.high) == [12.0, 10.0] and list(b.low) == [9.0, 8.0]
    assert list(b.close) == [12.0, 10.0]
    assert T.bars5([], []) is None


def test_path_keeps_only_session_minutes():
    t0 = 1_767_000_000
    base_min = (t0 + 19800) % 86400 // 60
    # build timestamps relative to 09:15 IST of that day
    day_start = t0 - ((t0 + 19800) % 86400) + 0
    open_ts = day_start + (9 * 60 + 15) * 60 - 0
    ts = [open_ts - 60, open_ts, open_ts + 600]
    assert base_min >= 0
    out = T.path(ts, [99.0, 100.0, 101.0], 100.0)
    assert out == [[0, 0.0], [10, 1.0]]


def test_next_expiry_rules():
    assert T.next_expiry("NIFTY", date(2026, 11, 12), CAL) == "2026-11-17"  # next Tuesday
    assert T.next_expiry("SENSEX", date(2026, 11, 12), CAL) == "2026-11-12"  # a Thursday today
    assert T.next_expiry("BANKNIFTY", date(2026, 11, 12), CAL) == "2026-11-24"  # last Tuesday of the month


def test_trail_is_value_fifteen_minutes_ago():
    t0 = 1_767_000_000 - ((1_767_000_000 + 19800) % 86400) + (9 * 60 + 15) * 60 - 19800 + 19800
    ts = [t0 + 60 * k for k in range(31)]
    close = [100 + k * 0.1 for k in range(31)]
    nse = {"name": "F", "open": 100.0, "prev": 100.0, "ltp": 103.0}
    got = {"ts": ts, "close": close, "ltp": 103.0, "prev": 100.0, "time": ts[-1]}
    r = T.stock_row("F", nse, got, None, "", 30, {}, False)
    assert r["trail"] == pytest.approx(round(float(close[15] / 100.0 - 1) * 100, 3))


class FakeSources:
    def __init__(self, spark: dict[str, Any]) -> None:
        self.spark = spark

    def yahoo_spark(self, symbols: list[str]) -> dict[str, Any]:
        return {s: self.spark[s] for s in symbols if s in self.spark}


def test_snapshot_matches_market_lab_feed():
    inp = json.loads((FX / "inputs_tape.json").read_text(encoding="utf-8"))
    daily_in = json.loads((FX / "inputs_1d_stock.json").read_text(encoding="utf-8"))
    from stocksense.marketlab.bars import Bars

    daily = Bars.from_arrays(*(daily_in[k] for k in ("ts", "open", "high", "low", "close", "volume")))
    clock = FixedClock(datetime.fromtimestamp(inp["now_epoch"], IST))
    store = LabStore()
    eng = T.TapeEngine(None, clock, CAL, FakeSources(inp["spark"]), store)  # type: ignore[arg-type]
    t = store.tape
    t["meta"], t["daily"], t["industry"] = inp["meta"], inp["daily"], inp["industry"]
    t["daily_idx"] = {k: daily for k in T.INDICES}
    eng.cycle(True, True)
    snap = eng.snapshot(True)
    gold = json.loads((FX / "golden_tape_snapshot.json").read_text(encoding="utf-8"))
    snap.pop("time")
    assert snap["market"] == "open"
    for ix in snap["indices"].values():
        ix["expiry"] = None  # Market Lab's holiday lookup is stubbed in the golden; ours is tested above
    same(snap, gold)


def test_snapshot_without_paths_drops_them():
    clock = FixedClock(datetime(2026, 11, 12, 11, 15, 30, tzinfo=IST))
    store = LabStore()
    eng = T.TapeEngine(None, clock, CAL, FakeSources({}), store)  # type: ignore[arg-type]
    store.tape["stocks"] = {"A": row(sym="A", path=[[0, 0.0]], rank=1, side="bull")}
    assert "path" not in eng.snapshot(False)["stocks"][0] and "path" in eng.snapshot(True)["stocks"][0]


def test_feed_manager_runs_a_loop_and_stops_fast():
    clock = FixedClock(datetime(2026, 11, 12, 11, 15, 30, tzinfo=IST))
    store = LabStore()
    mgr = FeedManager(None, clock, store)  # type: ignore[arg-type]
    hits: list[int] = []
    done = threading.Event()

    def step() -> float:
        hits.append(1)
        done.set()
        return 30.0

    mgr.add("tape", step, 4.0)
    mgr.start()
    assert done.wait(2.0)
    mgr.stop()
    assert not mgr.threads[0].is_alive() and hits == [1]
    assert store.snapshot_status()["feeds"]["tape"]["last_ok"] is not None


def test_feed_loop_survives_an_exception_and_counts_it():
    clock = FixedClock(datetime(2026, 11, 12, 11, 15, 30, tzinfo=IST))
    store = LabStore()
    mgr = FeedManager(None, clock, store)  # type: ignore[arg-type]
    seen = threading.Event()

    def step() -> float:
        seen.set()
        raise RuntimeError("boom")

    mgr.add("bad", step, 4.0)
    mgr.start()
    assert seen.wait(2.0)
    mgr.stop()
    assert store.feeds["bad"].errors >= 1 and store.feeds["bad"].last_err.startswith("RuntimeError")
