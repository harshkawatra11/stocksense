from __future__ import annotations

from datetime import date, datetime, time

import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.flow.preopen import PreopenEngine, board_rows, follow_through
from stocksense.marketlab.store import LabStore

DAY = date(2026, 11, 12)
CAL = TradingCalendar.from_dates([date(2026, 11, 10), date(2026, 11, 11)])


def _payload(rows):
    return {
        "data": [
            {
                "metadata": {"symbol": s, "previousClose": prev, "iep": iep},
                "detail": {
                    "preOpenMarket": {"totalBuyQuantity": b, "totalSellQuantity": q, "finalPrice": fin}
                },
            }
            for s, prev, iep, b, q, fin in rows
        ]
    }


class FakeSources:
    def __init__(self, clock, final_from: time | None):
        self.clock, self.final_from, self.calls = clock, final_from, []

    def nse_preopen(self, key):
        now = self.clock.now()
        self.calls.append((key, now))
        fin = 100.0 if self.final_from and now.time() >= self.final_from else 0
        if key == "NIFTY":
            return _payload([("AAA", 100.0, 101.0, 300, 100, fin)])
        return _payload([("AAA", 100.0, 101.0, 300, 100, fin), ("BBB", 50.0, 49.0, 100, 300, fin)])


def _run(final_from):
    clock = SteppingClock(datetime(2026, 11, 12, 8, 59, tzinfo=IST))
    src, store, caps = FakeSources(clock, final_from), LabStore(), []
    eng = PreopenEngine(clock, CAL, src, store, capture=lambda: caps.append(clock.now()) or 5)
    end = datetime(2026, 11, 12, 9, 16, tzinfo=IST)
    while clock.now() < end:
        clock.sleep(eng.step())
    return src, store, caps


def test_imbalance_and_gap_hand_numbers():
    df = pl.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "prev_close": [100.0, 100.0, None],
            "indicative_price": [102.0, 99.0, 5.0],
            "buy_qty": [300.0, 0.0, 10.0],
            "sell_qty": [100.0, 0.0, 10.0],
        }
    )
    rows = {r["symbol"]: r for r in board_rows(df)}
    assert rows["A"]["imb"] == pytest.approx(0.5)
    assert rows["A"]["iep_gap_pct"] == pytest.approx(2.0)
    assert rows["A"]["total_qty"] == 400
    assert rows["B"]["imb"] is None and rows["B"]["iep_gap_pct"] == pytest.approx(-1.0)
    assert rows["C"]["iep_gap_pct"] is None


def test_loop_calls_only_inside_window_and_captures_once_when_final_appears():
    src, store, caps = _run(time(9, 8))
    times = [t.time() for _, t in src.calls]
    assert times and min(times) == time(9, 0, 30) and max(times) == time(9, 14, 30)
    assert caps == [datetime(2026, 11, 12, 9, 8, tzinfo=IST)]
    pre = store.flow["preopen"]
    assert pre["observed"]["first_final"].startswith("09:08:00") and pre["observed"]["captured_at"]
    assert pre["rows"]["AAA"]["imb"] == pytest.approx(0.5)


def test_capture_falls_back_to_0912_30_once_when_no_final_price():
    _, store, caps = _run(None)
    assert caps == [datetime(2026, 11, 12, 9, 12, 30, tzinfo=IST)]
    assert store.flow["preopen"]["observed"]["first_final"] is None


def test_non_session_day_makes_no_calls():
    clock = SteppingClock(datetime(2026, 11, 14, 9, 5, tzinfo=IST))  # Saturday
    src = FakeSources(clock, None)
    eng = PreopenEngine(clock, CAL, src, LabStore(), capture=lambda: 0)
    eng.step()
    assert src.calls == []


def test_follow_through_sign_at_0930():
    rows = [
        {"symbol": "A", "iep_gap_pct": 1.0},
        {"symbol": "B", "iep_gap_pct": -1.0},
        {"symbol": "C", "iep_gap_pct": 0.5},
        {"symbol": "D", "iep_gap_pct": 0.5},
    ]
    tape = {
        "A": {"path": [[14, 0.2], [15, 0.4]]},
        "B": {"path": [[15, 0.3]]},
        "C": {"path": [[3, 0.3]]},  # nothing near 09:30
    }
    out = follow_through(rows, tape)
    assert out == {"A": True, "B": False, "C": None, "D": None}


def test_payload_phases_and_breadth():
    clock = FixedClock(datetime(2026, 11, 12, 9, 40, tzinfo=IST))
    store = LabStore()
    eng = PreopenEngine(clock, CAL, FakeSources(clock, None), store, capture=lambda: 0)
    assert eng.payload()["phase"] == "after" and eng.payload()["rows"] == []
    store.flow["preopen"] = {
        "day": DAY.isoformat(),
        "rows": {
            "A": {"symbol": "A", "iep": 101.0, "iep_gap_pct": 1.0, "imb": 0.5, "total_qty": 400},
            "B": {"symbol": "B", "iep": 49.0, "iep_gap_pct": -2.0, "imb": -0.5, "total_qty": 400},
        },
        "nifty": {"iep": 24000.0, "gap_pct": 0.4},
        "observed": {},
    }
    store.tape["stocks"] = {"A": {"path": [[15, 0.4]]}, "B": {"path": [[15, 0.4]]}}
    p = eng.payload()
    assert p["descriptive"] is True and p["nifty"] == {"iep": 24000.0, "gap_pct": 0.4}
    assert {r["symbol"]: r["follow_through"] for r in p["rows"]} == {"A": True, "B": False}
    assert p["breadth"] == {"adv": 1, "dec": 1}
    early = PreopenEngine(
        FixedClock(datetime(2026, 11, 12, 9, 5, tzinfo=IST)), CAL, FakeSources(clock, None), store
    )
    assert early.payload()["phase"] == "live" and "follow_through" not in early.payload()["rows"][0]
