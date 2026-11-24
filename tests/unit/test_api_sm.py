from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.sources import SourceError
from stocksense.marketlab.store import LabStore
from stocksense.marketlab.tape import INDICES, TapeEngine

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"


class FakeSources:
    def __init__(self, spark):
        self.spark = spark

    def yahoo_spark(self, symbols):
        return {s: self.spark[s] for s in symbols if s in self.spark}

    def yahoo_chart(self, ysym, interval, rng):
        raise SourceError("yahoo", "down")


@pytest.fixture
def client(tmp_data_root, tmp_path):
    inp = json.loads((FX / "inputs_tape.json").read_text(encoding="utf-8"))
    d = json.loads((FX / "inputs_1d_stock.json").read_text(encoding="utf-8"))
    daily = Bars.from_arrays(*(d[k] for k in ("ts", "open", "high", "low", "close", "volume")))
    clock = FixedClock(datetime.fromtimestamp(inp["now_epoch"], IST))
    store = LabStore()
    cal = TradingCalendar.from_dates([date(2026, 11, 10), date(2026, 11, 11)])
    eng = TapeEngine(tmp_data_root, clock, cal, FakeSources(inp["spark"]), store)
    t = store.tape
    t["meta"], t["daily"], t["industry"] = inp["meta"], inp["daily"], inp["industry"]
    t["daily_idx"] = {k: daily for k in INDICES}
    eng.cycle(True, True)
    app = create_app(
        tmp_data_root, start_feeds=False, clock=clock, store=store, tape=eng, web_out=tmp_path / "n"
    )
    with TestClient(app) as c:
        yield c


def test_live_key_tree(client, web_fixture):
    r = client.get("/api/sm/live")
    assert r.status_code == 200
    j = r.json()
    web_fixture("sm_live", j)
    assert set(j) == {"time", "market", "errors", "indices", "stocks", "sectors", "bull", "bear"}
    assert j["market"] == "open" and set(j["indices"]) == {"NIFTY", "BANKNIFTY", "SENSEX"}
    assert j["bull"] and j["bear"]
    ranks = [s["rank"] for s in j["stocks"] if s["side"] == "bull"]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))
    assert all("path" in s for s in j["stocks"])
    needed = {
        "sym",
        "name",
        "sector",
        "ltp",
        "pct",
        "from_open",
        "disp",
        "value_cr",
        "ffmc",
        "trail",
        "rank",
        "side",
    }
    assert needed <= set(j["stocks"][0])


def test_live_without_paths(client):
    j = client.get("/api/sm/live?paths=0").json()
    assert all("path" not in s for s in j["stocks"])


def test_chart_bad_timeframe_and_upstream_failure(client):
    assert client.get("/api/sm/chart?sym=TCS&tf=3m").status_code == 400
    assert client.get("/api/sm/chart?sym=TCS&tf=5m").status_code == 502
    assert client.get("/api/sm/quote?sym=TCS").status_code == 502


def test_no_tape_gives_503(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=tmp_path / "n")
    with TestClient(app) as c:
        assert c.get("/api/sm/live").status_code == 503
