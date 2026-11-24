from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.data import io
from stocksense.flow.engine import FlowEngine
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.sources import SourceError
from stocksense.marketlab.store import LabStore
from stocksense.marketlab.tape import INDICES, TapeEngine

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"
D1, D0 = date(2026, 11, 11), date(2026, 11, 10)


class FakeSources:
    def __init__(self, spark):
        self.spark = spark

    def yahoo_spark(self, symbols):
        return {s: self.spark[s] for s in symbols if s in self.spark}

    def yahoo_chart(self, ysym, interval, rng):
        raise SourceError("yahoo", "down")

    def nse_preopen(self, key):
        raise SourceError("nse", "not used")


def _f(n, rows):
    return pl.DataFrame(rows).with_columns(pl.col(c).cast(pl.Float64) for c in n)


def _seed(settings):
    sids = {"RELIANCE": 1, "TCS": 2}
    p = pl.DataFrame(
        [{"sid": sid, "date": d, "symbol": s, "series": "EQ"} for s, sid in sids.items() for d in (D0, D1)]
    )
    io.upsert(settings, "panel_daily", p)
    io.upsert(
        settings,
        "flow_delivery_daily",
        pl.DataFrame(
            {
                "date": [D1, D1],
                "sid": [1, 2],
                "deliv_pct": [60.0, 40.0],
                "deliv_z20": [2.5, 0.1],
                "deliv_value_cr": [10.0, 5.0],
                "deliv_spike": [True, False],
            }
        ),
    )
    oi = [
        ("RELIANCE", 1.0, "long buildup", 5.0),
        ("TCS", -1.0, "short buildup", 3.0),
        ("NIFTY", 0.5, "short covering", -1.0),
    ]
    io.upsert(
        settings,
        "flow_oi_daily",
        pl.DataFrame(
            {
                "date": [D1] * 3,
                "symbol": [o[0] for o in oi],
                "fut_close": [100.0] * 3,
                "fut_oi": [1000.0] * 3,
                "oi_chg_pct": [o[3] for o in oi],
                "price_chg_pct": [1.0] * 3,
                "basis_pct": [0.2] * 3,
                "rollover_pct": [10.0] * 3,
                "quad": [o[1] for o in oi],
                "quad_label": [o[2] for o in oi],
            }
        ),
    )
    part = {"date": [D0, D1]}
    for c in ("fii", "dii", "pro", "client"):
        for m in ("ratio", "ratio_chg1", "ratio_chg5", "call_net", "put_net"):
            part[f"{c}_{m}"] = [0.4, 0.45]
    part["fii_cash_net"], part["dii_cash_net"] = [-100.0, 250.0], [80.0, 120.0]
    io.upsert(settings, "flow_participants_daily", pl.DataFrame(part))
    opt_cols = [
        "spot",
        "atm_iv",
        "skew",
        "pcr_oi",
        "pcr_doi",
        "max_pain",
        "max_pain_dist_pct",
        "call_wall",
        "put_wall",
        "gex_total_cr",
        "zero_gamma",
        "zero_gamma_dist_pct",
    ]
    rows = []
    for und, spot in (("NIFTY", 24000.0), ("BANKNIFTY", 52000.0), ("RELIANCE", 2500.0)):
        for d in (D0, D1):
            rows.append(
                {"date": d, "underlying": und, "near_expiry": date(2026, 11, 17)}
                | dict.fromkeys(opt_cols, 1.0)
                | {"spot": spot, "gex_total_cr": -5.0 if und == "RELIANCE" else 12.5, "pcr_oi": 1.5}
            )
    io.upsert(settings, "flow_options_daily", pl.DataFrame(rows))
    st = [
        {
            "date": D1,
            "underlying": "NIFTY",
            "expiry": e,
            "strike": k,
            "call_oi": 100,
            "put_oi": 200,
            "call_doi": 1,
            "put_doi": -1,
            "call_iv": 0.15,
            "put_iv": 0.16,
            "gex_cr": 0.5,
        }
        for e in (date(2026, 11, 17), date(2026, 11, 24))
        for k in (23900.0, 24000.0)
    ]
    io.upsert(settings, "flow_options_strikes", pl.DataFrame(st))
    days = [D1 - timedelta(days=i) for i in range(30)]
    sec = [
        {
            "date": d,
            "sector": s,
            "flow_dir": fd,
            "value_cr": v,
            "turnover_share": v / 100,
            "n": 5,
            "up": 3,
            "down": 2,
            "deliv_spikes": 1,
        }
        for d in days
        for s, fd, v in (("IT", 0.5, 70.0 if d > D1 - timedelta(days=5) else 40.0), ("BANK", -0.1, 30.0))
    ]
    io.upsert(settings, "flow_sector_daily", pl.DataFrame(sec))
    io.upsert(
        settings,
        "flow_deals_daily",
        pl.DataFrame(
            {
                "date": [D0, D1],
                "symbol": ["TCS", "RELIANCE"],
                "buy_value_cr": [5.0, 9.0],
                "sell_value_cr": [1.0, 2.0],
                "net_value_cr": [4.0, 7.0],
                "n_deals": [1, 2],
                "top_client": ["X", "Y"],
                "flagged": [False, True],
            }
        ),
    )
    io.upsert(settings, "india_vix", pl.DataFrame({"date": [D0, D1], "close": [13.0, 14.5]}))


@pytest.fixture
def parts(tmp_data_root, tmp_path):
    inp = json.loads((FX / "inputs_tape.json").read_text(encoding="utf-8"))
    d = json.loads((FX / "inputs_1d_stock.json").read_text(encoding="utf-8"))
    daily = Bars.from_arrays(*(d[k] for k in ("ts", "open", "high", "low", "close", "volume")))
    clock = FixedClock(datetime.fromtimestamp(inp["now_epoch"], IST))
    store = LabStore()
    cal = TradingCalendar.from_dates([D0 - timedelta(days=1), D0, D1])
    src = FakeSources(inp["spark"])
    eng = TapeEngine(tmp_data_root, clock, cal, src, store)
    t = store.tape
    t["meta"], t["daily"], t["industry"] = inp["meta"], inp["daily"], inp["industry"]
    t["meta"]["RELIANCE"]["volume"] = 3_000.0
    t["daily_idx"] = {k: daily for k in INDICES}
    eng.cycle(True, True)
    _seed(tmp_data_root)
    flow = FlowEngine(tmp_data_root, clock, cal, src, store)
    store.flow["vol_curve"] = {"RELIANCE": {"x": [0.0, 375.0], "y": [0.0, 375_000.0]}}
    return tmp_data_root, clock, store, eng, flow, tmp_path


@pytest.fixture
def client(parts):
    settings, clock, store, eng, flow, tmp_path = parts
    app = create_app(
        settings, start_feeds=False, clock=clock, store=store, tape=eng, flow=flow, web_out=tmp_path / "n"
    )
    with TestClient(app) as c:
        yield c


def test_market(client, web_fixture):
    j = client.get("/api/flow/market").json()
    web_fixture("flow_market", j)
    assert j["descriptive"] is True and j["asof"] == "2026-11-11" and j["live"] is True
    assert j["fii"] == {"cash_net": 250.0, "cash_net_5d": 150.0, "idx_fut_ratio": 0.45, "ratio_chg5": 0.45}
    assert j["dii"]["cash_net"] == 120.0 and j["pro"]["cash_net"] is None
    assert j["indices"]["NIFTY"]["regime"] == "positive" and j["indices"]["NIFTY"]["spot"] == 24000.0
    assert j["vix"] == {"close": 14.5, "chg1": pytest.approx(1.5)}
    assert j["breadth"]["adv"] + j["breadth"]["dec"] > 0 and j["preopen"] is None


def test_stocks_sorted_with_cards(client, web_fixture):
    j = client.get("/api/flow/stocks?sort=disp").json()
    web_fixture("flow_stocks", j)
    rows = j["rows"]
    assert j["descriptive"] is True and len(rows) > 5
    disps = [r["disp"] for r in rows]
    assert disps == sorted(disps, reverse=True)
    rel = next(r for r in rows if r["sym"] == "RELIANCE")
    assert rel["oi_quad_label"] == "long buildup" and rel["deliv_z20_prev"] == 2.5
    assert rel["options"]["gex_sign"] == -1 and rel["rvol"] is not None
    assert {"sym", "name", "sector", "ltp", "pct", "flow_score", "components_used"} <= set(rel)
    assert client.get("/api/flow/stocks?sort=nope").status_code == 400
    by_score = client.get("/api/flow/stocks").json()["rows"]
    assert by_score[0]["flow_score"] >= by_score[-1]["flow_score"]


def test_oi_scopes(client, web_fixture):
    st = client.get("/api/flow/oi?scope=stocks").json()
    ix = client.get("/api/flow/oi?scope=index").json()
    web_fixture("flow_oi", st)
    assert {r["symbol"] for r in st["rows"]} == {"RELIANCE", "TCS"}
    assert [r["symbol"] for r in ix["rows"]] == ["NIFTY"] and ix["rows"][0]["quad_label"] == "short covering"
    assert client.get("/api/flow/oi?scope=x").status_code == 422


def test_options(client, web_fixture):
    j = client.get("/api/flow/options?und=nifty").json()
    web_fixture("flow_options", j)
    assert (
        j["und"] == "NIFTY" and j["expiry"] == "2026-11-17" and j["expiries"] == ["2026-11-17", "2026-11-24"]
    )
    assert [s["strike"] for s in j["strikes"]] == [23900.0, 24000.0]
    assert j["totals"]["gex_total_cr"] == 12.5 and len(j["history"]) == 2
    later = client.get("/api/flow/options?und=NIFTY&expiry=2026-11-24").json()
    assert later["expiry"] == "2026-11-24" and len(later["strikes"]) == 2
    assert client.get("/api/flow/options?und=NOPE").status_code == 404
    assert client.get("/api/flow/options?expiry=bad").status_code == 422


def test_sectors_ranges(client, web_fixture):
    live = client.get("/api/flow/sectors?range=today").json()
    web_fixture("flow_sectors_today", live)
    assert live["descriptive"] is True and live["rows"] and "rotation" in live["rows"][0]
    w1 = client.get("/api/flow/sectors?range=w1").json()
    web_fixture("flow_sectors_w1", w1)
    it = next(r for r in w1["rows"] if r["name"] == "IT")
    assert it["flow_dir"] == 0.5 and it["up"] == 15 and it["deliv_spikes"] == 5
    assert it["share_ratio"] == pytest.approx(0.7 / 0.4) and it["rotation"] == "in"
    bank = next(r for r in w1["rows"] if r["name"] == "BANK")
    assert bank["rotation"] == "neutral"
    assert client.get("/api/flow/sectors?range=y1").status_code == 422


def test_preopen_before_any_data(client, web_fixture):
    j = client.get("/api/flow/preopen").json()
    web_fixture("flow_preopen", j)
    assert j["descriptive"] is True and j["rows"] == [] and j["phase"] == "after" and j["nifty"] is None


def test_deals_and_participants(client, web_fixture):
    d = client.get("/api/flow/deals?days=1").json()
    web_fixture("flow_deals", d)
    assert [r["symbol"] for r in d["rows"]] == ["RELIANCE"] and d["rows"][0]["flagged"] is True
    assert len(client.get("/api/flow/deals?days=5").json()["rows"]) == 2
    p = client.get("/api/flow/participants?days=60").json()
    web_fixture("flow_participants", p)
    assert [r["date"] for r in p["rows"]] == ["2026-11-10", "2026-11-11"] and p["rows"][1][
        "fii_cash_net"
    ] == 250.0
    assert client.get("/api/flow/deals?days=0").status_code == 422


def test_sm_live_rows_gain_flow(client):
    rows = client.get("/api/sm/live").json()["stocks"]
    assert all("flow" in r for r in rows)
    rel = next(r for r in rows if r["sym"] == "RELIANCE")
    assert rel["flow"]["oi_quad_label"] == "long buildup" and rel["flow"]["descriptive"] is True


def test_no_flow_engine_gives_503_and_null_flow(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=tmp_path / "n")
    with TestClient(app) as c:
        for path in ("market", "stocks", "oi", "options", "sectors", "preopen", "deals", "participants"):
            assert c.get(f"/api/flow/{path}").status_code == 503
        assert c.post("/api/flow/market").status_code == 405


def test_engine_step_builds_curves_values_and_skips_failed_cvd(parts):
    settings, clock, store, _eng, flow, _ = parts
    assert flow.step() == 30.0  # market open at the fixture time
    assert set(store.flow["vol_curve"]) == set(store.tape["meta"])  # no stored 5m bars: all None
    assert all(v is None for v in store.flow["vol_curve"].values())
    assert flow.history.root.exists() and any(flow.history.root.glob("*.json"))
    assert store.flow["cvd"] == {}  # Yahoo down: nothing cached, loop survives
