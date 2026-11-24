from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.strategies import base, scan
from tests.unit.test_explore_scan import ASOF_IDX, sessions, write_panel


@pytest.fixture
def client(tmp_data_root, tmp_path, monkeypatch):
    write_panel(tmp_data_root)
    days = sessions()
    monkeypatch.setattr(
        TradingCalendar, "from_store", classmethod(lambda cls, settings, clock: cls.from_dates(days))
    )
    clock = FixedClock(
        datetime.combine(days[-1] + timedelta(days=3), datetime.min.time(), IST).replace(hour=10)
    )
    app = create_app(tmp_data_root, start_feeds=False, clock=clock, web_out=tmp_path / "none")
    with TestClient(app) as c:
        yield c


def test_trades_shape_for_a_stock(client, web_fixture):
    r = client.get("/api/lab/trades?sym=s00&tf=1d&strat=20-day breakout&exit=trail3")
    assert r.status_code == 200
    j = r.json()
    web_fixture("lab_trades", j)
    assert j["exploratory"] is True and j["strat"] == "ml_swing_breakout20" and j["gated"] is None
    assert j["costs_model"].startswith("stocksense.costs") and j["exit"] == "trail3"
    assert j["trades"] and {"t_in", "p_in", "stop", "target", "status"} <= set(j["trades"][0])
    by_key = client.get("/api/lab/trades?sym=S00&strat=ml_swing_breakout20&exit=trail3").json()
    assert by_key == j


def test_trades_defaults_to_the_candidate(client):
    j = client.get("/api/lab/trades?sym=S01").json()
    assert j["strat"] == "ml_swing_vcp" and j["exit"] == "rr3_40" and j["candidate"] is True


def test_index_swing_note_and_errors(client):
    j = client.get("/api/lab/trades?sym=NIFTY&tf=1d").json()
    assert j["trades"] == [] and "Swing rules are for stocks" in j["note"]
    r = client.get("/api/lab/trades?sym=S00&strat=nope")
    assert r.status_code == 400 and r.json() == {"error": "unknown strategy"}
    r = client.get("/api/lab/trades?sym=S00&strat=20-day breakout&exit=bogus")
    assert r.status_code == 400 and r.json() == {"error": "unknown exit"}
    assert client.get("/api/lab/trades?sym=S00&tf=3m").status_code == 400
    assert client.get("/api/lab/trades?sym=S00&tf=60m").json()["trades"] == []


def test_matrix_rows_are_the_sum_of_allowed_exits(client, web_fixture):
    r = client.get("/api/lab/matrix?sym=S02&tf=1d")
    assert r.status_code == 200
    j = r.json()
    web_fixture("lab_matrix", j)
    want = sum(len(s.exits) for s in base.load_library().values() if s.timeframe == "1d" and s.signal)
    assert len(j["rows"]) == want == j["n_tests"] and j["expected_max_sharpe_null"] > 0
    assert {
        "strat",
        "name",
        "tier",
        "source",
        "exit",
        "n",
        "win",
        "avgR",
        "totR",
        "pct",
        "is_avgR",
        "oos_avgR",
        "gated",
    } == set(j["rows"][0])
    assert client.get("/api/lab/matrix").status_code == 400


def test_strategies_list(client, web_fixture):
    j = client.get("/api/lab/strategies").json()
    web_fixture("lab_strategies", j)
    assert len(j) == len(base.load_library())
    assert {
        "key",
        "name",
        "source",
        "tier",
        "timeframe",
        "kind",
        "direction",
        "exits",
        "description",
        "reference",
        "gated",
    } == set(j[0])


def test_swing_summary_before_and_after_a_scan(client, tmp_data_root, web_fixture):
    assert "error" in client.get("/api/lab/swing").json()
    scan.run_scan(tmp_data_root, sessions()[ASOF_IDX], workers=1)
    j = client.get("/api/lab/swing").json()
    web_fixture("lab_swing", j)
    assert {"strategies", "profile", "best_stocks", "candidate", "live", "list", "exits"} <= set(j)
    assert j["candidate"]["exploratory"] is True


def test_intraday_summary_and_journal(client, web_fixture):
    j = client.get("/api/lab/summary").json()
    web_fixture("lab_summary", j)
    names = [s.name for s in base.load_library().values() if s.timeframe == "5m"]
    assert j["list"] == names and set(j["strategies"]) == set(names)
    jr = client.get("/api/lab/journal").json()
    web_fixture("lab_journal", jr)
    assert jr == {"trades": [], "paper": True}


def test_no_write_methods_on_lab_routes(client):
    for path in ("trades", "swing", "summary", "journal", "matrix", "strategies"):
        for m in ("post", "put", "patch", "delete"):
            assert getattr(client, m)(f"/api/lab/{path}").status_code == 405
