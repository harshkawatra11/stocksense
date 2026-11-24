from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.marketlab.sources import SourceError
from tests.unit.test_ml_sectors import GOLD, INP, build_engine


class DownSources:
    """yahoo_chart fails; everything else the engine needs is served by the fake news and cached state."""

    def nse_symbol_data(self, sym):
        raise SourceError("nse", "down")

    def yahoo_chart(self, ysym, interval, rng):
        raise SourceError("yahoo", "down")


@pytest.fixture
def client(tmp_data_root, tmp_path):
    eng = build_engine(tmp_data_root)
    eng.src = DownSources()
    app = create_app(
        tmp_data_root, start_feeds=False, clock=eng.clock, store=eng.store, sector=eng, web_out=tmp_path / "n"
    )
    with TestClient(app) as c:
        yield c


def test_horizon_route(client, web_fixture):
    r = client.get("/api/horizon?range=today")
    assert r.status_code == 200
    web_fixture("horizon", r.json())
    assert set(r.json()) == {
        "range",
        "nifty",
        "sectors",
        "groups",
        "climb",
        "themes",
        "news",
        "chart",
        "snaps",
    }
    assert r.json()["sectors"][1]["name"] == GOLD["horizon_today"]["sectors"][1]["name"]
    assert client.get("/api/horizon?range=y1").status_code == 400


def test_sector_route(client, web_fixture):
    r = client.get("/api/sector?name=Energy&range=today")
    assert r.status_code == 200
    web_fixture("sector", r.json())
    assert r.json()["name"] == "Energy" and r.json()["n"] == 15
    assert client.get("/api/sector?name=Nope").status_code == 404


def test_stock_route(client, web_fixture):
    r = client.get("/api/stock?sym=eng03")
    assert r.status_code == 200
    web_fixture("stock", r.json())
    assert r.json()["sym"] == "ENG03" and r.json()["sector"]["name"] == "Energy"


def test_search_route(client, web_fixture):
    r = client.get("/api/search?q=EN")
    assert r.status_code == 200
    web_fixture("search", r.json())
    assert r.json()[0] == {"type": "sector", "name": "Energy"}


def test_tick_route_caps_at_150_symbols(client, web_fixture):
    syms = ",".join(f"S{i}" for i in range(400))
    r = client.get(f"/api/tick?syms={syms}")
    assert r.status_code == 200
    web_fixture("tick", r.json())
    assert len(r.json()["stocks"]) == 150
    assert r.json()["market"] == "open" and set(r.json()) == {"stocks", "sectors", "nifty", "time", "market"}


def test_chart_route_errors(client):
    assert client.get("/api/chart?sym=TCS&tf=3m").status_code == 400
    assert client.get("/api/chart?sym=TCS&tf=5m").status_code == 502


def test_no_sector_engine_gives_503(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=tmp_path / "n")
    with TestClient(app) as c:
        for path in ("/api/horizon", "/api/sector", "/api/stock?sym=TCS", "/api/search?q=a", "/api/tick"):
            assert c.get(path).status_code == 503


def test_status_reports_classification_and_history_progress(client):
    j = client.get("/api/status").json()
    assert j["classified"] == len(INP["cls"]) and j["history"] > 0
    assert {"classified", "history", "progress"} <= set(j)
