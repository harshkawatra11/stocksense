from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from stocksense.api.app import create_app
from stocksense.cli import app as cli_app


@pytest.fixture
def client(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=tmp_path / "noweb")
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
@pytest.mark.parametrize("path", ["/health", "/api/status", "/"])
def test_write_methods_405(client, method, path):
    assert getattr(client, method)(path).status_code == 405


def test_fallback_page(client):
    assert "run-lab.bat" in client.get("/").text


def test_static_served(tmp_data_root, fixed_clock, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text("<p>hi</p>", encoding="utf-8")
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=out)
    with TestClient(app) as c:
        assert c.get("/").text == "<p>hi</p>"


def test_api_no_store(client):
    assert client.get("/api/status").headers["cache-control"] == "no-store"


def test_status_shape(client):
    j = client.get("/api/status").json()
    assert set(j) >= {"ready", "feeds", "catchup", "time", "market"}
    assert j["market"] == "closed" and j["ready"] is False


def test_status_reports_feed_errors(tmp_data_root, fixed_clock, tmp_path):
    from stocksense.marketlab.store import LabStore

    store = LabStore()
    store.ok("tape", fixed_clock.now())
    store.fail("sector", fixed_clock.now(), ValueError("x" * 300))
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, store=store, web_out=tmp_path / "n")
    with TestClient(app) as c:
        feeds = c.get("/api/status").json()["feeds"]
    assert feeds["tape"]["errors"] == 0 and feeds["tape"]["last_ok"].startswith("2026-11-12T10:00")
    assert feeds["sector"]["errors"] == 1
    assert (
        feeds["sector"]["last_err"].startswith("ValueError: xxx") and len(feeds["sector"]["last_err"]) <= 140
    )


def test_feed_manager_stops_quickly(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=True, clock=fixed_clock, web_out=tmp_path / "n")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
    assert app.state.feeds is not None


def test_serve_help():
    r = CliRunner().invoke(cli_app, ["serve", "--help"])
    assert r.exit_code == 0 and "8765" in r.output
