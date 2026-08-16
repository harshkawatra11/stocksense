"""Desktop control-room API tests, run against a real seeded store with
FastAPI's TestClient (httpx under the hood, no Electron in the loop --
exactly the plan's point: the API is independently testable). Verifies
each endpoint returns correct JSON from real data, and that pandas
NaN/Timestamp/numpy-scalar values never leak through unserialized."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stocksense.data.store import Store


@pytest.fixture()
def client_with_store(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(db_path))

    store = Store(db_path)
    positions = pd.DataFrame([
        {
            "position_id": "p1", "symbol": "AAA", "segment": "equity_delivery",
            "open_date": "2024-01-01", "open_time": "09:20", "close_date": "2024-01-05",
            "close_time": "15:00", "quantity": 10, "entry_price": 100.0, "exit_price": 110.0,
            "gross_pnl": 100.0, "charges": 10.0, "net_pnl": 90.0, "holding_seconds": 20000,
            "is_intraday": False, "mae": None, "mfe": None,
        },
        {
            "position_id": "p2", "symbol": "BBB", "segment": "equity_intraday",
            "open_date": "2024-01-02", "open_time": "09:20", "close_date": "2024-01-02",
            "close_time": "15:00", "quantity": 5, "entry_price": 50.0, "exit_price": 45.0,
            "gross_pnl": -25.0, "charges": 5.0, "net_pnl": -30.0, "holding_seconds": 20000,
            "is_intraday": True, "mae": None, "mfe": None,
        },
    ])
    store.write_positions(positions)

    diag = pd.DataFrame([
        {"run_id": "r1", "as_of": "2024-01-06", "metric_name": "cost_drag", "metric_value": 0.15,
         "metric_unit": "fraction", "severity": "notable", "cohort": "all", "detail_json": "{}"},
        {"run_id": "r1", "as_of": "2024-01-06", "metric_name": "expectancy", "metric_value": 30.0,
         "metric_unit": "inr_per_trade", "severity": "ok", "cohort": "all", "detail_json": "{}"},
    ])
    store.write_diagnostics(diag)

    cf = pd.DataFrame([
        {"run_id": "r1", "scenario_name": "remove_worst_trade", "actual_pnl": 60.0,
         "scenario_pnl": 90.0, "delta_pnl": 30.0, "n_trades_affected": 1, "detail_json": "{}"},
    ])
    store.write_counterfactuals(cf)

    store.start_job_run("j1", "reconcile_grade", datetime.now(timezone.utc))
    store.finish_job_run("j1", "completed", datetime.now(timezone.utc))

    store.insert_agent_run({
        "agent_run_id": "a1", "job_run_id": None, "skill_name": "claude-report-writing",
        "prompt_hash": "abc", "input_json": "{}", "output_text": "some narrative",
        "model": "sonnet", "started_at": datetime.now(timezone.utc), "finished_at": datetime.now(timezone.utc),
        "status": "unverified_numbers", "error": None, "cost_estimate": None,
    })
    store.close()

    from stocksense.server.app import app
    return TestClient(app)


def test_health_returns_ok(client_with_store) -> None:
    resp = client_with_store.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_summary_computes_from_real_positions(client_with_store) -> None:
    resp = client_with_store.get("/api/summary")
    data = resp.json()
    assert data["n_positions"] == 2
    assert data["total_net_pnl"] == 60.0  # 90 + (-30)
    assert data["win_rate"] == 0.5


def test_doshas_sorted_by_severity(client_with_store) -> None:
    resp = client_with_store.get("/api/doshas")
    doshas = resp.json()["doshas"]
    assert len(doshas) == 2
    assert doshas[0]["severity"] == "notable"  # ranks above 'ok'
    assert doshas[1]["severity"] == "ok"


def test_counterfactuals_returns_seeded_scenario(client_with_store) -> None:
    resp = client_with_store.get("/api/counterfactuals")
    cfs = resp.json()["counterfactuals"]
    assert len(cfs) == 1
    assert cfs[0]["scenario_name"] == "remove_worst_trade"
    assert cfs[0]["delta_pnl"] == 30.0


def test_positions_paginated(client_with_store) -> None:
    resp = client_with_store.get("/api/positions?limit=1&offset=0")
    data = resp.json()
    assert len(data["positions"]) == 1
    assert data["total"] == 2


def test_harness_returns_latest_per_job(client_with_store) -> None:
    resp = client_with_store.get("/api/harness")
    jobs = resp.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["job_name"] == "reconcile_grade"
    assert jobs[0]["status"] == "completed"


def test_registry_empty_when_no_models(client_with_store) -> None:
    resp = client_with_store.get("/api/registry")
    assert resp.json()["models"] == []


def test_agent_runs_flags_unverified_numbers(client_with_store) -> None:
    resp = client_with_store.get("/api/agent-runs")
    data = resp.json()
    assert data["n_unverified"] == 1
    assert len(data["agent_runs"]) == 1
    assert "input_json" not in data["agent_runs"][0]  # stripped, not leaked to the dashboard


def test_no_nan_or_unserializable_values_leak_through(client_with_store) -> None:
    """Every endpoint must return valid JSON with no NaN literal (which
    Python's json module would happily emit but is invalid per the JSON
    spec, and which breaks strict JSON parsers in a browser)."""
    for path in ["/api/health", "/api/summary", "/api/doshas", "/api/counterfactuals",
                 "/api/positions", "/api/harness", "/api/registry", "/api/agent-runs"]:
        resp = client_with_store.get(path)
        assert resp.status_code == 200
        assert "NaN" not in resp.text
        assert "Infinity" not in resp.text
