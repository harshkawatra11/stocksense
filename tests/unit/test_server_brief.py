"""Phase G5: /api/brief tests. Checks the three status branches (no
live model, live model with no predictions yet, the real happy path),
that weights/picks are correctly derived from the live model's latest
predictions, that min_capital_for_full_positions is computed from real
prices/weights rather than assumed, and -- the design invariant this
whole phase turns on -- that NO capital figure is ever read from or
written to the request/response on the server side."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stocksense.data.store import Store
from stocksense.server.app import app


@pytest.fixture()
def client_with_store(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(db_path))
    store = Store(db_path)
    store.close()
    return TestClient(app), db_path


def _insert_live_model(db_path, model_id: str, horizon: int = 10, top_n: int = 3) -> None:
    store = Store(db_path)
    store.con.execute(
        """
        INSERT INTO model_registry (
            model_id, model_type, horizon_bars, top_n, feature_schema_version,
            created_at, lifecycle_state, artifact_path, promoted_at
        ) VALUES (?, 'cross_sectional_ranker', ?, ?, 'v1', ?, 'live', 'unused.joblib', ?)
        """,
        [model_id, horizon, top_n, datetime.now(timezone.utc), datetime.now(timezone.utc)],
    )
    store.close()


def _insert_candles(db_path, prices: dict[str, float]) -> None:
    store = Store(db_path)
    rows = [
        {"symbol": s, "date": date(2026, 8, 20), "open": p, "high": p, "low": p,
         "close": p, "adj_close": p, "volume": 1000.0, "source": "test"}
        for s, p in prices.items()
    ]
    store.upsert_candles(pd.DataFrame(rows))
    store.close()


def _insert_predictions(db_path, model_id: str, horizon: int, rows: list[dict]) -> None:
    store = Store(db_path)
    df = pd.DataFrame([
        {
            "run_id": r.get("run_id", "run1"), "symbol": r["symbol"], "as_of_date": r["as_of_date"],
            "horizon_bars": horizon, "score": r["score"], "rank": r["rank"], "model_version": model_id,
            "horizon_type": "short", "predicted_return": r.get("predicted_return", r["score"]),
            "confidence": r.get("confidence"), "feature_snapshot_hash": "h",
        }
        for r in rows
    ])
    store.write_predictions(df)
    store.close()


def test_brief_no_live_model(client_with_store) -> None:
    client, db_path = client_with_store
    resp = client.get("/api/brief?horizon=10")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_live_model"


def test_brief_live_model_but_no_predictions(client_with_store) -> None:
    client, db_path = client_with_store
    _insert_live_model(db_path, "m1", horizon=10)
    resp = client.get("/api/brief?horizon=10")
    body = resp.json()
    assert body["status"] == "no_predictions"
    assert body["model_id"] == "m1"


def test_brief_happy_path_returns_top_n_picks_with_weights(client_with_store) -> None:
    client, db_path = client_with_store
    _insert_live_model(db_path, "m1", horizon=10, top_n=2)
    _insert_candles(db_path, {"AAA": 100.0, "BBB": 50.0, "CCC": 3000.0})
    _insert_predictions(db_path, "m1", 10, [
        {"symbol": "AAA", "as_of_date": date(2026, 8, 21), "score": 0.05, "rank": 1},
        {"symbol": "BBB", "as_of_date": date(2026, 8, 21), "score": 0.03, "rank": 2},
        {"symbol": "CCC", "as_of_date": date(2026, 8, 21), "score": 0.01, "rank": 3},  # below top_n=2, excluded
    ])

    resp = client.get("/api/brief?horizon=10")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_id"] == "m1"
    assert len(body["picks"]) == 2  # top_n=2
    symbols = {p["symbol"] for p in body["picks"]}
    assert symbols == {"AAA", "BBB"}
    for pick in body["picks"]:
        assert pick["weight"] == pytest.approx(0.5)  # equal-weighted top-2
        assert pick["last_close"] is not None


def test_brief_computes_min_capital_from_real_prices_not_assumed(client_with_store) -> None:
    client, db_path = client_with_store
    _insert_live_model(db_path, "m1", horizon=10, top_n=2)
    # AAA needs 100/0.5=200 to hold 1 share; BBB needs 50/0.5=100 -- binding is AAA
    _insert_candles(db_path, {"AAA": 100.0, "BBB": 50.0})
    _insert_predictions(db_path, "m1", 10, [
        {"symbol": "AAA", "as_of_date": date(2026, 8, 21), "score": 0.05, "rank": 1},
        {"symbol": "BBB", "as_of_date": date(2026, 8, 21), "score": 0.03, "rank": 2},
    ])

    resp = client.get("/api/brief?horizon=10")
    body = resp.json()
    assert body["min_capital_for_full_positions_inr"] == pytest.approx(200.0)


def test_brief_includes_yesterdays_graded_revision(client_with_store) -> None:
    client, db_path = client_with_store
    _insert_live_model(db_path, "m1", horizon=10, top_n=2)
    _insert_candles(db_path, {"AAA": 100.0})
    _insert_predictions(db_path, "m1", 10, [
        {"symbol": "AAA", "as_of_date": date(2026, 8, 21), "score": 0.05, "rank": 1, "run_id": "today"},
        {"symbol": "AAA", "as_of_date": date(2026, 8, 10), "score": 0.02, "rank": 1, "run_id": "yesterday"},
    ])
    store = Store(db_path)
    store.grade_prediction("yesterday", "AAA", date(2026, 8, 10), 10,
                            actual_return=-0.01, grade_json="{}", graded_at=datetime.now(timezone.utc))
    store.close()

    resp = client.get("/api/brief?horizon=10")
    body = resp.json()
    revision = body["yesterdays_revision"]
    assert len(revision) == 1
    assert revision[0]["symbol"] == "AAA"
    assert revision[0]["actual_return"] == pytest.approx(-0.01)
    assert revision[0]["direction_correct"] is False  # predicted +0.02, actual -0.01


def test_brief_never_accepts_or_returns_a_capital_field(client_with_store) -> None:
    """The design invariant this whole phase turns on: capital is never
    a system parameter. Confirms the endpoint has no capital query
    param and the response body contains no key that looks like an
    account-size figure."""
    client, db_path = client_with_store
    _insert_live_model(db_path, "m1", horizon=10, top_n=1)
    _insert_candles(db_path, {"AAA": 100.0})
    _insert_predictions(db_path, "m1", 10, [
        {"symbol": "AAA", "as_of_date": date(2026, 8, 21), "score": 0.05, "rank": 1},
    ])

    # passing a capital-looking query param must have no effect -- the
    # endpoint doesn't even declare such a parameter
    resp = client.get("/api/brief?horizon=10&capital=15000&account_size=999999")
    body = resp.json()
    forbidden_keys = {"capital", "capital_inr", "account_size", "account_size_inr", "rupees_per_day"}
    assert forbidden_keys.isdisjoint(body.keys())
    for pick in body["picks"]:
        assert forbidden_keys.isdisjoint(pick.keys())


def test_brief_no_nan_leaks(client_with_store) -> None:
    client, db_path = client_with_store
    _insert_live_model(db_path, "m1", horizon=10, top_n=1)
    _insert_predictions(db_path, "m1", 10, [
        {"symbol": "AAA", "as_of_date": date(2026, 8, 21), "score": 0.05, "rank": 1,
         "predicted_return": None, "confidence": None},
    ])
    resp = client.get("/api/brief?horizon=10")
    assert "NaN" not in resp.text
    assert "Infinity" not in resp.text
