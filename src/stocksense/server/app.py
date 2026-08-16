"""
Local JSON API for the desktop control room (docs/18-desktop-app.md).
Bound to 127.0.0.1 ONLY, never 0.0.0.0 -- this serves the user's private
trading data (P&L, positions, tax exposure) and must not be reachable
from the network. Every endpoint is READ-ONLY: it reads what the CLI
commands already computed and wrote to the store; it never triggers a
computation, a trade, or an agent call itself. The dashboard observes
state, it doesn't create it.

Runs standalone (`uvicorn stocksense.server.app:app`) for testing with
httpx and no Electron in the loop, exactly as the plan specifies -- the
Electron shell is a view over this API, never something the API depends
on.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd
from fastapi import FastAPI

from stocksense.core.config import get_settings
from stocksense.data.store import Store

app = FastAPI(title="StockSense Control Room API")

APP_VERSION = "0.1.0"


def _clean(value):
    """JSON-safe scalar: NaN/NaT -> None, pandas/numpy Timestamp -> ISO
    string, everything else passed through. FastAPI's default encoder
    chokes on NaN and numpy scalar types, which every DataFrame from
    Store hands back."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        return value.item()
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _store() -> Store:
    settings = get_settings()
    return Store(settings.duckdb_path)


@app.get("/api/health")
def health():
    settings = get_settings()
    store = _store()
    try:
        candles = store.con.execute("SELECT MAX(date) FROM candles").fetchone()
        latest_candle_date = candles[0].isoformat() if candles and candles[0] else None
    finally:
        store.close()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "db_path": str(settings.duckdb_path),
        "latest_candle_date": latest_candle_date,
        "checked_at": datetime.now().isoformat(),
    }


@app.get("/api/summary")
def summary():
    store = _store()
    try:
        positions = store.read_positions()
    finally:
        store.close()

    if positions.empty:
        return {"n_positions": 0, "total_net_pnl": 0.0, "total_gross_pnl": 0.0, "total_charges": 0.0, "win_rate": None}

    return {
        "n_positions": int(len(positions)),
        "total_net_pnl": float(positions["net_pnl"].sum()),
        "total_gross_pnl": float(positions["gross_pnl"].sum()),
        "total_charges": float(positions["charges"].sum()),
        "win_rate": float((positions["net_pnl"] > 0).mean()),
    }


@app.get("/api/doshas")
def doshas():
    store = _store()
    try:
        df = store.read_latest_diagnostics()
    finally:
        store.close()
    order = {"critical": 0, "high": 1, "notable": 2, "ok": 3}
    records = _df_to_records(df)
    records.sort(key=lambda r: order.get(r.get("severity"), 9))
    return {"doshas": records}


@app.get("/api/counterfactuals")
def counterfactuals():
    store = _store()
    try:
        diag = store.con.execute("SELECT run_id FROM diagnostics ORDER BY as_of DESC LIMIT 1").fetchone()
        if diag is None:
            return {"counterfactuals": []}
        df = store.read_counterfactuals(diag[0])
    finally:
        store.close()
    return {"counterfactuals": _df_to_records(df)}


@app.get("/api/positions")
def positions(limit: int = 100, offset: int = 0):
    store = _store()
    try:
        df = store.read_positions()
    finally:
        store.close()
    page = df.iloc[offset:offset + limit]
    return {"positions": _df_to_records(page), "total": int(len(df))}


@app.get("/api/harness")
def harness():
    store = _store()
    try:
        df = store.read_job_runs()
    finally:
        store.close()
    if df.empty:
        return {"jobs": []}
    latest_per_job = df.sort_values("started_at", ascending=False).groupby("job_name").first().reset_index()
    return {"jobs": _df_to_records(latest_per_job)}


@app.get("/api/registry")
def registry():
    store = _store()
    try:
        df = store.read_model_registry()
    finally:
        store.close()
    return {"models": _df_to_records(df)}


@app.get("/api/agent-runs")
def agent_runs(limit: int = 20):
    store = _store()
    try:
        df = store.read_agent_runs(limit=limit)
    finally:
        store.close()
    records = _df_to_records(df)
    for r in records:
        r.pop("input_json", None)  # may contain redacted-but-still-bulky fact payloads; the dashboard shows status, not full payloads
    return {"agent_runs": records, "n_unverified": sum(1 for r in records if r.get("status") == "unverified_numbers")}
