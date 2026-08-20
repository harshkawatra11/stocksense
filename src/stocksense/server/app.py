"""
Local JSON API for the desktop control room (docs/18-desktop-app.md).
Bound to 127.0.0.1 ONLY, never 0.0.0.0 -- this serves the user's private
trading data (P&L, positions, tax exposure) and must not be reachable
from the network.

AUDIT-NOTED DESIGN REVERSAL (Phase F1): this module's docstring used to
say every endpoint is read-only and never triggers anything. That is no
longer true, deliberately -- the desktop control center needs to trigger
backfills, statement ingestion, Kundli, and Foreman runs from the UI
without the user opening a terminal. The new safety story replacing
"can't do anything": a closed command allowlist (server/jobs.py's
COMMANDS, never arbitrary string execution), an explicit local
authorize/decline gate for anything that would invoke the Claude CLI
(Phase F2, `agent/access.py`), and the same 127.0.0.1-only binding this
module always had. The original 8 GET endpoints below are unchanged.

Runs standalone (`uvicorn stocksense.server.app:app`) for testing with
httpx and no Electron in the loop, exactly as the plan specifies -- the
Electron shell is a view over this API, never something the API depends
on.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from stocksense.core.config import REPO_ROOT, get_settings
from stocksense.data.store import Store
from stocksense.server.jobs import (
    COMMANDS,
    JobAlreadyRunningError,
    JobRegistry,
    MissingParamError,
    UnknownCommandError,
)

app = FastAPI(title="StockSense Control Room API")

APP_VERSION = "0.1.0"

_registry: JobRegistry | None = None

# Module-level (not a function-local import) specifically so tests can
# monkeypatch `stocksense.server.app.REPO_ROOT` to a tmp directory --
# the real .env holds live Upstox credentials, and a settings-endpoint
# test must never write to it.


def _job_registry() -> JobRegistry:
    """Lazily-created module-level singleton -- must be ONE instance for
    the server process's lifetime, since it holds the in-memory record
    of currently-running subprocesses (PIDs, output buffers) that no
    per-request object could track across requests."""
    global _registry
    if _registry is None:
        _registry = JobRegistry(get_settings().duckdb_path)
    return _registry


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
    """Every endpoint below calls this first. DuckDB gives whichever
    process holds a read-write connection an EXCLUSIVE file lock -- while
    a triggered job (a backfill, foreman run, etc.) is running, its
    subprocess holds that lock for its entire duration, and this call
    will fail here. Caught once, at this single choke point, rather than
    duplicated in every endpoint: surfaced as a 503 with a clear reason,
    not an unhandled 500 that looks like a real bug."""
    settings = get_settings()
    try:
        return Store(settings.duckdb_path)
    except duckdb.IOException as e:
        raise HTTPException(
            status_code=503,
            detail="database is busy -- a job is currently running and holds the write lock; try again once it finishes",
        ) from e


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


# ---- Phase F1: job management (trigger + watch) ----

@app.get("/api/statements-folder")
def list_statements_folder():
    """Lists files in the repo's `statements/` folder, so the desktop
    app's Statements panel can offer a picker instead of a free-text
    path field -- purely a directory listing, no file contents read."""
    from stocksense.core.config import REPO_ROOT

    folder = REPO_ROOT / "statements"
    if not folder.exists():
        return {"folder": str(folder), "files": []}
    files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls"))
    return {"folder": str(folder), "files": files}


@app.get("/api/job-commands")
def job_commands():
    """The allowlist itself, so the UI can build its trigger forms from
    the same source of truth the server enforces against -- a form for a
    command not in this list has nothing to submit to."""
    return {
        "commands": [
            {"name": spec.name, "params": list(spec.param_flags.keys()), "required": list(spec.required_params)}
            for spec in COMMANDS.values()
        ]
    }


@app.post("/api/jobs/{command}")
def trigger_job(command: str, params: dict | None = None):
    registry = _job_registry()
    try:
        job_id = registry.start(command, params or {})
    except UnknownCommandError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MissingParamError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except JobAlreadyRunningError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"job_id": job_id, "command": command}


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    registry = _job_registry()
    stopped = registry.stop(job_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"no running job with id {job_id!r}")
    return {"job_id": job_id, "status": "stopping"}


@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    """Merges live in-memory status (for anything currently running --
    no database read needed, so this works even while a job holds the
    write lock) with durable history from `ui_jobs`."""
    registry = _job_registry()
    active = {j["job_id"]: j for j in registry.list_active()}

    store = _store()
    try:
        df = store.read_ui_jobs(limit=limit)
    finally:
        store.close()

    history = _df_to_records(df)
    for row in history:
        live = active.pop(row.get("job_id"), None)
        if live:
            row["status"] = live["status"]  # live status supersedes the durable snapshot while running
    return {"jobs": history, "active": list(active.values())}


@app.websocket("/ws/jobs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: str):
    """Live stdout tail for one job. On connect, replays everything
    captured so far (so opening this mid-job isn't blank), then polls
    the in-memory ring buffer for new lines every 300ms until the job
    reaches a terminal status. A poll loop, not a true push callback --
    the registry's output pump runs on a plain thread (subprocess.Popen
    reading, not asyncio-native), and bridging that into this coroutine
    without a poll would need a cross-thread-safe async primitive this
    module doesn't otherwise need; 300ms is fast enough to feel live for
    a human watching a log scroll."""
    await websocket.accept()
    registry = _job_registry()
    sent = 0
    try:
        while True:
            lines = registry.tail(job_id)
            if len(lines) > sent:
                for line in lines[sent:]:
                    await websocket.send_json({"line": line})
                sent = len(lines)
            status = registry.status(job_id)
            if status is None or status["status"] != "running":
                await websocket.send_json({"done": True, "status": status["status"] if status else "unknown"})
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


# ---- Phase F4: live settings ----
#
# AUDIT FIX (found live while building the Settings panel): this used to
# read/write the `app_settings` DuckDB table -- but `core.config.Settings`
# is a pydantic-settings model that only ever reads environment variables
# and `.env` (verified directly: setting STOCKSENSE_PLANNER_MODEL as an
# env var changes get_settings().planner_model; nothing in Settings reads
# the DB). A PUT here was silently a no-op as far as actual pipeline
# behavior is concerned -- it "saved" to a table nothing consults. Fixed
# to read/write .env directly, which is the mechanism get_settings()
# ACTUALLY reacts to (fresh on every call, no caching to invalidate, so
# no restart is needed either). The app_settings DB table remains correct
# and unchanged for F2 (Claude access flag) and F3 (usage tracker state)
# -- neither of those is a Settings field, so DB storage was always right
# for them; only this endpoint was wired to the wrong store.

_SECRET_SETTINGS_FIELDS = {"upstox_api_key", "upstox_api_secret", "upstox_access_token"}


def _env_file_path() -> Path:
    return REPO_ROOT / ".env"


def _read_env_file() -> dict:
    path = _env_file_path()
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


def _write_env_file(updates: dict) -> None:
    """Merges into the existing .env rather than overwriting it --
    preserves comments, unrelated keys, and file order for every line
    not being changed."""
    path = _env_file_path()
    existing = _read_env_file()
    existing.update({k: v for k, v in updates.items() if v is not None})
    for k, v in updates.items():
        if v is None:
            existing.pop(k, None)
    lines = [f"{k}={v}" for k, v in existing.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@app.get("/api/settings")
def get_app_settings():
    """Returns every actual Settings field at its CURRENT effective
    value (defaults + any .env override already applied) -- not a raw
    dump of .env, so the UI shows what get_settings() will really
    return, not just what happens to be written down. Secret fields are
    masked, never echoed back in full."""
    settings = get_settings()
    values = settings.model_dump(mode="json")
    for field in _SECRET_SETTINGS_FIELDS:
        if values.get(field):
            values[field] = "••••••••" + str(values[field])[-4:]
    return {"settings": values}


@app.put("/api/settings")
def put_app_settings(values: dict):
    """Writes STOCKSENSE_<FIELD> to .env for each provided field --
    pydantic-settings re-reads .env fresh on every get_settings() call,
    so this takes effect on the NEXT invocation of anything that calls
    it, no server restart needed."""
    env_updates = {f"STOCKSENSE_{k.upper()}": (None if v is None else str(v)) for k, v in values.items()}
    _write_env_file(env_updates)
    return {"settings": values}


# ---- Phase F2: Claude CLI authorize/decline ----

@app.get("/api/claude/auth")
def claude_auth_status():
    """Read-only: proxies `claude auth status --json` plus whether THIS
    app currently has authorized access (the two are different facts --
    you can be logged into Claude Code without having granted
    StockSense permission to invoke it)."""
    from stocksense.agent.access import check_claude_auth, is_access_granted

    status = check_claude_auth()
    store = _store()
    try:
        granted = is_access_granted(store)
    finally:
        store.close()
    return {
        "logged_in": status.logged_in, "email": status.email, "plan": status.plan,
        "raw_error": status.raw_error, "access_granted": granted,
    }


@app.post("/api/claude/authorize")
def claude_authorize():
    from stocksense.agent.access import ClaudeAccessNotGranted, grant_access

    store = _store()
    try:
        status = grant_access(store)
    except ClaudeAccessNotGranted as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    finally:
        store.close()
    return {"access_granted": True, "email": status.email, "plan": status.plan}


@app.post("/api/claude/decline")
def claude_decline():
    from stocksense.agent.access import revoke_access

    store = _store()
    try:
        revoke_access(store)
    finally:
        store.close()
    return {"access_granted": False}


# ---- Phase F3: usage gauge (measured, not official -- see usage_tracker.py) ----

@app.get("/api/claude/usage")
def claude_usage():
    from stocksense.agent.usage_tracker import get_usage_summary

    store = _store()
    try:
        return get_usage_summary(store)
    finally:
        store.close()


@app.put("/api/claude/usage/soft-alarm")
def set_claude_usage_soft_alarm(tokens_5h: int | None = None):
    """A self-configured soft alarm, calibrated from the user's own
    experience of where they actually get rate-limited -- there is no
    official threshold to validate against (see usage_tracker.py)."""
    from stocksense.agent.usage_tracker import SOFT_ALARM_SETTING_KEY

    store = _store()
    try:
        store.set_app_setting(SOFT_ALARM_SETTING_KEY, None if tokens_5h is None else str(tokens_5h))
    finally:
        store.close()
    return {"soft_alarm_tokens_5h": tokens_5h}
