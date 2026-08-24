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


@app.post("/api/ask")
def ask_rag(payload: dict):
    """Phase G/B3: `index-corpus` was already triggerable from the
    Pipeline tab, but nothing could query the corpus it builds -- no
    UI, no endpoint. rag.agent.ask() already funnels through
    agent.claude_cli.invoke(), so the F2 authorize/decline gate applies
    here automatically, same as every other Claude-touching path; this
    endpoint adds no separate access check because there is exactly one
    enforcement point and this call already goes through it."""
    from stocksense.rag.agent import ask as rag_ask

    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="'question' is required")

    store = _store()
    try:
        return rag_ask(question, store)
    finally:
        store.close()


@app.get("/api/registry")
def registry():
    store = _store()
    try:
        df = store.read_model_registry()
    finally:
        store.close()
    return {"models": _df_to_records(df)}


@app.get("/api/brief")
def brief(horizon: int = 10, model_type: str = "cross_sectional_ranker"):
    """Phase G5: the daily brief -- today's top-N recommendation from
    the live model, plus yesterday's revision (matured/graded
    predictions), plus a capital-agnostic sizing note. NO capital
    figure is ever computed or assumed server-side; `min_capital_for_
    full_positions_inr` is the whole-share-divisibility floor derived
    from real prices and weights (optimizer/sizing.py), not a rupee
    target, and the client multiplies weights by whatever capital the
    viewer optionally types in -- that number never reaches this
    endpoint or gets persisted anywhere.

    Returns an explicit `status` so the UI can render a clear reason
    rather than an empty table: 'no_live_model' (nothing promoted to
    live for this horizon yet), 'no_predictions' (a live model exists
    but the reconcile loop hasn't recorded anything for it), or 'ok'.
    """
    from stocksense.optimizer.rebalance import recommend_todays_actions
    from stocksense.optimizer.sizing import min_capital_for_full_positions, round_trip_cost_bps
    from stocksense.portfolio.construct import target_weights_top_n

    store = _store()
    try:
        live = store.get_live_model(model_type, horizon)
        if live.empty:
            return {"status": "no_live_model", "horizon_bars": horizon}
        model_row = live.iloc[0]
        model_id = model_row["model_id"]
        top_n = int(model_row["top_n"]) if model_row["top_n"] else 10

        preds = store.read_predictions()
        preds = preds[preds["model_version"] == model_id]
        if preds.empty:
            return {"status": "no_predictions", "horizon_bars": horizon, "model_id": model_id}

        latest_date = preds["as_of_date"].max()
        # Defensive dedup -- see optimizer/rebalance.py's _weights_at
        # for the full story: record_predictions is idempotent per
        # (model, as_of_date, horizon) at the source now, but a
        # duplicate-run_id row from before that fix can still exist in
        # an already-populated database.
        latest = preds[preds["as_of_date"] == latest_date].drop_duplicates(subset=["symbol"], keep="last").copy()

        scores = latest.set_index("symbol")["score"]
        weights = target_weights_top_n(scores, top_n)
        weights = weights[weights > 0].sort_values(ascending=False)

        candles = store.read_candles()
        latest_prices: dict[str, float] = {}
        if not candles.empty:
            latest_close = candles.sort_values("date").groupby("symbol")["close"].last()
            latest_prices = {s: float(latest_close[s]) for s in weights.index if s in latest_close.index}

        # Phase H bug fix, found running this endpoint against the real
        # live model: a bhavcopy-point-in-time-trained model's picks
        # (e.g. SUMEETINDS, CUPID -- never in the 98-symbol `candles`
        # table at all) got last_close=None for every pick, because this
        # lookup only ever checked `candles`. Falls back to bhavcopy_eq
        # directly for whichever symbols the 98-symbol table didn't
        # have -- a targeted price lookup for the SPECIFIC symbols
        # already selected, not a full universe rebuild, so it's cheap
        # and needs no cap-band/price-source parameter on this endpoint.
        missing = [s for s in weights.index if s not in latest_prices]
        if missing:
            bhav = store.con.execute(
                "SELECT symbol, date, close FROM bhavcopy_eq WHERE symbol = ANY(?) ORDER BY date",
                [missing],
            ).fetchdf()
            if not bhav.empty:
                bhav_latest = bhav.groupby("symbol")["close"].last()
                for s in missing:
                    if s in bhav_latest.index:
                        latest_prices[s] = float(bhav_latest[s])

        picks = []
        for symbol, weight in weights.items():
            row = latest[latest["symbol"] == symbol].iloc[0]
            picks.append({
                "symbol": symbol,
                "weight": float(weight),
                "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
                "predicted_return": _clean(row.get("predicted_return")),
                "confidence": _clean(row.get("confidence")),
                "last_close": latest_prices.get(symbol),
            })

        # Yesterday's revision: the most recently graded predictions for
        # THIS model -- what was predicted, what actually happened.
        graded = preds[preds["graded_at"].notna()].sort_values("graded_at", ascending=False).head(20)
        revision = [
            {
                "symbol": r["symbol"], "as_of_date": str(r["as_of_date"]),
                "predicted_return": _clean(r.get("predicted_return")),
                "actual_return": _clean(r.get("actual_return")),
                "direction_correct": (
                    bool((r["predicted_return"] > 0) == (r["actual_return"] > 0))
                    if pd.notna(r.get("predicted_return")) and pd.notna(r.get("actual_return")) else None
                ),
            }
            for _, r in graded.iterrows()
        ]

        min_capital = min_capital_for_full_positions(latest_prices, weights.to_dict()) if latest_prices else None

        # Phase H4: actual buy/sell/hold moves, at the cadence the gate
        # was actually measured on (every `horizon` TRADING DAYS) --
        # NOT re-ranked into a fresh move list every day, which would
        # generate real churn nothing in research/verdict_bhavcopy_
        # rerun.md's PASS numbers ever paid for. next_rebalance_in_
        # trading_days is 0 exactly when today's own prediction run IS
        # the rebalance point; otherwise it's how many trading days
        # remain until the next one, and `actions` still reflects the
        # LAST real rebalance point's moves (unchanged since then).
        todays = recommend_todays_actions(store, model_id, horizon_bars=horizon, top_n=top_n)
        actions = None
        next_rebalance_in_trading_days = None
        if todays is not None:
            actions = [
                {
                    "symbol": a.symbol, "action": a.action,
                    "current_weight": a.current_weight, "target_weight": a.target_weight,
                    "estimated_cost_fraction": a.estimated_cost_inr,  # portfolio_value_inr=1.0 -- a fraction, not rupees
                }
                for a in todays.actions if a.action != "hold"
            ]
            next_rebalance_in_trading_days = todays.next_rebalance_in_trading_days

        return {
            "status": "ok",
            "horizon_bars": horizon,
            "model_id": model_id,
            "as_of_date": str(latest_date),
            "picks": picks,
            "yesterdays_revision": revision,
            "min_capital_for_full_positions_inr": min_capital,
            "round_trip_cost_bps_equity_delivery": round_trip_cost_bps("equity_delivery"),
            "actions": actions,
            "next_rebalance_in_trading_days": next_rebalance_in_trading_days,
        }
    finally:
        store.close()


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


RESEARCH_DOC_EXTENSIONS = (".md", ".csv")


@app.get("/api/research/docs")
def list_research_docs():
    """Phase G/Track C: lists readable files in `research/` -- pre-
    registrations, verdicts, fold-result CSVs -- so E4's answer (and
    every future research doc) is readable from the UI without a
    terminal. Listing is not a security boundary by itself; the
    traversal protection lives in get_research_doc below, which is the
    endpoint that actually reads file content from a caller-supplied
    name."""
    folder = REPO_ROOT / "research"
    if not folder.exists():
        return {"docs": []}
    docs = sorted(
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in RESEARCH_DOC_EXTENSIONS
    )
    return {"docs": docs}


@app.get("/api/research/doc/{name}")
def get_research_doc(name: str):
    """Serves one file's content by NAME, not by arbitrary path --
    deliberately restricted to a single path segment (no '/' allowed)
    resolved against `research/` and then verified to still resolve
    INSIDE that directory before ever being opened. Both checks matter:
    rejecting '/' stops the obvious 'name=../../.env' case, but
    resolving and re-checking containment is what actually stops a
    symlink or a cleverer traversal from escaping the directory --
    string-matching the input alone is not a sufficient guard."""
    if "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid document name")

    folder = (REPO_ROOT / "research").resolve()
    candidate = (folder / name).resolve()
    if not candidate.is_relative_to(folder):
        raise HTTPException(status_code=400, detail="invalid document name")
    if candidate.suffix.lower() not in RESEARCH_DOC_EXTENSIONS or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"no research document named {name!r}")

    return {"name": name, "content": candidate.read_text(encoding="utf-8", errors="replace")}


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


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str):
    """Phase G/A1: re-reads a job's output after the fact -- the console
    drawer only ever streams a job that's running RIGHT NOW
    (job-console.js clears on every new trigger), so a finished
    overnight backfill was otherwise unrecoverable even though its full
    output was already being written to disk the whole time
    (server/jobs.py's _pump_output, log_path column on `ui_jobs`).

    Still-running jobs are served from the in-memory ring buffer (no
    database read, same as /ws/jobs -- works even while this job or
    another one holds the write lock); finished jobs are read from their
    persisted log file on disk. Checking the job's STATUS, not just
    whether the registry has ever heard of it, matters: JobRegistry
    never evicts a finished job's in-memory entry for the life of the
    server process, so "registry knows about it" alone would never
    exercise the disk-read path at all once a job completes."""
    registry = _job_registry()
    live_status = registry.status(job_id)
    if live_status is not None and live_status["status"] == "running":
        return {"job_id": job_id, "lines": registry.tail(job_id), "source": "live"}

    store = _store()
    try:
        job = store.read_ui_job(job_id)
    finally:
        store.close()
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id!r}")

    log_path = job.get("log_path")
    if not log_path or not Path(log_path).exists():
        return {"job_id": job_id, "lines": [], "source": "disk", "note": "no log file found"}

    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {"job_id": job_id, "lines": lines, "source": "disk"}


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
