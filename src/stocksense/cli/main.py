"""
Nightly pipeline CLI, condensed to what Phase 1 actually needs:
train a candidate through walk-forward evaluation, gate it, register it,
and (separately) score today's cross-section with whichever model is
live. This is docs/05-nightly-pipeline.md's steps 6-11 as a runnable
command rather than a fully scheduled service — Electron/Telegram/
Ollama/Claude layers are deliberately not built yet (docs/00-overview.md
non-goals still apply at this stage; the point of Phase 0/1 is proving
the brain before building the shell around it).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import typer

from stocksense.core.config import REPO_ROOT, get_settings
from stocksense.data.store import Store
from stocksense.data.loader import load_candles, load_features_and_labels
from stocksense.data.validate import quarantine_symbols
from stocksense.features.engine import build_features
from stocksense.portfolio.construct import target_weights_top_n
from stocksense.statements.parsers import detect_parser
from stocksense.statements.parsers.base import file_hash
from stocksense.statements.positions import reconstruct_positions
from stocksense.statements.report import generate_kundli
from stocksense.foreman.adversary import red_team, has_blocking_finding
from stocksense.foreman.assess import propose_goals
from stocksense.foreman.budget import check_budget
from stocksense.foreman.executor import execute_goal, record_goal_result, record_ledger_entries, record_protected_violations
from stocksense.harness.loops import build_reconcile_graph, grade_matured_predictions, record_predictions
from stocksense.harness.retrain import build_weekly_retrain_graph
from stocksense.harness.runner import run_graph
from stocksense.optimizer.tax import compute_tax_liability
from stocksense.rag.agent import ask as rag_ask
from stocksense.rag.embed import embeddings_available
from stocksense.rag.index import index_document, rebuild_fts_index
from stocksense.data.nse_archive import fetch_range
from stocksense.data.corporate_actions import fetch_ca_range, parse_ca_frame
from stocksense.data.universe_pit import CAP_BANDS, universe_as_of
from stocksense.data.upstox_intraday import resolve_symbol_map, fetch_range as fetch_intraday_range
from stocksense.server.run import run as run_server

foreman_app = typer.Typer(help="The Foreman: self-building harness")

log = structlog.get_logger(__name__)
app = typer.Typer(help="StockSense nightly pipeline CLI")
app.add_typer(foreman_app, name="foreman")

MODEL_TYPE = "cross_sectional_ranker"

_PROGRESS_EVERY = 50  # print at most once per this many units of work, plus always on completion


def _emit_progress(current: int, total: int, force: bool = False) -> None:
    """Phase F1: a lightweight, parseable progress line for long-running
    backfills, which otherwise print nothing until they finish -- a UI
    job console watching only process-alive-or-not would show 'running'
    for hours with no finer signal. Printed to stdout (captured by the
    job registry's ring buffer) at a throttled cadence so a 6,500-day
    backfill doesn't flood the log with one line per day. Format is
    plain text, not JSON, so the command stays just as readable run by
    hand in a real terminal as it is parsed by the UI."""
    if total <= 0:
        return
    if not force and current % _PROGRESS_EVERY != 0 and current != total:
        return
    pct = min(100, round(100 * current / total))
    typer.echo(f"PROGRESS: {current}/{total} ({pct}%)")


# Phase G: the loading logic itself now lives in data/loader.py so
# harness/loops.py (the reconcile loop) can share it too without a
# circular import (this module already imports from harness.loops).
# Re-exported under the original private names -- callers and existing
# tests (test_price_source_wiring.py) import _load_candles/
# _load_features_and_labels from cli.main unchanged.
_load_candles = load_candles
_load_features_and_labels = load_features_and_labels


def _resolve_cap_band(cap_band: Optional[str], settings) -> tuple[float, float] | None:
    """Shared by train-candidate/reconcile/retrain-weekly's --cap-band
    option: validates against universe_pit.CAP_BANDS, mutates `settings`
    in place to force the point-in-time bhavcopy path (a cap band means
    nothing on the yfinance 'candles' path -- silently ignoring it would
    be worse than an explicit, logged override), and returns the
    resolved turnover_rank_band tuple (or None for 'full_pit'/when no
    --cap-band was given at all).

    Exits the process (typer.Exit) on an unknown cap_band value rather
    than raising, matching every other CLI-level validation error in
    this module."""
    if cap_band is None:
        return None
    if cap_band not in CAP_BANDS:
        typer.echo(f"Unknown --cap-band {cap_band!r}. Valid values: {sorted(CAP_BANDS)}")
        raise typer.Exit(code=1)
    if settings.price_source != "bhavcopy" or not settings.use_point_in_time_universe:
        typer.echo(
            f"--cap-band {cap_band!r} given: forcing price_source=bhavcopy, "
            f"use_point_in_time_universe=True for this run (was "
            f"price_source={settings.price_source!r}, "
            f"use_point_in_time_universe={settings.use_point_in_time_universe})."
        )
    settings.price_source = "bhavcopy"
    settings.use_point_in_time_universe = True
    return CAP_BANDS[cap_band]


@app.command()
def train_candidate(
    horizon: int = typer.Option(20, help="Prediction horizon in trading bars"),
    top_n: int = typer.Option(20, help="Portfolio size (top-N by rank)"),
    cost_bps: float = typer.Option(25.0, help="Round-trip cost assumption for gate evaluation, bps"),
    cap_band: Optional[str] = typer.Option(
        None, "--cap-band",
        help="Restrict to a liquidity-rank cap band on the point-in-time bhavcopy universe: "
             "'full_pit' (no restriction), 'large', 'mid', or 'small'. See data/universe_pit.py's "
             "CAP_BANDS -- research/verdict_bhavcopy_rerun.md is the evidence behind these bands. "
             "Forces price_source=bhavcopy + use_point_in_time_universe=True for this run.",
    ),
) -> None:
    """Walk-forward evaluate a candidate, run it through the gate, train
    the final full-history artifact, and register the outcome — win or
    lose — in the model registry.

    Gate criteria are NOT configurable from this command on purpose: the
    production path always uses the pre-registered defaults in
    GateCriteria (research/gate_criteria_preregistration.md). Allowing
    ad-hoc overrides here would recreate the exact evaluator-overfitting
    failure that pre-registration exists to prevent — someone could keep
    loosening criteria from the command line until a candidate passes.
    Experiment with alternative criteria in a research script, never here.

    Thin wrapper around models.train_candidate.train_candidate_core
    (Phase G4) — the actual logic is factored out so harness/retrain.py's
    weekly retrain graph can call it as a plain function, the same
    extraction Phase G2 did for record_predictions/grade_matured_
    predictions. This command's behavior is unchanged when --cap-band is
    omitted.
    """
    from stocksense.models.train_candidate import train_candidate_core

    settings = get_settings()
    turnover_rank_band = _resolve_cap_band(cap_band, settings)
    store = Store(settings.duckdb_path)

    result = train_candidate_core(horizon, top_n, cost_bps, store, settings=settings, turnover_rank_band=turnover_rank_band)

    if result.lifecycle_state is None:
        typer.echo("No fold results produced — insufficient data for this horizon/universe.")
        store.close()
        raise typer.Exit(code=1)

    typer.echo(f"\nGate verdict: {'PASS' if result.verdict.passed else 'FAIL'}")
    typer.echo(f"Reason: {result.verdict.reason}")
    typer.echo(f"Metrics: {json.dumps(result.verdict.metrics, indent=2)}")
    typer.echo(f"\nRegistered: {result.model_id}")
    typer.echo(f"Lifecycle state: {result.lifecycle_state}")
    store.close()


@app.command("promote-model")
def promote_model_cmd(model_id: str = typer.Argument(..., help="Model ID to promote to 'live'")) -> None:
    """Promotes a 'shadow' model to 'live' -- the missing half of docs/06's
    stated design ("the gate and the shadow trial are two different
    things a model must earn separately"). No code path has ever set a
    model to 'live' before this command: apply_gate_decision only ever
    produces 'shadow' (pass) or 'archived' (fail); apply_forward_record_
    decision (Phase G2) only ever demotes 'live' back to 'shadow'.

    Deliberately manual and human-triggered -- there is no automatic
    shadow-trial-passed criterion in this codebase yet, so promoting
    here is a judgment call the operator makes after reviewing the
    gate's own verdict/metrics (`stocksense registry`), not something
    this command decides on its own.
    """
    settings = get_settings()
    store = Store(settings.duckdb_path)

    row = store.con.execute(
        "SELECT lifecycle_state FROM model_registry WHERE model_id = ?", [model_id],
    ).fetchdf()
    if row.empty:
        typer.echo(f"No model with id {model_id!r} in the registry.")
        store.close()
        raise typer.Exit(code=1)

    current_state = row.iloc[0]["lifecycle_state"]
    if current_state != "shadow":
        typer.echo(
            f"Model {model_id!r} is in state {current_state!r}, not 'shadow' -- refusing to promote. "
            "Only a model that has already passed the gate (state='shadow') can be promoted to 'live'."
        )
        store.close()
        raise typer.Exit(code=1)

    store.update_model_lifecycle(model_id, "live", promoted_at=datetime.now(timezone.utc))
    store.close()
    typer.echo(f"Promoted {model_id!r}: shadow -> live")


@app.command()
def predict(
    horizon: int = typer.Option(20),
    lifecycle: str = typer.Option("shadow", help="Which model to use: 'live' or 'shadow'"),
) -> None:
    """Score the most recent available cross-section with the current
    live/shadow model and print the resulting target portfolio."""
    from stocksense.models.registry import load_model

    settings = get_settings()
    store = Store(settings.duckdb_path)

    row = store.con.execute(
        "SELECT * FROM model_registry WHERE model_type = ? AND horizon_bars = ? AND lifecycle_state = ? "
        "ORDER BY created_at DESC LIMIT 1",
        [MODEL_TYPE, horizon, lifecycle],
    ).fetchdf()
    if row.empty:
        typer.echo(f"No '{lifecycle}' model found for horizon={horizon}. Run train-candidate first.")
        store.close()
        raise typer.Exit(code=1)

    model_id = row.iloc[0]["model_id"]
    artifact_path = row.iloc[0]["artifact_path"]
    top_n = int(row.iloc[0]["top_n"])
    ranker = load_model(artifact_path)

    candles = store.read_candles()
    store.close()
    candles, quarantined = quarantine_symbols(candles)
    if quarantined:
        typer.echo(f"(quarantined, excluded from scoring: {quarantined})")
    feats = build_features(candles)

    latest_date = feats["date"].max()
    today = feats[feats["date"] == latest_date].dropna(subset=ranker.feature_names_, how="any")
    if today.empty:
        typer.echo("No clean feature rows for the latest date.")
        raise typer.Exit(code=1)

    scores = pd.Series(ranker.predict(today[ranker.feature_names_]).values, index=today["symbol"].values)
    weights = target_weights_top_n(scores, top_n)
    weights = weights[weights > 0].sort_values(ascending=False)

    typer.echo(f"\nModel: {model_id}  ({lifecycle}, horizon={horizon}b, top_n={top_n})")
    typer.echo(f"As of: {latest_date.date()}")
    typer.echo("\nTarget portfolio:")
    for symbol, w in weights.items():
        typer.echo(f"  {symbol:>12}  score={scores[symbol]:+.5f}  weight={w:.2%}")


@app.command()
def registry() -> None:
    """List everything in the model registry."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    df = store.read_model_registry()
    store.close()
    if df.empty:
        typer.echo("Model registry is empty.")
        return
    cols = ["model_id", "horizon_bars", "top_n", "lifecycle_state", "gate_decision", "created_at"]
    typer.echo(df[cols].to_string(index=False))


@app.command("statement-ingest")
def statement_ingest(file: str = typer.Argument(..., help="Path to a broker tradebook CSV/XLSX export")) -> None:
    """Parse a broker statement, normalize to canonical trades, and store
    it. Content-hashed: re-ingesting the same file is a no-op."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    path = Path(file)
    if not path.exists():
        typer.echo(f"File not found: {file}")
        raise typer.Exit(code=1)

    h = file_hash(path)
    existing = store.find_statement_by_hash(h)
    if not existing.empty:
        typer.echo(f"Already ingested (statement_id={existing.iloc[0]['statement_id']}), skipping.")
        store.close()
        return

    parser = detect_parser(path)
    if parser is None:
        typer.echo(f"Could not detect broker format for {file}. Supported: zerodha, upstox.")
        store.close()
        raise typer.Exit(code=1)

    trades = parser.parse(path)
    statement_id = f"{parser.broker}-{h[:12]}"
    store.insert_statement(
        {
            "statement_id": statement_id, "broker": parser.broker, "statement_type": "tradebook",
            "file_path": str(path), "file_hash": h,
            "period_start": trades["trade_date"].min() if len(trades) else None,
            "period_end": trades["trade_date"].max() if len(trades) else None,
            "ingested_at": datetime.now(timezone.utc), "row_count": len(trades), "parse_status": "ok",
        }
    )
    trades["statement_id"] = statement_id
    store.write_trades(trades)
    store.close()
    typer.echo(f"Ingested {len(trades)} trades from {parser.broker} statement (statement_id={statement_id}).")


@app.command()
def kundli(broker: Optional[str] = typer.Option(None, help="Filter to one broker's trades, or all if omitted")) -> None:
    """Generate the Kundli report: behavioral diagnostics + counterfactuals
    + narrated verdict, from all ingested statements."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    trades = store.read_trades(broker=broker)
    if trades.empty:
        typer.echo("No trades ingested yet. Run `statement-ingest <file>` first.")
        store.close()
        raise typer.Exit(code=1)

    positions = reconstruct_positions(trades)
    if positions.empty:
        typer.echo("No completed (closed) positions found — all trades appear to still be open.")
        store.close()
        raise typer.Exit(code=1)

    # AUDIT FIX: positions were reconstructed in memory and handed to
    # generate_kundli, but never persisted -- store.write_positions was
    # never called anywhere on this path. The `positions` table stayed
    # permanently empty regardless of how many times kundli ran, which
    # silently starved /api/summary (the desktop dashboard's P&L panel)
    # and optimizer/tax.py (nothing to compute tax on). Idempotent via
    # write_positions' ON CONFLICT DO NOTHING -- safe to call every run.
    store.write_positions(positions)

    result = generate_kundli(positions, store=store)
    store.close()

    typer.echo(f"\n=== KUNDLI (run_id={result['run_id']}) ===")
    typer.echo(f"Positions analyzed: {result['fact_sheet']['n_positions']}")
    typer.echo(f"Net P&L: {result['fact_sheet']['total_net_pnl']:,.2f}")
    typer.echo(f"Total charges: {result['fact_sheet']['total_charges']:,.2f}")
    if result["agent_status"] != "ok":
        typer.echo(f"(agent status: {result['agent_status']})")
    typer.echo("\n" + result["narrative"])


@app.command("record-predictions")
def record_predictions_cmd(
    horizon: int = typer.Option(20, help="Prediction horizon in trading bars"),
    lifecycle: str = typer.Option("live", help="Which model to score with: 'live' or 'shadow'"),
) -> None:
    """The reconcile loop, step 1: score today's cross-section with the
    current live (or shadow) model and freeze the predictions to the
    predictions table. Closes CRITICAL-1 -- this is the write that never
    happened before."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    result = record_predictions(store, horizon_bars=horizon, lifecycle=lifecycle)
    store.close()

    if result is None:
        typer.echo(f"No '{lifecycle}' model registered for horizon={horizon}, or no clean data to score. Nothing recorded.")
        raise typer.Exit(code=1)

    typer.echo(f"Recorded {result.n_predictions} predictions (run_id={result.run_id}, model={result.model_id}, as_of={result.as_of_date})")


@app.command("grade")
def grade_cmd(horizon: int = typer.Option(20, help="Prediction horizon in trading bars")) -> None:
    """The reconcile loop, step 2: grade every prediction whose horizon
    has actually elapsed against realized relative forward return,
    computed with the same function training uses."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    result = grade_matured_predictions(store, horizon_bars=horizon)
    store.close()

    typer.echo(f"Graded {result.n_graded} predictions.")
    if result.n_graded:
        typer.echo(f"Direction correct: {result.n_correct_direction}/{result.n_graded} ({result.n_correct_direction / result.n_graded:.0%})")
        typer.echo(f"Mean absolute error: {result.mean_abs_error:.5f}")


@app.command("serve")
def serve_cmd(port: Optional[int] = typer.Option(None, help="Port to bind (default: auto-select from 8420)")) -> None:
    """Run the desktop control room's local JSON API. Bound to 127.0.0.1
    only -- this serves private trading data and must never be reachable
    from the network. The Electron shell (desktop/) launches this as a
    child process; running it standalone here is for direct use or
    testing without Electron in the loop."""
    run_server(port=port)


@app.command("backfill-nse-archive")
def backfill_nse_archive_cmd(
    start: str = typer.Option(..., help="Start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD"),
    kind: str = typer.Option("cm", help="'cm' (equity), 'delivery', or 'fo'"),
) -> None:
    """Backfill NSE bhavcopy into the store, day by day, genuinely
    resumable: each day is written to the database as soon as it's
    fetched (fetch_range is a generator, consumed lazily here), not
    batched until the whole range finishes -- interrupting this command
    at any point keeps every day already fetched both on disk (content-
    hash cached) and in the database, and re-running the same range
    replays quickly through the cached prefix and continues writing from
    there. A full 2001-2026 CM backfill is ~6,500 trading days; at the
    polite 1 req/sec rate this is a multi-hour job -- safe to stop and
    resume across sessions."""
    from datetime import datetime as _dt

    start_d = _dt.strptime(start, "%Y-%m-%d").date()
    end_d = _dt.strptime(end, "%Y-%m-%d").date()

    settings = get_settings()
    store = Store(settings.duckdb_path)

    total_days = max(1, len(pd.bdate_range(start_d, end_d)))  # business days as the progress denominator; a few public holidays inside it don't materially skew a progress percentage
    results = fetch_range(start_d, end_d, kind=kind)
    n_ok, n_holiday = 0, 0
    for i, (d, df) in enumerate(results, start=1):
        if df is None:
            n_holiday += 1
        else:
            if kind == "cm":
                store.write_bhavcopy_eq(df)
            elif kind == "delivery":
                store.write_bhavcopy_delivery(df)
            elif kind == "fo":
                store.write_bhavcopy_fo(df)
            n_ok += 1
        _emit_progress(i, total_days)

    store.close()
    _emit_progress(total_days, total_days, force=True)
    typer.echo(f"Backfilled {n_ok} trading day(s) of '{kind}' data ({n_holiday} holidays/weekends skipped).")


@app.command("backfill-corporate-actions")
def backfill_corporate_actions_cmd(
    start: str = typer.Option(..., help="Start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD"),
) -> None:
    """Backfill NSE corporate actions (splits/bonuses/dividends) into the
    store, quarterly window by window, genuinely resumable in the same
    shape as backfill-nse-archive: fetch_ca_range yields each window as
    it's fetched and this loop writes it immediately, so an interruption
    keeps every window already fetched. Required before any bhavcopy
    price is usable as a feature -- see data/adjust.py."""
    from datetime import datetime as _dt

    start_d = _dt.strptime(start, "%Y-%m-%d").date()
    end_d = _dt.strptime(end, "%Y-%m-%d").date()

    settings = get_settings()
    store = Store(settings.duckdb_path)

    from stocksense.data.corporate_actions import _WINDOW_DAYS
    total_windows = max(1, -(-(end_d - start_d).days // (_WINDOW_DAYS + 1)))  # ceil division, matches fetch_ca_range's own windowing

    n_windows = n_actions = n_unparsed = 0
    for raw in fetch_ca_range(start_d, end_d):
        parsed = parse_ca_frame(raw)
        if not parsed.empty:
            store.write_corporate_actions(parsed)
            n_actions += len(parsed)
            n_unparsed += int((parsed["parse_status"] == "unparsed").sum())
        n_windows += 1
        _emit_progress(n_windows, total_windows)

    store.close()
    _emit_progress(n_windows, total_windows, force=True)
    typer.echo(f"Backfilled {n_windows} window(s), {n_actions} corporate action(s) ({n_unparsed} unparsed).")


@app.command("backfill-intraday")
def backfill_intraday_cmd(
    start: str = typer.Option("2022-01-01", help="Start date, YYYY-MM-DD (clamped to Upstox's 2022-01 1-min history floor)"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD"),
    top_n: int = typer.Option(250, help="Number of most-liquid EQ symbols (by universe_as_of on the latest bhavcopy date) to fetch"),
) -> None:
    """Backfill Upstox 1-minute intraday bars into the store (Phase E1),
    symbol by symbol, month-window by month-window, genuinely resumable
    in the same shape as backfill-nse-archive: fetch_range is a generator
    consumed lazily here, and each window is written to the DB as soon as
    it's fetched, not batched until the whole run finishes. Interrupting
    this command loses no progress -- content-hash disk cache means a
    re-run never re-downloads a window, and the DB already has everything
    written before the process died (the exact resumability property
    fixed in 731c262 for the daily bhavcopy backfill).

    The symbol universe is resolved once at the START of this command
    (today's most-liquid names), not re-resolved per historical date --
    intraday research needs the SAME symbols consistently fetched across
    the whole range; point-in-time universe filtering happens later, at
    feature-build time, via universe_pit.filter_to_point_in_time_universe."""
    from datetime import datetime as _dt

    start_d = _dt.strptime(start, "%Y-%m-%d").date()
    end_d = _dt.strptime(end, "%Y-%m-%d").date()

    settings = get_settings()
    store = Store(settings.duckdb_path)

    latest_date = store.con.execute("SELECT MAX(date) FROM bhavcopy_eq").fetchone()[0]
    if latest_date is None:
        store.close()
        typer.echo("bhavcopy_eq is empty -- run backfill-nse-archive first to establish a liquid universe.")
        raise typer.Exit(code=1)

    turnover = store.con.execute(
        """
        SELECT symbol, AVG(turnover_inr) AS avg_turnover
        FROM bhavcopy_eq
        WHERE series = 'EQ' AND date >= ? AND date <= ?
        GROUP BY symbol ORDER BY avg_turnover DESC LIMIT ?
        """,
        [latest_date - timedelta(days=60), latest_date, top_n],
    ).fetchdf()
    symbols = sorted(turnover["symbol"].tolist())
    typer.echo(f"Resolved {len(symbols)} most-liquid EQ symbol(s) as of {latest_date}.")

    instrument_map = resolve_symbol_map(symbols)
    store.write_upstox_instrument_map(instrument_map)
    n_resolved = int(instrument_map["resolved"].sum())
    n_unresolved = len(instrument_map) - n_resolved
    if n_unresolved:
        unresolved_syms = instrument_map.loc[~instrument_map["resolved"], "symbol"].tolist()
        typer.echo(f"WARNING: {n_unresolved} symbol(s) unmapped to an Upstox instrument (excluded, not silently skipped): {unresolved_syms}")

    from stocksense.data.upstox_intraday import EARLIEST_1MIN_DATE, _month_windows
    clamped_start = max(start_d, EARLIEST_1MIN_DATE)
    windows_per_symbol = max(1, sum(1 for _ in _month_windows(clamped_start, end_d)))
    total_windows = max(1, n_resolved * windows_per_symbol)

    n_windows = n_bars = n_failed = 0
    for symbol, window_start, window_end, df in fetch_intraday_range(instrument_map, start_d, end_d):
        n_windows += 1
        if df is None:
            n_failed += 1
        elif not df.empty:
            store.write_intraday_bars(df)
            n_bars += len(df)
        _emit_progress(n_windows, total_windows)

    store.close()
    _emit_progress(n_windows, total_windows, force=True)
    typer.echo(
        f"Backfilled {n_bars} intraday bar(s) across {n_windows} symbol-window(s) "
        f"for {n_resolved} symbol(s) ({n_failed} window(s) failed and should be re-run)."
    )


@app.command("universe-as-of")
def universe_as_of_cmd(as_of: str = typer.Option(..., help="Date, YYYY-MM-DD")) -> None:
    """Point-in-time tradeable universe as of a given date, from ingested
    bhavcopy data (requires `backfill-nse-archive --kind cm` covering
    that date's trailing lookback window first)."""
    from datetime import datetime as _dt

    d = _dt.strptime(as_of, "%Y-%m-%d").date()
    settings = get_settings()
    store = Store(settings.duckdb_path)
    symbols = universe_as_of(store, d)
    store.close()

    if not symbols:
        typer.echo(f"No symbols found for {as_of} -- has bhavcopy data been backfilled for this date range?")
        raise typer.Exit(code=1)
    typer.echo(f"{len(symbols)} symbols tradeable as of {as_of}:")
    typer.echo(", ".join(symbols))


@app.command("index-corpus")
def index_corpus_cmd() -> None:
    """Index docs/, research/, and skills/ into the RAG corpus. Safe to
    re-run any time -- content-hashed, so unchanged files are a no-op
    and only genuinely changed ones get re-chunked."""
    settings = get_settings()
    store = Store(settings.duckdb_path)

    # Check embedding availability ONCE, not once per chunk -- each
    # attempt costs a real ~2s Ollama round trip even to determine
    # "unavailable," which turns indexing dozens of files into minutes
    # for no benefit if the answer is the same for every one of them.
    embed_ok = embeddings_available()
    typer.echo(f"Embeddings {'available' if embed_ok else 'unavailable (FTS-only mode)'}.")

    n_indexed = 0
    n_skipped = 0
    for pattern, source_type in [("docs/*.md", "docs"), ("research/*.md", "research"), ("skills/*/SKILL.md", "skill")]:
        for path in sorted(REPO_ROOT.glob(pattern)):
            content = path.read_text(encoding="utf-8", errors="replace")
            result = index_document(
                store, source_type, str(path.relative_to(REPO_ROOT)).replace("\\", "/"), path.stem, content,
                embed_chunks=embed_ok,
            )
            if result["reindexed"]:
                n_indexed += 1
            else:
                n_skipped += 1

    rebuild_fts_index(store)
    store.close()
    typer.echo(f"Indexed {n_indexed} document(s), {n_skipped} unchanged (skipped).")


@app.command("ask")
def ask_cmd(question: str = typer.Argument(..., help="Question to ask the RAG agent")) -> None:
    """Query the RAG corpus. Answers only from indexed content, with
    citations -- run `index-corpus` first if the corpus is empty."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    result = rag_ask(question, store)
    store.close()

    typer.echo(f"\n{result['answer']}\n")
    if result["citations"]:
        typer.echo("Sources:")
        for c in result["citations"]:
            typer.echo(f"  [{c['index']}] {c['title']} ({c['source_ref']})")


@app.command("tax-summary")
def tax_summary_cmd(
    fy_start: str = typer.Option(..., help="Financial year start date, e.g. 2024-04-01"),
    fy_end: str = typer.Option(..., help="Financial year end date, e.g. 2025-03-31"),
    ltcg_exemption_used: float = typer.Option(0.0, help="LTCG exemption already used elsewhere this FY (e.g. mutual funds)"),
) -> None:
    """Realized-gains tax summary for one financial year, from ingested
    statement positions. Statutory rates only (STCG 20%, LTCG 12.5%
    above Rs 1.25L exemption/FY, 4% cess) -- not tax advice, and does
    not account for F&O business income or other income heads."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    trades = store.read_trades()
    store.close()

    if trades.empty:
        typer.echo("No trades ingested yet. Run `statement-ingest <file>` first.")
        raise typer.Exit(code=1)

    positions = reconstruct_positions(trades)
    fy_positions = positions[
        (positions["close_date"].astype(str) >= fy_start) & (positions["close_date"].astype(str) <= fy_end)
    ]
    if fy_positions.empty:
        typer.echo(f"No positions closed between {fy_start} and {fy_end}.")
        raise typer.Exit(code=1)

    summary = compute_tax_liability(fy_positions, ltcg_exemption_used_this_fy=ltcg_exemption_used)
    typer.echo(f"\n=== TAX SUMMARY: {fy_start} to {fy_end} ===")
    typer.echo(f"Total STCG (realized gains only): Rs {summary.total_stcg:,.2f}")
    typer.echo(f"Total LTCG (realized gains only): Rs {summary.total_ltcg:,.2f}")
    typer.echo(f"LTCG exemption used: Rs {summary.ltcg_exemption_used:,.2f}")
    typer.echo(f"Taxable LTCG: Rs {summary.taxable_ltcg:,.2f}")
    typer.echo(f"STCG tax (20%): Rs {summary.stcg_tax:,.2f}")
    typer.echo(f"LTCG tax (12.5%): Rs {summary.ltcg_tax:,.2f}")
    typer.echo(f"Cess (4%): Rs {summary.cess:,.2f}")
    typer.echo(f"TOTAL TAX: Rs {summary.total_tax:,.2f}")
    typer.echo("\n(Statutory rates only, not tax advice. Excludes F&O/intraday business income and other income heads.)")


@app.command("reconcile")
def reconcile_cmd(
    horizon: int = typer.Option(20, help="Prediction horizon in trading bars"),
    lifecycle: str = typer.Option("live", help="Which model to score with: 'live' or 'shadow'"),
    cap_band: Optional[str] = typer.Option(
        None, "--cap-band",
        help="MUST match the cap band the target model was trained with (train-candidate's own "
             "--cap-band) -- otherwise the ledger silently scores/grades against a different "
             "universe than the model was trained and gated on. See data/universe_pit.py's CAP_BANDS.",
    ),
) -> None:
    """The reconcile loop, run as a harness graph: grade matured
    predictions, then record today's. Idempotency-keyed by calendar
    date, so running this twice in one day is a no-op the second time --
    the property docs/05-nightly-pipeline.md requires of every step."""
    settings = get_settings()
    turnover_rank_band = _resolve_cap_band(cap_band, settings)
    store = Store(settings.duckdb_path)
    graph = build_reconcile_graph(store, horizon_bars=horizon, lifecycle=lifecycle,
                                   turnover_rank_band=turnover_rank_band, settings=settings)
    result = run_graph(graph, store)
    store.close()

    for outcome in result.outcomes:
        typer.echo(f"  [{outcome.status:>9}] {outcome.name}")
    if not result.all_succeeded:
        typer.echo(f"Failed: {result.failed_nodes()}")
        raise typer.Exit(code=1)

    grade_out = result.context.get("grade_matured_predictions", {})
    record_out = result.context.get("record_predictions", {})
    if grade_out:
        typer.echo(f"Graded: {grade_out.get('n_graded', 0)}")
    if record_out.get("recorded"):
        typer.echo(f"Recorded: {record_out.get('n_predictions', 0)} predictions (run_id={record_out.get('run_id')})")


@app.command("retrain-weekly")
def retrain_weekly_cmd(
    horizon: int = typer.Option(10, help="Prediction horizon in trading bars"),
    top_n: int = typer.Option(10, help="Portfolio size (top-N by rank)"),
    cost_bps: float = typer.Option(25.0, help="Round-trip cost assumption for gate evaluation, bps"),
    cap_band: Optional[str] = typer.Option(
        None, "--cap-band",
        help="Should match the cap band the live model this feeds runs on (see train-candidate's "
             "--cap-band) -- a freshly retrained candidate on the WRONG universe would still "
             "register and gate, just against evidence that doesn't match what's live.",
    ),
) -> None:
    """The weekly retrain loop (Phase G4), run as a harness graph:
    walk-forward evaluate, gate, and register a fresh candidate.
    Idempotency-keyed by ISO week, so running this more than once in
    the same week is a no-op after the first success -- the weekly
    analogue of `reconcile`'s daily idempotency."""
    settings = get_settings()
    turnover_rank_band = _resolve_cap_band(cap_band, settings)
    store = Store(settings.duckdb_path)
    graph = build_weekly_retrain_graph(store, horizon=horizon, top_n=top_n, cost_bps=cost_bps,
                                        turnover_rank_band=turnover_rank_band, settings=settings)
    result = run_graph(graph, store)
    store.close()

    for outcome in result.outcomes:
        typer.echo(f"  [{outcome.status:>9}] {outcome.name}")
    if not result.all_succeeded:
        typer.echo(f"Failed: {result.failed_nodes()}")
        raise typer.Exit(code=1)

    train_out = result.context.get("train_candidate", {})
    if train_out:
        typer.echo(f"Model: {train_out.get('model_id')}")
        typer.echo(f"Gate passed: {train_out.get('gate_passed')} ({train_out.get('gate_reason')})")
        typer.echo(f"Lifecycle state: {train_out.get('lifecycle_state')}")


@foreman_app.command("run")
def foreman_run(
    goal: str = typer.Argument(..., help="Plain-language goal, e.g. 'add a determinism test for the new parser'"),
    max_attempts: int = typer.Option(3, help="Max execution attempts before marking the goal blocked"),
) -> None:
    """Plan, execute, verify, and route (auto-merge or PR) a single goal.
    On-demand: runs once for this goal and exits. Never touches
    protected paths (evaluation/gate.py, walkforward.py, cost_model.py,
    leakage/determinism/gate tests, preregistration files) -- those
    always route to a human-reviewed PR regardless of how green
    everything else is."""
    settings = get_settings()
    store = Store(settings.duckdb_path)

    budget_status = check_budget(store)
    if not budget_status.within_budget:
        typer.echo(f"Blocked by daily budget: {budget_status.reason}")
        store.close()
        raise typer.Exit(code=1)

    # AUDIT FIX: this invocation previously never counted against its own
    # budget (invocations was never incremented anywhere), so check_budget
    # compared 0 against the cap forever and the cap could never fire.
    store.increment_budget(date.today(), invocations=1)

    # AUDIT FIX: insert_goal had zero call sites before this, so
    # record_goal_result's later UPDATE was always a no-op against a row
    # that never existed and the goals table stayed permanently empty.
    # Inserted here (not inside execute_goal) to keep execute_goal free
    # of store write side effects beyond agent_runs/job_runs, matching
    # record_goal_result's own stated separation-of-concerns reason.
    goal_id = str(uuid.uuid4())[:12]
    store.insert_goal({
        "goal_id": goal_id, "source": "user", "prompt": goal, "status": "executing",
        "priority": 5, "created_at": datetime.now(timezone.utc), "completed_at": None,
        "parent_goal_id": None, "result_summary": None,
    })

    outcome = execute_goal(goal, store, goal_id=goal_id, max_attempts=max_attempts)

    changed: list[str] = []
    if outcome.run_result is not None:
        changed = [
            v.get("data", {}).get("path")
            for v in outcome.run_result.context.values()
            if isinstance(v, dict) and v.get("tool") == "write_patch"
        ]
        changed = [c for c in changed if c]
        if changed:
            findings = red_team(changed)
            if findings:
                typer.echo("\n=== ADVERSARY FINDINGS ===")
                for f in findings:
                    typer.echo(f"  [{f.severity}] {f.check} in {f.file}: {f.detail}")
                if has_blocking_finding(findings) and outcome.status == "pushed":
                    typer.echo("Blocking finding on an otherwise-pushed result -- downgrading to blocked. Review required.")
                    outcome = type(outcome)(outcome.goal_id, "blocked", "adversary found a blocking issue post-push-check", outcome.branch, outcome.verification, outcome.run_result)

    record_goal_result(store, outcome)
    record_ledger_entries(store, goal_id, outcome)
    record_protected_violations(store, goal_id, outcome, changed)
    store.close()

    typer.echo(f"\n=== FOREMAN: {outcome.status.upper()} ===")
    typer.echo(f"goal_id: {outcome.goal_id}")
    typer.echo(f"reason: {outcome.reason}")
    if outcome.branch:
        typer.echo(f"branch: {outcome.branch}")
    if outcome.verification:
        for g in outcome.verification.gates:
            typer.echo(f"  [{'PASS' if g.passed else 'FAIL'}] {g.name}")


@foreman_app.command("assess")
def foreman_assess() -> None:
    """Self-assessment only: read project state, propose ranked goals,
    execute nothing. Use this to see what the Foreman WOULD work on."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    goals = propose_goals(store)
    store.close()

    if not goals:
        typer.echo("No goals proposed (or the agent response could not be parsed).")
        return

    typer.echo("=== PROPOSED GOALS ===")
    for g in sorted(goals, key=lambda x: x.get("priority", 99)):
        typer.echo(f"\n[{g.get('priority', '?')}] {g.get('goal')}")
        typer.echo(f"    reason: {g.get('reason', '(none given)')}")


@foreman_app.command("status")
def foreman_status() -> None:
    """Recent goals, their outcomes, and today's budget usage."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    goals = store.read_goals()
    from datetime import date
    budget = store.get_or_create_budget(date.today())
    store.close()

    typer.echo(f"Today's budget: {budget['invocations']} invocations, {budget['goals_completed']} goals completed")
    if goals.empty:
        typer.echo("No goals recorded yet.")
        return
    typer.echo("\n=== RECENT GOALS ===")
    for _, row in goals.head(20).iterrows():
        typer.echo(f"  [{row['status']:>10}] {row['prompt'][:70]}")


@foreman_app.command("ledger")
def foreman_ledger(goal_id: Optional[str] = typer.Option(None, help="Filter to one goal")) -> None:
    """The build ledger: every tool invocation the Foreman has made."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    ledger = store.read_ledger(goal_id=goal_id)
    store.close()

    if ledger.empty:
        typer.echo("Ledger is empty.")
        return
    for _, row in ledger.iterrows():
        typer.echo(f"  [{row['verdict']:>20}] {row['tool']:>15} — {row['task_name']}")


if __name__ == "__main__":
    app()
