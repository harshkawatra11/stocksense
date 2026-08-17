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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import typer

from stocksense.core.calendar import trading_days_index
from stocksense.core.config import REPO_ROOT, get_settings
from stocksense.data.store import Store
from stocksense.data.validate import quarantine_symbols
from stocksense.evaluation.backtest import simulate_portfolio, train_and_score_fold
from stocksense.evaluation.gate import GateCriteria, apply_gate_decision, evaluate_gate
from stocksense.evaluation.walkforward import make_folds
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import CrossSectionalRanker, RankerConfig
from stocksense.models.registry import register_candidate
from stocksense.portfolio.construct import target_weights_top_n
from stocksense.statements.parsers import detect_parser
from stocksense.statements.parsers.base import file_hash
from stocksense.statements.positions import reconstruct_positions
from stocksense.statements.report import generate_kundli
from stocksense.foreman.adversary import red_team, has_blocking_finding
from stocksense.foreman.assess import propose_goals
from stocksense.foreman.budget import check_budget
from stocksense.foreman.executor import execute_goal, record_goal_result
from stocksense.harness.loops import build_reconcile_graph, grade_matured_predictions, record_predictions
from stocksense.harness.runner import run_graph
from stocksense.optimizer.tax import compute_tax_liability
from stocksense.rag.agent import ask as rag_ask
from stocksense.rag.embed import embeddings_available
from stocksense.rag.index import index_document, rebuild_fts_index
from stocksense.data.nse_archive import fetch_range
from stocksense.data.corporate_actions import fetch_ca_range, parse_ca_frame
from stocksense.data.adjust import read_adjusted_candles, quarantine_unexplained_jumps
from stocksense.data.universe_pit import universe_as_of, filter_to_point_in_time_universe
from stocksense.server.run import run as run_server

foreman_app = typer.Typer(help="The Foreman: self-building harness")

log = structlog.get_logger(__name__)
app = typer.Typer(help="StockSense nightly pipeline CLI")
app.add_typer(foreman_app, name="foreman")

MODEL_TYPE = "cross_sectional_ranker"


def _load_candles(settings, store) -> pd.DataFrame:
    """Source switch (docs/17-data-spine.md, Phase D2): 'candles'
    preserves the exact Phase 0 path (yfinance, fixed 98-symbol
    universe) unchanged, so those numbers stay reproducible. 'bhavcopy'
    pulls point-in-time NSE data through the corporate-action adjustment
    layer (data/adjust.py) -- the widest candidate symbol set is read
    first, then narrowed to the point-in-time-tradeable universe per
    date if use_point_in_time_universe is set, rather than narrowed to
    a hardcoded list up front."""
    if settings.price_source == "candles":
        return store.read_candles()

    if settings.price_source != "bhavcopy":
        raise ValueError(f"unknown price_source {settings.price_source!r}, expected 'candles' or 'bhavcopy'")

    bounds = store.con.execute("SELECT MIN(date), MAX(date) FROM bhavcopy_eq").fetchone()
    if bounds[0] is None:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    min_d, max_d = bounds
    symbols = store.con.execute("SELECT DISTINCT symbol FROM bhavcopy_eq WHERE series = 'EQ'").fetchdf()["symbol"].tolist()
    candles = read_adjusted_candles(store, symbols, min_d, max_d, basis=settings.return_basis)

    if settings.use_point_in_time_universe:
        candles = filter_to_point_in_time_universe(
            store, candles, min_turnover_inr=settings.min_avg_daily_turnover_inr, min_price_inr=settings.min_price_inr
        )
    return candles


def _load_features_and_labels(horizon: int):
    settings = get_settings()
    store = Store(settings.duckdb_path)
    candles = _load_candles(settings, store)

    # Quarantine symbols with a detected adjustment anomaly before
    # anything downstream touches them. The detector is source-dependent
    # and this branch matters: data/validate.quarantine_symbols compares
    # adj_close to raw close, which is only a bug signal when close is
    # ALREADY split-adjusted at the source (true for yfinance's `candles`
    # — see the ADANIENT 8.6x-jump bug in research/phase0_verdict.md's
    # "Run 3"). For bhavcopy-sourced data close is raw by construction
    # (data/corporate_actions.py), so adj_close/close legitimately steps
    # at every real split/bonus — applying the yfinance check there is
    # itself a bug: it quarantined RELIANCE, TCS, and ~600 other names
    # for their genuine corporate actions, found live during Phase D2.
    # The bhavcopy-appropriate check instead flags adjusted-price jumps
    # with NO matching corporate-action record.
    if settings.price_source == "bhavcopy":
        candles, quarantined = quarantine_unexplained_jumps(store, candles)
    else:
        candles, quarantined = quarantine_symbols(candles)
    store.close()
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined, price_source=settings.price_source)

    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]

    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)
    return candles, feats, fcols, labeled


@app.command()
def train_candidate(
    horizon: int = typer.Option(20, help="Prediction horizon in trading bars"),
    top_n: int = typer.Option(20, help="Portfolio size (top-N by rank)"),
    cost_bps: float = typer.Option(25.0, help="Round-trip cost assumption for gate evaluation, bps"),
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
    """
    settings = get_settings()
    store = Store(settings.duckdb_path)

    candles, feats, fcols, labeled = _load_features_and_labels(horizon)
    trading_dates = trading_days_index(feats["date"])
    test_window = max(21, horizon * 12)
    folds = make_folds(trading_dates, horizon_bars=horizon, test_window_bars=test_window)
    log.info("folds_built", horizon=horizon, n_folds=len(folds))

    ranker_config = RankerConfig(random_state=settings.random_seed)
    fold_results = []
    for fold in folds:
        scored = train_and_score_fold(feats, labeled, fcols, fold, horizon_bars=horizon, ranker_config=ranker_config)
        if scored is None:
            continue
        result = simulate_portfolio(scored, top_n=top_n, round_trip_cost_bps=cost_bps)
        if result is not None:
            fold_results.append(result)

    if not fold_results:
        typer.echo("No fold results produced — insufficient data for this horizon/universe.")
        raise typer.Exit(code=1)

    incumbent = store.get_live_model(MODEL_TYPE, horizon)
    incumbent_alpha = None
    if not incumbent.empty:
        incumbent_metrics = json.loads(incumbent.iloc[0]["metrics_json"])
        incumbent_alpha = incumbent_metrics.get("mean_alpha_net")

    verdict = evaluate_gate(
        fold_results,
        criteria=GateCriteria(),  # pre-registered defaults, always — see docstring above
        incumbent_mean_alpha_net=incumbent_alpha,
    )

    typer.echo(f"\nGate verdict: {'PASS' if verdict.passed else 'FAIL'}")
    typer.echo(f"Reason: {verdict.reason}")
    typer.echo(f"Metrics: {json.dumps(verdict.metrics, indent=2)}")

    # Train the artifact that would actually be deployed: full available
    # history, not just one fold. Walk-forward folds prove out-of-sample
    # validity; this is the model that scores tomorrow's cross-section.
    full_merged = feats.merge(
        labeled[["symbol", "date", f"fwd_ret_{horizon}b", f"fwd_ret_{horizon}b_rel"]],
        on=["symbol", "date"], how="inner",
    ).dropna(subset=[f"fwd_ret_{horizon}b_rel"])

    final_ranker = CrossSectionalRanker(ranker_config)
    final_ranker.fit(full_merged[fcols], full_merged[f"fwd_ret_{horizon}b_rel"])

    metrics = dict(verdict.metrics)
    metrics["fold_alphas"] = [f.alpha_net for f in fold_results]
    metrics["cost_bps_used"] = cost_bps

    model_id = register_candidate(
        final_ranker,
        model_type=MODEL_TYPE,
        horizon_bars=horizon,
        top_n=top_n,
        training_start=str(full_merged["date"].min().date()),
        training_end=str(full_merged["date"].max().date()),
        metrics=metrics,
        store=store,
    )
    apply_gate_decision(model_id, verdict, store)
    typer.echo(f"\nRegistered: {model_id}")
    typer.echo(f"Lifecycle state: {'shadow' if verdict.passed else 'archived'}")
    store.close()


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

    results = fetch_range(start_d, end_d, kind=kind)
    n_ok, n_holiday = 0, 0
    for d, df in results:
        if df is None:
            n_holiday += 1
            continue
        if kind == "cm":
            store.write_bhavcopy_eq(df)
        elif kind == "delivery":
            store.write_bhavcopy_delivery(df)
        elif kind == "fo":
            store.write_bhavcopy_fo(df)
        n_ok += 1

    store.close()
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

    n_windows = n_actions = n_unparsed = 0
    for raw in fetch_ca_range(start_d, end_d):
        parsed = parse_ca_frame(raw)
        if not parsed.empty:
            store.write_corporate_actions(parsed)
            n_actions += len(parsed)
            n_unparsed += int((parsed["parse_status"] == "unparsed").sum())
        n_windows += 1

    store.close()
    typer.echo(f"Backfilled {n_windows} window(s), {n_actions} corporate action(s) ({n_unparsed} unparsed).")


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
) -> None:
    """The reconcile loop, run as a harness graph: grade matured
    predictions, then record today's. Idempotency-keyed by calendar
    date, so running this twice in one day is a no-op the second time --
    the property docs/05-nightly-pipeline.md requires of every step."""
    settings = get_settings()
    store = Store(settings.duckdb_path)
    graph = build_reconcile_graph(store, horizon_bars=horizon, lifecycle=lifecycle)
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

    goal_id = None
    outcome = execute_goal(goal, store, max_attempts=max_attempts)

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
                if has_blocking_finding(findings) and outcome.status == "merged":
                    typer.echo("Blocking finding on an otherwise-merged result -- downgrading to blocked. Review required.")
                    outcome = type(outcome)(outcome.goal_id, "blocked", "adversary found a blocking issue post-merge-check", outcome.branch, outcome.verification, outcome.run_result)

    record_goal_result(store, outcome)
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
