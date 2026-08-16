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
from stocksense.core.config import get_settings
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

log = structlog.get_logger(__name__)
app = typer.Typer(help="StockSense nightly pipeline CLI")

MODEL_TYPE = "cross_sectional_ranker"


def _load_features_and_labels(horizon: int):
    settings = get_settings()
    store = Store(settings.duckdb_path)
    candles = store.read_candles()
    store.close()

    # Quarantine symbols with a detected adjustment-factor discontinuity
    # before anything downstream touches them — a real bug found during
    # Phase 0 stress testing (ADANIENT's adj_close jumped 8.6x day-over-
    # day in 2003 while close barely moved), which fabricated an extreme
    # "return" and inflated one walk-forward fold's alpha. See
    # stocksense.data.validate and research/phase0_verdict.md's "Run 3".
    candles, quarantined = quarantine_symbols(candles)
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined)

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


if __name__ == "__main__":
    app()
