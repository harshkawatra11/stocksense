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
    min_pct_folds_positive: float = typer.Option(0.6),
) -> None:
    """Walk-forward evaluate a candidate, run it through the gate, train
    the final full-history artifact, and register the outcome — win or
    lose — in the model registry."""
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
        criteria=GateCriteria(min_pct_folds_positive=min_pct_folds_positive),
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


if __name__ == "__main__":
    app()
