"""
Phase G4: the train-candidate-through-gate-through-registry sequence,
factored out of cli/main.py's `train_candidate` command so it can run
as a harness.Graph node (harness/retrain.py's weekly retrain graph) and
not just as an interactively-triggered CLI command -- the same
extraction Phase G2 did for record_predictions/grade_matured_predictions
(harness/loops.py), for the same reason: a nightly/weekly loop needs a
plain function it can call, not a typer command that echoes to stdout
and calls sys.exit.

cli/main.py's `train_candidate` command becomes a thin wrapper around
train_candidate_core -- no behavior change, verified by the existing
CLI tests continuing to pass unmodified.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass

from stocksense.core.config import get_settings
from stocksense.data.loader import load_features_and_labels
from stocksense.data.store import Store
from stocksense.evaluation.backtest import simulate_portfolio, train_and_score_fold
from stocksense.evaluation.gate import GateCriteria, GateVerdict, apply_gate_decision, evaluate_gate
from stocksense.evaluation.walkforward import make_folds
from stocksense.models.ranker import CrossSectionalRanker, QuantileRanker, RankerConfig

MODEL_TYPE = "cross_sectional_ranker"
QUANTILE_ARTIFACT_NAME = "quantile_model.joblib"

log = structlog.get_logger(__name__)


def _fit_and_save_quantile_ranker(model_id: str, ranker_config: RankerConfig, full_merged, fcols, horizon: int, settings) -> None:
    """Phase H2: a QuantileRanker (models/ranker.py, Phase G3), fit on
    the SAME full-history data as the point CrossSectionalRanker above,
    saved as a sibling artifact next to it -- never fed into the gate.
    Every PASS in research/verdict_bhavcopy_rerun.md was earned by
    CrossSectionalRanker's walk-forward folds; this is purely additive,
    so record_predictions (harness/loops.py) can populate a real
    predicted_return/confidence band instead of leaving confidence
    permanently NULL. Failure here (e.g. too little data) is logged and
    swallowed -- the point model and gate verdict already registered
    successfully are the load-bearing outputs; a missing quantile band
    degrades to today's NULL-confidence behavior, not a failed run."""
    import joblib

    try:
        quantile_ranker = QuantileRanker(ranker_config)
        quantile_ranker.fit(full_merged[fcols], full_merged[f"fwd_ret_{horizon}b_rel"])
        model_dir = settings.parquet_dir.parent / "models" / model_id
        joblib.dump(quantile_ranker, model_dir / QUANTILE_ARTIFACT_NAME)
    except Exception:
        log.warning("quantile_ranker_fit_failed", model_id=model_id, exc_info=True)


@dataclass(frozen=True)
class TrainCandidateResult:
    model_id: str | None
    verdict: GateVerdict | None
    lifecycle_state: str | None  # 'shadow' (passed) | 'archived' (failed) | None (no fold results at all)
    n_fold_results: int


def train_candidate_core(
    horizon: int, top_n: int, cost_bps: float, store: Store,
    settings=None, turnover_rank_band: tuple[float, float] | None = None,
) -> TrainCandidateResult:
    """Walk-forward evaluate a candidate, run it through the gate, train
    the final full-history artifact, and register the outcome — win or
    lose — in the model registry.

    Gate criteria are NOT configurable on purpose: this always uses the
    pre-registered defaults in GateCriteria
    (research/gate_criteria_preregistration.md). Allowing ad-hoc
    overrides here would recreate the exact evaluator-overfitting
    failure that pre-registration exists to prevent — a caller could
    keep loosening criteria until a candidate passes. Experiment with
    alternative criteria in a research script, never here.

    `turnover_rank_band`: see universe_pit.py's docstring — restricts
    training to a liquidity-rank-proxied cap band (Phase G1 found the
    edge concentrated in mid/small caps, not large — see
    research/verdict_bhavcopy_rerun.md), only meaningful when
    settings.use_point_in_time_universe is set.

    Returns a TrainCandidateResult with lifecycle_state=None (and no
    model_id) if there were no fold results at all — insufficient
    data for this horizon/universe, not an error the caller should
    treat as a crash.
    """
    settings = settings or get_settings()

    candles, feats, fcols, labeled = load_features_and_labels(
        horizon, turnover_rank_band=turnover_rank_band, settings=settings, store=store,
    )

    from stocksense.core.calendar import trading_days_index
    trading_dates = trading_days_index(feats["date"])
    test_window = max(21, horizon * 12)
    folds = make_folds(trading_dates, horizon_bars=horizon, test_window_bars=test_window)

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
        return TrainCandidateResult(model_id=None, verdict=None, lifecycle_state=None, n_fold_results=0)

    incumbent = store.get_live_model(MODEL_TYPE, horizon)
    incumbent_alpha = None
    if not incumbent.empty:
        import json
        incumbent_metrics = json.loads(incumbent.iloc[0]["metrics_json"])
        incumbent_alpha = incumbent_metrics.get("mean_alpha_net")

    verdict = evaluate_gate(
        fold_results,
        criteria=GateCriteria(),  # pre-registered defaults, always — see docstring above
        incumbent_mean_alpha_net=incumbent_alpha,
    )

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

    from stocksense.models.registry import register_candidate

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
    _fit_and_save_quantile_ranker(model_id, ranker_config, full_merged, fcols, horizon, settings)

    return TrainCandidateResult(
        model_id=model_id, verdict=verdict,
        lifecycle_state="shadow" if verdict.passed else "archived",
        n_fold_results=len(fold_results),
    )
