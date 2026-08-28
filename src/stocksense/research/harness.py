"""
Phase K2.2: the reusable sweep runner.

WHY THIS EXISTS. `research/phase0_sweep.py`, `research/bhavcopy_rerun_sweep.py`,
`research/intraday_sweep.py`, and `research/fno_signal_sweep.py` are each an
independent ~200-line script implementing the identical shape: load -> build
folds -> train once per fold -> simulate many (top_n, cost) combinations ->
gate at one pre-registered cost point -> write a CSV + a hand-written verdict
doc. A search loop that evaluates hundreds of candidates cannot afford to
copy-paste a 200-line script per candidate. This module is that shape,
written once, parameterised over WHICH candles/features go in.

REUSES, UNCHANGED: `evaluation.backtest.train_and_score_fold`,
`evaluation.backtest.simulate_portfolio`, `evaluation.walkforward.make_folds`,
`evaluation.gate.evaluate_gate`, `data.loader.load_candles`,
`data.adjust.quarantine_unexplained_jumps`, `data.liquidity.segment_symbols_by_trading_gap`.
None of `evaluation/backtest.py`, `evaluation/gate.py`, or
`evaluation/walkforward.py` (all protected) is modified by this module.

ACCEPTANCE GATE -- do this before trusting a single new result out of this
module: re-run it against `cap_band='full_pit', horizon=10, top_n=10,
cost_bps=25.0` and confirm it reproduces
`research/verdict_bhavcopy_rerun.md`'s own number for that exact combination
-- 25 folds, mean net alpha +0.01672262970534454 (+1.672%/rebalance). See
`research/verify_harness_acceptance.py` for the runnable check (real-database,
several minutes -- not a unit test). If it does not reproduce, this module has
a bug and nothing built on top of it should be trusted yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

from stocksense.data.adjust import quarantine_unexplained_jumps
from stocksense.data.liquidity import segment_symbols_by_trading_gap
from stocksense.data.loader import load_candles
from stocksense.evaluation.backtest import FoldResult, simulate_portfolio, train_and_score_fold
from stocksense.evaluation.gate import GateCriteria, GateVerdict, evaluate_gate
from stocksense.evaluation.walkforward import make_folds
from stocksense.features.engine import build_features, feature_columns
from stocksense.features.registry import apply_registered_factors
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import RankerConfig

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SweepConfig:
    """Every knob a sweep needs, gathered in one place so a caller states its
    grid once rather than writing nested `for` loops by hand. All defaults
    match what the four existing sweeps used, so a config built with only
    `cap_bands` and `horizon_grid` set behaves like the historical scripts."""

    cap_bands: tuple[tuple[str, tuple[float, float] | None], ...]
    horizon_grid: tuple[int, ...] = (10, 20)
    top_n_grid: tuple[int, ...] = (10, 20)
    cost_grid_bps: tuple[float, ...] = (10.0, 15.0, 25.0, 35.0)
    gate_cost_bps: float = 25.0
    """Which cost_grid_bps entry the gate is actually evaluated at -- matching
    every prior gate evaluation in this project, which gates one pre-
    registered cost point and records the others only for the viability
    surface."""
    gate_criteria: GateCriteria = field(default_factory=GateCriteria)
    extra_factor_names: tuple[str, ...] | None = None
    """Names from features.registry.FACTOR_REGISTRY to append to the standard
    engine.py feature set -- how a searched candidate factor enters a sweep
    without editing engine.py. None means no extra factors (the historical
    behaviour); an empty tuple is different from None only in that it still
    calls apply_registered_factors with zero names, which is a harmless no-op."""


@dataclass(frozen=True)
class SweepResult:
    fold_results: pd.DataFrame
    """One row per (cap_band, horizon, top_n, cost_bps, fold_id) -- the exact
    schema the four historical *_fold_results.csv files use, so existing
    analysis code and verdict-doc writers keep working unmodified."""
    verdicts: list[dict]
    """One entry per (cap_band, horizon, top_n) at gate_cost_bps: {"cap_band",
    "horizon_bars", "top_n", "cost_bps", "verdict": GateVerdict}."""


def load_and_prepare(settings, store, cap_band: tuple[float, float] | None,
                      extra_factor_names: tuple[str, ...] | None = None):
    """Identical to every prior sweep's own `_load_and_prepare`, generalised
    to optionally append registered factors. Returns (candles, feats, fcols)
    -- feats/fcols are None if the resulting universe is empty."""
    candles = load_candles(settings, store, turnover_rank_band=cap_band)
    if candles.empty:
        return candles, None, None

    # Same branch data/loader.py:load_features_and_labels and every prior
    # sweep use for price_source="bhavcopy": adjusted-price jumps with NO
    # matching corporate-action record, not the yfinance-appropriate check.
    candles, quarantined = quarantine_unexplained_jumps(store, candles)
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined)

    # The halted-symbol reopening-print fix (data/liquidity.py's own finding,
    # first hit running bhavcopy_rerun_sweep.py): substitute the segment
    # symbol so bar-sequence pct_change/shift resets at a halt instead of
    # spanning it. Left substituted for the rest of this function -- FoldResult
    # carries no per-symbol detail, so nothing downstream needs the real
    # ticker restored (unlike data/loader.py's production path).
    candles = candles.assign(symbol=segment_symbols_by_trading_gap(candles))

    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]

    if extra_factor_names:
        extra = apply_registered_factors(candles, names=list(extra_factor_names))
        feats = feats.merge(extra, on=["symbol", "date"], how="left")
        fcols = fcols + list(extra_factor_names)

    return candles, feats, fcols


def run_sweep(settings, store, config: SweepConfig, random_seed: int | None = None) -> SweepResult:
    """The reusable loop. `store` must already be open (this function does
    not close it, matching train_candidate_core's own convention of taking an
    open Store rather than owning its lifecycle)."""
    random_seed = random_seed if random_seed is not None else settings.random_seed
    all_rows: list[dict] = []
    verdicts: list[dict] = []

    for cap_label, cap_band in config.cap_bands:
        candles, feats, fcols = load_and_prepare(settings, store, cap_band, config.extra_factor_names)
        if feats is None or feats.empty:
            log.warning("cap_band_empty_universe", cap_band=cap_label)
            continue
        log.info("cap_band_loaded", cap_band=cap_label, rows=len(feats), n_symbols=feats["symbol"].nunique())

        trading_dates = pd.DatetimeIndex(sorted(feats["date"].unique()))

        for horizon in config.horizon_grid:
            labeled = add_forward_return_labels(candles, horizon_bars=horizon)
            labeled = add_relative_forward_return(labeled, horizon_bars=horizon)

            test_window = max(21, horizon * 12)
            folds = make_folds(trading_dates, horizon_bars=horizon, test_window_bars=test_window)
            log.info("folds_built", cap_band=cap_label, horizon=horizon, n_folds=len(folds))

            scored_by_fold = {}
            for fold in folds:
                scored = train_and_score_fold(
                    feats, labeled, fcols, fold, horizon_bars=horizon,
                    ranker_config=RankerConfig(random_state=random_seed),
                )
                if scored is not None:
                    scored_by_fold[fold.fold_id] = scored

            for top_n in config.top_n_grid:
                for cost_bps in config.cost_grid_bps:
                    fold_results: list[FoldResult] = []
                    for fold_id, scored in scored_by_fold.items():
                        result = simulate_portfolio(scored, top_n=top_n, round_trip_cost_bps=cost_bps)
                        if result is None:
                            continue
                        fold_results.append(result)
                        all_rows.append({
                            "cap_band": cap_label, "horizon_bars": horizon, "top_n": top_n,
                            "cost_bps": cost_bps, "fold_id": fold_id,
                            "n_rebalances": result.n_rebalances,
                            "gross_expectancy": result.gross_expectancy,
                            "net_expectancy": result.net_expectancy,
                            "benchmark_expectancy": result.benchmark_expectancy,
                            "alpha_gross": result.alpha_gross, "alpha_net": result.alpha_net,
                            "mean_turnover": result.mean_turnover, "ic": result.information_coefficient,
                            "hit_rate": result.hit_rate,
                        })

                    if cost_bps == config.gate_cost_bps:
                        verdict = evaluate_gate(fold_results, criteria=config.gate_criteria)
                        verdicts.append({
                            "cap_band": cap_label, "horizon_bars": horizon, "top_n": top_n,
                            "cost_bps": cost_bps, "verdict": verdict,
                        })
                        log.info(
                            "gate_verdict", cap_band=cap_label, horizon=horizon, top_n=top_n,
                            passed=verdict.passed, reason=verdict.reason,
                        )

    return SweepResult(fold_results=pd.DataFrame(all_rows), verdicts=verdicts)


def write_verdict_doc(verdicts: list[dict], path) -> None:
    """Identical output shape to every prior sweep's own `write_verdict_doc`,
    factored out once so a new sweep gets it for free instead of copying it a
    fifth time."""
    any_pass = any(v["verdict"].passed for v in verdicts)
    lines = [
        f"# Sweep Verdict: {path.stem}",
        "",
        f"**Date:** {pd.Timestamp.now().date()}",
        "",
        f"## Overall: {'AT LEAST ONE COMBINATION PASSES' if any_pass else 'ALL COMBINATIONS FAIL'}",
        "",
        "Evaluated independently per (cap_band, horizon, top_n) at the pre-registered "
        "gate_cost_bps, using evaluation/gate.py unmodified.",
        "",
        "## Per-combination verdicts",
        "",
        "| cap_band | horizon | top_n | passed | reason | n_folds | mean_alpha_net | hit_rate_pvalue |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        m = v["verdict"].metrics
        lines.append(
            f"| {v['cap_band']} | {v['horizon_bars']} | {v['top_n']} | "
            f"{'PASS' if v['verdict'].passed else 'FAIL'} | {v['verdict'].reason} | "
            f"{m.get('n_folds', '-')} | {m.get('mean_alpha_net', float('nan')):+.5f} | "
            f"{m.get('hit_rate_pvalue', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote_verdict_doc", path=str(path))
