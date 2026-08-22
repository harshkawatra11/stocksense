"""
Bhavcopy point-in-time gate re-run -- the decisive experiment for
Phase G. Pre-registered in research/preregistration_bhavcopy_rerun.md
(committed before this script's first real run against full data):
does Run 4's h=10/h=20 GATE PASS survive on the real point-in-time
bhavcopy universe (8.17M rows, 2010-2026), and does any mid/small-cap
liquidity-rank band clear the same pre-registered gate?

Reuses data/loader.py's load_candles (the bhavcopy + point-in-time-
universe + turnover_rank_band path, shared with cli/main.py's
train_candidate and harness/loops.py's reconcile loop) rather than
reading candles directly, so this sweep and the production training
path can never diverge -- the exact fix preregistration_bhavcopy_rerun.md
names as missing from the original phase0_sweep.py.

Usage: python research/bhavcopy_rerun_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import structlog

from stocksense.core.config import get_settings
from stocksense.data.adjust import quarantine_unexplained_jumps
from stocksense.data.loader import load_candles
from stocksense.data.store import Store
from stocksense.evaluation.backtest import simulate_portfolio, train_and_score_fold
from stocksense.evaluation.gate import GateCriteria, evaluate_gate
from stocksense.evaluation.walkforward import make_folds
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import RankerConfig

log = structlog.get_logger(__name__)

# Pre-registered in research/preregistration_bhavcopy_rerun.md -- fixed
# before this run. Do not change any value below based on this run's
# own result.
HORIZON_GRID = (10, 20)
TOP_N_GRID = (10, 20)
COST_GRID_BPS = (10.0, 15.0, 25.0, 35.0)

# (label, turnover_rank_band) -- None is the full point-in-time
# universe, the direct comparison against Run 4 isolating the
# survivorship fix from the cap-band restriction.
CAP_BANDS = (
    ("full_pit", None),
    ("large", (0.8, 1.0)),
    ("mid", (0.5, 0.8)),
    ("small", (0.15, 0.5)),
)

GATE_CRITERIA = GateCriteria()  # research/gate_criteria_preregistration.md defaults, unmodified


def _load_and_prepare(settings, store, cap_band: tuple[float, float] | None):
    candles = load_candles(settings, store, turnover_rank_band=cap_band)
    if candles.empty:
        return candles, None, None

    # Same branch cli/main.py._load_features_and_labels uses for
    # price_source="bhavcopy": adjusted-price jumps with NO matching
    # corporate-action record, not the yfinance-appropriate check (which
    # would misfire on every genuine bhavcopy split/bonus -- see
    # cli/main.py's own comment on this exact bug, found live in D2).
    candles, quarantined = quarantine_unexplained_jumps(store, candles)
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined)

    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]
    return candles, feats, fcols


def main() -> None:
    settings = get_settings()
    settings.price_source = "bhavcopy"
    settings.use_point_in_time_universe = True

    RESEARCH_DIR = Path(__file__).resolve().parent
    all_rows = []
    verdicts = []

    for cap_label, cap_band in CAP_BANDS:
        t0 = time.time()
        store = Store(settings.duckdb_path)
        candles, feats, fcols = _load_and_prepare(settings, store, cap_band)
        store.close()
        if feats is None or feats.empty:
            log.warning("cap_band_empty_universe", cap_band=cap_label)
            continue
        log.info(
            "cap_band_loaded", cap_band=cap_label, rows=len(feats),
            n_symbols=feats["symbol"].nunique(), elapsed_s=round(time.time() - t0, 1),
        )

        trading_dates = pd.DatetimeIndex(sorted(feats["date"].unique()))

        for horizon in HORIZON_GRID:
            labeled = add_forward_return_labels(candles, horizon_bars=horizon)
            labeled = add_relative_forward_return(labeled, horizon_bars=horizon)

            test_window = max(21, horizon * 12)
            folds = make_folds(trading_dates, horizon_bars=horizon, test_window_bars=test_window)
            log.info("folds_built", cap_band=cap_label, horizon=horizon, n_folds=len(folds))

            scored_by_fold = {}
            for fold in folds:
                scored = train_and_score_fold(
                    feats, labeled, fcols, fold, horizon_bars=horizon,
                    ranker_config=RankerConfig(random_state=settings.random_seed),
                )
                if scored is not None:
                    scored_by_fold[fold.fold_id] = scored

            for top_n in TOP_N_GRID:
                for cost_bps in COST_GRID_BPS:
                    fold_results = []
                    for fold_id, scored in scored_by_fold.items():
                        result = simulate_portfolio(scored, top_n=top_n, round_trip_cost_bps=cost_bps)
                        if result is not None:
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

                    # Gate at the pre-registered cost only (25bps, matching
                    # every prior gate evaluation in this project) -- the
                    # other cost points are recorded for the viability
                    # surface but are not separately gated, exactly as
                    # Run 4 gated one cost point per (horizon, top_n).
                    if cost_bps == 25.0:
                        verdict = evaluate_gate(fold_results, criteria=GATE_CRITERIA)
                        verdicts.append({
                            "cap_band": cap_label, "horizon_bars": horizon, "top_n": top_n,
                            "cost_bps": cost_bps, "verdict": verdict,
                        })
                        log.info(
                            "gate_verdict", cap_band=cap_label, horizon=horizon, top_n=top_n,
                            passed=verdict.passed, reason=verdict.reason,
                        )

    df = pd.DataFrame(all_rows)
    df.to_csv(RESEARCH_DIR / "bhavcopy_rerun_fold_results.csv", index=False)
    log.info("fold_results_saved", n_rows=len(df))

    write_verdict_doc(verdicts, RESEARCH_DIR / "verdict_bhavcopy_rerun.md")
    log.info("sweep_complete")


def write_verdict_doc(verdicts: list[dict], path: Path) -> None:
    any_pass = any(v["verdict"].passed for v in verdicts)
    lines = [
        "# Bhavcopy Point-in-Time Gate Re-Run: Verdict",
        "",
        f"**Date:** {pd.Timestamp.now().date()}",
        "**Pre-registration:** research/preregistration_bhavcopy_rerun.md (parameters fixed before this run)",
        "",
        f"## Overall: {'AT LEAST ONE COMBINATION PASSES' if any_pass else 'ALL COMBINATIONS FAIL'}",
        "",
        "Evaluated independently per (cap_band, horizon, top_n) at the pre-registered 25bps cost point, "
        "using evaluation/gate.py unmodified with research/gate_criteria_preregistration.md's defaults.",
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


if __name__ == "__main__":
    main()
