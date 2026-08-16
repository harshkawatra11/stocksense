"""
Phase 0 sweep — the decisive experiment.

Question: does any (horizon x top_n x cost) configuration produce
net-positive, fold-stable, out-of-sample alpha? Answers by walk-forward
backtesting every combination in the config grid and writing a viability
surface + break-even cost per configuration.

Usage: python research/phase0_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import structlog

from stocksense.core.config import get_settings
from stocksense.data.store import Store
from stocksense.data.validate import quarantine_symbols
from stocksense.evaluation.backtest import simulate_portfolio, train_and_score_fold
from stocksense.evaluation.walkforward import make_folds
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import RankerConfig

log = structlog.get_logger(__name__)


def main() -> None:
    settings = get_settings()
    store = Store(settings.duckdb_path)
    candles = store.read_candles()
    store.close()
    log.info("loaded_candles", rows=len(candles), symbols=candles["symbol"].nunique())

    # AUDIT FIX: this script was never updated with the adjustment-anomaly
    # quarantine (data/validate.py) that cli/main.py received after the
    # ADANIENT data bug was found. Re-running this sweep without it would
    # silently resurrect the exact contamination Run 3 fixed.
    candles, quarantined = quarantine_symbols(candles)
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined)

    feats = build_features(candles)
    fcols = feature_columns(feats)
    fcols = [c for c in fcols if c not in ("mkt_ret_1b",)]  # keep as context but avoid redundant target leakage risk
    log.info("built_features", rows=len(feats), n_features=len(fcols))

    trading_dates = pd.DatetimeIndex(sorted(feats["date"].unique()))

    all_fold_results = []
    all_scored_cache = {}  # (horizon, fold_id) -> ScoredFold, so top_n/cost don't retrain

    for horizon in settings.horizon_grid:
        labeled = add_forward_return_labels(candles, horizon_bars=horizon)
        labeled = add_relative_forward_return(labeled, horizon_bars=horizon)

        test_window = max(21, horizon * 12)  # scale test window with horizon so folds aren't tiny
        folds = make_folds(trading_dates, horizon_bars=horizon, test_window_bars=test_window)
        log.info("folds_built", horizon=horizon, n_folds=len(folds))

        for fold in folds:
            scored = train_and_score_fold(
                feats, labeled, fcols, fold, horizon_bars=horizon,
                ranker_config=RankerConfig(random_state=settings.random_seed),
            )
            if scored is None:
                continue
            all_scored_cache[(horizon, fold.fold_id)] = scored
            log.info(
                "fold_trained", horizon=horizon, fold=fold.fold_id,
                train_rows=scored.n_train_rows, n_rebalances=len(scored.rebalance_dates),
            )

    log.info("training_complete", n_scored_folds=len(all_scored_cache))

    for (horizon, fold_id), scored in all_scored_cache.items():
        for top_n in settings.top_n_grid:
            for cost_bps in settings.cost_grid_bps:
                result = simulate_portfolio(scored, top_n=top_n, round_trip_cost_bps=cost_bps)
                if result is not None:
                    all_fold_results.append(result)

    rows = [
        {
            "horizon_bars": r.horizon_bars,
            "fold_id": r.fold_id,
            "top_n": r.top_n,
            "cost_bps": r.round_trip_cost_bps,
            "n_rebalances": r.n_rebalances,
            "gross_expectancy": r.gross_expectancy,
            "net_expectancy": r.net_expectancy,
            "benchmark_expectancy": r.benchmark_expectancy,
            "alpha_gross": r.alpha_gross,
            "alpha_net": r.alpha_net,
            "mean_turnover": r.mean_turnover,
            "ic": r.information_coefficient,
            "hit_rate": r.hit_rate,
        }
        for r in all_fold_results
    ]
    df = pd.DataFrame(rows)
    RESEARCH_DIR = Path(__file__).resolve().parent
    df.to_csv(RESEARCH_DIR / "phase0_fold_results.csv", index=False)
    log.info("fold_results_saved", n_rows=len(df))

    # ---- aggregate to viability surface: per (horizon, top_n, cost), across folds ----
    agg = df.groupby(["horizon_bars", "top_n", "cost_bps"]).agg(
        n_folds=("alpha_net", "count"),
        mean_alpha_net=("alpha_net", "mean"),
        std_alpha_net=("alpha_net", "std"),
        pct_folds_positive=("alpha_net", lambda s: float((s > 0).mean())),
        mean_ic=("ic", "mean"),
        mean_turnover=("mean_turnover", "mean"),
        mean_gross_alpha=("alpha_gross", "mean"),
    ).reset_index()

    # break-even cost: gross alpha per unit turnover tells us the cost bps
    # at which net alpha crosses zero, holding selectivity fixed
    gross_by_config = df.groupby(["horizon_bars", "top_n"]).agg(
        mean_alpha_gross=("alpha_gross", "mean"),
        mean_turnover=("mean_turnover", "mean"),
    ).reset_index()
    gross_by_config["breakeven_cost_bps"] = (
        gross_by_config["mean_alpha_gross"] / (gross_by_config["mean_turnover"].clip(lower=1e-6)) * 10_000
    )

    agg.to_csv(RESEARCH_DIR / "phase0_viability_surface.csv", index=False)
    gross_by_config.to_csv(RESEARCH_DIR / "phase0_breakeven_cost.csv", index=False)

    print("\n=== VIABILITY SURFACE (top configs by mean net alpha) ===")
    print(agg.sort_values("mean_alpha_net", ascending=False).head(20).to_string(index=False))

    print("\n=== BREAK-EVEN COST BY (horizon, top_n) ===")
    print(gross_by_config.sort_values("breakeven_cost_bps", ascending=False).to_string(index=False))

    log.info("sweep_complete")


if __name__ == "__main__":
    main()
