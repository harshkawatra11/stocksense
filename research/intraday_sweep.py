"""
Phase E4 sweep -- the decisive experiment for the intraday track.

Question: does the intraday signal clear costs on the real 91M-bar
Upstox spine, under the parameters fixed in
research/preregistration_intraday.md (stop_pct=1.0%, target_pct=1.5%,
max_holding_minutes=60), judged by evaluation/gate.py completely
unchanged? No threshold here may be adjusted after seeing a result --
if this fails the gate, that is the finding.

Usage: python research/intraday_sweep.py
Expected runtime: full-scale resample ~1min (SQL), feature build ~78min
on first run (then cached to parquet, seconds on any re-run), the
sweep loop itself is the remaining variable -- reported as it runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import structlog

from stocksense.core.config import get_settings
from stocksense.data.store import Store
from stocksense.evaluation.gate import evaluate_gate
from stocksense.evaluation.intraday_backtest import (
    add_relative_session_return,
    make_session_folds,
    session_split,
    simulate_intraday_trades_for_fold,
    trades_to_fold_result,
    train_intraday_ranker,
)
from stocksense.features.intraday import (
    build_intraday_features_cached,
    feature_columns,
    resample_to_bars_sql,
)
from stocksense.labels.intraday_labels import add_session_forward_return
from stocksense.models.ranker import RankerConfig

log = structlog.get_logger(__name__)

# Pre-registered in research/preregistration_intraday.md -- fixed before
# any result from this script exists. Do not change these after seeing
# an output; that is the exact failure this project already corrected
# once (research/gate_criteria_preregistration.md).
STOP_PCT = 0.01
TARGET_PCT = 0.015
MAX_HOLDING_MINUTES = 60
TOP_N = 10
EXPOSURE_INR = 75_000.0
LABEL_HORIZON_BARS = MAX_HOLDING_MINUTES // 5  # 12 five-min bars ~ the max-hold window
MIN_TRAIN_SESSIONS = 500
TEST_WINDOW_SESSIONS = 42
EMBARGO_SESSIONS = 1

# Excludes raw price levels (or_high/or_low/vwap) -- not comparable
# across symbols, unlike the normalized dist_from_vwap/or_breakout_*
# fields that already carry the same information scale-invariantly.
FEATURE_COLS = [
    "minutes_since_open", "or_breakout_up", "or_breakout_down",
    "dist_from_vwap", "rsi_14", "session_ret_so_far", "volume_spike_ratio",
]


def main() -> None:
    settings = get_settings()
    store = Store(settings.duckdb_path)

    t0 = time.time()
    symbols = store.con.execute(
        "SELECT DISTINCT symbol FROM intraday_bars"
    ).fetchdf()["symbol"].tolist()
    log.info("resolved_universe", n_symbols=len(symbols))

    bars_1min = store.read_intraday_bars(symbols=symbols, interval="1minute")
    store.close()
    log.info("loaded_1min_bars", rows=len(bars_1min), elapsed_s=round(time.time() - t0, 1))

    t0 = time.time()
    bars_5min = resample_to_bars_sql(bars_1min, interval="5min")
    log.info("resampled_5min", rows=len(bars_5min), elapsed_s=round(time.time() - t0, 1))

    t0 = time.time()
    cache_key = f"{len(symbols)}syms_{bars_5min['ts'].min().date()}_{bars_5min['ts'].max().date()}"
    feats = build_intraday_features_cached(bars_5min, cache_key=cache_key)
    log.info("built_features", rows=len(feats), cache_key=cache_key, elapsed_s=round(time.time() - t0, 1))

    # Only rebalance once the opening-range window has actually closed --
    # or_breakout_up/down and or_high/or_low are NaN-by-design before
    # then (features/intraday.py), so scoring on them is scoring on
    # incomplete information the feature contract itself says isn't
    # ready yet.
    feats = feats[feats["minutes_since_open"] >= 15].reset_index(drop=True)

    labeled = add_session_forward_return(bars_5min, horizon_bars=LABEL_HORIZON_BARS, label_col="fwd_ret")
    labeled = add_relative_session_return(labeled, "fwd_ret")
    labeled = labeled[["symbol", "ts", "fwd_ret", "fwd_ret_rel"]]

    session_dates = pd.to_datetime(feats["ts"]).dt.normalize().unique()
    folds = make_session_folds(
        session_dates, min_train_sessions=MIN_TRAIN_SESSIONS,
        test_window_sessions=TEST_WINDOW_SESSIONS, embargo_sessions=EMBARGO_SESSIONS,
    )
    log.info("session_folds_built", n_folds=len(folds))

    fold_results = []
    for fold in folds:
        t0 = time.time()
        ranker = train_intraday_ranker(feats, labeled, FEATURE_COLS, fold, "fwd_ret_rel", RankerConfig(random_state=settings.random_seed))
        if ranker is None:
            log.warning("fold_skipped_insufficient_train_rows", fold_id=fold.fold_id)
            continue

        merged = feats.merge(labeled, on=["symbol", "ts"], how="inner")
        _, test_df = session_split(merged, fold)
        test_df = test_df.dropna(subset=FEATURE_COLS)
        if test_df.empty:
            log.warning("fold_skipped_empty_test", fold_id=fold.fold_id)
            continue

        scores_by_ts = {}
        for ts, grp in test_df.groupby("ts"):
            preds = pd.Series(ranker.predict(grp[FEATURE_COLS]).values, index=grp["symbol"].values)
            scores_by_ts[ts] = preds

        _, bars_5min_test = session_split(bars_5min, fold)
        _, bars_1min_test = session_split(bars_1min, fold)
        benchmark_by_ts = test_df.groupby("ts")["fwd_ret"].mean().to_dict()

        trades = simulate_intraday_trades_for_fold(
            scores_by_ts, bars_5min_test, bars_1min_test, benchmark_by_ts,
            top_n=TOP_N, stop_pct=STOP_PCT, target_pct=TARGET_PCT,
            max_holding_minutes=MAX_HOLDING_MINUTES, exposure_inr=EXPOSURE_INR,
        )
        result = trades_to_fold_result(fold.fold_id, trades)
        if result is not None:
            fold_results.append(result)
        log.info(
            "fold_done", fold_id=fold.fold_id, n_trades=len(trades),
            net_expectancy=None if result is None else round(result.net_expectancy, 5),
            elapsed_s=round(time.time() - t0, 1),
        )

    verdict = evaluate_gate(fold_results)
    log.info("gate_verdict", passed=verdict.passed, reason=verdict.reason, metrics=verdict.metrics)

    out_path = Path(__file__).resolve().parent / "intraday_fold_results.csv"
    pd.DataFrame([{
        "fold_id": r.fold_id, "n_trades": r.n_rebalances, "gross_expectancy": r.gross_expectancy,
        "net_expectancy": r.net_expectancy, "benchmark_expectancy": r.benchmark_expectancy,
        "alpha_gross": r.alpha_gross, "alpha_net": r.alpha_net, "hit_rate": r.hit_rate,
    } for r in fold_results]).to_csv(out_path, index=False)
    log.info("wrote_fold_results", path=str(out_path))

    write_verdict_doc(fold_results, verdict)


def write_verdict_doc(fold_results, verdict) -> None:
    path = Path(__file__).resolve().parent / "verdict_intraday.md"
    lines = [
        "# Intraday Gate Verdict",
        "",
        f"**Date:** {pd.Timestamp.now().date()}",
        f"**Pre-registration:** research/preregistration_intraday.md (parameters fixed before this run)",
        "",
        f"## Verdict: {'GATE PASS' if verdict.passed else 'GATE FAIL'}",
        "",
        f"{verdict.reason}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in verdict.metrics.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"## Folds: {len(fold_results)}",
        "",
        "| fold_id | n_trades | net_expectancy | alpha_net | hit_rate |",
        "|---|---|---|---|---|",
    ]
    for r in fold_results:
        lines.append(f"| {r.fold_id} | {r.n_rebalances} | {r.net_expectancy:+.5f} | {r.alpha_net:+.5f} | {r.hit_rate:.2%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote_verdict_doc", path=str(path))


if __name__ == "__main__":
    main()
