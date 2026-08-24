"""
F&O positioning signal sweep -- the decisive experiment for Phase I.
Pre-registered in research/preregistration_fno_signal.md (committed
before this script's first real run): does yesterday's-close options-
positioning data (open interest, put/call ratio, days-to-expiry --
features/fno.py, built and unit-tested since an earlier phase but never
backfilled or wired into training) carry same-day (h=1) directional
signal that raw OHLCV features didn't, on the F&O-eligible universe
where this data actually exists?

Two feature sets run through the IDENTICAL walk-forward/gate pipeline,
on the SAME F&O-restricted universe, so any difference is attributable
to the new features, not a universe change:
  A (baseline)  = existing price features only
  B (treatment) = A + build_oi_features/build_put_call_ratio/days_to_expiry

Reuses data/loader.py's load_candles, data/liquidity.py's
segment_symbols_by_trading_gap, and the identical
train_and_score_fold/simulate_portfolio/evaluate_gate pipeline every
prior sweep in this project has used -- the only new logic here is the
F&O-eligible universe restriction and the treatment feature merge.

Usage: python research/fno_signal_sweep.py
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import structlog

from stocksense.core.config import get_settings
from stocksense.data.adjust import quarantine_unexplained_jumps
from stocksense.data.liquidity import segment_symbols_by_trading_gap
from stocksense.data.loader import load_candles
from stocksense.data.store import Store
from stocksense.evaluation.backtest import simulate_portfolio, train_and_score_fold
from stocksense.evaluation.gate import GateCriteria, evaluate_gate
from stocksense.evaluation.walkforward import make_folds
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import RankerConfig

log = structlog.get_logger(__name__)

# Pre-registered in research/preregistration_fno_signal.md -- fixed
# before this run. Do not change any value below based on this run's
# own result.
HORIZON_BARS = 1
TOP_N_GRID = (10, 20)
COST_GRID_BPS = (10.0, 15.0, 25.0, 35.0)
FO_ELIGIBILITY_LOOKBACK_DAYS = 400  # generous trailing window over a symbol's own FUTSTK history -- see the pre-registration's named limitation on this being a coarser reconstruction than universe_pit.py's equity-side filter

GATE_CRITERIA = GateCriteria()  # research/gate_criteria_preregistration.md defaults, unmodified


def _fo_eligible_symbol_windows(store: Store) -> pd.DataFrame:
    """Per-symbol [first, last] FUTSTK date range from bhavcopy_fo
    itself -- the point-in-time F&O-eligibility reconstruction named in
    the pre-registration (a coarser proxy than an independent NSE
    eligibility archive would give, disclosed there rather than
    assumed away). A single up-front aggregation, not a per-date query
    loop -- efficient regardless of how many trading dates this sweep
    ultimately covers."""
    fut = store.con.execute(
        "SELECT symbol, MIN(date) AS fo_first_date, MAX(date) AS fo_last_date "
        "FROM bhavcopy_fo WHERE instrument = 'FUTSTK' GROUP BY symbol"
    ).fetchdf()
    return fut


def _restrict_to_fo_universe(candles: pd.DataFrame, fo_windows: pd.DataFrame) -> pd.DataFrame:
    """Drops every (symbol, date) row outside that symbol's own
    [fo_first_date, fo_last_date] window -- point-in-time safe by
    construction (a date before a symbol's first FUTSTK listing, or
    after its last, is excluded), and never lets a symbol with NO F&O
    history at all appear in this restricted universe at all (an inner
    join, not left)."""
    merged = candles.merge(fo_windows, on="symbol", how="inner")
    mask = (merged["date"] >= merged["fo_first_date"]) & (merged["date"] <= merged["fo_last_date"])
    return merged.loc[mask].drop(columns=["fo_first_date", "fo_last_date"])


def _build_fo_daily_features(store: Store, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """One row per (symbol, date) present in bhavcopy_fo. Even after the
    symbol/date restriction, pulling raw per-contract rows into pandas
    still crashed with a MemoryError -- the F&O-eligible universe is
    exactly the liquid names that carry the deepest option chains (found
    live: ~150M of bhavcopy_fo's ~158M rows are option rows, and most of
    those belong to precisely this restricted symbol set). So the
    aggregation itself is pushed into DuckDB via SQL, replicating
    features/fno.py's build_oi_features/build_put_call_ratio/
    days_to_expiry arithmetic exactly (same filters: instrument LIKE
    '%FUT%' for OI, option_type IN ('CE','PE') for PCR, nearest expiry
    per symbol/date) -- only the small per-(symbol,date) aggregated
    result crosses into pandas, not the ~150M raw rows behind it."""
    if not symbols:
        return pd.DataFrame(columns=["symbol", "date"])
    placeholders = ", ".join("?" for _ in symbols)
    base_params = [*symbols, start, end]

    oi = store.con.execute(
        f"""
        SELECT symbol, date,
               SUM(open_interest) AS total_oi,
               SUM(chg_in_oi) AS total_chg_in_oi,
               AVG(close) AS futures_close
        FROM bhavcopy_fo
        WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?
          AND instrument LIKE '%FUT%'
        GROUP BY symbol, date
        """,
        base_params,
    ).fetchdf()
    # replace(0, pd.NA) on a float column upcasts it to object dtype,
    # which LightGBM's _check_for_bad_pandas_dtypes rejects outright --
    # found live, the treatment run crashed with "pandas dtypes must be
    # int, float or bool" after the baseline run had already completed.
    # np.nan keeps the column float64 throughout.
    denom = (oi["total_oi"] - oi["total_chg_in_oi"]).replace(0, float("nan"))
    oi["oi_pct_change"] = (oi["total_chg_in_oi"] / denom).astype("float64")

    pcr_raw = store.con.execute(
        f"""
        SELECT symbol, date, option_type, SUM(open_interest) AS oi
        FROM bhavcopy_fo
        WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?
          AND option_type IN ('CE', 'PE')
        GROUP BY symbol, date, option_type
        """,
        base_params,
    ).fetchdf()
    if pcr_raw.empty:
        pcr = pd.DataFrame(columns=["symbol", "date", "put_oi", "call_oi", "pcr_oi"])
    else:
        pivot = pcr_raw.pivot_table(index=["symbol", "date"], columns="option_type", values="oi", fill_value=0.0)
        pcr = pd.DataFrame(index=pivot.index)
        pcr["put_oi"] = pivot.get("PE", 0.0)
        pcr["call_oi"] = pivot.get("CE", 0.0)
        pcr_denom = pcr["call_oi"].replace(0, float("nan"))
        pcr["pcr_oi"] = (pcr["put_oi"] / pcr_denom).astype("float64")
        pcr = pcr.reset_index()

    # Same era-dependent expiry formats nse_archive.py normalizes on
    # ingest: "DD-Mon-YYYY" (legacy) and ISO "YYYY-MM-DD" (UDiFF) --
    # TRY_STRPTIME on each, COALESCE to whichever parses.
    dte = store.con.execute(
        f"""
        SELECT symbol, date,
               MIN(DATE_DIFF('day', date,
                   COALESCE(TRY_STRPTIME(expiry_date, '%d-%b-%Y')::DATE,
                            TRY_STRPTIME(expiry_date, '%Y-%m-%d')::DATE)
               )) AS days_to_nearest_expiry
        FROM bhavcopy_fo
        WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?
        GROUP BY symbol, date
        """,
        base_params,
    ).fetchdf()
    # DuckDB's DATE_DIFF over a column with NULLs (unparseable expiry,
    # essentially none expected but not guaranteed) round-trips into
    # pandas as nullable Int64, another dtype LightGBM's pandas-dtype
    # check rejects -- cast to plain float64 for the same reason as
    # oi_pct_change/pcr_oi above.
    dte["days_to_nearest_expiry"] = dte["days_to_nearest_expiry"].astype("float64")

    if oi.empty and pcr.empty and dte.empty:
        return pd.DataFrame(columns=["symbol", "date"])

    merged = oi.merge(pcr, on=["symbol", "date"], how="outer").merge(dte, on=["symbol", "date"], how="outer")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged.sort_values(["symbol", "date"])


def _attach_fo_features_point_in_time_safe(feats: pd.DataFrame, fo_daily: pd.DataFrame) -> pd.DataFrame:
    """merge_asof, per symbol, matching each equity feature row's date
    to the MOST RECENT F&O snapshot strictly BEFORE it
    (allow_exact_matches=False) -- yesterday's-or-earlier close OI/PCR
    predicting today's forward return, never today's own F&O close
    (which already reflects today's price action and would leak)."""
    if fo_daily.empty:
        out = feats.copy()
        for col in ("total_oi", "total_chg_in_oi", "oi_pct_change", "put_oi", "call_oi", "pcr_oi", "days_to_nearest_expiry"):
            out[col] = pd.NA
        return out

    feats = feats.sort_values(["symbol", "date"]).copy()
    feats["date"] = pd.to_datetime(feats["date"])
    fo_daily = fo_daily.sort_values(["symbol", "date"])

    # feats' `symbol` column holds halt-SEGMENTED names (e.g.
    # "RELIANCE__seg0", from segment_symbols_by_trading_gap upstream);
    # fo_daily holds real symbol names from bhavcopy_fo, which has no
    # concept of segments. Matching the segmented name directly against
    # fo_daily would never hit -- found before this sweep's first real
    # run: every symbol would silently fall into the "no F&O data"
    # branch below, making feature set B collapse to feature set A on
    # every F&O column (all NaN) without ever raising an error.
    parts = []
    for symbol, g in feats.groupby("symbol", group_keys=False):
        base_symbol = symbol.split("__seg")[0]
        fo_g = fo_daily[fo_daily["symbol"] == base_symbol]
        if fo_g.empty:
            parts.append(g)
            continue
        merged = pd.merge_asof(
            g, fo_g.drop(columns=["symbol"]), on="date", direction="backward", allow_exact_matches=False,
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def _load_and_prepare(settings, store: Store):
    candles = load_candles(settings, store, turnover_rank_band=None)
    if candles.empty:
        return None, None, None, None

    candles, quarantined = quarantine_unexplained_jumps(store, candles)
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined)

    fo_windows = _fo_eligible_symbol_windows(store)
    if fo_windows.empty:
        return None, None, None, None
    candles = _restrict_to_fo_universe(candles, fo_windows)
    if candles.empty:
        return None, None, None, None

    # Same halt-segmentation fix Phase G1 found necessary -- applies
    # equally here since this is still the same bhavcopy candle source.
    candles = candles.assign(symbol=segment_symbols_by_trading_gap(candles))

    feats_a = build_features(candles)
    fcols_a = [c for c in feature_columns(feats_a) if c != "mkt_ret_1b"]

    fo_symbols = sorted(candles["symbol"].astype(str).str.split("__seg").str[0].unique())
    start_str = pd.to_datetime(candles["date"]).min().strftime("%Y-%m-%d")
    end_str = pd.to_datetime(candles["date"]).max().strftime("%Y-%m-%d")
    fo_daily = _build_fo_daily_features(store, fo_symbols, start_str, end_str)
    feats_b = _attach_fo_features_point_in_time_safe(feats_a, fo_daily)
    fo_cols = ["total_oi", "total_chg_in_oi", "oi_pct_change", "put_oi", "call_oi", "pcr_oi", "days_to_nearest_expiry"]
    fcols_b = fcols_a + fo_cols

    return candles, feats_a, fcols_a, (feats_b, fcols_b)


def _run_feature_set(label, feats, fcols, candles, settings, trading_dates, all_rows, verdicts):
    labeled = add_forward_return_labels(candles, horizon_bars=HORIZON_BARS)
    labeled = add_relative_forward_return(labeled, horizon_bars=HORIZON_BARS)

    test_window = max(21, HORIZON_BARS * 12)
    folds = make_folds(trading_dates, horizon_bars=HORIZON_BARS, test_window_bars=test_window)
    log.info("folds_built", feature_set=label, n_folds=len(folds))

    scored_by_fold = {}
    for fold in folds:
        scored = train_and_score_fold(
            feats, labeled, fcols, fold, horizon_bars=HORIZON_BARS,
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
                        "feature_set": label, "top_n": top_n, "cost_bps": cost_bps, "fold_id": fold_id,
                        "n_rebalances": result.n_rebalances,
                        "gross_expectancy": result.gross_expectancy, "net_expectancy": result.net_expectancy,
                        "benchmark_expectancy": result.benchmark_expectancy,
                        "alpha_gross": result.alpha_gross, "alpha_net": result.alpha_net,
                        "mean_turnover": result.mean_turnover, "ic": result.information_coefficient,
                        "hit_rate": result.hit_rate,
                    })

            if cost_bps == 25.0:
                verdict = evaluate_gate(fold_results, criteria=GATE_CRITERIA)
                verdicts.append({"feature_set": label, "top_n": top_n, "cost_bps": cost_bps, "verdict": verdict})
                log.info("gate_verdict", feature_set=label, top_n=top_n, passed=verdict.passed, reason=verdict.reason)


def main() -> None:
    settings = get_settings()
    settings.price_source = "bhavcopy"
    settings.use_point_in_time_universe = True

    RESEARCH_DIR = Path(__file__).resolve().parent
    all_rows: list[dict] = []
    verdicts: list[dict] = []

    t0 = time.time()
    store = Store(settings.duckdb_path)
    candles, feats_a, fcols_a, treatment = _load_and_prepare(settings, store)
    store.close()

    if candles is None:
        log.error("fo_universe_empty_or_no_data")
        return

    feats_b, fcols_b = treatment
    log.info(
        "fo_universe_loaded", n_symbols=feats_a["symbol"].nunique(),
        rows_a=len(feats_a), rows_b=len(feats_b), elapsed_s=round(time.time() - t0, 1),
    )

    trading_dates = pd.DatetimeIndex(sorted(feats_a["date"].unique()))

    _run_feature_set("baseline_price_only", feats_a, fcols_a, candles, settings, trading_dates, all_rows, verdicts)
    _run_feature_set("treatment_price_plus_fno", feats_b, fcols_b, candles, settings, trading_dates, all_rows, verdicts)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESEARCH_DIR / "fno_signal_fold_results.csv", index=False)
    log.info("fold_results_saved", n_rows=len(df))

    write_verdict_doc(verdicts, RESEARCH_DIR / "verdict_fno_signal.md")
    log.info("sweep_complete")


def write_verdict_doc(verdicts: list[dict], path: Path) -> None:
    baseline_passed = any(v["verdict"].passed for v in verdicts if v["feature_set"] == "baseline_price_only")
    treatment_passed = any(v["verdict"].passed for v in verdicts if v["feature_set"] == "treatment_price_plus_fno")

    if treatment_passed and not baseline_passed:
        headline = "TREATMENT PASSES, BASELINE FAILS -- F&O features appear to add real signal"
    elif treatment_passed and baseline_passed:
        headline = "BOTH PASS -- inconclusive whether F&O features add anything beyond the universe restriction itself"
    else:
        headline = "NO PASS -- same-day signal not found, with or without F&O positioning data"

    lines = [
        "# F&O Positioning Signal: Verdict",
        "",
        f"**Date:** {pd.Timestamp.now().date()}",
        "**Pre-registration:** research/preregistration_fno_signal.md (parameters fixed before this run)",
        "",
        f"## Overall: {headline}",
        "",
        "Baseline and treatment run on the IDENTICAL F&O-eligible point-in-time universe, "
        "horizon=1, using evaluation/gate.py unmodified with research/gate_criteria_preregistration.md's defaults.",
        "",
        "## Per-combination verdicts",
        "",
        "| feature_set | top_n | passed | reason | n_folds | mean_alpha_net | hit_rate_pvalue |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        m = v["verdict"].metrics
        lines.append(
            f"| {v['feature_set']} | {v['top_n']} | "
            f"{'PASS' if v['verdict'].passed else 'FAIL'} | {v['verdict'].reason} | "
            f"{m.get('n_folds', '-')} | {m.get('mean_alpha_net', float('nan')):+.5f} | "
            f"{m.get('hit_rate_pvalue', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote_verdict_doc", path=str(path))


if __name__ == "__main__":
    main()
