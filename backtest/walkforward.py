"""
Genuine walk-forward backtest for the LightGBM BUY-signal layer.

Fixes the key flaw in research/edge_by_year.py (per the Stage-2 audit): that
script scores every year against a SINGLE model trained once on the full
2001-2026 history, so early years are in-sample and the "edge" it reports is
partly an artifact of the model having already seen those years' patterns.

This harness instead retrains a fresh LightGBM model at each expanding-window
checkpoint using ONLY data strictly before that checkpoint, then scores/trades
the following quarter out-of-sample. No window ever sees its own future.

Design choices (documented, not hidden):
  - Window cadence: QUARTERLY, not monthly. LightGBM training on the full
    expanding history (up to ~2,000+ tickers x 20+ years of daily bars) takes
    real wall-clock time; monthly windows over 2019-2026 would mean ~28
    retrains vs. ~28 already at quarterly -- wait, quarterly over 7 years is
    28 windows regardless of choice of "month" label; the actual saving is
    that monthly would be 84 retrains for the same period. Quarterly keeps
    the harness runnable in one sitting while still giving 28+ genuine OOS
    folds from 2019-2026.
  - Model/threshold logic: reuses models.ml.train's LGBM_PARAMS and
    compute_optimal_threshold (same hyperparameters, same F1-optimal
    threshold selection as the live training path) rather than reimplementing
    them. It does NOT reuse train.py's 5-fold TimeSeriesSplit CV loop --
    running 5-fold CV at every one of ~28 walk-forward checkpoints would
    multiply an already-heavy training job 5x for marginal benefit (CV in
    train.py is for hyperparameter/model selection; here the hyperparameters
    are already fixed). Instead each window fits ONE model on the training
    slice with a simple time-ordered train/validation split for early
    stopping and threshold selection -- same class, same params, same
    threshold-selection function, cheaper cadence.
  - Trade simulation: reuses intelligence/signal_pipeline.py's horizon_stops()
    for stop/target sizing (ATR x sqrt(hold_days), clamped 0.5%-8%, target at
    2R) -- the exact function the live pipeline uses for its ATR-derived
    stop/target when the quantile regressors are unavailable. This backtest
    does NOT invoke the quantile-derived target/stop logic used in live
    signal generation; it is honest about testing the ATR-stop swing-trade
    logic specifically.
  - Horizon tested: 5 trading days (the "5D" entry in config.TIMEFRAMES).
    This is a DAILY-BAR SWING backtest, not an intraday backtest -- the
    deployed LightGBM model and its features (data/pipeline/feature_engineering.py)
    are computed from ohlcv_daily, so that is what is genuinely being tested
    here. Do not read the results as validating intraday performance.
  - Costs: intelligence.costs.net_return(), called with product="MIS" against
    whatever signature the file currently exposes (see _net_return() below --
    this repo has another agent concurrently adding a `product` kwarg to
    costs.py; this harness calls the new signature first and falls back to
    the old single-arg signature via TypeError so it works either way).
  - Benchmark: NIFTYBEES buy-and-hold (ETF, ohlcv_daily has clean full-history
    coverage 2002-2026 for this ticker; checked against the DB directly, no
    NSEI/index-column data source needed).
  - 2001-2003 exclusion: the FIRST possible OOS window under `--from 2019`
    already starts in 2019, so 2001-2003 never appears in this harness's
    output at all -- unlike edge_by_year.py, which trains once on everything
    and therefore CAN show 2001-2003. If a user passes `--from` before 2004,
    those early windows are still explicitly flagged in the per-window table
    (see report.py) as thin-history / early-model windows, though they are
    never in-sample the way edge_by_year.py's were (each window's model still
    only ever sees data before that window).

Run:  python -m backtest.walkforward --from 2019
Smoke test:  python -m backtest.walkforward --from 2024 --to 2025
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import asyncpg
import lightgbm as lgb

from data.pipeline.feature_engineering import compute_features, get_all_feature_columns
from models.ml.train import LGBM_PARAMS, compute_optimal_threshold
from intelligence.signal_pipeline import horizon_stops
from intelligence import costs as costs_module
from config import settings

log = logging.getLogger("backtest.walkforward")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MIN_ROWS_PER_TICKER = 300          # matches train.py / edge_by_year.py
MIN_TRAIN_ROWS = 5000              # skip windows with too little training history
HOLD_DAYS = 5                      # matches config.TIMEFRAMES "5D" entry
BENCHMARK_TICKER = "NIFTYBEES"
EXCLUDE_YEARS_FROM_HEADLINE = {2001, 2002, 2003}   # in-sample bias per audit finding


# ------------------------------------------------------------------ #
# Cost model — defensive against the concurrent costs.py edit         #
# ------------------------------------------------------------------ #
def _net_return(gross_return_pct: float, product: str = "MIS") -> float:
    """
    Call intelligence.costs.net_return() with whatever signature it currently
    exposes. Another agent is landing a `product` kwarg concurrently; this
    checks the live signature first (cheap, via inspect) and falls back to
    the old single-arg call on TypeError so this file works before/after
    that change lands, without needing to coordinate.
    """
    try:
        sig = inspect.signature(costs_module.net_return)
        if "product" in sig.parameters:
            return costs_module.net_return(gross_return_pct, product=product)
    except (TypeError, ValueError):
        pass
    try:
        return costs_module.net_return(gross_return_pct, product)
    except TypeError:
        return costs_module.net_return(gross_return_pct)


# ------------------------------------------------------------------ #
# Data loading                                                        #
# ------------------------------------------------------------------ #
async def load_ohlcv(conn) -> pd.DataFrame:
    rows = await conn.fetch(
        "SELECT time, ticker, open, high, low, close, volume "
        "FROM ohlcv_daily WHERE close IS NOT NULL"
    )
    df = pd.DataFrame(rows, columns=["time", "ticker", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def build_features_per_ticker(raw: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Compute features ONCE per ticker over its full available history (cheap,
    pure pandas/ta-style rolling windows -- no leakage risk since features at
    time t only use data up to t). Re-used across every walk-forward window;
    only the LightGBM TRAINING slice changes per window, not the features.
    Returns {ticker: feat_df} where feat_df also carries raw high/low/close
    for trade simulation.
    """
    out = {}
    for ticker, group in raw.groupby("ticker"):
        if len(group) < MIN_ROWS_PER_TICKER:
            continue
        try:
            feat = compute_features(group, ticker=ticker)
        except Exception as e:
            log.warning("feature error %s: %s", ticker, e)
            continue
        feat["close_raw"] = group["close"]
        feat["high_raw"] = group["high"]
        feat["low_raw"] = group["low"]
        feat["ticker"] = ticker
        out[ticker] = feat
    return out


# ------------------------------------------------------------------ #
# Per-window training (reuses train.py's params + threshold logic)    #
# ------------------------------------------------------------------ #
def train_window_model(train_df: pd.DataFrame, feature_cols: list[str]):
    """
    Fit one LightGBM model on data strictly before the OOS window, using a
    time-ordered 90/10 split for early stopping + F1-optimal threshold
    selection. Same LGBM_PARAMS and compute_optimal_threshold as
    models/ml/train.py -- deliberately skips train.py's 5-fold CV loop (see
    module docstring) for per-window compute tractability.
    """
    X = train_df[feature_cols]
    y = train_df["target_buy"]

    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
    if n_pos < 20 or n_neg < 20:
        return None, None

    scale_pos_weight = n_neg / (n_pos + 1e-9)
    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    split = int(len(X) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    if X_val.empty or y_val.nunique() < 2:
        return None, None

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    threshold = compute_optimal_threshold(model, X_val, y_val)
    return model, threshold


# ------------------------------------------------------------------ #
# Trade simulation                                                    #
# ------------------------------------------------------------------ #
def simulate_trade(feat_df: pd.DataFrame, entry_pos: int, entry_price: float,
                    stop: float, target: float, hold_days: int) -> tuple[float, str, int]:
    """
    Walk forward up to `hold_days` trading days from entry_pos in feat_df
    (which carries high_raw/low_raw/close_raw). Stop is checked before target
    on any day both could trigger (conservative assumption, documented).
    Returns (exit_price, outcome, days_held).
    """
    n = len(feat_df)
    for h in range(1, hold_days + 1):
        pos = entry_pos + h
        if pos >= n:
            break
        lo = feat_df["low_raw"].iloc[pos]
        hi = feat_df["high_raw"].iloc[pos]
        if pd.notna(lo) and lo <= stop:
            return stop, "hit_stop", h
        if pd.notna(hi) and hi >= target:
            return target, "hit_target", h
    exit_pos = min(entry_pos + hold_days, n - 1)
    exit_price = feat_df["close_raw"].iloc[exit_pos]
    return float(exit_price), "expired", exit_pos - entry_pos


# ------------------------------------------------------------------ #
# Walk-forward driver                                                 #
# ------------------------------------------------------------------ #
def quarter_bounds(year: int, q: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_month = (q - 1) * 3 + 1
    start = pd.Timestamp(year=year, month=start_month, day=1, tz="UTC")
    end = start + pd.DateOffset(months=3)
    return start, end


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def _save_checkpoint(window_rows: list[dict], all_trades: list[dict], completed_through_year: int) -> None:
    """
    Called after every completed year (all 4 quarters) during a long
    walk-forward run. Writes the accumulated windows/trades so far to disk
    so progress is inspectable without waiting for the full run to finish,
    and so a killed/crashed run doesn't lose everything already computed.
    Overwrites the same two files each time (not one file per year) — the
    checkpoint is always "everything through the last completed year".
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pd.DataFrame(window_rows).to_csv(
        os.path.join(CHECKPOINT_DIR, "windows_checkpoint.csv"), index=False
    )
    pd.DataFrame(all_trades).to_csv(
        os.path.join(CHECKPOINT_DIR, "trades_checkpoint.csv"), index=False
    )
    with open(os.path.join(CHECKPOINT_DIR, "progress.txt"), "w") as f:
        f.write(
            f"Completed through year {completed_through_year}. "
            f"{len(window_rows)} windows, {len(all_trades)} trades so far. "
            f"Updated: {pd.Timestamp.now(tz='UTC').isoformat()}\n"
        )
    log.info(
        "Checkpoint saved: through year %d (%d windows, %d trades so far)",
        completed_through_year, len(window_rows), len(all_trades),
    )


def run_walkforward(raw: pd.DataFrame, from_year: int, to_year: int | None,
                     confidence_threshold: float | None) -> dict:
    feature_cols = get_all_feature_columns()
    per_ticker = build_features_per_ticker(raw, feature_cols)
    log.info("Built features for %d tickers", len(per_ticker))

    combined = pd.concat(per_ticker.values())
    combined = combined.sort_index()
    max_date = combined.index.max()
    if to_year is None:
        to_year = max_date.year

    window_rows = []
    all_trades = []

    year, q = from_year, 1
    while True:
        w_start, w_end = quarter_bounds(year, q)
        if w_start.year > to_year:
            break
        if w_start > max_date:
            break

        train_slice = combined[combined.index < w_start].dropna(
            subset=feature_cols + ["target_buy"]
        )
        oos_slice = combined[(combined.index >= w_start) & (combined.index < w_end)].dropna(
            subset=feature_cols
        )

        label = f"{year}-Q{q}"
        if len(train_slice) < MIN_TRAIN_ROWS or oos_slice.empty:
            log.info("%s: skipped (train_rows=%d, oos_rows=%d)", label, len(train_slice), len(oos_slice))
            q += 1
            if q > 4:
                _save_checkpoint(window_rows, all_trades, completed_through_year=year)
                q = 1
                year += 1
            continue

        model, threshold = train_window_model(train_slice, feature_cols)
        if model is None:
            log.info("%s: skipped (model training failed / degenerate class balance)", label)
            q += 1
            if q > 4:
                _save_checkpoint(window_rows, all_trades, completed_through_year=year)
                q = 1
                year += 1
            continue

        gate = confidence_threshold if confidence_threshold is not None else threshold
        probs = model.predict_proba(oos_slice[feature_cols])[:, 1]
        fired_mask = probs >= gate
        log.info("%s: train_rows=%d oos_rows=%d threshold=%.3f fired=%d",
                  label, len(train_slice), len(oos_slice), threshold, int(fired_mask.sum()))

        window_trades = []
        for (idx, row), fired in zip(oos_slice.iterrows(), fired_mask):
            if not fired:
                continue
            ticker = row["ticker"]
            fdf = per_ticker[ticker]
            entry_pos = fdf.index.get_loc(idx)
            if isinstance(entry_pos, slice) or isinstance(entry_pos, np.ndarray):
                continue  # duplicate timestamp edge case, skip defensively
            entry_price = float(row["close_raw"])
            atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else None
            stop, target = horizon_stops(entry_price, "BUY", atr, HOLD_DAYS)
            exit_price, outcome, days_held = simulate_trade(
                fdf, entry_pos, entry_price, stop, target, HOLD_DAYS
            )
            gross = (exit_price - entry_price) / entry_price
            net = _net_return(gross, product="MIS")
            window_trades.append({
                "window": label, "ticker": ticker, "entry_date": idx,
                "entry_price": entry_price, "exit_price": exit_price,
                "stop": stop, "target": target, "outcome": outcome,
                "days_held": days_held, "gross_return": gross, "net_return": net,
            })

        all_trades.extend(window_trades)
        n = len(window_trades)
        net_rets = np.array([t["net_return"] for t in window_trades]) if n else np.array([])
        window_rows.append({
            "window": label, "year": year, "quarter": q,
            "train_rows": len(train_slice), "oos_rows": len(oos_slice),
            "threshold": threshold, "trades": n,
            "expectancy_net_bps": float(net_rets.mean() * 10000) if n else float("nan"),
            "winrate": float((net_rets > 0).mean()) if n else float("nan"),
            "hit_target": sum(1 for t in window_trades if t["outcome"] == "hit_target"),
            "hit_stop": sum(1 for t in window_trades if t["outcome"] == "hit_stop"),
            "expired": sum(1 for t in window_trades if t["outcome"] == "expired"),
            "in_sample_flagged": year in EXCLUDE_YEARS_FROM_HEADLINE,
        })

        q += 1
        if q > 4:
            # A full year (all 4 quarters, or as many as had enough data/
            # fired trades) just completed — checkpoint before moving on.
            _save_checkpoint(window_rows, all_trades, completed_through_year=year)
            q = 1
            year += 1

    return {
        "windows": pd.DataFrame(window_rows),
        "trades": pd.DataFrame(all_trades),
    }


async def load_benchmark(conn, from_year: int, to_year: int | None) -> pd.DataFrame:
    rows = await conn.fetch(
        "SELECT time, close FROM ohlcv_daily WHERE ticker = $1 AND close IS NOT NULL ORDER BY time",
        BENCHMARK_TICKER,
    )
    df = pd.DataFrame(rows, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.set_index("time").sort_index()
    start = pd.Timestamp(year=from_year, month=1, day=1, tz="UTC")
    df = df[df.index >= start]
    if to_year:
        end = pd.Timestamp(year=to_year + 1, month=1, day=1, tz="UTC")
        df = df[df.index < end]
    return df


def benchmark_buy_and_hold(bench_df: pd.DataFrame) -> dict:
    if bench_df.empty or len(bench_df) < 2:
        return {"total_return_pct": float("nan"), "cagr_pct": float("nan")}
    start_px, end_px = float(bench_df["close"].iloc[0]), float(bench_df["close"].iloc[-1])
    total_return = (end_px - start_px) / start_px
    years = max((bench_df.index[-1] - bench_df.index[0]).days / 365.25, 0.01)
    cagr = (1 + total_return) ** (1 / years) - 1
    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "start_date": bench_df.index[0].date().isoformat(),
        "end_date": bench_df.index[-1].date().isoformat(),
        "start_price": start_px,
        "end_price": end_px,
    }


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #
async def main():
    ap = argparse.ArgumentParser(description="Genuine walk-forward backtest of the LightGBM BUY layer.")
    ap.add_argument("--from", dest="from_year", type=int, default=2019,
                     help="First OOS year (expanding window trains only on data before it). Default 2019.")
    ap.add_argument("--to", dest="to_year", type=int, default=None,
                     help="Last OOS year (inclusive). Default: latest year in the DB.")
    ap.add_argument("--confidence-threshold", type=float, default=None,
                     help="Override the per-window F1-optimal threshold with a fixed gate.")
    args = ap.parse_args()

    # NOTE: the DB connection used to load `raw` was previously held open
    # across the entire run_walkforward() call below -- which, for a
    # multi-year run, is many minutes to hours of pure CPU-bound synchronous
    # LightGBM training that blocks the asyncio event loop completely (no
    # await points, so the connection is never touched but also never gets
    # a chance to send/receive a keepalive). Docker Desktop's networking
    # (WSL2 vswitch) resets long-idle TCP connections, which surfaced as a
    # "ConnectionResetError: [WinError 10054]" crash — not a bug in the
    # walk-forward logic itself, an artifact of holding a DB connection open
    # and idle for far longer than any real query needs it. Fix: close the
    # connection immediately after loading the raw data (all of it is
    # already in memory as a DataFrame at that point, no further DB access
    # happens during training/simulation), then open a FRESH connection
    # afterward only for the benchmark load.
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        log.info("Loading ohlcv_daily...")
        raw = await load_ohlcv(conn)
        log.info("Loaded %d rows, %d tickers", len(raw), raw["ticker"].nunique())
    finally:
        await conn.close()

    result = run_walkforward(raw, args.from_year, args.to_year, args.confidence_threshold)

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        log.info("Loading benchmark (%s)...", BENCHMARK_TICKER)
        bench_df = await load_benchmark(conn, args.from_year, args.to_year)
        bench = benchmark_buy_and_hold(bench_df)
    finally:
        await conn.close()

    from backtest.report import render_report
    render_report(result["windows"], result["trades"], bench, args.from_year, args.to_year)


if __name__ == "__main__":
    asyncio.run(main())
