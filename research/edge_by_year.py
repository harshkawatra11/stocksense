"""
Phase-0 edge re-check — per-year diagnostic for the LightGBM BUY signal.

Rebuilds the (lost) research/ by-year blotter analysis as a committed, repeatable
script. Answers the only question that matters before building infra on top of the
model: does the edge GENERALISE, or is it concentrated in one year and inverted in
the live-relevant recent regime?

What it does:
  1. Loads ohlcv_daily from the DB (same query as models.ml.train).
  2. Rebuilds features per ticker via the exact training pipeline (compute_features).
  3. Scores every row with the saved model (lgbm_latest.pkl) at its own threshold.
  4. For rows where the model fires BUY (prob >= threshold), measures the realised
     next-day return (target_1d_return) grouped by calendar year.
  5. Compares model-conditional expectancy to the UNCONDITIONAL base rate of the
     same year (the alpha-vs-beta test) and reports net-of-cost expectancy.

Run:  python -m research.edge_by_year
Optional: --cost-bps 25  (round-trip cost in basis points, default 25 = 0.25%)
          --min-rows 300 (minimum history per ticker, matches train.py)

Pure CPU/DB — safe to run alongside GPU training.
"""
from __future__ import annotations

import argparse
import asyncio
import pickle
import json
import os
import logging

import numpy as np
import pandas as pd
import asyncpg

from data.pipeline.feature_engineering import compute_features
from config import settings

log = logging.getLogger("edge_by_year")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SAVED_DIR = os.path.join(str(settings.MODELS_DIR), "ml", "saved")
MODEL_PATH = os.path.join(SAVED_DIR, "lgbm_latest.pkl")
META_PATH = os.path.join(SAVED_DIR, "lgbm_20260604_1048_meta.json")
THRESH_PATH = os.path.join(SAVED_DIR, "lgbm_latest_threshold.json")


def _load_model_and_meta():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(THRESH_PATH) as f:
        threshold = json.load(f)["threshold"]
    return model, meta["feature_cols"], threshold


async def _load_ohlcv(conn) -> pd.DataFrame:
    rows = await conn.fetch(
        "SELECT time, ticker, open, high, low, close, volume "
        "FROM ohlcv_daily WHERE close IS NOT NULL"
    )
    df = pd.DataFrame(rows, columns=["time", "ticker", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def _accumulate(acc: dict, year: int, rets: np.ndarray):
    """Accumulate per-year sums for a vector of realised next-day returns."""
    if year not in acc:
        acc[year] = {"n": 0, "sum_ret": 0.0, "n_win": 0, "n_hit": 0}
    a = acc[year]
    a["n"] += len(rets)
    a["sum_ret"] += float(rets.sum())
    a["n_win"] += int((rets > 0).sum())
    a["n_hit"] += int((rets > 0.015).sum())


def run_analysis(df: pd.DataFrame, model, feature_cols, threshold, min_rows: int, cost: float):
    sig_acc: dict[int, dict] = {}   # model-conditional (BUY fired)
    base_acc: dict[int, dict] = {}  # unconditional benchmark (all rows)

    n_tickers = 0
    for ticker, group in df.groupby("ticker"):
        if len(group) < min_rows:
            continue
        try:
            feat = compute_features(group, ticker=ticker)
        except Exception as e:
            log.warning("feature error %s: %s", ticker, e)
            continue
        feat = feat.dropna(subset=feature_cols + ["target_1d_return"])
        if feat.empty:
            continue
        n_tickers += 1

        prob = model.predict_proba(feat[feature_cols])[:, 1]
        ret = feat["target_1d_return"].to_numpy()
        years = feat.index.year.to_numpy()

        # unconditional benchmark, per year
        for y in np.unique(years):
            _accumulate(base_acc, int(y), ret[years == y])

        # model-conditional, per year
        fired = prob >= threshold
        if fired.any():
            fy, fr = years[fired], ret[fired]
            for y in np.unique(fy):
                _accumulate(sig_acc, int(y), fr[fy == y])

    log.info("Processed %d tickers (>= %d rows)", n_tickers, min_rows)

    rows = []
    for y in sorted(set(sig_acc) | set(base_acc)):
        s = sig_acc.get(y)
        b = base_acc.get(y, {"n": 0, "sum_ret": 0.0, "n_win": 0})
        bench_exp = (b["sum_ret"] / b["n"]) if b["n"] else float("nan")
        if s and s["n"]:
            exp = s["sum_ret"] / s["n"]
            rows.append({
                "year": y,
                "signals": s["n"],
                "expectancy_gross": exp,
                "expectancy_net": exp - cost,
                "winrate": s["n_win"] / s["n"],
                "hit_rate_1.5pct": s["n_hit"] / s["n"],
                "bench_exp": bench_exp,
                "alpha": exp - bench_exp,
            })
        else:
            rows.append({
                "year": y, "signals": 0, "expectancy_gross": float("nan"),
                "expectancy_net": float("nan"), "winrate": float("nan"),
                "hit_rate_1.5pct": float("nan"), "bench_exp": bench_exp, "alpha": float("nan"),
            })
    return pd.DataFrame(rows).set_index("year")


def print_report(tbl: pd.DataFrame, cost: float):
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print(f"PHASE-0 EDGE BY YEAR  —  lgbm_latest, round-trip cost = {cost*100:.2f}%")
    print("  expectancy = mean realised next-day return when model fires BUY")
    print("  bench_exp  = mean next-day return of ALL rows that year (unconditional)")
    print("  alpha      = expectancy - bench_exp (selection skill, beta-removed)")
    print("=" * 90)
    show = tbl.copy()
    for c in ["expectancy_gross", "expectancy_net", "winrate", "hit_rate_1.5pct", "bench_exp", "alpha"]:
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:+.4f}")
    print(show.to_string())

    # ---- Verdict ----
    traded = tbl[tbl["signals"] > 0]
    total_pnl = (tbl["expectancy_net"] * tbl["signals"]).sum()
    recent = [y for y in (2025, 2026) if y in tbl.index and tbl.loc[y, "signals"] > 0]
    print("\n" + "-" * 90)
    print("VERDICT")
    if not len(traded):
        print("  No signals fired — cannot assess edge.")
        return
    # concentration: top year's share of total net PnL
    yearly_pnl = (tbl["expectancy_net"] * tbl["signals"]).dropna()
    pos_total = yearly_pnl[yearly_pnl > 0].sum()
    if pos_total > 0:
        top_year = yearly_pnl.idxmax()
        share = yearly_pnl.max() / pos_total
        print(f"  Net PnL concentration: {top_year} = {share*100:.0f}% of all positive-year PnL")
    print(f"  Total net expectancy-weighted PnL across years: {total_pnl:+.3f}")
    for y in recent:
        r = tbl.loc[y]
        flag = "INVERTED" if r["alpha"] < 0 else "positive"
        print(f"  {y}: net exp {r['expectancy_net']:+.4f}  winrate {r['winrate']:.3f}  "
              f"alpha {r['alpha']:+.4f}  -> selection {flag}")
    print("-" * 90)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=25.0, help="round-trip cost in bps (default 25 = 0.25%)")
    ap.add_argument("--min-rows", type=int, default=300)
    args = ap.parse_args()
    cost = args.cost_bps / 10000.0

    model, feature_cols, threshold = _load_model_and_meta()
    log.info("Model loaded: %d features, threshold=%.2f", len(feature_cols), threshold)

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        log.info("Loading ohlcv_daily...")
        df = await _load_ohlcv(conn)
        log.info("Loaded %d rows, %d tickers", len(df), df["ticker"].nunique())
    finally:
        await conn.close()

    tbl = run_analysis(df, model, feature_cols, threshold, args.min_rows, cost)
    print_report(tbl, cost)

    out = os.path.join(os.path.dirname(__file__), "edge_by_year.csv")
    tbl.to_csv(out)
    log.info("Saved %s", out)


if __name__ == "__main__":
    asyncio.run(main())
