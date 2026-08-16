"""
Phase 2B: bound the survivorship-bias damage before paying for full
point-in-time bhavcopy ingestion.

Method: inject synthetic delisting shocks into the real, trained h=20
backtest. At each rebalance, a fraction of the currently-held positions
is randomly selected (weighted toward lower model-score names, since
delistings are not uniform across quality) and their realized return for
that period is overridden to -100% (total loss) instead of their actual
historical return. Sweep the annual shock rate and find where net alpha
crosses zero.

This does NOT reconstruct the true point-in-time universe (that is
Phase 2C, ~6,500 bhavcopy files). It answers a narrower, immediately
actionable question: how much delisting damage would the edge have to
absorb before it stops clearing costs? If the edge survives shock rates
well above any realistic NSE delisting rate, full ingestion drops in
priority. If it breaks at a rate below what's realistic, ingestion is
mandatory before anything else.

Usage: python research/survivorship_bound.py
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
from stocksense.evaluation.backtest import ScoredFold, train_and_score_fold
from stocksense.evaluation.walkforward import make_folds
from stocksense.execution.cost_model import apply_turnover_cost
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import RankerConfig
from stocksense.portfolio.construct import (
    apply_no_trade_band,
    enforce_turnover_budget,
    one_way_turnover,
    target_weights_top_n,
)

log = structlog.get_logger(__name__)

HORIZON = 20
TOP_N = 20
COST_BPS = 25.0
BARS_PER_YEAR = 252
ANNUAL_SHOCK_RATES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]  # 0% to 25%/year
N_MONTE_CARLO_PER_RATE = 200
SEED = 42


def annual_to_period_prob(annual_rate: float, horizon_bars: int) -> float:
    """Convert an annual per-name failure probability to a per-rebalance-
    period probability, assuming independence across periods."""
    period_frac = horizon_bars / BARS_PER_YEAR
    return 1.0 - (1.0 - annual_rate) ** period_frac


def simulate_with_delisting_shocks(
    scored: ScoredFold, top_n: int, round_trip_cost_bps: float, annual_shock_rate: float, rng: np.random.Generator,
    no_trade_band: float = 0.02, max_turnover_per_rebalance: float = 1.0,
) -> float | None:
    """One Monte Carlo draw of the h=20/top_n=20 backtest with delisting
    shocks injected. Returns mean alpha_net across rebalances, or None if
    the fold produced no usable periods."""
    period_prob = annual_to_period_prob(annual_shock_rate, scored.horizon_bars)
    current_weights = pd.Series(dtype=float)
    net_alphas = []

    for rdate in scored.rebalance_dates:
        scores = scored.scores_by_date[rdate]
        raw_actual = scored.raw_actual_by_date[rdate].copy()

        target = target_weights_top_n(scores, top_n, symbols=list(scores.index))
        target = apply_no_trade_band(target, current_weights, band=no_trade_band)
        target = enforce_turnover_budget(target, current_weights, max_turnover=max_turnover_per_rebalance)

        held_symbols = target[target > 0].index
        if period_prob > 0 and len(held_symbols) > 0:
            # weight shock probability toward LOWER-score (weaker) held
            # names: rank ascending by score, invert to weights so the
            # weakest name is most likely to be the one shocked. This is
            # an approximation for "failures cluster in weaker names" —
            # not a substitute for real point-in-time delisting data.
            held_scores = scores.reindex(held_symbols)
            ranks = held_scores.rank(ascending=True)  # 1 = weakest
            weakness_weight = (len(ranks) - ranks + 1) / ranks.sum()
            shocked_mask = rng.random(len(held_symbols)) < (period_prob * weakness_weight * len(held_symbols))
            shocked_symbols = held_symbols[shocked_mask]
            for sym in shocked_symbols:
                if sym in raw_actual.index:
                    raw_actual[sym] = -1.0  # total loss, simulating delisting/failure

        turnover = one_way_turnover(target, current_weights)
        cost = apply_turnover_cost(turnover, round_trip_cost_bps)

        held = target.reindex(raw_actual.index, fill_value=0.0)
        gross_ret = float((held * raw_actual).sum())
        net_ret = gross_ret - cost
        bench_ret = float(raw_actual.mean())
        net_alphas.append(net_ret - bench_ret)

        drifted_value = held * (1 + raw_actual)
        total_value = float(drifted_value.sum())
        current_weights = (drifted_value / total_value)[lambda s: s > 0] if total_value > 0 else pd.Series(dtype=float)

    return float(np.mean(net_alphas)) if net_alphas else None


def main() -> None:
    settings = get_settings()
    store = Store(settings.duckdb_path)
    candles = store.read_candles()
    store.close()
    candles, quarantined = quarantine_symbols(candles)
    log.info("data_loaded", rows=len(candles), quarantined=quarantined)

    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]
    labeled = add_forward_return_labels(candles, horizon_bars=HORIZON)
    labeled = add_relative_forward_return(labeled, horizon_bars=HORIZON)

    trading_dates = pd.DatetimeIndex(sorted(feats["date"].unique()))
    test_window = max(21, HORIZON * 12)
    folds = make_folds(trading_dates, horizon_bars=HORIZON, test_window_bars=test_window)
    log.info("folds_built", n_folds=len(folds))

    scored_folds = []
    for fold in folds:
        scored = train_and_score_fold(feats, labeled, fcols, fold, horizon_bars=HORIZON, ranker_config=RankerConfig(random_state=settings.random_seed))
        if scored is not None:
            scored_folds.append(scored)
    log.info("folds_scored", n_scored=len(scored_folds))

    rng = np.random.default_rng(SEED)
    rows = []
    for annual_rate in ANNUAL_SHOCK_RATES:
        mc_means = []
        for _ in range(N_MONTE_CARLO_PER_RATE):
            fold_alphas = []
            for scored in scored_folds:
                alpha = simulate_with_delisting_shocks(scored, TOP_N, COST_BPS, annual_rate, rng)
                if alpha is not None:
                    fold_alphas.append(alpha)
            if fold_alphas:
                mc_means.append(float(np.mean(fold_alphas)))

        mc_arr = np.array(mc_means)
        rows.append(
            {
                "annual_shock_rate": annual_rate,
                "mean_alpha_net": float(mc_arr.mean()),
                "p5_alpha_net": float(np.percentile(mc_arr, 5)),
                "p95_alpha_net": float(np.percentile(mc_arr, 95)),
                "pct_mc_draws_positive": float((mc_arr > 0).mean()),
            }
        )
        log.info(
            "shock_rate_result", annual_rate=annual_rate,
            mean_alpha=rows[-1]["mean_alpha_net"], pct_positive=rows[-1]["pct_mc_draws_positive"],
        )

    df = pd.DataFrame(rows)
    out_path = Path(__file__).parent / "survivorship_bound_results.csv"
    df.to_csv(out_path, index=False)

    print("\n=== SURVIVORSHIP SHOCK-RATE SENSITIVITY (h=20, top_n=20, 25bps) ===")
    print(df.to_string(index=False))

    # find approximate break-even shock rate via linear interpolation on mean_alpha_net
    breakeven_rate = None
    for i in range(len(df) - 1):
        a, b = df.iloc[i], df.iloc[i + 1]
        if a["mean_alpha_net"] > 0 >= b["mean_alpha_net"]:
            frac = a["mean_alpha_net"] / (a["mean_alpha_net"] - b["mean_alpha_net"])
            breakeven_rate = a["annual_shock_rate"] + frac * (b["annual_shock_rate"] - a["annual_shock_rate"])
            break

    print(f"\nApproximate annual shock rate where mean net alpha crosses zero: "
          f"{breakeven_rate:.1%}" if breakeven_rate is not None else
          "\nEdge did not break within the tested shock-rate range (0-25%/year).")


if __name__ == "__main__":
    main()
