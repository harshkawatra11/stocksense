"""
Phase 0 adversarial stress battery, part 2: Monte Carlo path reshuffling
and hyperparameter perturbation on the winning configuration (h=20,
top_n=20), per docs/10-evaluation.md sections 10-11. Completes the two
items research/phase0_verdict.md named as still outstanding.

Usage: python research/phase0_stress.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
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

WINNING_HORIZON = 20
WINNING_TOP_N = 20
REFERENCE_COST_BPS = 25.0
N_MONTE_CARLO_PATHS = 5000


def load_scored_folds(horizon: int, ranker_config: RankerConfig):
    settings = get_settings()
    store = Store(settings.duckdb_path)
    candles = store.read_candles()
    store.close()
    candles, _ = quarantine_symbols(candles)  # AUDIT FIX: this script never got the ADANIENT-bug quarantine

    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]

    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)

    trading_dates = pd.DatetimeIndex(sorted(feats["date"].unique()))
    test_window = max(21, horizon * 12)
    folds = make_folds(trading_dates, horizon_bars=horizon, test_window_bars=test_window)

    scored_list = []
    for fold in folds:
        scored = train_and_score_fold(feats, labeled, fcols, fold, horizon_bars=horizon, ranker_config=ranker_config)
        if scored is not None:
            scored_list.append(scored)
    return scored_list


def monte_carlo(net_returns: np.ndarray, n_paths: int, seed: int = 42) -> dict:
    """Reshuffle the observed per-rebalance return sequence n_paths times,
    build cumulative equity curves, and report the outcome distribution —
    per docs/10-evaluation.md section 11. This does NOT generate new
    returns; it resamples the order of the ones actually observed, so it
    tests sequence/path risk given the empirical return distribution."""
    rng = np.random.default_rng(seed)
    n = len(net_returns)
    terminal_returns = np.empty(n_paths)
    max_drawdowns = np.empty(n_paths)

    for i in range(n_paths):
        path = rng.permutation(net_returns)
        equity = np.cumprod(1 + path)
        terminal_returns[i] = equity[-1] - 1.0
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdowns[i] = drawdown.min()

    return {
        "n_rebalances": n,
        "n_paths": n_paths,
        "mean_terminal_return": float(terminal_returns.mean()),
        "median_terminal_return": float(np.median(terminal_returns)),
        "p5_terminal_return": float(np.percentile(terminal_returns, 5)),
        "p95_terminal_return": float(np.percentile(terminal_returns, 95)),
        "prob_terminal_negative": float((terminal_returns < 0).mean()),
        "mean_max_drawdown": float(max_drawdowns.mean()),
        "p5_max_drawdown": float(np.percentile(max_drawdowns, 5)),  # worst 5% of paths
        "prob_drawdown_worse_than_10pct": float((max_drawdowns < -0.10).mean()),
        "prob_drawdown_worse_than_20pct": float((max_drawdowns < -0.20).mean()),
    }


def run_monte_carlo() -> None:
    print("\n=== MONTE CARLO: reshuffling the observed h=20/n=20 return sequence ===")
    scored_folds = load_scored_folds(WINNING_HORIZON, RankerConfig(random_state=42))
    all_net_returns: list[float] = []
    for scored in scored_folds:
        result = simulate_portfolio(scored, top_n=WINNING_TOP_N, round_trip_cost_bps=REFERENCE_COST_BPS)
        if result is not None:
            all_net_returns.extend(result.net_returns)

    net_arr = np.array(all_net_returns)
    print(f"pooled rebalance-period returns: n={len(net_arr)}, mean={net_arr.mean():+.4%}, std={net_arr.std():.4%}")

    mc = monte_carlo(net_arr, N_MONTE_CARLO_PATHS)
    for k, v in mc.items():
        if isinstance(v, float) and abs(v) < 1:
            print(f"  {k}: {v:+.2%}" if "return" in k or "drawdown" in k else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    pd.DataFrame([mc]).to_csv(Path(__file__).parent / "phase0_montecarlo.csv", index=False)


def run_parameter_perturbation() -> None:
    """Per docs/10-evaluation.md section 10: does a small change to the
    ranker's hyperparameters collapse the result? If so, the original
    configuration was fitted to noise, not signal."""
    print("\n=== PARAMETER PERTURBATION: ranker hyperparameters +/-20% ===")
    base = RankerConfig(random_state=42)
    variants = {
        "base": base,
        "num_leaves-20%": replace(base, num_leaves=int(base.num_leaves * 0.8)),
        "num_leaves+20%": replace(base, num_leaves=int(base.num_leaves * 1.2)),
        "learning_rate-20%": replace(base, learning_rate=base.learning_rate * 0.8),
        "learning_rate+20%": replace(base, learning_rate=base.learning_rate * 1.2),
        "n_estimators-20%": replace(base, n_estimators=int(base.n_estimators * 0.8)),
        "n_estimators+20%": replace(base, n_estimators=int(base.n_estimators * 1.2)),
        "seed=7": replace(base, random_state=7),
        "seed=123": replace(base, random_state=123),
    }

    rows = []
    for name, config in variants.items():
        scored_folds = load_scored_folds(WINNING_HORIZON, config)
        fold_alphas = []
        for scored in scored_folds:
            result = simulate_portfolio(scored, top_n=WINNING_TOP_N, round_trip_cost_bps=REFERENCE_COST_BPS)
            if result is not None:
                fold_alphas.append(result.alpha_net)
        arr = np.array(fold_alphas)
        rows.append(
            {
                "variant": name,
                "n_folds": len(arr),
                "mean_alpha_net": float(arr.mean()) if len(arr) else float("nan"),
                "pct_folds_positive": float((arr > 0).mean()) if len(arr) else float("nan"),
            }
        )
        print(f"  {name:>20}: mean_alpha_net={rows[-1]['mean_alpha_net']:+.4%}  pct_positive={rows[-1]['pct_folds_positive']:.0%}")

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "phase0_parameter_perturbation.csv", index=False)

    base_alpha = df[df.variant == "base"]["mean_alpha_net"].iloc[0]
    others = df[df.variant != "base"]["mean_alpha_net"]
    collapsed = (others <= 0).any()
    print(f"\n  base mean_alpha_net={base_alpha:+.4%}; any perturbed variant <= 0: {collapsed}")


if __name__ == "__main__":
    run_monte_carlo()
    run_parameter_perturbation()
    print("\n=== STRESS BATTERY COMPLETE ===")
