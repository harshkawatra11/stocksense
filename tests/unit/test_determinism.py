"""
Determinism: identical config + seed must produce identical features,
model predictions, and fold metrics. Claimed in the Phase 0 plan's
verification section but never actually tested (audit finding MED-12)
— retraining rigor (docs/06) depends on this holding, since the gate
compares candidate metrics against a fixed bar and a nondeterministic
pipeline would make every comparison noise of unknown magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocksense.evaluation.backtest import simulate_portfolio, train_and_score_fold
from stocksense.evaluation.walkforward import make_folds
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import RankerConfig


def _synthetic_candles(n_symbols: int = 8, n_days: int = 900, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for s in range(n_symbols):
        symbol = f"SYN{s}"
        price = 100.0 + s * 10
        for d in dates:
            ret = rng.normal(0.0005, 0.015)
            price *= 1 + ret
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            open_ = price * (1 + rng.normal(0, 0.003))
            vol = abs(rng.normal(1_000_000, 200_000))
            rows.append(
                {
                    "symbol": symbol, "date": d,
                    "open": open_, "high": max(high, open_, price), "low": min(low, open_, price),
                    "close": price, "adj_close": price, "volume": vol, "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def _run_once(candles: pd.DataFrame, horizon: int, seed: int):
    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]
    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)

    dates = pd.DatetimeIndex(sorted(feats["date"].unique()))
    folds = make_folds(dates, horizon_bars=horizon, test_window_bars=max(21, horizon * 6), min_train_bars=400)
    assert folds, "test fixture must produce at least one fold"

    scored = train_and_score_fold(feats, labeled, fcols, folds[0], horizon_bars=horizon, ranker_config=RankerConfig(random_state=seed))
    assert scored is not None
    result = simulate_portfolio(scored, top_n=3, round_trip_cost_bps=25.0)
    assert result is not None
    return feats, scored, result


def test_features_are_deterministic() -> None:
    candles = _synthetic_candles()
    feats_a = build_features(candles)
    feats_b = build_features(candles)
    pd.testing.assert_frame_equal(feats_a, feats_b)


def test_full_pipeline_identical_seed_produces_identical_metrics() -> None:
    candles = _synthetic_candles()
    _, _, result_a = _run_once(candles, horizon=10, seed=42)
    _, _, result_b = _run_once(candles, horizon=10, seed=42)

    assert result_a.gross_expectancy == result_b.gross_expectancy
    assert result_a.net_expectancy == result_b.net_expectancy
    assert result_a.information_coefficient == result_b.information_coefficient
    assert result_a.net_returns == result_b.net_returns


def test_different_seed_can_produce_different_metrics() -> None:
    """Sanity check on the sanity check: if two different seeds ALSO
    produced bit-identical results, the determinism tests above would be
    trivially passing because nothing in the pipeline actually depends on
    the seed — not because it's truly deterministic."""
    candles = _synthetic_candles()
    _, _, result_a = _run_once(candles, horizon=10, seed=42)
    _, _, result_b = _run_once(candles, horizon=10, seed=999)
    assert result_a.net_returns != result_b.net_returns
