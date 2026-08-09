"""
Fold-level backtest, split into two stages so the sweep doesn't pay for
retraining LightGBM at every (top_n, cost) grid point:

  1. train_and_score_fold  — expensive: fits the ranker once per
     (horizon, fold), produces per-rebalance-date scores + realized
     returns. Depends only on horizon (the label) and the fold.
  2. simulate_portfolio     — cheap: given cached scores, constructs the
     portfolio and charges costs for one (top_n, cost) combination. This
     is what research/phase0_sweep.py calls at every grid point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stocksense.evaluation.walkforward import Fold, split
from stocksense.execution.cost_model import apply_turnover_cost
from stocksense.models.ranker import CrossSectionalRanker, RankerConfig
from stocksense.portfolio.construct import (
    apply_no_trade_band,
    enforce_turnover_budget,
    one_way_turnover,
    target_weights_top_n,
)


@dataclass
class ScoredFold:
    """Cached output of training + scoring one (horizon, fold)."""

    fold_id: int
    horizon_bars: int
    n_train_rows: int
    # one entry per rebalance date: (scores, raw_actual, rel_actual) indexed by symbol
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    scores_by_date: dict = field(default_factory=dict)
    raw_actual_by_date: dict = field(default_factory=dict)
    rel_actual_by_date: dict = field(default_factory=dict)
    feature_importance: pd.Series | None = None


@dataclass
class FoldResult:
    fold_id: int
    horizon_bars: int
    top_n: int
    round_trip_cost_bps: float
    n_rebalances: int
    gross_expectancy: float
    net_expectancy: float
    benchmark_expectancy: float
    alpha_gross: float
    alpha_net: float
    mean_turnover: float
    information_coefficient: float
    hit_rate: float


def train_and_score_fold(
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    feature_cols: list[str],
    fold: Fold,
    horizon_bars: int,
    ranker_config: RankerConfig | None = None,
) -> ScoredFold | None:
    raw_col = f"fwd_ret_{horizon_bars}b"
    rel_col = f"{raw_col}_rel"

    merged = feats.merge(
        labeled[["symbol", "date", raw_col, rel_col]], on=["symbol", "date"], how="inner"
    )
    train_df, test_df = split(merged, fold)
    train_df = train_df.dropna(subset=[rel_col])
    if len(train_df) < 500 or test_df.empty:
        return None

    ranker = CrossSectionalRanker(ranker_config)
    ranker.fit(train_df[feature_cols], train_df[rel_col])

    test_dates = sorted(test_df["date"].unique())
    rebalance_dates = test_dates[::horizon_bars]
    if len(rebalance_dates) < 2:
        return None

    out = ScoredFold(
        fold_id=fold.fold_id,
        horizon_bars=horizon_bars,
        n_train_rows=len(train_df),
        feature_importance=ranker.feature_importance(),
    )

    for rdate in rebalance_dates:
        day_df = test_df[test_df["date"] == rdate].dropna(subset=feature_cols, how="any")
        if day_df.empty:
            continue
        scores = pd.Series(
            ranker.predict(day_df[feature_cols]).values, index=day_df["symbol"].values
        )
        raw_actual = pd.Series(day_df[raw_col].values, index=day_df["symbol"].values)
        rel_actual = pd.Series(day_df[rel_col].values, index=day_df["symbol"].values)

        out.rebalance_dates.append(rdate)
        out.scores_by_date[rdate] = scores
        out.raw_actual_by_date[rdate] = raw_actual
        out.rel_actual_by_date[rdate] = rel_actual

    if len(out.rebalance_dates) < 2:
        return None
    return out


def simulate_portfolio(
    scored: ScoredFold,
    top_n: int,
    round_trip_cost_bps: float,
    no_trade_band: float = 0.02,
    max_turnover_per_rebalance: float = 1.0,
) -> FoldResult | None:
    current_weights = pd.Series(dtype=float)
    gross_returns, net_returns, bench_returns, turnovers = [], [], [], []
    all_scores, all_rel_returns = [], []

    for rdate in scored.rebalance_dates:
        scores = scored.scores_by_date[rdate]
        raw_actual = scored.raw_actual_by_date[rdate]
        rel_actual = scored.rel_actual_by_date[rdate]

        all_scores.append(scores)
        all_rel_returns.append(rel_actual)

        target = target_weights_top_n(scores, top_n, symbols=list(scores.index))
        target = apply_no_trade_band(target, current_weights, band=no_trade_band)
        target = enforce_turnover_budget(target, current_weights, max_turnover=max_turnover_per_rebalance)

        turnover = one_way_turnover(target, current_weights)
        cost = apply_turnover_cost(turnover, round_trip_cost_bps)

        held = target.reindex(raw_actual.index, fill_value=0.0)
        gross_ret = float((held * raw_actual).sum())
        net_ret = gross_ret - cost
        bench_ret = float(raw_actual.mean())

        gross_returns.append(gross_ret)
        net_returns.append(net_ret)
        bench_returns.append(bench_ret)
        turnovers.append(turnover)

        current_weights = target[target > 0]

    if not net_returns:
        return None

    scores_concat = pd.concat(all_scores)
    rel_concat = pd.concat(all_rel_returns)
    valid = scores_concat.notna() & rel_concat.notna()
    ic = (
        float(np.corrcoef(scores_concat[valid], rel_concat[valid])[0, 1])
        if valid.sum() > 10
        else float("nan")
    )

    gross_arr = np.array(gross_returns)
    net_arr = np.array(net_returns)
    bench_arr = np.array(bench_returns)

    return FoldResult(
        fold_id=scored.fold_id,
        horizon_bars=scored.horizon_bars,
        top_n=top_n,
        round_trip_cost_bps=round_trip_cost_bps,
        n_rebalances=len(net_returns),
        gross_expectancy=float(gross_arr.mean()),
        net_expectancy=float(net_arr.mean()),
        benchmark_expectancy=float(bench_arr.mean()),
        alpha_gross=float((gross_arr - bench_arr).mean()),
        alpha_net=float((net_arr - bench_arr).mean()),
        mean_turnover=float(np.mean(turnovers)),
        information_coefficient=ic,
        hit_rate=float((net_arr > 0).mean()),
    )
