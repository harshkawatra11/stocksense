"""
Intraday backtest loop (Phase E4). Cannot reuse
evaluation.backtest.simulate_portfolio -- that loop is PORTFOLIO-WEIGHT
shaped (target weights, turnover, drift, cost as a fraction of book).
The intraday strategy is TRADE-shaped: discrete entries, each exiting on
a first-touch stop/target/time outcome (labels/intraday_labels.py).
Forcing one into the other would either fabricate a turnover number or
throw away the path-dependent exit that is the entire point of the
first-touch label.

This module emits the SAME evaluation.backtest.FoldResult dataclass the
daily pipeline does, so evaluation.gate.evaluate_gate judges an intraday
sweep with EXACTLY the same, unmodified code -- only the aggregation
unit differs (per-trade, not per-calendar-rebalance), which FoldResult's
`net_returns: list[float]` already accommodates generically.

Session-based folds reuse evaluation.walkforward.make_folds UNCHANGED:
passing one Timestamp per trading SESSION (midnight-normalized) as the
'trading_dates' index, with bar-count parameters reinterpreted as
session-counts, produces session-aligned fold boundaries with no new
fold-construction code -- make_folds is generic over its index's units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stocksense.evaluation.backtest import FoldResult
from stocksense.evaluation.walkforward import Fold, make_folds
from stocksense.execution.cost_model import compute_charges
from stocksense.execution.fill_model import simulate_fill
from stocksense.labels.intraday_labels import first_touch_label, precompute_sessions
from stocksense.models.ranker import CrossSectionalRanker, RankerConfig


def make_session_folds(
    session_dates,
    min_train_sessions: int = 500,
    test_window_sessions: int = 42,
    embargo_sessions: int = 1,
) -> list[Fold]:
    """Thin session-unit wrapper around make_folds -- horizon_bars=0
    because the intraday label's own purge is handled differently (a
    trade never straddles a session boundary at all, by construction of
    first_touch_label), so only the embargo term is needed here."""
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(session_dates)).dt.normalize().unique()))
    return make_folds(
        idx, horizon_bars=0, test_window_bars=test_window_sessions,
        min_train_bars=min_train_sessions, embargo_buffer_bars=embargo_sessions,
    )


def session_split(bars: pd.DataFrame, fold: Fold, ts_col: str = "ts") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like walkforward.split, but compares each bar's SESSION DATE
    (normalized) against the fold's session-date boundaries -- a plain
    timestamp comparison would incorrectly exclude every bar after
    midnight on the fold's boundary sessions."""
    session_date = pd.to_datetime(bars[ts_col]).dt.normalize()
    train = bars.loc[(session_date >= fold.train_start) & (session_date <= fold.train_end)]
    test = bars.loc[(session_date >= fold.test_start) & (session_date <= fold.test_end)]
    return train, test


def add_relative_session_return(labeled: pd.DataFrame, raw_col: str, ts_col: str = "ts") -> pd.DataFrame:
    """Cross-sectional relative return at each rebalance timestamp --
    the intraday analogue of labels.forward_return.add_relative_forward_return,
    which groups by a 'date' column this frame doesn't have (it has 'ts')."""
    rel_col = f"{raw_col}_rel"
    out = labeled.copy()
    out[rel_col] = out[raw_col] - out.groupby(ts_col)[raw_col].transform("mean")
    return out


@dataclass
class IntradayTrade:
    symbol: str
    entry_ts: pd.Timestamp
    fill_price: float
    exit_ts: pd.Timestamp | None
    exit_price: float | None
    outcome: str  # 'stop' | 'target' | 'time_exit' | 'no_fill' | 'no_data'
    quantity: float
    gross_ret: float
    net_ret: float
    benchmark_ret: float


def train_intraday_ranker(
    feats: pd.DataFrame, labeled: pd.DataFrame, feature_cols: list[str],
    fold: Fold, rel_col: str, ranker_config: RankerConfig | None = None,
) -> CrossSectionalRanker | None:
    """Trains on TRAIN sessions only, using the path-INDEPENDENT relative
    session return as the ranking target (matching CrossSectionalRanker's
    existing contract) -- the path-DEPENDENT first-touch outcome is only
    ever used at evaluation time, in simulate_intraday_trades_for_fold,
    never as a training signal (it doesn't exist until an entry has
    already been decided)."""
    merged = feats.merge(labeled[["symbol", "ts", rel_col]], on=["symbol", "ts"], how="inner")
    train_df, _ = session_split(merged, fold)
    train_df = train_df.dropna(subset=[rel_col] + feature_cols)
    if len(train_df) < 500:
        return None

    ranker = CrossSectionalRanker(ranker_config)
    ranker.fit(train_df[feature_cols], train_df[rel_col])
    return ranker


def simulate_intraday_trades_for_fold(
    scores_by_ts: dict,
    bars_5min_test: pd.DataFrame,
    bars_1min: pd.DataFrame,
    benchmark_by_ts: dict,
    top_n: int,
    stop_pct: float,
    target_pct: float,
    max_holding_minutes: int,
    exposure_inr: float,
    leverage_table: dict | None = None,
    max_participation_pct: float = 0.1,
    half_spread_bps: float = 2.5,
) -> list[IntradayTrade]:
    """The core trade loop: at each rebalance timestamp, among symbols
    with NO currently-open position, rank by predicted score, take
    top-N, attempt a fill against that symbol's OWN next 5-min bar
    (never the signal bar's own close -- fill_model.simulate_fill
    enforces this), and for every successful fill simulate the exit via
    first_touch_label against the true 1-minute path. A symbol with an
    open position is skipped for new entries until its exit resolves --
    this is what makes 'top-N' a real capacity constraint instead of
    double-counting the same name.

    PERFORMANCE FIX (found live 2026-08-20): this loop can call
    first_touch_label hundreds of times in a single fold (once per
    filled trade). It used to rebuild bars_1min's full session grouping
    from scratch on every one of those calls -- O(n_trades x
    fold_data_size), which stalled the real 244-symbol sweep for 6.5+
    hours on its FIRST fold alone (confirmed alive via CPU time, not
    hung -- just redoing the same expensive regroup repeatedly). Now
    built once via labels.intraday_labels.precompute_sessions and reused
    for every trade this fold considers.
    """
    bars_5min_test = bars_5min_test.sort_values(["symbol", "ts"]).reset_index(drop=True)
    bars_by_symbol = {sym: g.reset_index(drop=True) for sym, g in bars_5min_test.groupby("symbol")}
    precomputed_sessions = precompute_sessions(bars_1min)

    open_until: dict[str, pd.Timestamp] = {}
    trades: list[IntradayTrade] = []

    for rebalance_ts in sorted(scores_by_ts.keys()):
        open_until = {s: t for s, t in open_until.items() if t is not None and t > rebalance_ts}

        scores = scores_by_ts[rebalance_ts]
        available = scores.drop(index=[s for s in open_until if s in scores.index], errors="ignore")
        top = available.sort_values(ascending=False).head(top_n)
        benchmark_ret = benchmark_by_ts.get(rebalance_ts, float("nan"))

        for symbol in top.index:
            g = bars_by_symbol.get(symbol)
            if g is None:
                continue
            future = g[g["ts"] > rebalance_ts]
            if future.empty:
                continue
            next_bar = future.iloc[0]

            est_qty = exposure_inr / float(next_bar["open"])
            fill = simulate_fill(
                symbol=symbol, direction="buy", order_qty=est_qty,
                next_bar_open=float(next_bar["open"]), next_bar_high=float(next_bar["high"]),
                next_bar_low=float(next_bar["low"]), next_bar_volume=float(next_bar["volume"]),
                leverage_table=leverage_table, max_participation_pct=max_participation_pct,
                half_spread_bps=half_spread_bps,
            )
            if not fill.filled:
                continue

            quantity = exposure_inr / fill.fill_price
            entry_df = pd.DataFrame([{"symbol": symbol, "entry_ts": next_bar["ts"], "entry_price": fill.fill_price}])
            exit_row = first_touch_label(
                bars_1min, entry_df, stop_pct=stop_pct, target_pct=target_pct,
                max_holding_minutes=max_holding_minutes, sessions=precomputed_sessions,
            ).iloc[0]

            if exit_row["outcome"] == "no_data":
                continue

            entry_charges = compute_charges("equity_intraday", "buy", quantity, fill.fill_price)
            exit_charges = compute_charges("equity_intraday", "sell", quantity, exit_row["exit_price"])
            notional = quantity * fill.fill_price
            gross_pnl = (exit_row["exit_price"] - fill.fill_price) * quantity
            net_pnl = gross_pnl - entry_charges.total_charges - exit_charges.total_charges

            trades.append(IntradayTrade(
                symbol=symbol, entry_ts=next_bar["ts"], fill_price=fill.fill_price,
                exit_ts=exit_row["exit_ts"], exit_price=exit_row["exit_price"], outcome=exit_row["outcome"],
                quantity=quantity, gross_ret=gross_pnl / notional, net_ret=net_pnl / notional,
                benchmark_ret=benchmark_ret,
            ))
            open_until[symbol] = exit_row["exit_ts"]

    return trades


def trades_to_fold_result(fold_id: int, trades: list[IntradayTrade]) -> FoldResult | None:
    """Aggregates trade-level results into the SAME FoldResult shape
    evaluation.gate.evaluate_gate already consumes, unmodified -- one
    row per FOLD, net_returns holding one entry per TRADE rather than
    per calendar rebalance (FoldResult's own field already documents
    'per-rebalance realized net returns, for Monte Carlo' generically
    enough to accommodate this)."""
    if not trades:
        return None

    gross = np.array([t.gross_ret for t in trades])
    net = np.array([t.net_ret for t in trades])
    bench = np.array([t.benchmark_ret for t in trades])
    valid_bench = ~np.isnan(bench)

    return FoldResult(
        fold_id=fold_id, horizon_bars=0, top_n=0, round_trip_cost_bps=float("nan"),
        n_rebalances=len(trades),
        gross_expectancy=float(gross.mean()),
        net_expectancy=float(net.mean()),
        benchmark_expectancy=float(bench[valid_bench].mean()) if valid_bench.any() else float("nan"),
        alpha_gross=float((gross[valid_bench] - bench[valid_bench]).mean()) if valid_bench.any() else float(gross.mean()),
        alpha_net=float((net[valid_bench] - bench[valid_bench]).mean()) if valid_bench.any() else float(net.mean()),
        mean_turnover=float("nan"),  # not a portfolio-weight concept for a trade-shaped strategy
        information_coefficient=float("nan"),
        hit_rate=float((net > 0).mean()),
        net_returns=list(net),
    )
