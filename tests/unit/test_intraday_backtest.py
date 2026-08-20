"""Intraday backtest loop tests (Phase E4). Covers: session-aligned fold
construction and splitting (the exact bug a naive timestamp comparison
would introduce -- excluding bars after midnight on a boundary session),
cross-sectional relative-return labeling, and the core trade loop --
open-position capacity (a symbol already held is skipped for new
entries), a rejected fill produces no trade, and FoldResult aggregation
is gate.py-compatible."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.evaluation.intraday_backtest import (
    IntradayTrade,
    add_relative_session_return,
    make_session_folds,
    session_split,
    simulate_intraday_trades_for_fold,
    trades_to_fold_result,
    train_intraday_ranker,
)
from stocksense.evaluation.walkforward import Fold


def _bar5(symbol, ts, o, h, l, c, v=50_000.0):
    return {"symbol": symbol, "ts": pd.Timestamp(ts), "interval": "5min", "open": o, "high": h, "low": l, "close": c, "volume": v}


def _bar1(symbol, ts, price, v=5_000.0):
    return {"symbol": symbol, "ts": pd.Timestamp(ts), "open": price, "high": price + 0.2, "low": price - 0.2, "close": price, "volume": v}


# ---- make_session_folds / session_split ----

def test_make_session_folds_produces_session_aligned_boundaries() -> None:
    sessions = pd.bdate_range("2022-01-01", periods=600)  # business days as a stand-in for trading sessions
    folds = make_session_folds(sessions, min_train_sessions=500, test_window_sessions=42, embargo_sessions=1)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.test_start
        assert f.test_start <= f.test_end


def test_session_split_includes_bars_after_midnight_on_boundary_session() -> None:
    """The exact bug a naive `ts <= fold.train_end` comparison would
    introduce: fold.train_end is a midnight-normalized session date, so
    a bar at e.g. 14:30 on that same session must still count as
    in-sample, not be excluded for being 'after' midnight."""
    fold = Fold(fold_id=0, train_start=pd.Timestamp("2022-01-01"), train_end=pd.Timestamp("2022-01-05"),
                test_start=pd.Timestamp("2022-01-06"), test_end=pd.Timestamp("2022-01-10"))
    bars = pd.DataFrame([
        _bar5("X", "2022-01-05 09:15", 100, 100, 100, 100),
        _bar5("X", "2022-01-05 14:30", 101, 101, 101, 101),  # same boundary session, later in the day
        _bar5("X", "2022-01-06 09:15", 102, 102, 102, 102),  # first test session
    ])
    train, test = session_split(bars, fold)
    assert len(train) == 2  # both 01-05 bars included
    assert len(test) == 1


# ---- add_relative_session_return ----

def test_relative_session_return_demeans_cross_sectionally() -> None:
    labeled = pd.DataFrame([
        {"symbol": "A", "ts": pd.Timestamp("2022-01-05 09:30"), "fwd_ret_3b": 0.02},
        {"symbol": "B", "ts": pd.Timestamp("2022-01-05 09:30"), "fwd_ret_3b": -0.01},
        {"symbol": "A", "ts": pd.Timestamp("2022-01-05 09:35"), "fwd_ret_3b": 0.05},
        {"symbol": "B", "ts": pd.Timestamp("2022-01-05 09:35"), "fwd_ret_3b": 0.03},
    ])
    out = add_relative_session_return(labeled, "fwd_ret_3b")
    # first ts: mean=0.005, so A_rel=0.015, B_rel=-0.015
    row_a1 = out[(out["symbol"] == "A") & (out["ts"] == pd.Timestamp("2022-01-05 09:30"))].iloc[0]
    assert row_a1["fwd_ret_3b_rel"] == pytest.approx(0.015)


# ---- train_intraday_ranker ----

def _synthetic_feats_and_labels(n_rows_per_symbol: int, symbols=("A", "B", "C")):
    rng = np.random.RandomState(0)
    rows_feat, rows_lab = [], []
    for sym in symbols:
        for i in range(n_rows_per_symbol):
            ts = pd.Timestamp("2022-01-05 09:30") + pd.Timedelta(minutes=5 * i)
            rows_feat.append({"symbol": sym, "ts": ts, "f1": rng.randn(), "f2": rng.randn()})
            rows_lab.append({"symbol": sym, "ts": ts, "fwd_ret_3b_rel": rng.randn() * 0.01})
    return pd.DataFrame(rows_feat), pd.DataFrame(rows_lab)


def test_train_intraday_ranker_fits_with_enough_rows() -> None:
    feats, labeled = _synthetic_feats_and_labels(n_rows_per_symbol=250)  # 3 symbols * 250 = 750 rows
    fold = Fold(fold_id=0, train_start=pd.Timestamp("2022-01-01"), train_end=pd.Timestamp("2022-12-31"),
                test_start=pd.Timestamp("2023-01-01"), test_end=pd.Timestamp("2023-06-30"))
    ranker = train_intraday_ranker(feats, labeled, ["f1", "f2"], fold, "fwd_ret_3b_rel")
    assert ranker is not None
    preds = ranker.predict(feats[["f1", "f2"]].head(5))
    assert len(preds) == 5


def test_train_intraday_ranker_returns_none_with_too_few_rows() -> None:
    feats, labeled = _synthetic_feats_and_labels(n_rows_per_symbol=5, symbols=("A",))  # 5 rows total
    fold = Fold(fold_id=0, train_start=pd.Timestamp("2022-01-01"), train_end=pd.Timestamp("2022-12-31"),
                test_start=pd.Timestamp("2023-01-01"), test_end=pd.Timestamp("2023-06-30"))
    ranker = train_intraday_ranker(feats, labeled, ["f1", "f2"], fold, "fwd_ret_3b_rel")
    assert ranker is None


# ---- simulate_intraday_trades_for_fold ----

def _simple_scores(ts, symbols_scores: dict) -> pd.Series:
    return pd.Series(symbols_scores)


def test_top_n_selection_produces_a_filled_and_exited_trade() -> None:
    rebalance_ts = pd.Timestamp("2022-01-05 09:30")
    scores_by_ts = {rebalance_ts: _simple_scores(rebalance_ts, {"A": 1.0, "B": 0.5})}

    bars_5min = pd.DataFrame([
        _bar5("A", rebalance_ts, 100, 100.5, 99.5, 100),
        _bar5("A", "2022-01-05 09:35", 100, 100.5, 99.5, 100.2),  # next bar -> fill source
        _bar5("B", rebalance_ts, 200, 200.5, 199.5, 200),
        _bar5("B", "2022-01-05 09:35", 200, 200.5, 199.5, 200.1),
    ])
    bars_1min = pd.DataFrame([
        _bar1("A", "2022-01-05 09:35", 100.0),
        _bar1("A", "2022-01-05 09:36", 103.0),  # target hit (2% target)
    ])

    trades = simulate_intraday_trades_for_fold(
        scores_by_ts, bars_5min, bars_1min, benchmark_by_ts={rebalance_ts: 0.001},
        top_n=1, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60, exposure_inr=75_000,
    )
    assert len(trades) == 1
    assert trades[0].symbol == "A"  # higher score wins the top_n=1 slot
    assert trades[0].outcome == "target"
    assert trades[0].benchmark_ret == 0.001


def test_open_position_blocks_a_new_entry_in_the_same_symbol() -> None:
    ts1 = pd.Timestamp("2022-01-05 09:30")
    ts2 = pd.Timestamp("2022-01-05 09:35")
    scores_by_ts = {
        ts1: _simple_scores(ts1, {"A": 1.0}),
        ts2: _simple_scores(ts2, {"A": 1.0}),  # A scores high again, but should already be "open"
    }
    bars_5min = pd.DataFrame([
        _bar5("A", ts1, 100, 100.5, 99.5, 100),
        _bar5("A", ts2, 100, 100.5, 99.5, 100.1),
        _bar5("A", "2022-01-05 09:40", 100, 100.5, 99.5, 100.1),
    ])
    # A's position from ts1 stays open the WHOLE session (never touches
    # stop/target) -- max_holding_minutes governs the actual exit
    bars_1min = pd.DataFrame([_bar1("A", "2022-01-05 09:35", 100.0)] + [
        _bar1("A", f"2022-01-05 09:{35+i}" if 35 + i < 60 else f"2022-01-05 10:{35+i-60}", 100.0)
        for i in range(1, 25)
    ])

    trades = simulate_intraday_trades_for_fold(
        scores_by_ts, bars_5min, bars_1min, benchmark_by_ts={ts1: 0.0, ts2: 0.0},
        top_n=1, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60, exposure_inr=75_000,
    )
    # only ONE entry for A -- the second rebalance must not re-enter a symbol already held
    assert len(trades) == 1
    assert trades[0].entry_ts == pd.Timestamp("2022-01-05 09:35")


def test_circuit_locked_next_bar_produces_no_trade() -> None:
    ts1 = pd.Timestamp("2022-01-05 09:30")
    scores_by_ts = {ts1: _simple_scores(ts1, {"A": 1.0})}
    bars_5min = pd.DataFrame([
        _bar5("A", ts1, 100, 100.5, 99.5, 100),
        _bar5("A", "2022-01-05 09:35", 100, 100.0, 100.0, 100.0),  # circuit-locked next bar (high==low)
    ])
    bars_1min = pd.DataFrame([_bar1("A", "2022-01-05 09:35", 100.0)])

    trades = simulate_intraday_trades_for_fold(
        scores_by_ts, bars_5min, bars_1min, benchmark_by_ts={ts1: 0.0},
        top_n=1, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60, exposure_inr=75_000,
    )
    assert trades == []


def test_no_next_bar_available_produces_no_trade() -> None:
    ts1 = pd.Timestamp("2022-01-05 15:25")  # near session close, no next 5-min bar
    scores_by_ts = {ts1: _simple_scores(ts1, {"A": 1.0})}
    bars_5min = pd.DataFrame([_bar5("A", ts1, 100, 100.5, 99.5, 100)])
    bars_1min = pd.DataFrame([_bar1("A", "2022-01-05 15:26", 100.0)])

    trades = simulate_intraday_trades_for_fold(
        scores_by_ts, bars_5min, bars_1min, benchmark_by_ts={ts1: 0.0},
        top_n=1, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60, exposure_inr=75_000,
    )
    assert trades == []


# ---- trades_to_fold_result ----

def test_trades_to_fold_result_empty_returns_none() -> None:
    assert trades_to_fold_result(0, []) is None


def test_trades_to_fold_result_aggregates_correctly() -> None:
    trades = [
        IntradayTrade(symbol="A", entry_ts=pd.Timestamp("2022-01-05 09:35"), fill_price=100.0,
                       exit_ts=pd.Timestamp("2022-01-05 09:40"), exit_price=102.0, outcome="target",
                       quantity=750.0, gross_ret=0.02, net_ret=0.018, benchmark_ret=0.005),
        IntradayTrade(symbol="B", entry_ts=pd.Timestamp("2022-01-05 09:35"), fill_price=200.0,
                       exit_ts=pd.Timestamp("2022-01-05 09:50"), exit_price=197.0, outcome="stop",
                       quantity=375.0, gross_ret=-0.015, net_ret=-0.017, benchmark_ret=0.005),
    ]
    result = trades_to_fold_result(fold_id=3, trades=trades)
    assert result is not None
    assert result.fold_id == 3
    assert result.n_rebalances == 2
    assert result.net_expectancy == pytest.approx((0.018 - 0.017) / 2)
    assert result.hit_rate == pytest.approx(0.5)
    assert result.alpha_net == pytest.approx(((0.018 - 0.005) + (-0.017 - 0.005)) / 2)
    assert len(result.net_returns) == 2


def test_trades_to_fold_result_handles_missing_benchmark_gracefully() -> None:
    trades = [
        IntradayTrade(symbol="A", entry_ts=pd.Timestamp("2022-01-05 09:35"), fill_price=100.0,
                       exit_ts=pd.Timestamp("2022-01-05 09:40"), exit_price=102.0, outcome="target",
                       quantity=750.0, gross_ret=0.02, net_ret=0.018, benchmark_ret=float("nan")),
    ]
    result = trades_to_fold_result(fold_id=0, trades=trades)
    assert result is not None
    assert result.net_expectancy == pytest.approx(0.018)
    assert result.alpha_net == pytest.approx(0.018)  # falls back to raw net when no valid benchmark
