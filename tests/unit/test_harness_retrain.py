"""Phase G4: the weekly retrain graph. Checks it actually runs
train_candidate_core end to end through harness.runner, and that it is
idempotent within the same ISO week -- the weekly analogue of the
reconcile graph's daily idempotency (test_harness_loops.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.harness.retrain import build_weekly_retrain_graph
from stocksense.harness.runner import run_graph


def _synthetic_candles(n_symbols: int = 15, n_days: int = 900, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
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
            rows.append({
                "symbol": symbol, "date": d,
                "open": open_, "high": max(high, open_, price), "low": min(low, open_, price),
                "close": price, "adj_close": price, "volume": vol, "source": "synthetic",
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def test_weekly_retrain_graph_runs_train_candidate_core(tmp_store) -> None:
    candles = _synthetic_candles()
    tmp_store.upsert_candles(candles)

    graph = build_weekly_retrain_graph(tmp_store, horizon=20, top_n=5)
    result = run_graph(graph, tmp_store)

    assert result.all_succeeded
    train_out = result.context["train_candidate"]
    assert train_out["n_fold_results"] >= 0  # ran, whatever the fold count turned out to be
    assert "gate_passed" in train_out


def test_weekly_retrain_graph_is_idempotent_within_the_same_iso_week(tmp_store) -> None:
    candles = _synthetic_candles()
    tmp_store.upsert_candles(candles)

    graph = build_weekly_retrain_graph(tmp_store, horizon=20, top_n=5)
    run_graph(graph, tmp_store)
    n_models_after_first = len(tmp_store.read_model_registry())

    # A second graph instance (same week, same horizon/top_n) must skip
    # rather than register a second candidate
    graph2 = build_weekly_retrain_graph(tmp_store, horizon=20, top_n=5)
    result2 = run_graph(graph2, tmp_store)

    n_models_after_second = len(tmp_store.read_model_registry())
    assert n_models_after_second == n_models_after_first  # not doubled
    assert all(o.status == "skipped" for o in result2.outcomes)


def test_weekly_retrain_graph_different_horizons_are_independent(tmp_store) -> None:
    """Idempotency is keyed by (horizon, top_n, cap_band, week) -- a
    different horizon in the SAME week must still run, not be skipped
    because some other horizon already ran this week."""
    candles = _synthetic_candles()
    tmp_store.upsert_candles(candles)

    graph_h20 = build_weekly_retrain_graph(tmp_store, horizon=20, top_n=5)
    run_graph(graph_h20, tmp_store)

    graph_h10 = build_weekly_retrain_graph(tmp_store, horizon=10, top_n=5)
    result_h10 = run_graph(graph_h10, tmp_store)

    assert all(o.status != "skipped" for o in result_h10.outcomes)
