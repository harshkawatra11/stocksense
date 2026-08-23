"""Reconcile loop tests (docs/00-overview.md's founding claim, closes
CRITICAL-1): record_predictions actually writes to `predictions`, and
grade_matured_predictions grades them against real computed forward
returns once the horizon has elapsed -- end to end, on synthetic data
with a real trained model, not mocked."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.features.engine import build_features, feature_columns
from stocksense.harness.loops import build_reconcile_graph, grade_matured_predictions, record_predictions
from stocksense.harness.runner import run_graph
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.ranker import CrossSectionalRanker, RankerConfig
from stocksense.models.registry import register_candidate


def _synthetic_candles(n_symbols: int = 10, n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
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


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


@pytest.fixture()
def store_with_live_model(tmp_store, tmp_path, monkeypatch):
    """A store with real candle data and a genuinely trained, promoted
    ('live') model -- register_candidate writes the artifact under
    settings.parquet_dir, so point that at a temp dir."""
    from stocksense.core import config as config_mod

    monkeypatch.setattr(config_mod, "DATA_STORE", tmp_path / "data_store")

    candles = _synthetic_candles()
    tmp_store.upsert_candles(candles)

    horizon = 20
    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]
    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)

    merged = feats.merge(labeled[["symbol", "date", f"fwd_ret_{horizon}b_rel"]], on=["symbol", "date"], how="inner")
    merged = merged.dropna(subset=[f"fwd_ret_{horizon}b_rel"])

    ranker = CrossSectionalRanker(RankerConfig(random_state=42))
    ranker.fit(merged[fcols], merged[f"fwd_ret_{horizon}b_rel"])

    model_id = register_candidate(
        ranker, model_type="cross_sectional_ranker", horizon_bars=horizon, top_n=5,
        training_start=str(candles["date"].min()), training_end=str(candles["date"].max()),
        metrics={"mean_alpha_net": 0.01}, store=tmp_store,
    )
    tmp_store.update_model_lifecycle(model_id, "live", promoted_at=datetime.now(timezone.utc))
    return tmp_store, horizon, model_id


def test_record_predictions_writes_rows_for_every_symbol(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model
    result = record_predictions(store, horizon_bars=horizon, lifecycle="live")

    assert result is not None
    assert result.model_id == model_id
    assert result.n_predictions == 10  # all 10 synthetic symbols have clean features on the latest date

    predictions = store.read_predictions()
    assert len(predictions) == 10
    assert set(predictions["symbol"]) == {f"SYN{i}" for i in range(10)}
    assert all(predictions["model_version"] == model_id)
    assert predictions["feature_snapshot_hash"].notna().all()


def test_record_predictions_returns_none_when_no_live_model(tmp_store) -> None:
    result = record_predictions(tmp_store, horizon_bars=20, lifecycle="live")
    assert result is None


def test_grade_matured_predictions_grades_real_outcomes(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model

    # Manually insert a prediction dated well before the end of the
    # synthetic history, so its horizon has actually elapsed within the
    # available candle data (the fixture's 400 days gives 380 bars of
    # room after any date in the first half).
    candles = store.read_candles()
    early_date = sorted(candles["date"].unique())[100]

    pred_row = pd.DataFrame(
        {
            "run_id": ["test-run-1"], "symbol": ["SYN0"], "as_of_date": [early_date],
            "horizon_bars": [horizon], "score": [0.01], "rank": [1], "model_version": [model_id],
            "horizon_type": ["monthly"], "predicted_return": [0.01], "confidence": [None],
            "feature_snapshot_hash": ["abc123"],
        }
    )
    store.write_predictions(pred_row)

    result = grade_matured_predictions(store, horizon_bars=horizon, as_of_before=date.today())

    assert result.n_graded == 1
    predictions = store.read_predictions()
    graded = predictions[predictions["run_id"] == "test-run-1"].iloc[0]
    assert graded["graded_at"] is not None
    assert graded["actual_return"] == graded["actual_return"]  # not NaN


def test_grade_matured_predictions_skips_unmatured(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model
    candles = store.read_candles()
    latest_date = sorted(candles["date"].unique())[-1]  # no room left for the horizon to elapse

    pred_row = pd.DataFrame(
        {
            "run_id": ["test-run-2"], "symbol": ["SYN0"], "as_of_date": [latest_date],
            "horizon_bars": [horizon], "score": [0.01], "rank": [1], "model_version": [model_id],
            "horizon_type": ["monthly"], "predicted_return": [0.01], "confidence": [None],
            "feature_snapshot_hash": ["abc123"],
        }
    )
    store.write_predictions(pred_row)

    result = grade_matured_predictions(store, horizon_bars=horizon, as_of_before=date.today())
    assert result.n_graded == 0  # horizon hasn't elapsed -- should not be graded


def test_grade_matured_predictions_no_ungraded_returns_zero(tmp_store) -> None:
    result = grade_matured_predictions(tmp_store, horizon_bars=20)
    assert result.n_graded == 0
    assert result.mean_abs_error is None


def test_grade_prediction_uses_same_label_function_as_training(store_with_live_model) -> None:
    """The load-bearing property: grading must compute 'actual return'
    with the exact same function training used, so the two can never
    silently disagree about what a correct prediction even means."""
    store, horizon, model_id = store_with_live_model
    candles = store.read_candles()
    early_date = sorted(candles["date"].unique())[100]

    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)
    expected = labeled[(labeled["symbol"] == "SYN0") & (labeled["date"] == early_date)][f"fwd_ret_{horizon}b_rel"].iloc[0]

    pred_row = pd.DataFrame(
        {
            "run_id": ["test-run-3"], "symbol": ["SYN0"], "as_of_date": [early_date],
            "horizon_bars": [horizon], "score": [0.01], "rank": [1], "model_version": [model_id],
            "horizon_type": ["monthly"], "predicted_return": [0.01], "confidence": [None],
            "feature_snapshot_hash": ["abc123"],
        }
    )
    store.write_predictions(pred_row)
    grade_matured_predictions(store, horizon_bars=horizon, as_of_before=date.today())

    graded = store.read_predictions()
    graded_row = graded[graded["run_id"] == "test-run-3"].iloc[0]
    assert abs(float(graded_row["actual_return"]) - float(expected)) < 1e-9


def test_reconcile_graph_runs_grade_before_record(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model
    graph = build_reconcile_graph(store, horizon_bars=horizon)
    order = graph.topological_order()
    assert order.index("grade_matured_predictions") < order.index("record_predictions")


def test_reconcile_graph_end_to_end_records_predictions(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model
    graph = build_reconcile_graph(store, horizon_bars=horizon)
    result = run_graph(graph, store)

    assert result.all_succeeded
    assert result.context["record_predictions"]["recorded"] is True
    predictions = store.read_predictions()
    assert len(predictions) == 10


def test_reconcile_graph_is_idempotent_within_the_same_day(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model
    graph = build_reconcile_graph(store, horizon_bars=horizon)

    run_graph(graph, store)
    predictions_after_first = store.read_predictions()

    # a second graph instance (same day, same horizon) must skip both
    # nodes rather than double-record
    graph2 = build_reconcile_graph(store, horizon_bars=horizon)
    result2 = run_graph(graph2, store)

    predictions_after_second = store.read_predictions()
    assert len(predictions_after_second) == len(predictions_after_first)  # not doubled
    assert all(o.status == "skipped" for o in result2.outcomes)


# ---- Phase G2: check_forward_record node in the reconcile graph ----

def test_reconcile_graph_check_forward_record_noops_for_shadow_lifecycle(store_with_live_model) -> None:
    store, horizon, model_id = store_with_live_model
    graph = build_reconcile_graph(store, horizon_bars=horizon, lifecycle="shadow")
    result = run_graph(graph, store)

    assert result.context["check_forward_record"]["checked"] is False
    assert result.context["check_forward_record"]["reason"].startswith("only the live model")


def test_reconcile_graph_check_forward_record_noops_when_no_live_model(tmp_store) -> None:
    graph = build_reconcile_graph(tmp_store, horizon_bars=20, lifecycle="live")
    result = run_graph(graph, tmp_store)

    assert result.context["check_forward_record"]["checked"] is False
    assert result.context["check_forward_record"]["reason"] == "no live model for this horizon"


def test_reconcile_graph_demotes_underperforming_live_model_before_recording(store_with_live_model) -> None:
    """The end-to-end property Phase G2 exists for: a live model with a
    significantly-below-chance forward record gets rolled back to
    'shadow' by the SAME reconcile run that would otherwise have
    recorded a fresh prediction against it -- and that fresh prediction
    must NOT be recorded once the model is no longer live (fail-closed,
    not a leftover stale prediction from a demoted model)."""
    store, horizon, model_id = store_with_live_model

    # Seed a bad forward record for the real registered live model_id:
    # 6 correct out of 30, well below chance.
    rows = []
    for i in range(6):
        rows.append({
            "run_id": f"r{i}", "symbol": f"SYN{i % 10}", "as_of_date": pd.Timestamp("2023-06-01") + pd.Timedelta(days=i),
            "horizon_bars": horizon, "score": 0.01, "rank": 1, "model_version": model_id,
            "horizon_type": "monthly", "predicted_return": 0.01, "confidence": None, "feature_snapshot_hash": "h",
        })
    for i in range(24):
        rows.append({
            "run_id": f"w{i}", "symbol": f"SYN{i % 10}", "as_of_date": pd.Timestamp("2023-06-01") + pd.Timedelta(days=i),
            "horizon_bars": horizon, "score": 0.01, "rank": 1, "model_version": model_id,
            "horizon_type": "monthly", "predicted_return": 0.01, "confidence": None, "feature_snapshot_hash": "h",
        })
    store.write_predictions(pd.DataFrame(rows))
    now = datetime.now(timezone.utc)
    for i in range(6):
        store.grade_prediction(f"r{i}", f"SYN{i % 10}", pd.Timestamp("2023-06-01") + pd.Timedelta(days=i), horizon,
                                actual_return=0.02, grade_json="{}", graded_at=now)
    for i in range(24):
        store.grade_prediction(f"w{i}", f"SYN{i % 10}", pd.Timestamp("2023-06-01") + pd.Timedelta(days=i), horizon,
                                actual_return=-0.02, grade_json="{}", graded_at=now)

    graph = build_reconcile_graph(store, horizon_bars=horizon, lifecycle="live")
    result = run_graph(graph, store)

    assert result.context["check_forward_record"]["checked"] is True
    assert result.context["check_forward_record"]["demoted"] is True
    assert result.context["record_predictions"]["recorded"] is False  # no live model left to score with

    row = store.con.execute("SELECT lifecycle_state FROM model_registry WHERE model_id = ?", [model_id]).fetchdf().iloc[0]
    assert row["lifecycle_state"] == "shadow"


# ---- Phase H2: QuantileRanker sibling artifact consumption ----

@pytest.fixture()
def store_with_live_model_and_quantile(store_with_live_model):
    """Same live model as store_with_live_model, plus a real sibling
    quantile_model.joblib fit on the same data -- simulates a model
    trained after Phase H2 (train_candidate_core._fit_and_save_
    quantile_ranker), without paying the cost of a full walk-forward
    gate run."""
    import joblib
    from stocksense.core.config import get_settings
    from stocksense.models.ranker import QuantileRanker, RankerConfig
    from stocksense.models.train_candidate import QUANTILE_ARTIFACT_NAME

    store, horizon, model_id = store_with_live_model

    candles = _synthetic_candles()
    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]
    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)
    merged = feats.merge(labeled[["symbol", "date", f"fwd_ret_{horizon}b_rel"]], on=["symbol", "date"], how="inner")
    merged = merged.dropna(subset=[f"fwd_ret_{horizon}b_rel"])

    quantile_ranker = QuantileRanker(RankerConfig(random_state=42))
    quantile_ranker.fit(merged[fcols], merged[f"fwd_ret_{horizon}b_rel"])

    settings = get_settings()
    model_dir = settings.parquet_dir.parent / "models" / model_id
    joblib.dump(quantile_ranker, model_dir / QUANTILE_ARTIFACT_NAME)

    return store, horizon, model_id


def test_record_predictions_populates_confidence_when_quantile_sibling_exists(store_with_live_model_and_quantile) -> None:
    store, horizon, model_id = store_with_live_model_and_quantile
    result = record_predictions(store, horizon_bars=horizon, lifecycle="live")
    assert result is not None

    predictions = store.read_predictions()
    assert predictions["confidence"].notna().all()
    assert (predictions["confidence"] >= 0).all()
    assert predictions["predicted_return"].notna().all()


def test_record_predictions_confidence_stays_null_without_quantile_sibling(store_with_live_model) -> None:
    """The fallback path -- a model registered without Phase H2's
    sibling artifact (e.g. trained before this phase) must keep scoring
    exactly as before: predicted_return from the point ranker's raw
    score, confidence NULL."""
    store, horizon, model_id = store_with_live_model
    result = record_predictions(store, horizon_bars=horizon, lifecycle="live")
    assert result is not None

    predictions = store.read_predictions()
    assert predictions["confidence"].isna().all()
    assert predictions["predicted_return"].notna().all()  # still populated from the point ranker's score
