"""Phase J0.2: ledger_status is the honest progress bar toward this
project's first real forward track record. These tests pin down the
counting/threshold arithmetic and the "no live model" / "no predictions
yet" degenerate cases -- the actual live database currently sits in the
zero-graded state this file tests directly."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.evaluation.gate import ForwardRecordCriteria
from stocksense.evaluation.ledger_status import ledger_status


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _register_live_model(store, model_id: str, horizon_bars: int = 10) -> None:
    store.con.execute(
        """
        INSERT INTO model_registry
            (model_id, model_type, horizon_bars, top_n, feature_schema_version,
             created_at, lifecycle_state, artifact_path, promoted_at)
        VALUES (?, 'cross_sectional_ranker', ?, 10, 'phase0-v1', ?, 'live', 'unused', ?)
        """,
        [model_id, horizon_bars, datetime.now(timezone.utc), datetime.now(timezone.utc)],
    )


def _write_predictions(store, model_id: str, as_of_dates: list[date], horizon_bars: int, n_graded: int) -> None:
    rows = []
    for i, d in enumerate(as_of_dates):
        rows.append({
            "run_id": f"run-{i}", "symbol": "AAA", "as_of_date": d, "horizon_bars": horizon_bars,
            "score": 0.1, "rank": 1, "model_version": model_id, "horizon_type": "monthly",
            "predicted_return": 0.01, "confidence": None,
            "feature_snapshot_hash": "deadbeef",
        })
    store.write_predictions(pd.DataFrame(rows))
    graded_dates = as_of_dates[:n_graded]
    for i, d in enumerate(graded_dates):
        store.grade_prediction(
            f"run-{i}", "AAA", d, horizon_bars,
            actual_return=0.02, grade_json="{}", graded_at=datetime.now(timezone.utc),
        )


def _write_calendar(store, dates: list[date]) -> None:
    rows = [
        {"symbol": "AAA", "series": "EQ", "date": d, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "prev_close": 1.0, "volume": 100, "turnover_inr": 100.0, "era": "udiff"}
        for d in dates
    ]
    store.write_bhavcopy_eq(pd.DataFrame(rows))


def test_no_live_model_reports_has_live_model_false(tmp_store) -> None:
    status = ledger_status(tmp_store, horizon_bars=10)
    assert status.has_live_model is False
    assert status.n_recorded == 0
    assert status.n_graded == 0


def test_live_model_with_zero_graded_matches_real_db_state(tmp_store) -> None:
    """The exact state the live database is actually in right now:
    predictions recorded, none graded."""
    _register_live_model(tmp_store, "m1", horizon_bars=10)
    dates = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24)]
    _write_predictions(tmp_store, "m1", dates, horizon_bars=10, n_graded=0)

    status = ledger_status(tmp_store, horizon_bars=10)
    assert status.has_live_model is True
    assert status.model_id == "m1"
    assert status.n_recorded == 3
    assert status.n_graded == 0
    assert status.n_ungraded == 3
    assert status.min_graded_required == ForwardRecordCriteria().min_graded_predictions
    assert status.predictions_until_threshold == ForwardRecordCriteria().min_graded_predictions


def test_threshold_met_reports_zero_remaining(tmp_store) -> None:
    _register_live_model(tmp_store, "m1", horizon_bars=10)
    dates = [date(2026, 1, 1) + pd.Timedelta(days=i) for i in range(35)]
    _write_predictions(tmp_store, "m1", dates, horizon_bars=10, n_graded=32)

    status = ledger_status(tmp_store, horizon_bars=10)
    assert status.n_graded == 32
    assert status.predictions_until_threshold == 0


def test_only_counts_the_live_model_and_matching_horizon(tmp_store) -> None:
    """A shadow model's predictions, or predictions at a different
    horizon, must not inflate the live model's own count -- otherwise
    the progress bar lies about how close the ACTUAL forward-record
    check is to having enough data."""
    _register_live_model(tmp_store, "live-model", horizon_bars=10)
    _write_predictions(tmp_store, "live-model", [date(2026, 1, 1)], horizon_bars=10, n_graded=1)
    # a different model_version at the same horizon
    _write_predictions(tmp_store, "shadow-model", [date(2026, 1, 2)], horizon_bars=10, n_graded=1)
    # the live model, but at a different horizon
    _write_predictions(tmp_store, "live-model", [date(2026, 1, 3)], horizon_bars=20, n_graded=1)

    status = ledger_status(tmp_store, horizon_bars=10)
    assert status.n_recorded == 1
    assert status.n_graded == 1


def test_estimated_maturity_date_uses_observed_trading_calendar(tmp_store) -> None:
    _register_live_model(tmp_store, "m1", horizon_bars=2)
    calendar_dates = [date(2026, 1, d) for d in (5, 6, 7, 8, 9)]  # 5 consecutive trading days
    _write_calendar(tmp_store, calendar_dates)
    _write_predictions(tmp_store, "m1", [date(2026, 1, 5)], horizon_bars=2, n_graded=0)

    status = ledger_status(tmp_store, horizon_bars=2)
    # 2 trading bars after 2026-01-05 in this 5-day calendar is 2026-01-07
    assert status.estimated_first_maturity_date == "2026-01-07"
    assert status.latest_calendar_date == "2026-01-09"


def test_estimated_maturity_date_is_none_when_calendar_does_not_reach_that_far(tmp_store) -> None:
    _register_live_model(tmp_store, "m1", horizon_bars=10)
    calendar_dates = [date(2026, 1, d) for d in (5, 6, 7)]  # only 3 known trading days
    _write_calendar(tmp_store, calendar_dates)
    _write_predictions(tmp_store, "m1", [date(2026, 1, 5)], horizon_bars=10, n_graded=0)

    status = ledger_status(tmp_store, horizon_bars=10)
    assert status.estimated_first_maturity_date is None
