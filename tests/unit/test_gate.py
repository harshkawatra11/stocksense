from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.evaluation.backtest import FoldResult
from stocksense.evaluation.gate import (
    ForwardRecordCriteria,
    GateCriteria,
    _one_sided_binomial_pvalue,
    apply_forward_record_decision,
    evaluate_forward_record,
    evaluate_gate,
)


def _fold(alpha_net: float, fold_id: int = 0) -> FoldResult:
    return FoldResult(
        fold_id=fold_id,
        horizon_bars=20,
        top_n=20,
        round_trip_cost_bps=25.0,
        n_rebalances=12,
        gross_expectancy=alpha_net + 0.001,
        net_expectancy=alpha_net,
        benchmark_expectancy=0.0,
        alpha_gross=alpha_net + 0.001,
        alpha_net=alpha_net,
        mean_turnover=0.5,
        information_coefficient=0.05,
        hit_rate=0.6,
        net_returns=[alpha_net] * 12,
    )


def test_binomial_pvalue_matches_hand_computation() -> None:
    # P(X >= 9) for X ~ Binomial(11, 0.5): sum of C(11,9)+C(11,10)+C(11,11) terms / 2^11
    import math
    n, k = 11, 9
    expected = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    assert abs(_one_sided_binomial_pvalue(k, n) - expected) < 1e-12


def test_binomial_pvalue_at_exactly_half_is_high() -> None:
    # 5 of 10 positive is exactly the null expectation — should not be significant
    assert _one_sided_binomial_pvalue(5, 10) > 0.5


def test_binomial_pvalue_all_positive_is_highly_significant() -> None:
    assert _one_sided_binomial_pvalue(11, 11) < 0.001


def test_gate_passes_broadly_distributed_significant_edge() -> None:
    # 11 folds, 9 positive (matches Phase 0's real h=20/n=20 shape), no
    # single fold dominates.
    folds = [_fold(a, i) for i, a in enumerate(
        [0.023, 0.003, 0.017, 0.012, 0.005, 0.002, 0.007, 0.010, 0.011, -0.006, -0.001]
    )]
    verdict = evaluate_gate(folds)
    assert verdict.passed
    assert verdict.metrics["mean_alpha_net"] > 0
    assert verdict.metrics["hit_rate_pvalue"] <= 0.10
    assert verdict.metrics["mean_alpha_excl_best_k"] > 0


def test_gate_rejects_edge_carried_by_outliers() -> None:
    # 9 of 11 folds individually positive (clears hit-rate significance)
    # but two outsized folds carry the whole mean — the shape research/
    # phase0_verdict.md found for h=10/top_n=20 on the 2010-2026 sample.
    folds = [_fold(a, i) for i, a in enumerate(
        [0.0088, 0.0079, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -0.0006, -0.0006]
    )]
    verdict = evaluate_gate(folds)
    assert not verdict.passed
    assert "best-trade-removal" in verdict.reason


def test_gate_rejects_negative_mean_alpha() -> None:
    folds = [_fold(a, i) for i, a in enumerate(
        [-0.01, -0.02, 0.005, -0.008, 0.001, -0.01, -0.01, 0.002, -0.01, -0.01]
    )]
    verdict = evaluate_gate(folds)
    assert not verdict.passed
    assert "mean net alpha" in verdict.reason


def test_gate_rejects_hit_rate_not_statistically_significant() -> None:
    # 6 of 10 positive: directionally fine but not significant vs a fair
    # coin at alpha=0.10 (P(X>=6 | n=10, p=0.5) ~= 0.377).
    folds = [_fold(a, i) for i, a in enumerate(
        [0.05, 0.04, 0.001, 0.001, 0.001, 0.001, -0.005, -0.005, -0.005, -0.005]
    )]
    verdict = evaluate_gate(folds, criteria=GateCriteria())
    assert not verdict.passed
    assert "not significant" in verdict.reason


def test_gate_rejects_insufficient_folds() -> None:
    folds = [_fold(0.01, i) for i in range(9)]  # one below the min_folds_required=10 floor
    verdict = evaluate_gate(folds)
    assert not verdict.passed
    assert "insufficient folds" in verdict.reason


def test_gate_requires_beating_incumbent() -> None:
    folds = [_fold(a, i) for i, a in enumerate(
        [0.010, 0.008, 0.009, 0.011, 0.010, 0.009, 0.008, 0.010, 0.009, 0.010]
    )]
    verdict = evaluate_gate(folds, incumbent_mean_alpha_net=0.02)
    assert not verdict.passed
    assert "incumbent" in verdict.reason


def test_gate_passes_when_beating_incumbent() -> None:
    folds = [_fold(a, i) for i, a in enumerate(
        [0.023, 0.003, 0.017, 0.012, 0.005, 0.002, 0.007, 0.010, 0.011, -0.006, -0.001]
    )]
    verdict = evaluate_gate(folds, incumbent_mean_alpha_net=0.001)
    assert verdict.passed


def test_drop_fraction_is_scale_invariant_not_a_fixed_count() -> None:
    """Regression test for the audit finding: best_fold_drop_fraction must
    scale with fold count, not silently behave like the old fixed n=2."""
    # 20 folds: 0.15 * 20 = 3 folds dropped, not 2.
    strong = [0.05, 0.04, 0.03] + [0.001] * 14 + [-0.001, -0.001, -0.001]
    folds = [_fold(a, i) for i, a in enumerate(strong)]
    verdict = evaluate_gate(folds)
    assert verdict.metrics["n_folds_dropped_for_stress_test"] == 3


# ---- Phase G2: forward-record feedback (evaluate_forward_record / apply_forward_record_decision) ----

@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _insert_model(store: Store, model_id: str, lifecycle_state: str = "live") -> None:
    store.con.execute(
        """
        INSERT INTO model_registry (
            model_id, model_type, horizon_bars, top_n, feature_schema_version,
            created_at, lifecycle_state, artifact_path
        ) VALUES (?, 'cross_sectional_ranker', 10, 10, 'v1', ?, ?, 'unused.joblib')
        """,
        [model_id, datetime.now(timezone.utc), lifecycle_state],
    )


def _write_graded_predictions(store: Store, model_id: str, n_correct: int, n_wrong: int) -> None:
    """n_correct rows where sign(predicted) == sign(actual), n_wrong where they disagree."""
    rows = []
    for i in range(n_correct):
        rows.append({
            "run_id": f"r{i}", "symbol": f"S{i}", "as_of_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "horizon_bars": 10, "score": 0.01, "rank": 1, "model_version": model_id,
            "horizon_type": "short", "predicted_return": 0.01, "confidence": None,
            "feature_snapshot_hash": "h",
        })
    for i in range(n_wrong):
        rows.append({
            "run_id": f"w{i}", "symbol": f"W{i}", "as_of_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "horizon_bars": 10, "score": 0.01, "rank": 1, "model_version": model_id,
            "horizon_type": "short", "predicted_return": 0.01, "confidence": None,
            "feature_snapshot_hash": "h",
        })
    store.write_predictions(pd.DataFrame(rows))

    now = datetime.now(timezone.utc)
    for i in range(n_correct):
        store.grade_prediction(f"r{i}", f"S{i}", pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), 10,
                                actual_return=0.02, grade_json="{}", graded_at=now)  # same sign as predicted (0.01)
    for i in range(n_wrong):
        store.grade_prediction(f"w{i}", f"W{i}", pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), 10,
                                actual_return=-0.02, grade_json="{}", graded_at=now)  # opposite sign


def test_evaluate_forward_record_insufficient_predictions(tmp_store) -> None:
    _insert_model(tmp_store, "m1")
    _write_graded_predictions(tmp_store, "m1", n_correct=5, n_wrong=5)  # only 10 < default 30 required

    verdict = evaluate_forward_record(tmp_store, "m1")
    assert not verdict.demote
    assert "insufficient" in verdict.reason


def test_evaluate_forward_record_significantly_underperforming_demotes(tmp_store) -> None:
    _insert_model(tmp_store, "m1")
    # 30 graded, only 6 correct -- well below chance, should be significant
    _write_graded_predictions(tmp_store, "m1", n_correct=6, n_wrong=24)

    verdict = evaluate_forward_record(tmp_store, "m1")
    assert verdict.demote
    assert "significantly below chance" in verdict.reason
    assert verdict.metrics["n_graded"] == 30


def test_evaluate_forward_record_near_chance_does_not_demote(tmp_store) -> None:
    _insert_model(tmp_store, "m1")
    _write_graded_predictions(tmp_store, "m1", n_correct=16, n_wrong=14)  # close to 50/50

    verdict = evaluate_forward_record(tmp_store, "m1")
    assert not verdict.demote


def test_evaluate_forward_record_respects_custom_criteria(tmp_store) -> None:
    _insert_model(tmp_store, "m1")
    _write_graded_predictions(tmp_store, "m1", n_correct=6, n_wrong=24)

    verdict = evaluate_forward_record(tmp_store, "m1", criteria=ForwardRecordCriteria(min_graded_predictions=100))
    assert not verdict.demote
    assert "insufficient" in verdict.reason


def test_apply_forward_record_decision_demotes_live_to_shadow(tmp_store) -> None:
    _insert_model(tmp_store, "m1", lifecycle_state="live")
    _write_graded_predictions(tmp_store, "m1", n_correct=6, n_wrong=24)
    verdict = evaluate_forward_record(tmp_store, "m1")
    assert verdict.demote

    apply_forward_record_decision("m1", verdict, tmp_store)

    row = tmp_store.con.execute("SELECT lifecycle_state, rolled_back_at, gate_decision FROM model_registry WHERE model_id = 'm1'").fetchdf().iloc[0]
    assert row["lifecycle_state"] == "shadow"
    assert row["rolled_back_at"] is not None
    assert row["gate_decision"] == "forward_record_demoted"


def test_apply_forward_record_decision_noop_when_not_demoting(tmp_store) -> None:
    _insert_model(tmp_store, "m1", lifecycle_state="live")
    verdict = evaluate_forward_record(tmp_store, "m1")  # insufficient data -> demote=False
    assert not verdict.demote

    apply_forward_record_decision("m1", verdict, tmp_store)

    row = tmp_store.con.execute("SELECT lifecycle_state, rolled_back_at FROM model_registry WHERE model_id = 'm1'").fetchdf().iloc[0]
    assert row["lifecycle_state"] == "live"  # untouched
    assert pd.isna(row["rolled_back_at"])
