from __future__ import annotations

from stocksense.evaluation.backtest import FoldResult
from stocksense.evaluation.gate import GateCriteria, _one_sided_binomial_pvalue, evaluate_gate


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
