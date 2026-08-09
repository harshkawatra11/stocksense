from __future__ import annotations

from stocksense.evaluation.backtest import FoldResult
from stocksense.evaluation.gate import GateCriteria, evaluate_gate


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


def test_gate_passes_broadly_distributed_positive_edge() -> None:
    # 9 of 11 folds positive, no single fold dominates — the h=20/n=20 shape
    folds = [_fold(a, i) for i, a in enumerate(
        [0.023, 0.003, 0.017, 0.012, 0.005, 0.002, 0.007, 0.010, 0.011, -0.006, -0.001]
    )]
    verdict = evaluate_gate(folds)
    assert verdict.passed
    assert verdict.metrics["mean_alpha_net"] > 0
    assert verdict.metrics["mean_alpha_excl_best_k"] > 0


def test_gate_rejects_edge_carried_by_outliers() -> None:
    # 7 of 9 folds individually positive (clears the hit-rate bar) but two
    # outsized folds carry the whole mean — the actual shape research/
    # phase0_verdict.md found for h=10/top_n=20 on the 2010-2026 sample.
    folds = [_fold(a, i) for i, a in enumerate(
        [0.0088, 0.0079, 0.0002, 0.0002, 0.0002, 0.0002, -0.0005, -0.0005, -0.0005]
    )]
    verdict = evaluate_gate(folds)
    assert not verdict.passed
    assert "best-trade-removal" in verdict.reason


def test_gate_rejects_negative_mean_alpha() -> None:
    folds = [_fold(a, i) for i, a in enumerate([-0.01, -0.02, 0.005, -0.008, 0.001])]
    verdict = evaluate_gate(folds)
    assert not verdict.passed
    assert "mean net alpha" in verdict.reason


def test_gate_rejects_low_hit_rate_even_if_mean_positive() -> None:
    # One huge winner, mostly losers — positive mean but low hit rate
    folds = [_fold(a, i) for i, a in enumerate([0.5, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])]
    verdict = evaluate_gate(folds, criteria=GateCriteria(min_pct_folds_positive=0.6))
    assert not verdict.passed


def test_gate_rejects_insufficient_folds() -> None:
    folds = [_fold(0.01, 0), _fold(0.01, 1)]
    verdict = evaluate_gate(folds)
    assert not verdict.passed
    assert "insufficient folds" in verdict.reason


def test_gate_requires_beating_incumbent() -> None:
    folds = [_fold(a, i) for i, a in enumerate(
        [0.010, 0.008, 0.009, 0.011, 0.010, 0.009, 0.008]
    )]
    # candidate mean ~0.0093, incumbent already achieves 0.02 — candidate must not pass
    verdict = evaluate_gate(folds, incumbent_mean_alpha_net=0.02)
    assert not verdict.passed
    assert "incumbent" in verdict.reason


def test_gate_passes_when_beating_incumbent() -> None:
    folds = [_fold(a, i) for i, a in enumerate(
        [0.023, 0.003, 0.017, 0.012, 0.005, 0.002, 0.007, 0.010, 0.011, -0.006, -0.001]
    )]
    verdict = evaluate_gate(folds, incumbent_mean_alpha_net=0.001)
    assert verdict.passed
