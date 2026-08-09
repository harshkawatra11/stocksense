"""
The Gate: promote-or-reject, per docs/06-retraining-rigor.md and
docs/01-architecture.md's ownership split ("06 keeps ownership of the
mechanism; criteria are owned by 10-evaluation.md's scorecard").

This implements the specific criteria Phase 0 actually validated for the
h=20 configuration (research/phase0_verdict.md) — net-of-cost alpha,
fold hit-rate, and the best-trade-removal check that distinguished a
broadly-distributed edge from one carried by outliers. It is a real
subset of the full docs/10-evaluation.md battery (Monte Carlo and
parameter perturbation run separately in research/phase0_stress.py;
regime stratification, drift detection, and shadow-trial automation are
not yet built) — this is recorded honestly rather than claimed as complete.

Passing the gate promotes to 'shadow', not 'live' — per docs/06's
explicit statement that the gate and the shadow trial are two different
things a model must earn separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from stocksense.data.store import Store
from stocksense.evaluation.backtest import FoldResult


@dataclass(frozen=True)
class GateCriteria:
    min_mean_alpha_net: float = 0.0
    min_pct_folds_positive: float = 0.6
    require_best_trade_removal_positive: bool = True
    n_best_folds_to_drop: int = 2
    min_folds_required: int = 5  # below this, there isn't enough sample to gate on at all


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str
    metrics: dict


def evaluate_gate(
    fold_results: list[FoldResult],
    criteria: GateCriteria | None = None,
    incumbent_mean_alpha_net: float | None = None,
) -> GateVerdict:
    criteria = criteria or GateCriteria()

    if len(fold_results) < criteria.min_folds_required:
        return GateVerdict(
            False,
            f"insufficient folds: {len(fold_results)} < {criteria.min_folds_required} required",
            {"n_folds": len(fold_results)},
        )

    alphas = np.array([f.alpha_net for f in fold_results])
    mean_alpha = float(alphas.mean())
    pct_positive = float((alphas > 0).mean())

    sorted_desc = np.sort(alphas)[::-1]
    k = criteria.n_best_folds_to_drop
    remaining = sorted_desc[k:] if len(sorted_desc) > k else sorted_desc
    mean_excl_best = float(remaining.mean()) if len(remaining) else float("nan")

    metrics = {
        "n_folds": len(fold_results),
        "mean_alpha_net": mean_alpha,
        "pct_folds_positive": pct_positive,
        "mean_alpha_excl_best_k": mean_excl_best,
        "incumbent_mean_alpha_net": incumbent_mean_alpha_net,
    }

    if mean_alpha <= criteria.min_mean_alpha_net:
        return GateVerdict(False, f"mean net alpha {mean_alpha:+.4%} <= threshold {criteria.min_mean_alpha_net:+.4%}", metrics)

    if pct_positive < criteria.min_pct_folds_positive:
        return GateVerdict(False, f"only {pct_positive:.0%} of folds positive, < {criteria.min_pct_folds_positive:.0%} required", metrics)

    if criteria.require_best_trade_removal_positive and mean_excl_best <= 0:
        return GateVerdict(
            False,
            f"fails best-trade-removal stress test: mean excl. best {k} folds = {mean_excl_best:+.4%} <= 0 "
            "(edge concentrated in outlier folds, not broadly distributed)",
            metrics,
        )

    if incumbent_mean_alpha_net is not None and mean_alpha <= incumbent_mean_alpha_net:
        return GateVerdict(
            False,
            f"candidate mean alpha {mean_alpha:+.4%} does not beat incumbent {incumbent_mean_alpha_net:+.4%}",
            metrics,
        )

    return GateVerdict(True, "all criteria passed", metrics)


def apply_gate_decision(model_id: str, verdict: GateVerdict, store: Store) -> None:
    """Record the gate's decision and update lifecycle state. PASS moves
    the candidate to 'shadow' (earns the right to be graded on live data,
    not yet the right to reach the user — docs/06's distinction). FAIL
    archives it with the reason preserved for audit."""
    import json

    now = datetime.now(timezone.utc)
    decision = "promote" if verdict.passed else "reject"
    new_state = "shadow" if verdict.passed else "archived"

    store.con.execute(
        "UPDATE model_registry SET gate_decision = ?, gate_reason = ?, metrics_json = ? WHERE model_id = ?",
        [decision, verdict.reason, json.dumps(verdict.metrics), model_id],
    )
    store.update_model_lifecycle(model_id, new_state, promoted_at=now if verdict.passed else None)
