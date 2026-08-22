"""
The Gate: promote-or-reject, per docs/06-retraining-rigor.md and
docs/01-architecture.md's ownership split ("06 keeps ownership of the
mechanism; criteria are owned by 10-evaluation.md's scorecard").

CRITICAL AUDIT FINDING, fixed here: the first version of this gate used
GateCriteria(min_pct_folds_positive=0.6, n_best_folds_to_drop=2) —
thresholds chosen AFTER seeing Run 2/3 results, which happened to show
77-82% of folds positive. That is evaluator overfitting in miniature,
exactly the failure docs/10-evaluation.md section 17 names as the
deepest risk in the whole design, committed in the same session that
documented the risk.

This version replaces both hand-picked numbers with criteria derived
from statistical principle rather than from having looked at the data:

- The hit-rate check is now a one-sided EXACT BINOMIAL TEST against the
  null hypothesis that folds are positive purely by chance (p=0.5), at
  a pre-registered significance level. This is a real hypothesis test,
  not a percentage threshold that happens to fit past observations. The
  answer changes fold-by-fold with the actual data, but the RULE does
  not — it was fixed before any corrected-methodology numbers existed.
- The best-trade-removal check now drops a FRACTION of folds
  (`best_fold_drop_fraction`), not a fixed count of 2. Two folds "felt
  right" only because Phase 0 happened to have ~9-11 folds; a fraction
  is scale-invariant and doesn't quietly re-encode the old fitted choice
  under a different name.

See research/gate_criteria_preregistration.md for the dated record of
these choices, written and committed BEFORE the corrected-methodology
sweep re-run these criteria are meant to judge.

Passing the gate promotes to 'shadow', not 'live' — per docs/06's
explicit statement that the gate and the shadow trial are two different
things a model must earn separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from stocksense.data.store import Store
from stocksense.evaluation.backtest import FoldResult


def _one_sided_binomial_pvalue(n_success: int, n_trials: int, p_null: float = 0.5) -> float:
    """P(X >= n_success) under X ~ Binomial(n_trials, p_null). Exact, no
    normal approximation — walk-forward fold counts are small (single or
    low double digits), where the normal approximation is unreliable.
    """
    if n_trials == 0:
        return 1.0
    total = 0.0
    for k in range(n_success, n_trials + 1):
        total += math.comb(n_trials, k) * (p_null**k) * ((1 - p_null) ** (n_trials - k))
    return min(1.0, total)


@dataclass(frozen=True)
class GateCriteria:
    min_mean_alpha_net: float = 0.0  # net-of-cost edge must be strictly positive; not a fitted number
    hit_rate_significance_alpha: float = 0.10  # one-sided binomial test vs p=0.5; loose because folds are few and correlated
    best_fold_drop_fraction: float = 0.15  # scale-invariant; ~2 of 11-13 folds, ~3 of 18-23 — not a fixed count
    require_best_trade_removal_positive: bool = True
    min_folds_required: int = 10  # need enough folds for both the binomial test and the drop-fraction test to mean anything


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
    n = len(alphas)
    mean_alpha = float(alphas.mean())
    n_positive = int((alphas > 0).sum())
    pct_positive = n_positive / n

    hit_rate_pvalue = _one_sided_binomial_pvalue(n_positive, n)

    sorted_desc = np.sort(alphas)[::-1]
    k = max(1, round(n * criteria.best_fold_drop_fraction))
    remaining = sorted_desc[k:] if n > k else sorted_desc
    mean_excl_best = float(remaining.mean()) if len(remaining) else float("nan")

    metrics = {
        "n_folds": n,
        "mean_alpha_net": mean_alpha,
        "n_folds_positive": n_positive,
        "pct_folds_positive": pct_positive,
        "hit_rate_pvalue": hit_rate_pvalue,
        "n_folds_dropped_for_stress_test": k,
        "mean_alpha_excl_best_k": mean_excl_best,
        "incumbent_mean_alpha_net": incumbent_mean_alpha_net,
    }

    if mean_alpha <= criteria.min_mean_alpha_net:
        return GateVerdict(False, f"mean net alpha {mean_alpha:+.4%} <= threshold {criteria.min_mean_alpha_net:+.4%}", metrics)

    if hit_rate_pvalue > criteria.hit_rate_significance_alpha:
        return GateVerdict(
            False,
            f"hit rate {pct_positive:.0%} ({n_positive}/{n}) not significant vs chance: "
            f"one-sided binomial p={hit_rate_pvalue:.3f} > alpha={criteria.hit_rate_significance_alpha}",
            metrics,
        )

    if criteria.require_best_trade_removal_positive and mean_excl_best <= 0:
        return GateVerdict(
            False,
            f"fails best-trade-removal stress test: mean excl. best {k} of {n} folds = {mean_excl_best:+.4%} <= 0 "
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


@dataclass(frozen=True)
class ForwardRecordCriteria:
    """Phase G2: closes the gap docs/STATUS.md names -- "grading does
    not currently feed the gate... that's the natural next extension."
    Deliberately reuses the SAME statistical framework as GateCriteria
    (exact one-sided binomial test at the same 0.10 significance level)
    rather than inventing a new threshold, for the same reason
    gate_criteria_preregistration.md gives for its own choices: a
    demotion rule chosen independently of any live model's actual
    forward record is not tunable into firing (or not firing) on a
    specific result after the fact."""
    min_graded_predictions: int = 30  # ledger-row analogue of min_folds_required -- enough for the binomial test to mean something
    hit_rate_significance_alpha: float = 0.10  # same alpha as GateCriteria's promotion-side test


@dataclass(frozen=True)
class ForwardRecordVerdict:
    demote: bool
    reason: str
    metrics: dict


def evaluate_forward_record(store, model_id: str, criteria: ForwardRecordCriteria | None = None) -> ForwardRecordVerdict:
    """Checks a live model's OWN graded forward-prediction record (the
    `predictions` ledger Phase G2's reconcile loop writes to) for a
    significantly-below-chance hit rate -- the live analogue of the
    promotion gate's hit-rate check, run in the opposite direction. A
    model that cleared the historical walk-forward gate can still decay
    once live (regime shift, a bug in a later feature-pipeline change,
    etc.); this is what would actually catch that, rather than nothing
    ever re-checking a promoted model again.

    Direction-correct is defined the same way grading itself does
    (`grade_matured_predictions`'s `direction_correct`): sign(predicted)
    == sign(actual). Testing whether the WRONG-direction count is
    significantly high is the mirror image of GateCriteria's own test of
    whether the correct-direction count is significantly high -- same
    `_one_sided_binomial_pvalue` call, unmodified, just handed the
    complementary count.
    """
    criteria = criteria or ForwardRecordCriteria()
    preds = store.read_predictions()
    graded = preds[(preds["model_version"] == model_id) & preds["graded_at"].notna()]
    n = len(graded)

    if n < criteria.min_graded_predictions:
        return ForwardRecordVerdict(
            False,
            f"insufficient graded predictions: {n} < {criteria.min_graded_predictions} required",
            {"n_graded": n},
        )

    n_correct = int(((graded["predicted_return"] > 0) == (graded["actual_return"] > 0)).sum())
    n_wrong = n - n_correct
    pvalue_underperforming = _one_sided_binomial_pvalue(n_wrong, n)

    metrics = {
        "n_graded": n,
        "n_correct_direction": n_correct,
        "hit_rate": n_correct / n,
        "pvalue_underperforming": pvalue_underperforming,
    }

    if pvalue_underperforming <= criteria.hit_rate_significance_alpha:
        return ForwardRecordVerdict(
            True,
            f"forward hit rate {n_correct / n:.0%} ({n_correct}/{n}) significantly below chance: "
            f"one-sided binomial p={pvalue_underperforming:.3f} <= alpha={criteria.hit_rate_significance_alpha}",
            metrics,
        )

    return ForwardRecordVerdict(False, "forward record not significantly underperforming", metrics)


def apply_forward_record_decision(model_id: str, verdict: ForwardRecordVerdict, store: Store) -> None:
    """A demote verdict rolls a 'live' model back to 'shadow' -- not
    'archived', which is reserved for a candidate that never earned
    promotion in the first place (apply_gate_decision's FAIL branch).
    A model demoted here already proved itself once on walk-forward
    history; losing live status on a bad forward stretch is a
    downgrade to renewed probation, not a verdict that it was always
    worthless."""
    import json

    now = datetime.now(timezone.utc)
    if not verdict.demote:
        return

    store.con.execute(
        "UPDATE model_registry SET gate_decision = 'forward_record_demoted', gate_reason = ?, "
        "metrics_json = ? WHERE model_id = ?",
        [verdict.reason, json.dumps(verdict.metrics), model_id],
    )
    store.update_model_lifecycle(model_id, "shadow", rolled_back_at=now)
