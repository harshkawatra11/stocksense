"""The promotion gate. PROTECTED, frozen the moment it lands.

One-sided binomial test on the count of positive folds against pre-registered
thresholds. No threshold may change after a result is seen -- this project
committed that error once, documented it, and rebuilt from statistical
principle. That discipline carries.

Verdict is PASS | FAIL | INCONCLUSIVE, never a number to be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binomtest

GATE = dict(
    min_folds_required=10,
    min_mean_alpha_net=0.0,  # AFTER compute_charges, never gross
    max_binomial_p=0.05,  # H0: positive folds ~ Binomial(n, 0.5)
    max_drop_fraction=0.15,  # >15% of folds unusable -> inconclusive, not a pass
)


@dataclass(frozen=True)
class GateResult:
    verdict: str  # PASS | FAIL | INCONCLUSIVE
    n_folds_attempted: int
    n_folds_used: int
    n_folds_dropped: int
    drop_fraction: float
    n_positive: int
    mean_alpha_net: float
    binomial_p: float


def evaluate_gate(fold_alpha_net: list[float | None], gate: dict = GATE) -> GateResult:
    """Score a strategy's per-fold net alpha against the pre-registered gate.

    Args:
        fold_alpha_net: one net-of-charges alpha figure per CPCV fold, in the
            SAME order `walkforward.make_folds` produced them. A fold that
            could not be scored (e.g. zero trades fired) is `None` and is
            DROPPED, not treated as a zero -- a zero would be a real claim
            about that fold's performance, which "unscoreable" is not.
        gate: the threshold dict. Defaults to the frozen GATE above; a
            different dict is accepted only for testing this function itself,
            never for re-running a real hypothesis with softer numbers.

    Returns:
        GateResult with verdict PASS, FAIL, or INCONCLUSIVE. INCONCLUSIVE
        overrides PASS/FAIL whenever there are too few usable folds or too
        many were dropped -- "not enough evidence" is a distinct answer from
        "the evidence says no."
    """
    n_attempted = len(fold_alpha_net)
    used = [a for a in fold_alpha_net if a is not None]
    n_used = len(used)
    n_dropped = n_attempted - n_used
    drop_fraction = (n_dropped / n_attempted) if n_attempted else 1.0

    n_positive = sum(1 for a in used if a > 0)
    mean_alpha_net = (sum(used) / n_used) if n_used else float("nan")
    binomial_p = (
        binomtest(n_positive, n_used, 0.5, alternative="greater").pvalue if n_used else 1.0
    )

    if n_attempted == 0 or drop_fraction > gate["max_drop_fraction"]:
        verdict = "INCONCLUSIVE"
    elif n_used < gate["min_folds_required"]:
        verdict = "INCONCLUSIVE"
    elif (
        mean_alpha_net > gate["min_mean_alpha_net"]
        and binomial_p <= gate["max_binomial_p"]
    ):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return GateResult(
        verdict=verdict,
        n_folds_attempted=n_attempted,
        n_folds_used=n_used,
        n_folds_dropped=n_dropped,
        drop_fraction=drop_fraction,
        n_positive=n_positive,
        mean_alpha_net=mean_alpha_net,
        binomial_p=binomial_p,
    )
