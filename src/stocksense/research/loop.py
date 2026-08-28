"""
Phase K2.4: the loop driver -- generate, screen, record, repeat.

WHAT "EVALUATE" MEANS HERE, and why no LightGBM training happens per
candidate: a single raw factor's own predictive power is measured directly by
correlating its cross-sectional value against forward relative return, per
date -- exactly the standard single-factor IC screen quant desks run before a
factor is ever handed to a model. This is the fast, cheap filter; only a
candidate that survives it is worth the cost of a full walk-forward
train-and-gate cycle via research/harness.py. Reusing
evaluation/factor_metrics.py's `decay_curve` for this needs no changes to that
function -- it only requires an object with a `.predict(X) -> array-like`
method, so `_RawFactorRanker` below is a trivial adapter that just returns the
factor's own column, standing in for a fitted model.

THE MULTIPLICITY DISCIPLINE THIS ENFORCES: every single candidate in a batch
calls `evaluation.attempts.register_attempt`, unconditionally, whether it
passes or fails the screen. That is what makes N in the Deflated Sharpe Ratio
(evaluation/robustness.py) a COUNTED FACT rather than a guess -- the video's
"read the failure, feed it back in" step, made auditable rather than just a
mental note.

STOPPING RULE. This module does not decide when to stop searching -- the
`SearchBudget` a caller constructs IS the pre-registered stopping rule
(fixed iteration count and survivor count, agreed before the first batch
runs). `run_search_iteration` processes exactly one batch and returns; a
caller loop stops itself by exhausting `SearchBudget.max_iterations`, never by
"the results still don't look good enough."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from stocksense.evaluation.attempts import register_attempt
from stocksense.evaluation.factor_metrics import build_labeled_by_horizon, decay_curve, half_life, icir
from stocksense.features.registry import apply_registered_factors
from stocksense.research.search import Candidate, generate_candidates, register_candidates

log = structlog.get_logger(__name__)

LOW_IC = "low_ic"
UNSTABLE_IC = "unstable_ic"
FAST_DECAY = "fast_decay"
SURVIVED = "survived"
"""The four outcomes `screen_candidate` can report -- the video's own
language ("read why it failed") made into a closed, machine-checkable set
rather than free text a human has to interpret every time."""


class _RawFactorRanker:
    """Stands in for a fitted model in `decay_curve`'s `ranker` parameter --
    `.predict(X)` just returns the candidate's own raw column. This is the
    entire reason a candidate can be screened WITHOUT training a LightGBM
    ranker: at the screening stage, the factor IS the score being tested."""

    def __init__(self, feature_col: str) -> None:
        self._col = feature_col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self._col].to_numpy()


@dataclass(frozen=True)
class SearchBudget:
    """The pre-registered stopping rule, fixed before the first batch runs.
    `max_iterations * batch_size` is the total number of attempts this search
    will ever register against `hypothesis_id`'s holdout -- write this number
    down in the pre-registration doc BEFORE running anything."""

    batch_size: int
    max_iterations: int
    min_icir: float
    max_half_life_for_survival: float | None = None
    """A candidate whose half_life is BELOW this is too short-lived (the
    video's "decays in two days" rejection). None means no decay filter is
    applied (screen on ICIR alone)."""


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    attempt_id: str
    mean_ic: float
    icir: float
    half_life: float
    outcome: str  # one of LOW_IC, UNSTABLE_IC, FAST_DECAY, SURVIVED


def screen_candidate(
    feats_with_factor: pd.DataFrame,
    labeled_by_horizon: dict[int, pd.DataFrame],
    candidate_name: str,
    scoring_dates: list[pd.Timestamp],
    horizons: tuple[int, ...],
    min_icir: float,
    max_half_life_for_survival: float | None,
) -> tuple[float, float, float, str]:
    """Returns (mean_ic_at_shortest_horizon, icir, half_life, outcome).

    `mean_ic` and `icir` are read from the SHORTEST horizon in `horizons` --
    the horizon closest to how the candidate would actually be traded, per
    the decay curve's own `horizon` column, not an average across horizons
    (averaging across horizons that have already decayed at different rates
    would be a meaningless number).
    """
    ranker = _RawFactorRanker(candidate_name)
    curve = decay_curve(ranker, feats_with_factor, labeled_by_horizon, scoring_dates, [candidate_name], horizons=horizons)
    hl = half_life(curve)

    shortest = curve.sort_values("horizon").iloc[0]
    mean_ic = float(shortest["mean_ic"])
    candidate_icir = float(shortest["icir"])

    if not np.isfinite(candidate_icir) or abs(mean_ic) < 1e-9:
        return mean_ic, candidate_icir, hl, LOW_IC
    if abs(candidate_icir) < min_icir:
        return mean_ic, candidate_icir, hl, UNSTABLE_IC
    if max_half_life_for_survival is not None and np.isfinite(hl) and hl < max_half_life_for_survival:
        return mean_ic, candidate_icir, hl, FAST_DECAY
    return mean_ic, candidate_icir, hl, SURVIVED


def run_search_iteration(
    store,
    candles: pd.DataFrame,
    feature_cols_base: list[str],
    *,
    seed: int,
    primitives: list[str],
    budget: SearchBudget,
    hypothesis_id: str,
    preregistration_path: str | Path,
    holdout_spec: dict,
    horizons: tuple[int, ...] = (5, 10, 20),
    registered_by: str = "user",
) -> pd.DataFrame:
    """Runs exactly ONE batch (`budget.batch_size` candidates): generate,
    merge each candidate's column onto `candles`' feature frame, screen it,
    register it as an attempt regardless of outcome, and return a results
    table. `candles` must already be vault-sealed (i.e. loaded via
    `data.loader.load_candles` with no `unseal_token`) -- this function does
    not apply or check the seal itself, since it operates on whatever frame
    it is handed, matching every other function in this codebase that leaves
    universe/date selection to its caller.
    """
    candidates = generate_candidates(seed=seed, primitives=primitives, n=budget.batch_size)
    names = register_candidates(candidates)

    labeled_by_horizon = build_labeled_by_horizon(candles, horizons)
    scoring_dates = sorted(candles["date"].unique())

    factor_frame = apply_registered_factors(candles, names=names)
    feats = candles[["symbol", "date"]].merge(factor_frame, on=["symbol", "date"], how="left")

    rows = []
    for candidate in candidates:
        attempt = register_attempt(
            store, hypothesis_id=hypothesis_id, preregistration_path=preregistration_path,
            holdout_spec={**holdout_spec, "candidate_name": candidate.name, "candidate_expression": candidate.expression},
            registered_by=registered_by,
            notes=f"seed={seed} expression={candidate.expression}",
        )
        try:
            mean_ic, candidate_icir, hl, outcome = screen_candidate(
                feats, labeled_by_horizon, candidate.name, scoring_dates, horizons,
                budget.min_icir, budget.max_half_life_for_survival,
            )
        except Exception:  # noqa: BLE001 -- one bad candidate must not abort the whole batch
            log.warning("candidate_screen_failed", candidate=candidate.name, exc_info=True)
            mean_ic, candidate_icir, hl, outcome = float("nan"), float("nan"), float("nan"), LOW_IC

        rows.append({
            "candidate_name": candidate.name, "expression": candidate.expression,
            "primitive": candidate.primitive, "depth": candidate.depth,
            "attempt_id": attempt.attempt_id, "attempt_index": attempt.attempt_index,
            "mean_ic": mean_ic, "icir": candidate_icir, "half_life": hl, "outcome": outcome,
        })
        log.info(
            "candidate_screened", candidate=candidate.name, outcome=outcome,
            icir=candidate_icir, half_life=hl, attempt_index=attempt.attempt_index,
        )

    return pd.DataFrame(rows)
