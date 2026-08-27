"""
Phase J4c: the evaluation-attempt registry -- docs/09-open-questions.md's
OQ-11, finally built. `research/gate_criteria_preregistration.md` names
the exact gap this closes in its own text: pre-registering threshold
VALUES stops the specific overfitting failure this project already had
once, but "if this exact gate is run repeatedly against re-tunings of
the model... until one clears it, the gate becomes an overfitting
instrument again -- just one level removed." This project's own history
already shows the pattern: ~40+ distinct configurations swept across
Phase 0's Runs 1-4, the 16 bhavcopy cap-band/horizon/top_n combinations,
15 intraday folds, and 4 F&O feature-set variants, with no multiplicity
correction anywhere.

THIS FILE MUST NEVER LOOSEN A THRESHOLD -- it is a protected path
(foreman/policy.py) for exactly that reason, same as gate.py itself.
Every function below can only make evaluate_gate's own hit-rate
significance test STRICTER (via a smaller adjusted alpha), never more
lenient, and that is enforced by an assertion, not just a convention.

Deliberately does NOT modify evaluation/gate.py. GateCriteria is already
a frozen dataclass evaluate_gate accepts as a parameter
(evaluation/gate.py:79) -- this module constructs a stricter instance
and hands it in. gate.py's own thresholds remain the ceiling no attempt
can exceed.

Scope note, so a future session doesn't "helpfully" misapply this:
attempt-adjusted criteria apply to the RESEARCH SWEEP path
(research/*_sweep.py, a new hypothesis against a fixed holdout), never
to models/train_candidate.py's production retrain path, which
deliberately hardcodes GateCriteria() and forbids configurable criteria
in its own docstring -- a scheduled retrain of an ALREADY-VALIDATED
hypothesis is not a new attempt against a fresh holdout.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from stocksense.evaluation.gate import GateCriteria

BONFERRONI = "bonferroni"
"""Bonferroni, not Benjamini-Hochberg: BH controls false-discovery rate
across a SIMULTANEOUSLY specified family of tests. Sequential attempts
against one holdout, decided one at a time as each prior attempt's
result becomes known, are not that -- Bonferroni is the conservative,
one-sentence-explainable choice appropriate to a sequential process, and
the point here is defensibility over statistical power."""


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    hypothesis_id: str
    holdout_id: str
    attempt_index: int
    registered_at: str


def holdout_id_for(spec: dict) -> str:
    """sha256 of the canonicalized (sorted-key) spec -- two attempts
    touching the same universe/date-range/horizon/cost-model collide BY
    CONSTRUCTION, not by a researcher remembering to disclose it."""
    canonical = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _is_committed(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=path.parent, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def register_attempt(
    store, *, hypothesis_id: str, preregistration_path: str | Path, holdout_spec: dict,
    registered_by: str = "user", base_alpha: float = GateCriteria().hit_rate_significance_alpha,
    notes: str | None = None,
) -> Attempt:
    """Refuses to register against a pre-registration file that doesn't
    exist or isn't committed to git -- the whole point of pre-
    registration is that the criteria exist BEFORE the result, and an
    uncommitted file could still be edited after seeing one."""
    path = Path(preregistration_path)
    if not path.exists():
        raise FileNotFoundError(f"preregistration file does not exist: {path}")
    if not _is_committed(path):
        raise ValueError(
            f"preregistration file is not committed to git: {path} -- "
            "commit it BEFORE registering an attempt, or the pre-registration has no teeth"
        )

    preregistration_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    holdout_id = holdout_id_for(holdout_spec)
    attempt_id = str(uuid.uuid4())[:12]

    attempt_index = store.register_evaluation_attempt({
        "attempt_id": attempt_id, "hypothesis_id": hypothesis_id,
        "preregistration_path": str(path), "preregistration_hash": preregistration_hash,
        "holdout_id": holdout_id, "holdout_spec_json": json.dumps(holdout_spec, sort_keys=True, default=str),
        "registered_at": datetime.now(timezone.utc), "registered_by": registered_by,
        "status": "registered", "base_alpha": base_alpha, "gate_alpha_used": None,
        "result_verdict": None, "result_metrics_json": None, "notes": notes,
    })
    return Attempt(
        attempt_id=attempt_id, hypothesis_id=hypothesis_id, holdout_id=holdout_id,
        attempt_index=attempt_index, registered_at=datetime.now(timezone.utc).isoformat(),
    )


def attempt_count(store, holdout_id: str) -> int:
    return store.count_evaluation_attempts(holdout_id)


def adjusted_alpha(base_alpha: float, n_attempts: int) -> float:
    """Bonferroni: divide by the number of attempts made against this
    holdout so far (including the current one). n_attempts < 1 is
    treated as 1 -- a first attempt is not a discount."""
    n = max(1, n_attempts)
    return base_alpha / n


def criteria_for_attempt(store, attempt: Attempt, base: GateCriteria | None = None) -> GateCriteria:
    """Builds the criteria the CURRENT attempt must actually clear --
    strictly tighter than or equal to `base` (default GateCriteria()),
    never looser. This is the one-way ratchet, enforced by assertion,
    not just documentation."""
    base = base or GateCriteria()
    n = attempt_count(store, attempt.holdout_id)
    adj = adjusted_alpha(base.hit_rate_significance_alpha, n)
    assert adj <= base.hit_rate_significance_alpha, (
        "evaluation.attempts must only TIGHTEN hit_rate_significance_alpha, never loosen it"
    )
    return replace(base, hit_rate_significance_alpha=adj)


def close_attempt(store, attempt_id: str, *, verdict: str, gate_alpha_used: float, metrics: dict) -> None:
    if verdict not in ("pass", "fail", "inconclusive"):
        raise ValueError(f"verdict must be 'pass', 'fail', or 'inconclusive', got {verdict!r}")
    store.update_evaluation_attempt_result(
        attempt_id, status="run", gate_alpha_used=gate_alpha_used,
        result_verdict=verdict, result_metrics_json=json.dumps(metrics, default=str),
    )
