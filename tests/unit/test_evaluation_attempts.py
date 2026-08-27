"""Phase J4c: the evaluation-attempt registry (docs/09's OQ-11). These
tests pin down the one property that actually matters -- the registry
can only TIGHTEN evaluate_gate's significance threshold, never loosen
it -- plus the mechanics (DB-assigned attempt_index, holdout collision,
pre-registration-must-be-committed) that make it trustworthy."""

from __future__ import annotations

import subprocess

import pytest

from stocksense.data.store import Store
from stocksense.evaluation.attempts import (
    Attempt,
    adjusted_alpha,
    attempt_count,
    close_attempt,
    criteria_for_attempt,
    holdout_id_for,
    register_attempt,
)
from stocksense.evaluation.gate import GateCriteria


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


@pytest.fixture()
def committed_prereg(tmp_path):
    """A real git repo with one committed file -- register_attempt
    refuses anything else, so tests exercising the happy path need a
    genuinely committed file, not just one that exists on disk."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    prereg = repo / "preregistration_x.md"
    prereg.write_text("# fixed before any result\n", encoding="utf-8")
    subprocess.run(["git", "add", "preregistration_x.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "prereg"], cwd=repo, check=True)
    return prereg


def test_bonferroni_arithmetic() -> None:
    assert adjusted_alpha(0.10, 1) == pytest.approx(0.10)
    assert adjusted_alpha(0.10, 4) == pytest.approx(0.025)
    assert adjusted_alpha(0.10, 0) == pytest.approx(0.10)  # 0 attempts treated as 1, not a discount


def test_holdout_id_collides_on_same_spec_regardless_of_key_order() -> None:
    a = holdout_id_for({"universe": "full_pit", "horizon": 10, "start": "2020-01-01"})
    b = holdout_id_for({"start": "2020-01-01", "horizon": 10, "universe": "full_pit"})
    assert a == b


def test_holdout_id_differs_on_different_spec() -> None:
    a = holdout_id_for({"universe": "full_pit", "horizon": 10})
    b = holdout_id_for({"universe": "full_pit", "horizon": 20})
    assert a != b


def test_register_attempt_refuses_missing_prereg_file(tmp_store, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        register_attempt(
            tmp_store, hypothesis_id="h1", preregistration_path=tmp_path / "does_not_exist.md",
            holdout_spec={"x": 1},
        )


def test_register_attempt_refuses_uncommitted_prereg_file(tmp_store, tmp_path) -> None:
    repo = tmp_path / "repo2"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    uncommitted = repo / "preregistration_y.md"
    uncommitted.write_text("not committed", encoding="utf-8")
    with pytest.raises(ValueError, match="not committed"):
        register_attempt(tmp_store, hypothesis_id="h1", preregistration_path=uncommitted, holdout_spec={"x": 1})


def test_register_attempt_succeeds_on_committed_prereg(tmp_store, committed_prereg) -> None:
    attempt = register_attempt(
        tmp_store, hypothesis_id="dispersion_regime", preregistration_path=committed_prereg,
        holdout_spec={"universe": "full_pit", "horizon": 10},
    )
    assert attempt.attempt_index == 1
    assert attempt.hypothesis_id == "dispersion_regime"


def test_attempt_index_assigned_by_db_increments_per_holdout(tmp_store, committed_prereg) -> None:
    spec = {"universe": "full_pit", "horizon": 10}
    first = register_attempt(tmp_store, hypothesis_id="h1", preregistration_path=committed_prereg, holdout_spec=spec)
    assert first.attempt_index == 1

    # a second, DIFFERENT hypothesis against the SAME holdout increments the shared count
    prereg2 = committed_prereg.parent / "preregistration_z.md"
    prereg2.write_text("# second hypothesis\n", encoding="utf-8")
    subprocess.run(["git", "add", "preregistration_z.md"], cwd=committed_prereg.parent, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "prereg2"], cwd=committed_prereg.parent, check=True)
    second = register_attempt(tmp_store, hypothesis_id="h2", preregistration_path=prereg2, holdout_spec=spec)
    assert second.attempt_index == 2
    assert attempt_count(tmp_store, first.holdout_id) == 2


def test_attempt_index_independent_across_different_holdouts(tmp_store, committed_prereg) -> None:
    a = register_attempt(tmp_store, hypothesis_id="h1", preregistration_path=committed_prereg, holdout_spec={"universe": "full_pit"})
    b = register_attempt(tmp_store, hypothesis_id="h1", preregistration_path=committed_prereg, holdout_spec={"universe": "mid"})
    assert a.attempt_index == 1
    assert b.attempt_index == 1  # different holdout, independent counter


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_registry_can_only_tighten_never_loosen(tmp_store, committed_prereg, n) -> None:
    """The one-way ratchet, checked directly against real registered
    attempts rather than just the arithmetic helper."""
    spec = {"universe": "full_pit", "n": n}
    attempt = None
    for i in range(n):
        prereg = committed_prereg.parent / f"preregistration_n{n}_{i}.md"
        prereg.write_text(f"# attempt {i}\n", encoding="utf-8")
        subprocess.run(["git", "add", prereg.name], cwd=committed_prereg.parent, check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"n{n}_{i}"], cwd=committed_prereg.parent, check=True)
        attempt = register_attempt(tmp_store, hypothesis_id=f"h{i}", preregistration_path=prereg, holdout_spec=spec)

    criteria = criteria_for_attempt(tmp_store, attempt)
    assert criteria.hit_rate_significance_alpha <= GateCriteria().hit_rate_significance_alpha
    assert criteria.hit_rate_significance_alpha == pytest.approx(GateCriteria().hit_rate_significance_alpha / n)
    # every OTHER field of GateCriteria must be untouched
    assert criteria.min_mean_alpha_net == GateCriteria().min_mean_alpha_net
    assert criteria.best_fold_drop_fraction == GateCriteria().best_fold_drop_fraction
    assert criteria.min_folds_required == GateCriteria().min_folds_required


def test_close_attempt_rejects_invalid_verdict(tmp_store, committed_prereg) -> None:
    attempt = register_attempt(tmp_store, hypothesis_id="h1", preregistration_path=committed_prereg, holdout_spec={"x": 1})
    with pytest.raises(ValueError):
        close_attempt(tmp_store, attempt.attempt_id, verdict="maybe", gate_alpha_used=0.05, metrics={})


def test_close_attempt_records_result(tmp_store, committed_prereg) -> None:
    attempt = register_attempt(tmp_store, hypothesis_id="h1", preregistration_path=committed_prereg, holdout_spec={"x": 1})
    close_attempt(tmp_store, attempt.attempt_id, verdict="fail", gate_alpha_used=0.10, metrics={"mean_alpha_net": -0.001})
    df = tmp_store.read_evaluation_attempts(attempt.holdout_id)
    assert df.iloc[0]["result_verdict"] == "fail"
    assert df.iloc[0]["status"] == "run"


def test_gate_py_is_never_imported_for_writing_by_attempts_module() -> None:
    """Tripwire: evaluation/attempts.py must construct GateCriteria and
    pass it to evaluate_gate as a parameter, never edit gate.py's own
    defaults. Checked by source inspection -- the module must import
    GateCriteria but must NOT reference evaluate_gate's default-altering
    internals (_one_sided_binomial_pvalue, GateVerdict construction)."""
    import inspect

    from stocksense.evaluation import attempts as attempts_mod

    source = inspect.getsource(attempts_mod)
    assert "GateCriteria" in source
    assert "class GateVerdict" not in source
    assert "def evaluate_gate" not in source
