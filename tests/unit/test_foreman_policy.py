"""Protected-path policy tests. This is the single most important file
in the Foreman (docs/19-foreman.md) -- an autonomous harness with a
verifier it can quietly weaken is not a verifier. Every protected path
listed in the plan must actually be caught, in both absolute and
relative form, and the policy file must protect itself."""

from __future__ import annotations

from pathlib import Path

from stocksense.foreman.policy import PROTECTED_PATTERNS, check_patch, is_protected


def test_gate_py_is_protected() -> None:
    assert is_protected("src/stocksense/evaluation/gate.py")


def test_walkforward_py_is_protected() -> None:
    assert is_protected("src/stocksense/evaluation/walkforward.py")


def test_cost_model_is_protected() -> None:
    assert is_protected("src/stocksense/execution/cost_model.py")


def test_leakage_test_is_protected() -> None:
    assert is_protected("tests/unit/test_leakage.py")


def test_determinism_test_is_protected() -> None:
    assert is_protected("tests/unit/test_determinism.py")


def test_gate_test_is_protected() -> None:
    assert is_protected("tests/unit/test_gate.py")


def test_preregistration_files_are_protected() -> None:
    assert is_protected("research/gate_criteria_preregistration.md")
    assert is_protected("research/intraday_gate_preregistration.md")  # doesn't exist yet, still caught


def test_policy_file_protects_itself() -> None:
    assert is_protected("src/stocksense/foreman/policy.py")


def test_evaluation_attempts_registry_is_protected() -> None:
    """Phase J4c: the evaluation-attempt registry can only tighten
    evaluate_gate's significance threshold, never loosen it -- that
    property must go through human review to change, same as gate.py
    itself."""
    assert is_protected("src/stocksense/evaluation/attempts.py")


def test_ci_workflow_is_protected() -> None:
    assert is_protected(".github/workflows/verify.yml")
    assert is_protected(".github/workflows/anything.yml")


def test_absolute_path_form_also_caught() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    abs_path = repo_root / "src" / "stocksense" / "evaluation" / "gate.py"
    assert is_protected(abs_path)


def test_ordinary_source_file_is_not_protected() -> None:
    assert not is_protected("src/stocksense/statements/diagnostics.py")
    assert not is_protected("src/stocksense/agent/claude_cli.py")
    assert not is_protected("tests/unit/test_diagnostics.py")


def test_ordinary_docs_file_is_not_protected() -> None:
    assert not is_protected("docs/STATUS.md")
    assert not is_protected("research/phase0_verdict.md")


def test_check_patch_returns_only_protected_subset() -> None:
    patch_files = [
        "src/stocksense/statements/report.py",
        "src/stocksense/evaluation/gate.py",
        "tests/unit/test_report.py",
    ]
    flagged = check_patch(patch_files)
    assert flagged == ["src/stocksense/evaluation/gate.py"]


def test_check_patch_empty_when_all_clear() -> None:
    assert check_patch(["src/stocksense/statements/report.py", "docs/STATUS.md"]) == []


def test_no_protected_pattern_is_accidentally_empty_string() -> None:
    # a pattern of "" or "*" would protect nothing or everything by
    # accident -- guard the list itself, not just is_protected's logic
    assert all(p.strip() for p in PROTECTED_PATTERNS)
    assert "*" not in PROTECTED_PATTERNS
