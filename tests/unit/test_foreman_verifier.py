"""Verifier gate-chain tests: order and short-circuiting are the
properties under test, since a gate chain that runs expensive remote CI
on a result that already failed protected-path checking would be both
slow and (worse) capable of accidentally validating a blocked change."""

from __future__ import annotations

from unittest.mock import patch

from stocksense.foreman.tools.base import ToolResult
from stocksense.foreman.verifier import verify


def test_protected_path_stops_the_chain_immediately() -> None:
    result = verify(["src/stocksense/evaluation/gate.py"], require_remote_ci=False)
    assert result.passed is False
    assert len(result.gates) == 1  # never got to run_tests
    assert result.gates[0].name == "protected_paths"


@patch("stocksense.foreman.verifier.run_tests")
def test_local_test_failure_stops_before_leakage_suite(mock_run_tests) -> None:
    mock_run_tests.return_value = ToolResult(ok=False, output="1 failed")
    result = verify(["src/stocksense/statements/report.py"], require_remote_ci=False)
    assert result.passed is False
    names = [g.name for g in result.gates]
    assert names == ["protected_paths", "local_tests"]  # never reached leakage/determinism


@patch("stocksense.foreman.verifier.check_ci")
@patch("stocksense.foreman.verifier.run_tests")
def test_all_local_gates_pass_but_no_branch_fails_remote_ci_gate(mock_run_tests, mock_check_ci) -> None:
    mock_run_tests.return_value = ToolResult(ok=True, output="ok")
    result = verify(["src/stocksense/statements/report.py"], branch=None, require_remote_ci=True)
    assert result.passed is False
    assert result.first_failure.name == "remote_ci"
    mock_check_ci.assert_not_called()  # no branch -> don't even try


@patch("stocksense.foreman.verifier.check_ci")
@patch("stocksense.foreman.verifier.run_tests")
def test_full_green_chain_passes(mock_run_tests, mock_check_ci) -> None:
    mock_run_tests.return_value = ToolResult(ok=True, output="ok")
    mock_check_ci.return_value = ToolResult(ok=True, output="green")
    result = verify(["src/stocksense/statements/report.py"], branch="foreman/abc123", require_remote_ci=True)
    assert result.passed is True
    assert [g.name for g in result.gates] == ["protected_paths", "local_tests", "leakage_suite", "determinism_suite", "remote_ci"]


@patch("stocksense.foreman.verifier.check_ci")
@patch("stocksense.foreman.verifier.run_tests")
def test_remote_ci_red_fails_even_if_local_all_green(mock_run_tests, mock_check_ci) -> None:
    mock_run_tests.return_value = ToolResult(ok=True, output="ok")
    mock_check_ci.return_value = ToolResult(ok=False, output="failure")
    result = verify(["src/stocksense/statements/report.py"], branch="foreman/abc123", require_remote_ci=True)
    assert result.passed is False
    assert result.first_failure.name == "remote_ci"


def test_local_only_check_does_not_require_ci() -> None:
    with patch("stocksense.foreman.verifier.run_tests") as mock_run_tests:
        mock_run_tests.return_value = ToolResult(ok=True, output="ok")
        result = verify(["src/stocksense/statements/report.py"], require_remote_ci=False)
    assert result.passed is True
    assert "remote_ci" not in [g.name for g in result.gates]
