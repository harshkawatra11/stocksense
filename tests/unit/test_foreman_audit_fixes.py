"""Regression tests for four honesty bugs found by direct audit
(2026-08-18), all confirmed live before the fix by grep/inspection:

1. executor.py reported status "merged" when no merge tool exists
   anywhere in git_tools.py -- renamed to "pushed", the thing that
   actually happens.
2. insert_ledger_entry had zero call sites -- build_ledger was
   permanently empty regardless of Foreman activity.
3. insert_protected_violation had zero call sites --
   protected_violation_count was permanently 0.
4. budget.invocations was never incremented -- check_budget compared 0
   against the cap forever, so the daily cap could never fire.
5. (bonus, same audit) verifier.py hardcoded the research gate to
   passed=True, delegating to a foreman/tools/research.py that doesn't
   exist -- now fails closed.

Uses a real Store (tmp_path) rather than a stub, since the point of
these tests is that rows actually land in the tables, not that the
right method was called."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from stocksense.data.store import Store
from stocksense.foreman.executor import ExecutionOutcome, record_ledger_entries, record_protected_violations
from stocksense.foreman.tools.base import ToolResult
from stocksense.foreman.verifier import verify
from stocksense.harness.runner import NodeOutcome, RunResult


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _outcome(run_result=None, status="pushed", reason="all gates green, pushed for human merge") -> ExecutionOutcome:
    return ExecutionOutcome(
        goal_id="g1", status=status, reason=reason, branch="foreman/g1",
        verification=None, run_result=run_result,
    )


# ---- Fix 2: build_ledger gets written to ----

def test_record_ledger_entries_writes_one_row_per_node(tmp_store) -> None:
    run_result = RunResult(
        outcomes=[
            NodeOutcome(name="step1", status="completed", run_id="r1"),
            NodeOutcome(name="step2", status="failed", run_id="r2", error="boom"),
        ],
        context={
            "step1": {"tool": "write_patch", "output": "wrote file", "data": {"path": "scratch/x.py"}},
        },
    )
    outcome = _outcome(run_result=run_result)

    n = record_ledger_entries(tmp_store, "g1", outcome)
    assert n == 2

    ledger = tmp_store.read_ledger(goal_id="g1")
    assert len(ledger) == 2
    verdicts = set(ledger["verdict"])
    assert "ok" in verdicts and "failed" in verdicts


def test_record_ledger_entries_maps_tool_action_class(tmp_store) -> None:
    run_result = RunResult(
        outcomes=[NodeOutcome(name="step1", status="completed", run_id="r1")],
        context={"step1": {"tool": "write_patch", "output": "ok", "data": {"path": "scratch/x.py"}}},
    )
    record_ledger_entries(tmp_store, "g1", _outcome(run_result=run_result))
    ledger = tmp_store.read_ledger(goal_id="g1")
    assert ledger.iloc[0]["action"] == "write"  # write_patch is ActionClass.WRITE in the registry


def test_record_ledger_entries_noop_when_no_run_result(tmp_store) -> None:
    n = record_ledger_entries(tmp_store, "g1", _outcome(run_result=None))
    assert n == 0
    assert tmp_store.read_ledger(goal_id="g1").empty


# ---- Fix 3: protected_violations gets written to ----

def test_record_protected_violations_writes_a_row_for_a_protected_path(tmp_store) -> None:
    outcome = _outcome(status="pr_opened", reason="touches a protected path")
    n = record_protected_violations(
        tmp_store, "g1", outcome, changed_paths=["src/stocksense/evaluation/gate.py", "scratch/ok.py"],
    )
    assert n == 1  # only the actually-protected one counts

    violations = tmp_store.con.execute("SELECT * FROM protected_violations").fetchdf()
    assert len(violations) == 1
    assert violations.iloc[0]["path"] == "src/stocksense/evaluation/gate.py"
    assert violations.iloc[0]["action_taken"] == "blocked_routed_to_pr"


def test_record_protected_violations_noop_for_unrelated_block_reason(tmp_store) -> None:
    outcome = _outcome(status="blocked", reason="local verification failed at 'local_tests': ...")
    n = record_protected_violations(
        tmp_store, "g1", outcome, changed_paths=["src/stocksense/evaluation/gate.py"],
    )
    assert n == 0
    assert tmp_store.con.execute("SELECT COUNT(*) FROM protected_violations").fetchone()[0] == 0


# ---- Fix 4: budget.invocations actually increments ----

def test_increment_budget_invocations_makes_the_cap_reachable(tmp_store) -> None:
    from datetime import date

    from stocksense.foreman.budget import check_budget

    today = date.today()
    for _ in range(5):
        tmp_store.increment_budget(today, invocations=1)

    status = check_budget(tmp_store, max_invocations=5)
    assert status.within_budget is False
    assert status.invocations_used == 5


# ---- Fix 1: status is "pushed", never "merged" ----

def test_no_code_path_still_reports_literal_merged_status() -> None:
    """Checks executable code only (via ast), not comments/docstrings
    that legitimately discuss the old, now-fixed behavior by name."""
    import ast
    import inspect

    from stocksense.foreman import executor as executor_mod

    tree = ast.parse(inspect.getsource(executor_mod))
    string_constants = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    # exclude docstrings (first statement of the module/each function/class)
    docstrings = {ast.get_docstring(n) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
    code_strings = [s for s in string_constants if s not in docstrings]
    assert "merged" not in code_strings


# ---- Fix 5 (bonus): research gate fails closed, not open ----

@patch("stocksense.foreman.verifier.run_tests")
def test_research_gate_fails_closed_with_no_implementation(mock_run_tests) -> None:
    mock_run_tests.return_value = ToolResult(ok=True, output="ok")
    result = verify(changed_paths=["scratch/x.py"], require_remote_ci=False, is_research_goal=True)
    gate = next(g for g in result.gates if g.name == "pre_registered_gate")
    assert gate.passed is False
    assert not result.passed
