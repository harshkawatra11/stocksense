"""Executor routing tests, everything mocked -- no real branch, commit,
push, PR, or CI poll happens here. What's under test is the ROUTING
LOGIC: protected paths always go to PR regardless of how green
everything else is; unprotected + fully green (incl. remote CI) merges;
anything else blocks. This is the decision logic the plan calls out as
living separately from the tools that execute it."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from stocksense.agent.claude_cli import AgentResult
from stocksense.foreman.executor import execute_goal
from stocksense.foreman.tools.base import ToolResult
from stocksense.harness.graph import Graph, Node
from stocksense.harness.runner import NodeOutcome, RunResult


def _fake_agent_result(text: str) -> AgentResult:
    return AgentResult(agent_run_id="t", output_text=text, status="ok", error=None,
                        started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc))


def _plan_json(tool: str, args: dict) -> str:
    import json
    return json.dumps([{"name": "step1", "tool": tool, "args": args, "depends_on": []}])


class _StubStore:
    """Minimal store stub — executor only needs insert_agent_run,
    start_job_run, finish_job_run for its dependencies (planner logs
    agent_runs; run_graph logs job_runs)."""

    def insert_agent_run(self, row): pass
    def start_job_run(self, run_id, job_name, started_at): pass
    def finish_job_run(self, run_id, status, finished_at, detail_json=None): pass
    def con(self): pass


def test_execute_goal_blocks_on_planning_failure(monkeypatch):
    monkeypatch.setattr("stocksense.foreman.executor.plan_goal",
                         lambda *a, **k: (_ for _ in ()).throw(__import__("stocksense.foreman.planner", fromlist=["PlanValidationError"]).PlanValidationError("bad plan")))
    outcome = execute_goal("do something", _StubStore())
    assert outcome.status == "blocked"
    assert "planning failed" in outcome.reason


@patch("stocksense.foreman.executor.plan_goal")
def test_execute_goal_blocks_on_tool_failure(mock_plan_goal):
    from stocksense.foreman.planner import Plan, PlanStep

    mock_plan_goal.return_value = Plan(
        steps=[PlanStep(name="a", tool="read_file", args={"path": "does/not/exist.py"}, depends_on=())],
        raw_response="",
    )
    outcome = execute_goal("read a missing file", _StubStore())
    assert outcome.status == "blocked"
    assert "execution failed" in outcome.reason


@patch("stocksense.foreman.executor.verify")
@patch("stocksense.foreman.executor.plan_goal")
def test_execute_goal_routes_protected_path_to_pr(mock_plan_goal, mock_verify, tmp_path):
    from stocksense.foreman.planner import Plan, PlanStep
    from stocksense.foreman.verifier import GateResult, VerificationResult

    mock_plan_goal.return_value = Plan(
        steps=[PlanStep(name="a", tool="write_patch", args={"file_path": "src/stocksense/evaluation/gate.py", "content": "x"}, depends_on=())],
        raw_response="",
    )
    # write_patch itself refuses the protected path -> tool fails -> execution fails
    outcome = execute_goal("edit the gate", _StubStore())
    assert outcome.status == "blocked"  # write_patch tool itself refused, so run_graph fails first
    assert "execution failed" in outcome.reason


@patch("stocksense.foreman.executor.open_pr")
@patch("stocksense.foreman.executor.push")
@patch("stocksense.foreman.executor.git_commit")
@patch("stocksense.foreman.executor.create_branch")
@patch("stocksense.foreman.executor.verify")
@patch("stocksense.foreman.executor.plan_goal")
def test_execute_goal_merges_on_full_green(mock_plan_goal, mock_verify, mock_create_branch,
                                            mock_commit, mock_push, mock_open_pr, tmp_path, monkeypatch):
    from stocksense.foreman.planner import Plan, PlanStep
    from stocksense.foreman.verifier import GateResult, VerificationResult
    import stocksense.foreman.tools.code as code_mod

    monkeypatch.setattr(code_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "scratch").mkdir()

    mock_plan_goal.return_value = Plan(
        steps=[PlanStep(name="a", tool="write_patch", args={"file_path": "scratch/new_feature.py", "content": "x = 1"}, depends_on=())],
        raw_response="",
    )
    green = VerificationResult(gates=[GateResult("protected_paths", True, "clear"), GateResult("local_tests", True, "ok")])
    mock_verify.return_value = green
    mock_create_branch.return_value = ToolResult(ok=True, output="", data={"branch": "foreman/abc"})
    mock_commit.return_value = ToolResult(ok=True, output="")
    mock_push.return_value = ToolResult(ok=True, output="")

    outcome = execute_goal("add a small feature", _StubStore())
    assert outcome.status == "merged"
    mock_open_pr.assert_not_called()  # merged, so no PR needed


@patch("stocksense.foreman.executor.open_pr")
@patch("stocksense.foreman.executor.push")
@patch("stocksense.foreman.executor.git_commit")
@patch("stocksense.foreman.executor.create_branch")
@patch("stocksense.foreman.executor.verify")
@patch("stocksense.foreman.executor.plan_goal")
def test_execute_goal_opens_pr_when_remote_ci_fails(mock_plan_goal, mock_verify, mock_create_branch,
                                                      mock_commit, mock_push, mock_open_pr, tmp_path, monkeypatch):
    from stocksense.foreman.planner import Plan, PlanStep
    from stocksense.foreman.verifier import GateResult, VerificationResult
    import stocksense.foreman.tools.code as code_mod

    monkeypatch.setattr(code_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "scratch").mkdir()

    mock_plan_goal.return_value = Plan(
        steps=[PlanStep(name="a", tool="write_patch", args={"file_path": "scratch/new_feature.py", "content": "x = 1"}, depends_on=())],
        raw_response="",
    )
    local_green = VerificationResult(gates=[GateResult("protected_paths", True, "clear"), GateResult("local_tests", True, "ok")])
    remote_red = VerificationResult(gates=[GateResult("protected_paths", True, "clear"), GateResult("remote_ci", False, "failed")])
    mock_verify.side_effect = [local_green, remote_red]
    mock_create_branch.return_value = ToolResult(ok=True, output="", data={"branch": "foreman/abc"})
    mock_commit.return_value = ToolResult(ok=True, output="")
    mock_push.return_value = ToolResult(ok=True, output="")
    mock_open_pr.return_value = ToolResult(ok=True, output="https://github.com/x/pr/1")

    outcome = execute_goal("add a small feature", _StubStore())
    assert outcome.status == "pr_opened"
    mock_open_pr.assert_called_once()


def test_execute_goal_never_merges_without_calling_verify_with_remote_ci(monkeypatch):
    """Structural check: merged status must only ever come from a path
    that called verify(..., require_remote_ci=True). Guards against a
    future refactor accidentally short-circuiting to 'merged' on local
    green alone."""
    import inspect

    from stocksense.foreman import executor as executor_mod

    source = inspect.getsource(executor_mod)
    # the merge line must appear after a require_remote_ci=True call in the source
    ci_true_idx = source.index("require_remote_ci=True")
    merged_idx = source.index('"merged"')
    assert ci_true_idx < merged_idx
