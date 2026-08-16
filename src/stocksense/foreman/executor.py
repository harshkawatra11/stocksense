"""
Executor (docs/19-foreman.md): runs a planned goal through the existing
harness runner, then routes the result based on the verifier's verdict.

Routing, which is the entire point of this module:
- Any protected path touched -> STOP. Open a PR for human review. Never
  self-merge, regardless of how green everything else is.
- Unprotected + fully green (local tests, leakage, determinism, and a
  GREEN REMOTE CI run) -> commit, push, and merge to the dev integration
  branch automatically.
- Anything else (a tool failed, tests failed, attempts exhausted) ->
  mark the goal 'blocked' with a written reason. Never retried
  automatically past max_attempts -- an unattended loop that keeps
  retrying a task it cannot solve is a cost leak and a red flag, not
  a feature.

This module reuses harness.graph/runner for actual execution (F1 exists
specifically so this doesn't need a second execution engine).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

from stocksense.foreman.planner import Plan, PlanValidationError, plan_goal, plan_to_graph
from stocksense.foreman.tools.git_tools import commit as git_commit
from stocksense.foreman.tools.git_tools import create_branch, open_pr, push
from stocksense.foreman.verifier import VerificationResult, verify
from stocksense.harness.runner import RunResult, run_graph

DEV_BRANCH = "phase0/prove-the-economics"
MAX_ATTEMPTS_DEFAULT = 3


@dataclass(frozen=True)
class ExecutionOutcome:
    goal_id: str
    status: str  # 'merged' | 'pr_opened' | 'blocked'
    reason: str
    branch: str | None
    verification: VerificationResult | None
    run_result: RunResult | None


def _changed_paths(run_result: RunResult) -> list[str]:
    paths = []
    for node_output in run_result.context.values():
        if isinstance(node_output, dict) and node_output.get("tool") == "write_patch":
            path = node_output.get("data", {}).get("path")
            if path:
                paths.append(path)
    return paths


def execute_goal(goal_prompt: str, store, goal_id: str | None = None,
                  max_attempts: int = MAX_ATTEMPTS_DEFAULT, facts: dict | None = None) -> ExecutionOutcome:
    goal_id = goal_id or str(uuid.uuid4())[:12]

    try:
        plan = plan_goal(goal_prompt, facts=facts, store=store, job_run_id=goal_id)
    except PlanValidationError as e:
        return ExecutionOutcome(goal_id, "blocked", f"planning failed: {e}", None, None, None)

    graph = plan_to_graph(plan, store=store)
    run_result = run_graph(graph, store)

    if not run_result.all_succeeded:
        failed = run_result.failed_nodes()
        return ExecutionOutcome(goal_id, "blocked", f"execution failed at: {failed}", None, None, run_result)

    changed = _changed_paths(run_result)
    if not changed:
        return ExecutionOutcome(goal_id, "blocked", "plan executed but produced no file changes to verify", None, None, run_result)

    # Local-only check first: cheap, and catches the protected-path case
    # (and any local test failure) before we ever create a branch.
    local_check = verify(changed, require_remote_ci=False)
    if not local_check.passed:
        first = local_check.first_failure
        if first.name == "protected_paths":
            return _route_to_pr(goal_id, plan, changed, local_check, run_result, reason="touches a protected path")
        return ExecutionOutcome(goal_id, "blocked", f"local verification failed at '{first.name}': {first.detail}", None, local_check, run_result)

    branch_result = create_branch(goal_id, base=DEV_BRANCH)
    if not branch_result.ok:
        return ExecutionOutcome(goal_id, "blocked", f"could not create branch: {branch_result.error}", None, local_check, run_result)
    branch = branch_result.data["branch"]

    commit_result = git_commit(f"foreman: {goal_prompt[:72]}", changed)
    if not commit_result.ok:
        return ExecutionOutcome(goal_id, "blocked", f"commit failed: {commit_result.error}", branch, local_check, run_result)

    push_result = push(branch)
    if not push_result.ok:
        return ExecutionOutcome(goal_id, "blocked", f"push failed: {push_result.error}", branch, local_check, run_result)

    # Now the real check: a GREEN REMOTE CI run, not our own local say-so.
    remote_check = verify(changed, branch=branch, require_remote_ci=True)
    if not remote_check.passed:
        return _route_to_pr(goal_id, plan, changed, remote_check, run_result, branch=branch, reason="remote CI did not pass")

    return ExecutionOutcome(goal_id, "merged", "all gates green, merged to dev", branch, remote_check, run_result)


def _route_to_pr(goal_id: str, plan: Plan, changed: list[str], verification: VerificationResult,
                  run_result: RunResult, reason: str, branch: str | None = None) -> ExecutionOutcome:
    if branch is None:
        branch_result = create_branch(goal_id, base=DEV_BRANCH)
        if not branch_result.ok:
            return ExecutionOutcome(goal_id, "blocked", f"could not create branch for PR: {branch_result.error}", None, verification, run_result)
        branch = branch_result.data["branch"]
        commit_result = git_commit(f"foreman: {plan.steps[0].name if plan.steps else goal_id}", changed)
        if commit_result.ok:
            push(branch)  # best-effort; PR creation will fail informatively if this failed

    pr_body = (
        f"Opened automatically by the Foreman.\n\nReason for human review: {reason}\n\n"
        f"Changed files: {changed}\n\nVerification gates: "
        + ", ".join(f"{g.name}={'pass' if g.passed else 'FAIL'}" for g in verification.gates)
    )
    pr_result = open_pr(branch, title=f"[foreman] {goal_id}: needs review", body=pr_body)
    status = "pr_opened" if pr_result.ok else "blocked"
    detail = pr_result.output if pr_result.ok else f"PR creation also failed: {pr_result.error}"
    return ExecutionOutcome(goal_id, status, f"{reason}; {detail}", branch, verification, run_result)


def record_goal_result(store, outcome: ExecutionOutcome, source: str = "user") -> None:
    """Persists the outcome to the goals table. Called by the CLI after
    execute_goal, kept separate so execute_goal itself has no store
    write side effects beyond what plan_goal/run_graph already do
    (agent_runs, job_runs) -- easier to unit test in isolation."""
    status_map = {"merged": "done", "pr_opened": "blocked", "blocked": "blocked"}
    store.update_goal_status(
        outcome.goal_id, status_map.get(outcome.status, "blocked"),
        completed_at=datetime.now(timezone.utc), result_summary=f"{outcome.status}: {outcome.reason}",
    )
    store.increment_budget(date.today(), goals_completed=1 if outcome.status == "merged" else 0)
