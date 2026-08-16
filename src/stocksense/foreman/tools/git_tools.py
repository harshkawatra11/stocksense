"""
Git/GitHub tools (docs/19-foreman.md). These are the Foreman's only
NETWORK-class tools alongside knowledge.web_research -- create_branch
and commit are local (git objects, no network), but push and open_pr
reach GitHub, and check_ci polls it.

None of these tools decide WHETHER to push or merge -- that decision
belongs to foreman/verifier.py (did CI go green?) and foreman/executor.py
(is the path protected?). This module only executes what it's told,
which is why the decision logic living elsewhere, in a form that's
independently testable, matters.
"""

from __future__ import annotations

import shutil
import subprocess

from stocksense.core.config import REPO_ROOT
from stocksense.foreman.tools.base import ActionClass, Tool, ToolResult, registry

FOREMAN_BRANCH_PREFIX = "foreman/"


def _run(cmd: list[str], timeout_s: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=timeout_s)


def create_branch(goal_id: str, base: str = "phase0/prove-the-economics") -> ToolResult:
    branch_name = f"{FOREMAN_BRANCH_PREFIX}{goal_id}"
    proc = _run(["git", "checkout", "-b", branch_name, base])
    if proc.returncode != 0:
        return ToolResult(ok=False, output=proc.stdout, error=proc.stderr)
    return ToolResult(ok=True, output=f"created and checked out {branch_name}", data={"branch": branch_name})


def commit(message: str, paths: list[str]) -> ToolResult:
    add_proc = _run(["git", "add"] + paths)
    if add_proc.returncode != 0:
        return ToolResult(ok=False, output=add_proc.stdout, error=add_proc.stderr)
    commit_proc = _run(["git", "commit", "-m", message])
    return ToolResult(ok=commit_proc.returncode == 0, output=commit_proc.stdout, error=commit_proc.stderr or None)


def push(branch: str) -> ToolResult:
    """Reaches GitHub. Only ever called by the executor after the local
    verifier chain has already passed -- pushing a red branch is not
    forbidden by this function (it has no way to know), which is why
    the caller's ordering matters and is tested in test_foreman_verifier.py."""
    proc = _run(["git", "push", "-u", "origin", branch], timeout_s=60)
    return ToolResult(ok=proc.returncode == 0, output=proc.stdout, error=proc.stderr or None)


def open_pr(branch: str, title: str, body: str) -> ToolResult:
    gh = shutil.which("gh")
    if gh is None:
        return ToolResult(ok=False, output="", error="gh CLI not found on PATH")
    proc = _run([gh, "pr", "create", "--head", branch, "--title", title, "--body", body], timeout_s=60)
    return ToolResult(ok=proc.returncode == 0, output=proc.stdout, error=proc.stderr or None, data={"branch": branch})


def check_ci(branch: str) -> ToolResult:
    """Polls the REMOTE run for `branch` -- this is what makes CI an
    independent referee rather than a formality. A caller checking local
    `pytest` output and calling that 'CI passed' is exactly the failure
    mode this tool exists to prevent."""
    gh = shutil.which("gh")
    if gh is None:
        return ToolResult(ok=False, output="", error="gh CLI not found on PATH")
    proc = _run([gh, "run", "list", "--branch", branch, "--json", "status,conclusion,url", "--limit", "1"], timeout_s=30)
    if proc.returncode != 0:
        return ToolResult(ok=False, output=proc.stdout, error=proc.stderr)

    import json
    try:
        runs = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ToolResult(ok=False, output=proc.stdout, error="could not parse gh run list output")

    if not runs:
        return ToolResult(ok=False, output="", error="no CI run found for branch yet", data={"status": "not_found"})

    run = runs[0]
    is_green = run.get("status") == "completed" and run.get("conclusion") == "success"
    return ToolResult(ok=is_green, output=str(run), data=run)


registry.register(Tool("create_branch", ActionClass.EXEC, create_branch, "Create and checkout a foreman/<goal_id> branch."))
registry.register(Tool("commit", ActionClass.EXEC, commit, "Stage and commit the given paths with a message."))
registry.register(Tool("push", ActionClass.NETWORK, push, "Push the current branch to origin."))
registry.register(Tool("open_pr", ActionClass.NETWORK, open_pr, "Open a GitHub PR via gh CLI."))
registry.register(Tool("check_ci", ActionClass.NETWORK, check_ci, "Poll the remote CI run status for a branch."))
