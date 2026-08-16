"""Git/GitHub tool tests, all with subprocess mocked -- these tests must
never actually push, branch, or open a PR against the real repo. The
point under test is the tool's own logic (argument construction, return
mapping, error handling), not real git/gh behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stocksense.foreman.tools import registry
from stocksense.foreman.tools.git_tools import check_ci, commit, create_branch, open_pr, push


def _completed(returncode=0, stdout="", stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


@patch("stocksense.foreman.tools.git_tools._run")
def test_create_branch_uses_foreman_prefix(mock_run) -> None:
    mock_run.return_value = _completed(0)
    result = create_branch("abc123")
    assert result.ok is True
    assert result.data["branch"] == "foreman/abc123"
    args = mock_run.call_args[0][0]
    assert "foreman/abc123" in args


@patch("stocksense.foreman.tools.git_tools._run")
def test_create_branch_reports_failure(mock_run) -> None:
    mock_run.return_value = _completed(1, stderr="branch already exists")
    result = create_branch("abc123")
    assert result.ok is False
    assert "branch already exists" in result.error


@patch("stocksense.foreman.tools.git_tools._run")
def test_commit_stages_then_commits(mock_run) -> None:
    mock_run.return_value = _completed(0)
    result = commit("test message", ["a.py", "b.py"])
    assert result.ok is True
    assert mock_run.call_count == 2  # add, then commit


@patch("stocksense.foreman.tools.git_tools._run")
def test_commit_fails_if_add_fails(mock_run) -> None:
    mock_run.return_value = _completed(1, stderr="pathspec error")
    result = commit("msg", ["nonexistent.py"])
    assert result.ok is False


@patch("stocksense.foreman.tools.git_tools._run")
def test_push_reports_success(mock_run) -> None:
    mock_run.return_value = _completed(0)
    result = push("foreman/abc123")
    assert result.ok is True


@patch("stocksense.foreman.tools.git_tools.shutil.which", return_value=None)
def test_open_pr_fails_cleanly_without_gh_cli(mock_which) -> None:
    result = open_pr("foreman/abc", "title", "body")
    assert result.ok is False
    assert "gh CLI not found" in result.error


@patch("stocksense.foreman.tools.git_tools.shutil.which", return_value=None)
def test_check_ci_fails_cleanly_without_gh_cli(mock_which) -> None:
    result = check_ci("foreman/abc")
    assert result.ok is False


@patch("stocksense.foreman.tools.git_tools._run")
@patch("stocksense.foreman.tools.git_tools.shutil.which", return_value="/usr/bin/gh")
def test_check_ci_green_run_is_ok(mock_which, mock_run) -> None:
    mock_run.return_value = _completed(0, stdout='[{"status": "completed", "conclusion": "success", "url": "http://x"}]')
    result = check_ci("foreman/abc")
    assert result.ok is True


@patch("stocksense.foreman.tools.git_tools._run")
@patch("stocksense.foreman.tools.git_tools.shutil.which", return_value="/usr/bin/gh")
def test_check_ci_failed_run_is_not_ok(mock_which, mock_run) -> None:
    mock_run.return_value = _completed(0, stdout='[{"status": "completed", "conclusion": "failure", "url": "http://x"}]')
    result = check_ci("foreman/abc")
    assert result.ok is False


@patch("stocksense.foreman.tools.git_tools._run")
@patch("stocksense.foreman.tools.git_tools.shutil.which", return_value="/usr/bin/gh")
def test_check_ci_no_runs_yet_is_not_ok(mock_which, mock_run) -> None:
    mock_run.return_value = _completed(0, stdout="[]")
    result = check_ci("foreman/abc")
    assert result.ok is False
    assert result.data.get("status") == "not_found"


def test_network_tools_are_classified_correctly() -> None:
    from stocksense.foreman.tools.base import ActionClass

    assert registry.get("push").action_class == ActionClass.NETWORK
    assert registry.get("open_pr").action_class == ActionClass.NETWORK
    assert registry.get("check_ci").action_class == ActionClass.NETWORK
