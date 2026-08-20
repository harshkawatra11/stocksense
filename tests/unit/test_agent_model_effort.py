"""Verifies model/effort actually reach the subprocess command line --
the gap being closed here: AgentRequest had a `model` field nobody set
and no `effort` field at all, so every agent call silently used whatever
the CLI's own default was, with no way to route cheap decomposition
calls to a lower effort than expensive code-generation calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stocksense.agent.claude_cli import AgentRequest, invoke


def _mock_proc(stdout='{"result": "ok"}', returncode=0):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = ""
    return m


@patch("stocksense.agent.claude_cli._check_access")  # Phase F2's access gate is a separate concern from this file
@patch("stocksense.agent.claude_cli.subprocess.run")
@patch("stocksense.agent.claude_cli._resolve_claude_binary", return_value="claude")
def test_model_flag_passed_when_set(mock_resolve, mock_run, mock_access) -> None:
    mock_run.return_value = _mock_proc()
    invoke(AgentRequest(prompt="hi", model="opus"))
    cmd = mock_run.call_args[0][0]
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "opus"


@patch("stocksense.agent.claude_cli._check_access")
@patch("stocksense.agent.claude_cli.subprocess.run")
@patch("stocksense.agent.claude_cli._resolve_claude_binary", return_value="claude")
def test_effort_flag_passed_when_set(mock_resolve, mock_run, mock_access) -> None:
    mock_run.return_value = _mock_proc()
    invoke(AgentRequest(prompt="hi", model="sonnet", effort="medium"))
    cmd = mock_run.call_args[0][0]
    assert "--effort" in cmd
    assert cmd[cmd.index("--effort") + 1] == "medium"


@patch("stocksense.agent.claude_cli._check_access")
@patch("stocksense.agent.claude_cli.subprocess.run")
@patch("stocksense.agent.claude_cli._resolve_claude_binary", return_value="claude")
def test_neither_flag_passed_when_unset(mock_resolve, mock_run, mock_access) -> None:
    mock_run.return_value = _mock_proc()
    invoke(AgentRequest(prompt="hi"))
    cmd = mock_run.call_args[0][0]
    assert "--model" not in cmd
    assert "--effort" not in cmd


@patch("stocksense.agent.claude_cli.subprocess.run")
@patch("stocksense.agent.claude_cli._resolve_claude_binary", return_value="claude")
def test_effort_recorded_in_agent_runs_input_json(mock_resolve, mock_run) -> None:
    import json

    mock_run.return_value = _mock_proc()
    captured = {}

    class _StubStore:
        def insert_agent_run(self, row):
            captured.update(row)

    invoke(AgentRequest(prompt="hi", model="opus", effort="low"), store=_StubStore())
    assert captured["model"] == "opus"
    logged = json.loads(captured["input_json"])
    assert logged["effort"] == "low"
