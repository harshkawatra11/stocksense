"""Regression tests for two real bugs found live on 2026-08-18 while
running Kundli for the first time against real trade data -- both in
`invoke()`'s subprocess command construction, both would have kept
silently corrupting or outright failing EVERY multi-line Claude CLI call
on Windows if left unfixed:

1. The prompt was passed as a `-p <prompt>` positional argument. On
   Windows, `claude.cmd` can only be launched via `cmd.exe`'s own
   argument handling (a `.cmd` file isn't directly executable by
   CreateProcess), and cmd.exe parses its command line the way a line
   typed at a console would -- a raw newline inside a quoted argument
   terminates the line rather than staying literal. Every prompt this
   module builds is multi-line (instructions + a fenced JSON facts
   block), so this silently truncated every call at the first newline.
   Confirmed directly: Claude received only the first line and replied
   confused ("your message seems to have been cut off"). Fixed by
   passing the prompt via stdin instead.

2. `--setting-sources <SKILLS_DIR path>` -- this flag takes one of the
   enum values user|project|local, never an arbitrary directory.
   Confirmed directly: passing a path raised "Invalid setting source"
   and failed every single invoke() call, since SKILLS_DIR exists in
   this repo. Fixed to pass "project" (with cwd=REPO_ROOT already set,
   this is what makes Claude load this repo's project-level settings,
   including the skills directory under its own discovery convention).
"""

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
def test_prompt_is_never_passed_as_a_positional_cli_argument(mock_resolve, mock_run, mock_access) -> None:
    mock_run.return_value = _mock_proc()
    multiline_prompt = 'Line one has "quotes" in it.\nLine two is here.\nLine three ends it.'
    invoke(AgentRequest(prompt=multiline_prompt, facts={"x": 1}))

    cmd = mock_run.call_args[0][0]
    assert multiline_prompt not in cmd  # never present as a literal argv element
    assert "-p" in cmd
    # -p must be the print-mode flag only, never followed by the prompt text itself
    p_idx = cmd.index("-p")
    if p_idx + 1 < len(cmd):
        assert not cmd[p_idx + 1].startswith("Line one")


@patch("stocksense.agent.claude_cli._check_access")
@patch("stocksense.agent.claude_cli.subprocess.run")
@patch("stocksense.agent.claude_cli._resolve_claude_binary", return_value="claude")
def test_full_prompt_including_facts_is_sent_via_stdin(mock_resolve, mock_run, mock_access) -> None:
    mock_run.return_value = _mock_proc()
    multiline_prompt = "Instruction line one.\nInstruction line two."
    invoke(AgentRequest(prompt=multiline_prompt, facts={"net_pnl": -1309.74}))

    kwargs = mock_run.call_args.kwargs
    assert "input" in kwargs
    sent = kwargs["input"]
    assert "Instruction line one." in sent
    assert "Instruction line two." in sent
    assert "-1309.74" in sent  # the facts block reached stdin intact too


@patch("stocksense.agent.claude_cli._check_access")
@patch("stocksense.agent.claude_cli.subprocess.run")
@patch("stocksense.agent.claude_cli._resolve_claude_binary", return_value="claude")
def test_setting_sources_uses_a_valid_enum_value_not_a_path(mock_resolve, mock_run, mock_access, monkeypatch) -> None:
    import stocksense.agent.claude_cli as mod

    monkeypatch.setattr(mod, "SKILLS_DIR", mod.REPO_ROOT / "skills")  # exists in this repo
    mock_run.return_value = _mock_proc()
    invoke(AgentRequest(prompt="hi"))

    cmd = mock_run.call_args[0][0]
    assert "--setting-sources" in cmd
    value = cmd[cmd.index("--setting-sources") + 1]
    assert value in ("user", "project", "local")  # never a filesystem path
