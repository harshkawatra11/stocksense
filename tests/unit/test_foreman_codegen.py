"""Codegen tests: the Sonnet half of the two-model split. The agent call
is monkeypatched -- these verify codegen's own logic (fence-stripping,
model/effort routing), not Claude's code-writing quality."""

from __future__ import annotations

from datetime import datetime, timezone

from stocksense.agent.claude_cli import AgentResult
from stocksense.foreman.codegen import CODEGEN_EFFORT, CODEGEN_MODEL, generate_file_content


def _fake_result(text: str) -> AgentResult:
    return AgentResult(agent_run_id="t", output_text=text, status="ok", error=None,
                        started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc))


def test_generate_file_content_strips_fenced_code_block(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return _fake_result("```python\nx = 1\ny = 2\n```")

    monkeypatch.setattr("stocksense.foreman.codegen.invoke", fake_invoke)
    content = generate_file_content("foo.py", "a file with two assignments")
    assert content == "x = 1\ny = 2"
    assert "```" not in content


def test_generate_file_content_passes_through_unfenced_output(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return _fake_result("x = 1\ny = 2")

    monkeypatch.setattr("stocksense.foreman.codegen.invoke", fake_invoke)
    content = generate_file_content("foo.py", "spec")
    assert content == "x = 1\ny = 2"


def test_generate_file_content_uses_sonnet_medium(monkeypatch) -> None:
    captured = {}

    def fake_invoke(req, store=None, job_run_id=None):
        captured["model"] = req.model
        captured["effort"] = req.effort
        return _fake_result("content")

    monkeypatch.setattr("stocksense.foreman.codegen.invoke", fake_invoke)
    generate_file_content("foo.py", "spec")
    assert captured["model"] == CODEGEN_MODEL == "sonnet"
    assert captured["effort"] == CODEGEN_EFFORT == "medium"
