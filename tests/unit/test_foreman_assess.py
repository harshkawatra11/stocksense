"""Self-assessment tests. gather_signals is pure fact-gathering (no
agent call) and is tested directly; propose_goals' agent call is
monkeypatched so these tests verify parsing/validation, not Claude's
judgment."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from stocksense.agent.claude_cli import AgentResult
from stocksense.foreman.assess import gather_signals, propose_goals


class _StubStore:
    def read_goals(self):
        return pd.DataFrame({"status": ["done", "blocked", "done"]})

    def read_protected_violations(self):
        return pd.DataFrame({"path": ["src/stocksense/evaluation/gate.py"]})


def test_gather_signals_reflects_real_test_count() -> None:
    signals = gather_signals(_StubStore())
    assert signals["test_count"] > 100  # this suite alone has well over 100 tests


def test_gather_signals_reports_backlog_not_built() -> None:
    signals = gather_signals(_StubStore())
    # none of these marker files exist yet at this point in the plan
    assert "data_spine" in signals["backlog_not_built"]
    assert "skills_suite" in signals["backlog_not_built"]


def test_gather_signals_counts_recent_goals_and_violations() -> None:
    signals = gather_signals(_StubStore())
    assert signals["recent_goal_count"] == 3
    assert signals["protected_violation_count"] == 1


def test_propose_goals_parses_valid_json_array(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return AgentResult(
            agent_run_id="t", output_text='[{"goal": "build X", "reason": "Y", "priority": 1}]',
            status="ok", error=None, started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("stocksense.foreman.assess.invoke", fake_invoke)
    goals = propose_goals(_StubStore())
    assert len(goals) == 1
    assert goals[0]["goal"] == "build X"


def test_propose_goals_returns_empty_on_malformed_response(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return AgentResult(
            agent_run_id="t", output_text="not json at all",
            status="ok", error=None, started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("stocksense.foreman.assess.invoke", fake_invoke)
    goals = propose_goals(_StubStore())
    assert goals == []


def test_propose_goals_filters_malformed_entries(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return AgentResult(
            agent_run_id="t", output_text='[{"goal": "ok one"}, {"not_a_goal_field": "bad"}, "just a string"]',
            status="ok", error=None, started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("stocksense.foreman.assess.invoke", fake_invoke)
    goals = propose_goals(_StubStore())
    assert len(goals) == 1
    assert goals[0]["goal"] == "ok one"
