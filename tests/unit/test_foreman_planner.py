"""Planner tests: plan parsing, validation (esp. hallucinated-tool
rejection, since that's the actual safety property), and graph
conversion. The agent call is monkeypatched -- these tests verify the
planner's OWN logic, not Claude's output quality."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stocksense.agent.claude_cli import AgentResult
from stocksense.foreman.planner import (
    PLANNER_EFFORT,
    PLANNER_MODEL,
    Plan,
    PlanStep,
    PlanValidationError,
    _parse_plan,
    _validate_plan,
    plan_goal,
    plan_to_graph,
)


def _fake_result(text: str) -> AgentResult:
    return AgentResult(agent_run_id="t", output_text=text, status="ok", error=None,
                        started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc))


def test_parse_plan_handles_bare_json() -> None:
    raw = '[{"name": "a", "tool": "read_file", "args": {"path": "x.py"}, "depends_on": []}]'
    steps = _parse_plan(raw)
    assert len(steps) == 1
    assert steps[0].tool == "read_file"


def test_parse_plan_handles_fenced_json() -> None:
    raw = '```json\n[{"name": "a", "tool": "read_file", "args": {}, "depends_on": []}]\n```'
    steps = _parse_plan(raw)
    assert len(steps) == 1


def test_parse_plan_rejects_invalid_json() -> None:
    with pytest.raises(PlanValidationError):
        _parse_plan("not json at all")


def test_parse_plan_rejects_non_array() -> None:
    with pytest.raises(PlanValidationError):
        _parse_plan('{"name": "a"}')


def test_parse_plan_rejects_missing_required_field() -> None:
    with pytest.raises(PlanValidationError):
        _parse_plan('[{"tool": "read_file"}]')  # missing "name" (args/depends_on have defaults)


def test_validate_plan_rejects_empty() -> None:
    with pytest.raises(PlanValidationError):
        _validate_plan([])


def test_validate_plan_rejects_hallucinated_tool() -> None:
    steps = [PlanStep(name="a", tool="delete_everything", args={}, depends_on=())]
    with pytest.raises(PlanValidationError, match="unknown tool"):
        _validate_plan(steps)


def test_validate_plan_rejects_duplicate_names() -> None:
    steps = [
        PlanStep(name="a", tool="read_file", args={}, depends_on=()),
        PlanStep(name="a", tool="search_code", args={}, depends_on=()),
    ]
    with pytest.raises(PlanValidationError, match="duplicate"):
        _validate_plan(steps)


def test_validate_plan_rejects_unknown_dependency() -> None:
    steps = [PlanStep(name="a", tool="read_file", args={}, depends_on=("ghost",))]
    with pytest.raises(PlanValidationError, match="unknown step"):
        _validate_plan(steps)


def test_validate_plan_accepts_well_formed_plan() -> None:
    steps = [
        PlanStep(name="a", tool="read_file", args={"path": "x.py"}, depends_on=()),
        PlanStep(name="b", tool="run_tests", args={}, depends_on=("a",)),
    ]
    _validate_plan(steps)  # should not raise


def test_plan_goal_rejects_hallucinated_tool_end_to_end(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return _fake_result('[{"name": "a", "tool": "delete_everything", "args": {}, "depends_on": []}]')

    monkeypatch.setattr("stocksense.foreman.planner.invoke", fake_invoke)
    with pytest.raises(PlanValidationError):
        plan_goal("do something dangerous")


def test_plan_goal_succeeds_on_valid_plan(monkeypatch) -> None:
    def fake_invoke(req, store=None, job_run_id=None):
        return _fake_result('[{"name": "read", "tool": "read_file", "args": {"path": "pyproject.toml"}, "depends_on": []}]')

    monkeypatch.setattr("stocksense.foreman.planner.invoke", fake_invoke)
    plan = plan_goal("read the pyproject file")
    assert len(plan.steps) == 1


def test_plan_to_graph_produces_valid_topological_order() -> None:
    plan = Plan(
        steps=[
            PlanStep(name="a", tool="read_file", args={"path": "pyproject.toml"}, depends_on=()),
            PlanStep(name="b", tool="run_tests", args={"target": "tests/unit/test_foreman_policy.py"}, depends_on=("a",)),
        ],
        raw_response="",
    )
    graph = plan_to_graph(plan)
    order = graph.topological_order()
    assert order.index("a") < order.index("b")


def test_plan_to_graph_node_fn_executes_the_real_tool() -> None:
    plan = Plan(steps=[PlanStep(name="a", tool="read_file", args={"path": "pyproject.toml"}, depends_on=())], raw_response="")
    graph = plan_to_graph(plan)
    result = graph["a"].fn({})
    assert "stocksense" in result["output"]


def test_plan_to_graph_node_fn_raises_on_tool_failure() -> None:
    plan = Plan(steps=[PlanStep(name="a", tool="read_file", args={"path": "does/not/exist.py"}, depends_on=())], raw_response="")
    graph = plan_to_graph(plan)
    with pytest.raises(RuntimeError):
        graph["a"].fn({})


def test_plan_goal_uses_opus_low_effort_for_decomposition(monkeypatch) -> None:
    """The two-model split's planning half: decomposition must go
    through PLANNER_MODEL/PLANNER_EFFORT (opus/low), not whatever the
    CLI defaults to."""
    captured = {}

    def fake_invoke(req, store=None, job_run_id=None):
        captured["model"] = req.model
        captured["effort"] = req.effort
        return _fake_result('[{"name": "a", "tool": "read_file", "args": {"path": "x.py"}, "depends_on": []}]')

    monkeypatch.setattr("stocksense.foreman.planner.invoke", fake_invoke)
    plan_goal("do something")
    assert captured["model"] == PLANNER_MODEL == "opus"
    assert captured["effort"] == PLANNER_EFFORT == "low"


def test_plan_to_graph_routes_spec_through_codegen_not_literal_content(monkeypatch, tmp_path) -> None:
    """A write_patch step carrying 'spec' (not 'content') must resolve
    through codegen.generate_file_content -- the enforcement point for
    'Opus never writes code, only decides what code is needed.'"""
    import stocksense.foreman.tools.code as code_mod

    monkeypatch.setattr(code_mod, "REPO_ROOT", tmp_path)
    codegen_calls = []

    def fake_codegen(file_path, spec, context=None, store=None, job_run_id=None):
        codegen_calls.append((file_path, spec))
        return "generated content"

    monkeypatch.setattr("stocksense.foreman.planner.generate_file_content", fake_codegen)

    plan = Plan(
        steps=[PlanStep(name="write_it", tool="write_patch", args={"file_path": "scratch/out.py", "spec": "a module that does X"}, depends_on=())],
        raw_response="",
    )
    graph = plan_to_graph(plan)
    graph["write_it"].fn({})

    assert len(codegen_calls) == 1
    assert codegen_calls[0] == ("scratch/out.py", "a module that does X")
    assert (tmp_path / "scratch" / "out.py").read_text() == "generated content"


def test_plan_to_graph_skips_codegen_when_content_already_literal(monkeypatch, tmp_path) -> None:
    """A write_patch step that already has literal 'content' (e.g. a
    trivial config change the planner wrote inline) must NOT trigger an
    extra codegen call -- codegen is for generated code, not every write."""
    import stocksense.foreman.tools.code as code_mod

    monkeypatch.setattr(code_mod, "REPO_ROOT", tmp_path)
    codegen_calls = []
    monkeypatch.setattr("stocksense.foreman.planner.generate_file_content",
                         lambda *a, **k: codegen_calls.append(1))

    plan = Plan(
        steps=[PlanStep(name="write_it", tool="write_patch", args={"file_path": "scratch/out.txt", "content": "literal"}, depends_on=())],
        raw_response="",
    )
    graph = plan_to_graph(plan)
    graph["write_it"].fn({})

    assert codegen_calls == []
    assert (tmp_path / "scratch" / "out.txt").read_text() == "literal"
