"""
Planner (docs/19-foreman.md): turns a goal prompt into a harness.Graph of
tool invocations. Uses the existing agent bridge (agent/claude_cli.py),
so the same compute/narrate discipline and numeric tripwire apply here
too -- though a plan is structure, not a number, the same principle
holds: the planner's OUTPUT is validated against ground truth (the tool
registry) before anything downstream trusts it.

The critical safety property: an emitted step naming a tool that isn't
in the registry is a PLANNING FAILURE, not a runtime crash. Steps are
validated before a single one executes, so a hallucinated tool name
never reaches subprocess.run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from stocksense.agent.claude_cli import AgentRequest, invoke
from stocksense.foreman.tools import registry
from stocksense.harness.graph import Graph, Node


@dataclass(frozen=True)
class PlanStep:
    name: str
    tool: str
    args: dict
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    steps: list[PlanStep]
    raw_response: str


class PlanValidationError(Exception):
    pass


_PLANNER_PROMPT = """You are decomposing a software engineering goal into
a sequence of tool calls for an autonomous build harness working on the
StockSense NSE quant-trading codebase.

Available tools:
{tool_descriptions}

Rules:
- Prefer reading and reusing existing code (read_file, search_code)
  before writing new files.
- Every code change must be followed by a run_tests step.
- Never propose editing: evaluation/gate.py, evaluation/walkforward.py,
  execution/cost_model.py, test_leakage.py, test_determinism.py,
  test_gate.py, any *preregistration*.md file, foreman/policy.py, or
  .github/workflows/* -- these are protected; if the goal requires
  changing one of them, say so in your response instead of planning a
  write to it.
- Respond with ONLY a JSON array, no prose, where each element is:
  {{"name": "<unique step id>", "tool": "<tool name from the list above>",
    "args": {{...}}, "depends_on": ["<step id>", ...]}}

The goal:
{goal_prompt}
"""


def plan_goal(goal_prompt: str, facts: dict | None = None, store=None, job_run_id: str | None = None) -> Plan:
    prompt = _PLANNER_PROMPT.format(tool_descriptions=registry.describe_all(), goal_prompt=goal_prompt)
    result = invoke(
        AgentRequest(prompt=prompt, facts=facts or {}, skill="backtest-rigor"),
        store=store, job_run_id=job_run_id,
    )
    steps = _parse_plan(result.output_text)
    _validate_plan(steps)
    return Plan(steps=steps, raw_response=result.output_text)


def _parse_plan(raw: str) -> list[PlanStep]:
    text = raw.strip()
    # tolerate a fenced code block, since models often wrap JSON in ```json
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"planner did not return valid JSON: {e}\nraw: {raw[:500]}")

    if not isinstance(parsed, list):
        raise PlanValidationError(f"planner output must be a JSON array, got {type(parsed)}")

    steps = []
    for item in parsed:
        try:
            steps.append(PlanStep(
                name=item["name"], tool=item["tool"], args=item.get("args", {}),
                depends_on=tuple(item.get("depends_on", [])),
            ))
        except KeyError as e:
            raise PlanValidationError(f"plan step missing required field {e}: {item}")
    return steps


def _validate_plan(steps: list[PlanStep]) -> None:
    if not steps:
        raise PlanValidationError("planner produced an empty plan")

    names = [s.name for s in steps]
    if len(names) != len(set(names)):
        raise PlanValidationError(f"duplicate step names in plan: {names}")

    for step in steps:
        if step.tool not in registry:
            raise PlanValidationError(
                f"step '{step.name}' names unknown tool '{step.tool}' "
                f"(hallucinated? available: {registry.names()})"
            )
        for dep in step.depends_on:
            if dep not in names:
                raise PlanValidationError(f"step '{step.name}' depends on unknown step '{dep}'")


def plan_to_graph(plan: Plan) -> Graph:
    """Converts a validated Plan into a harness.Graph so execution reuses
    the existing resumable/idempotent runner rather than a second
    execution engine."""
    nodes = []
    for step in plan.steps:
        tool = registry.get(step.tool)

        def make_fn(tool=tool, args=step.args):
            def fn(ctx: dict) -> dict:
                result = tool.fn(**args)
                if not result.ok:
                    raise RuntimeError(f"tool '{tool.name}' failed: {result.error or result.output[:500]}")
                return {"tool": tool.name, "output": result.output, "data": result.data}
            return fn

        nodes.append(Node(name=step.name, fn=make_fn(), depends_on=step.depends_on))
    return Graph(nodes)
