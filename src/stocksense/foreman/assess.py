"""
Self-assessment (docs/19-foreman.md): reads the project's own state and
proposes ranked goals with reasons. This is the audit performed by hand
earlier in this project's history (docs/STATUS.md, the CRITICAL/HIGH/MED
findings), encoded as a repeatable function — the thing that makes the
Foreman *build itself* rather than execute a fixed backlog.

Deliberately conservative: this function computes facts (test count,
what's missing, what STATUS.md says) and hands them to the agent bridge
for prioritization narrative. It does not invent findings; it reads real
signals (file existence, test counts, ledger history) and lets Claude
rank and explain them — the same compute/narrate split as everywhere
else in this system.
"""

from __future__ import annotations

from pathlib import Path

from stocksense.agent.claude_cli import AgentRequest, invoke
from stocksense.core.config import REPO_ROOT
from stocksense.foreman.planner import PLANNER_EFFORT, PLANNER_MODEL

_KNOWN_BACKLOG_MARKERS = {
    "electron_desktop_app": "desktop/main.js",
    "reconcile_loop": "src/stocksense/harness/loops.py",
    "data_spine": "src/stocksense/data/nse_archive.py",
    "skills_suite": "skills/nse-market-structure/SKILL.md",
    "rag_agent": "src/stocksense/rag/agent.py",
    "optimizer": "src/stocksense/optimizer/risk_layer.py",
    "intraday_track": "src/stocksense/data/upstox_intraday.py",
}


def gather_signals(store) -> dict:
    """Deterministic facts about project state — no narrative here."""
    test_count = _count_tests()
    backlog_status = {name: (REPO_ROOT / marker).exists() for name, marker in _KNOWN_BACKLOG_MARKERS.items()}

    recent_goals = store.read_goals().head(20) if hasattr(store, "read_goals") else None
    violations = store.read_protected_violations() if hasattr(store, "read_protected_violations") else None

    return {
        "test_count": test_count,
        "backlog_built": {k: v for k, v in backlog_status.items() if v},
        "backlog_not_built": [k for k, v in backlog_status.items() if not v],
        "recent_goal_count": 0 if recent_goals is None else len(recent_goals),
        "recent_goal_statuses": [] if recent_goals is None else recent_goals["status"].tolist(),
        "protected_violation_count": 0 if violations is None else len(violations),
        "status_md_exists": (REPO_ROOT / "docs" / "STATUS.md").exists(),
    }


def _count_tests() -> int:
    tests_dir = REPO_ROOT / "tests" / "unit"
    if not tests_dir.exists():
        return 0
    count = 0
    for f in tests_dir.glob("test_*.py"):
        content = f.read_text(encoding="utf-8", errors="ignore")
        count += content.count("\ndef test_")
    return count


_ASSESS_PROMPT = """You are self-assessing a quant-trading codebase
(StockSense) to propose what it should build next. Below are computed
facts about its current state: what backlog items are already built
(marker file exists), what isn't, test count, recent goal outcomes, and
whether the agent has recently attempted to touch a protected path
(a red flag if the count is rising).

Propose 3-5 ranked goals as a JSON array of
{{"goal": "<one sentence, concrete and actionable>", "reason": "<why this, why now>", "priority": <1-10, 1=highest>}}.
Prefer goals that: close a correctness gap over one that adds a new
surface; are scoped to one coherent unit of work, not several bundled
together; and do not require touching a protected path (gate.py,
walkforward.py, cost_model.py, leakage/determinism/gate tests,
preregistration files) -- if the most valuable next thing DOES require
that, say so as a reason to route it to human review rather than
silently avoiding it.

Facts:
{facts}

Respond with ONLY the JSON array."""


def propose_goals(store) -> list[dict]:
    signals = gather_signals(store)
    import json

    result = invoke(
        AgentRequest(
            prompt=_ASSESS_PROMPT.format(facts=json.dumps(signals, indent=2)), facts=signals,
            model=PLANNER_MODEL, effort=PLANNER_EFFORT,  # judgment/ranking, same tier as goal decomposition
        ),
        store=store,
    )
    try:
        text = result.output_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        proposals = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return []

    if not isinstance(proposals, list):
        return []
    return [p for p in proposals if isinstance(p, dict) and "goal" in p]
