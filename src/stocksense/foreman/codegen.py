"""
Code generation (docs/19-foreman.md): the "Sonnet executes" half of the
two-model split named in the plan (Opus plans, Sonnet builds). Deliberately
separate from planner.py's decomposition call — planning is a small
number of architectural judgment calls (cheap, low effort is enough);
writing an actual file is a larger, more mechanical generation task that
benefits from a different effort budget.

Given a `spec` (what the file should contain, produced by the planner)
plus whatever upstream step outputs are available as context, this
returns the literal file content a write_patch step will write. It goes
through the same agent bridge as everything else, so the numeric
tripwire and secret redaction still apply even though most generated
files are code, not narrative containing figures.
"""

from __future__ import annotations

from stocksense.agent.claude_cli import AgentRequest, invoke
from stocksense.core.config import get_settings

# Phase F4: was CODEGEN_MODEL/CODEGEN_EFFORT module-level constants --
# now Settings fields, read fresh on every call. See planner.py's
# equivalent comment for why (live-editable from the desktop app).

_CODEGEN_PROMPT = """You are writing the complete contents of one file
for the StockSense NSE quant-trading codebase, as part of a larger
change an architect (a separate planning pass) already decomposed into
steps. Follow the codebase's existing conventions: docstrings explain
WHY not WHAT, dataclasses for structured returns, type hints, no
comments that just restate the code.

Target file: {file_path}

Specification (what this file must do):
{spec}

Context from prior steps in this change (file contents already read,
earlier files already written):
{context}

Respond with ONLY the complete raw file contents -- no markdown fences,
no explanation before or after."""


def generate_file_content(file_path: str, spec: str, context: dict | None = None,
                           store=None, job_run_id: str | None = None) -> str:
    settings = get_settings()
    prompt = _CODEGEN_PROMPT.format(file_path=file_path, spec=spec, context=context or {})
    result = invoke(
        AgentRequest(prompt=prompt, facts={}, skill="claude-report-writing", model=settings.codegen_model, effort=settings.codegen_effort),
        store=store, job_run_id=job_run_id,
    )
    text = result.output_text.strip()
    # tolerate the model wrapping output in a fenced code block despite
    # the instruction not to -- strip fences if present rather than
    # writing "```python" as the first line of a source file.
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text
