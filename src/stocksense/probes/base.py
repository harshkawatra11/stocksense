"""Probe scaffolding.

A probe answers one factual question about the environment in under a few
minutes, and writes the raw answer to research/probes/<name>.md. Every previous
build of this project lost days to an assumption that a ten-minute probe would
have killed, so probes run BEFORE the code that depends on them.

A probe never raises: a failure IS the finding, and gets recorded as one.
"""

from __future__ import annotations

import json
import platform
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from stocksense.core.config import REPO_ROOT
from stocksense.core.clock import now_ist

PROBE_DIR = REPO_ROOT / "research" / "probes"


@dataclass
class ProbeResult:
    name: str
    question: str
    verdict: str = "UNKNOWN"  # PASS | FAIL | BLOCKED | UNKNOWN
    findings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def note(self, line: str) -> None:
        self.notes.append(line)
        print(f"  {line}", flush=True)

    def to_markdown(self) -> str:
        lines = [
            f"# Probe: {self.name}",
            "",
            f"- **Question:** {self.question}",
            f"- **Verdict:** **{self.verdict}**",
            f"- **Run at:** {now_ist().isoformat()}",
            f"- **Machine:** {platform.platform()}",
            "",
            "## Findings",
            "",
            "```json",
            json.dumps(self.findings, indent=2, default=str),
            "```",
            "",
        ]
        if self.notes:
            lines += ["## Log", ""] + [f"- {n}" for n in self.notes] + [""]
        if self.error:
            lines += ["## Error", "", "```", self.error, "```", ""]
        return "\n".join(lines)

    def write(self) -> Path:
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROBE_DIR / f"{self.name}.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def run_probe(name: str, question: str, fn) -> ProbeResult:
    """Run `fn(result)`; capture any exception as the finding rather than crashing."""
    result = ProbeResult(name=name, question=question)
    print(f"\n=== probe: {name} ===", flush=True)
    print(f"    {question}", flush=True)
    started = datetime.now()
    try:
        fn(result)
    except Exception:
        result.verdict = "FAIL"
        result.error = traceback.format_exc()
        print(f"  ! raised: {result.error.strip().splitlines()[-1]}", flush=True)
    result.findings["elapsed_s"] = round((datetime.now() - started).total_seconds(), 2)
    path = result.write()
    print(f"  -> {result.verdict}  ({path.relative_to(REPO_ROOT)})", flush=True)
    return result
