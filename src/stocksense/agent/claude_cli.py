"""
Claude CLI bridge (docs/11-agent-harness.md).

The load-bearing rule this module enforces structurally, not just by
prompt wording: **Python computes every number, Claude writes every
narrative.** A request carries a `facts` dict — the only source of
figures — serialized into the prompt as a fenced JSON block with an
explicit instruction that no number outside it may appear in the
response. A post-check flags (not blocks) responses containing numeric
tokens that don't trace back to `facts`, so drift is caught rather than
silently trusted.

Windows invocation pitfalls, already paid for once in this project's
history — do not rediscover them:
- `subprocess` cannot find `claude` bare on Windows; the npm shim is
  `claude.cmd` and must be resolved via `shutil.which`.
- `--system` is not a valid flag and fails silently (exit 0, empty
  response) — use `--append-system-prompt`.
- Every invocation is timed and written to `agent_runs`, success or not,
  so "did the agent actually run and what did it say" is always
  answerable from the database, not from scrollback.
- THE PROMPT MUST GO ON STDIN, NEVER AS A `-p <prompt>` POSITIONAL
  ARGUMENT. Found live (2026-08-18): on Windows, `claude.cmd` can only be
  launched via `cmd.exe`'s own argument handling (a `.cmd` file isn't
  directly executable by `CreateProcess`), and `cmd.exe` parses its
  command line the way a line typed at a console prompt would — a raw
  newline inside a quoted argument is NOT preserved as literal content,
  it terminates the line. Every prompt in this module is multi-line
  (`_REPORT_PROMPT`-style instructions plus a fenced JSON facts block),
  so passing it as `[binary, "-p", full_prompt, ...]` silently truncated
  every such call at the FIRST newline — confirmed directly: Claude
  received only the first line and replied confused ("your message
  seems to have been cut off"), while `--output-format json`'s own
  output got corrupted into non-JSON text in the process. This means
  every prior multi-line Claude CLI call in this project's history that
  ran on Windows was plausibly receiving a truncated prompt, not the
  full instructions+facts. Fixed by passing the prompt via `stdin`
  (`subprocess.run(cmd, input=full_prompt, ...)`) with no prompt
  positional at all — verified directly to preserve a 3-line prompt
  with embedded quotes intact and return clean JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

from stocksense.agent.access import ClaudeAccessNotGranted
from stocksense.core.config import REPO_ROOT

log = structlog.get_logger(__name__)

SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_TIMEOUT_S = 300

# Patterns that must never reach a prompt. Conservative and over-inclusive
# on purpose — a false positive here just redacts something harmless; a
# false negative could leak a credential to a logged prompt.
_SECRET_PATTERNS = [
    # allow an optional closing quote between the key name and the colon,
    # since JSON serializes keys as "api_key": "value" — the quote sits
    # between the key text and the separator.
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret|password)['\"]?\s*[:=]\s*['\"]?[\w\-\.]{8,}"),
    re.compile(r"\b[A-Za-z0-9_-]{32,}\b"),  # long opaque tokens (JWTs, API keys)
]


def redact_secrets(text: str) -> str:
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


@dataclass(frozen=True)
class AgentRequest:
    prompt: str
    facts: dict = field(default_factory=dict)
    skill: str | None = None
    max_turns: int = 1
    timeout_s: int = DEFAULT_TIMEOUT_S
    model: str | None = None
    # low|medium|high|xhigh|max, per `claude --effort`. Cost/latency
    # control independent of model choice -- e.g. Opus at "low" effort
    # for a cheap architectural decision, Sonnet at "medium" for actual
    # code generation. Neither model nor effort is set by default here;
    # callers (planner.py, codegen.py, assess.py) choose deliberately.
    effort: str | None = None


@dataclass(frozen=True)
class AgentResult:
    agent_run_id: str
    output_text: str
    status: str  # 'ok' | 'unverified_numbers' | 'error' | 'timeout'
    error: str | None
    started_at: datetime
    finished_at: datetime


def _check_access(store) -> None:
    """Raises ClaudeAccessNotGranted unless the desktop app's local
    authorize flag is set for the currently-logged-in account. Opens a
    short-lived Store when the caller didn't pass one, so the gate
    still applies (see invoke()'s docstring) -- fails closed if that
    open itself fails, e.g. another job holds the write lock."""
    from stocksense.agent.access import require_claude_access

    if store is not None:
        require_claude_access(store)
        return

    try:
        from stocksense.core.config import get_settings
        from stocksense.data.store import Store

        temp_store = Store(get_settings().duckdb_path)
    except Exception as e:  # noqa: BLE001 -- fail closed: can't confirm access, so deny it
        raise ClaudeAccessNotGranted(f"could not verify Claude access (database unavailable: {e})") from e

    try:
        require_claude_access(temp_store)
    finally:
        temp_store.close()


def _resolve_claude_binary() -> str:
    """`shutil.which` finds the .cmd shim on Windows; bare subprocess
    calls to 'claude' silently fail to launch npm-installed CLIs there."""
    path = shutil.which("claude")
    if path is None:
        raise RuntimeError(
            "claude CLI not found on PATH. Install via `npm install -g @anthropic-ai/claude-code` "
            "or ensure the npm global bin directory is on PATH."
        )
    return path


def _build_prompt(req: AgentRequest) -> str:
    facts_json = json.dumps(req.facts, indent=2, default=str)
    facts_block = redact_secrets(facts_json)
    return (
        f"{req.prompt}\n\n"
        "The following JSON is the ONLY source of numbers you may use. "
        "Every figure in your response must trace directly to a value in "
        "this block. Do not compute, estimate, or invent any number not "
        "present here:\n\n```json\n" + facts_block + "\n```\n"
    )


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_fact_numbers(facts: dict) -> set[str]:
    numbers: set[str] = set()

    def walk(obj):
        if isinstance(obj, (int, float)):
            numbers.add(str(obj))
            numbers.add(f"{obj:.2f}" if isinstance(obj, float) else str(obj))
            numbers.add(f"{obj:.1f}" if isinstance(obj, float) else str(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            for m in _NUMBER_RE.findall(obj):
                numbers.add(m)

    walk(facts)
    return numbers


def _check_unverified_numbers(output_text: str, facts: dict) -> bool:
    """Returns True if the response contains a numeric token that does
    not appear (even approximately) among the facts. A tripwire, not a
    hard block — small formatting differences (e.g. '82.68' vs '83') are
    tolerated by rounding both sides before comparing."""
    fact_numbers = _extract_fact_numbers(facts)
    fact_rounded = {round(float(n.replace(",", "")), 0) for n in fact_numbers if _is_number(n)}

    for token in _NUMBER_RE.findall(output_text):
        clean = token.replace(",", "")
        if not _is_number(clean):
            continue
        val = float(clean)
        if abs(val) < 10:  # small numbers (percentages, ratios, counts) too noisy to police reliably
            continue
        if round(val, 0) not in fact_rounded and round(val, -1) not in {round(f, -1) for f in fact_rounded}:
            return True
    return False


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def invoke(req: AgentRequest, store=None, job_run_id: str | None = None) -> AgentResult:
    """Invoke the Claude CLI non-interactively with the given facts and
    prompt. Always records the invocation to agent_runs when a store is
    provided (best-effort — a logging failure must not mask the agent's
    actual result).

    Phase F2: this is the single enforcement point for the desktop app's
    Claude-CLI authorize/decline gate (agent/access.py) — every caller
    (foreman/planner.py, foreman/codegen.py, foreman/assess.py,
    statements/report.py, rag/agent.py) funnels through here, so none of
    them can accidentally bypass the gate by forgetting to check it
    themselves. If `store` is None, a short-lived Store is opened just
    for this check (fail-closed if that open itself fails, e.g. the
    database is busy) — the gate applies even to a caller that didn't
    think to pass one, rather than silently skipping the check.
    """
    agent_run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    full_prompt = _build_prompt(req)
    prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()[:16]

    status = "ok"
    error: str | None = None
    output_text = ""

    try:
        _check_access(store)
        binary = _resolve_claude_binary()
        # Prompt goes on stdin, never as a `-p <prompt>` argument -- see
        # the module docstring's "THE PROMPT MUST GO ON STDIN" note for
        # why (Windows' cmd.exe wrapping of the .cmd shim truncates a
        # multi-line argument at its first newline).
        cmd = [binary, "-p", "--output-format", "json"]
        if SKILLS_DIR.exists():
            # AUDIT FIX: `--setting-sources` takes one of the enum values
            # user|project|local (verified: passing a raw path raises
            # "Invalid setting source" and every invoke() call failed
            # whenever SKILLS_DIR existed, i.e. always in this repo). It
            # was never meant to point at an arbitrary directory -- with
            # cwd=REPO_ROOT (set below), "project" is what makes Claude
            # load this repo's own project-level settings, including
            # SKILLS_DIR itself under the project's skill-discovery
            # convention.
            cmd += ["--setting-sources", "project"]
        if req.model:
            cmd += ["--model", req.model]
        if req.effort:
            cmd += ["--effort", req.effort]

        proc = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True, timeout=req.timeout_s,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            status = "error"
            error = redact_secrets(proc.stderr[:2000])
        else:
            try:
                parsed = json.loads(proc.stdout)
                output_text = parsed.get("result", proc.stdout)
            except json.JSONDecodeError:
                output_text = proc.stdout

            if _check_unverified_numbers(output_text, req.facts):
                status = "unverified_numbers"
                log.warning("agent_unverified_numbers", agent_run_id=agent_run_id, skill=req.skill)

    except subprocess.TimeoutExpired:
        status = "timeout"
        error = f"exceeded {req.timeout_s}s"
    except ClaudeAccessNotGranted as e:
        status = "access_denied"
        error = str(e)
    except Exception as e:  # noqa: BLE001
        status = "error"
        error = str(e)

    finished_at = datetime.now(timezone.utc)

    if store is not None:
        try:
            store.insert_agent_run(
                {
                    "agent_run_id": agent_run_id,
                    "job_run_id": job_run_id,
                    "skill_name": req.skill,
                    "prompt_hash": prompt_hash,
                    "input_json": redact_secrets(json.dumps({"facts": req.facts, "effort": req.effort}, default=str)),
                    "output_text": redact_secrets(output_text),
                    "model": req.model,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": status,
                    "error": error,
                    "cost_estimate": None,
                }
            )
        except Exception as log_err:  # noqa: BLE001
            log.warning("agent_run_logging_failed", error=str(log_err))

    return AgentResult(
        agent_run_id=agent_run_id, output_text=output_text, status=status,
        error=error, started_at=started_at, finished_at=finished_at,
    )
