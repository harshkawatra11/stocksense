"""
Synthesis layer — bring-your-own CLI.

The final synthesis/validation calls run through whichever CLI the user connected:
Claude Code (Anthropic), Codex (OpenAI), or Gemini (Google). The active backend and
its models come from the provider config (intelligence/provider_config.py), chosen in
the first-run modal. All three are driven non-interactively with the prompt on stdin.

Public API:
    run(prompt, role, system="") -> str   # role: "fast" | "deep"; "" on failure (legacy)
    detect() -> dict                        # which CLIs are installed
    active_backend() / active_label()       # current choice, for tagging output
    get_component_status() -> dict          # Stage 0 truth-layer status of the last run() call

Stage 0 truth layer: run() still returns "" on failure for backward compatibility
(existing callers do `if out:` and no-op), but it now also records an explicit
status object — {status, detail, source} — for the most recent call, retrievable
via get_component_status(). Callers that care whether synthesis actually happened
(rather than silently no-op'ing) should check get_component_status() after calling
run(), or catch LLMUnavailable if they use run_strict().

No API keys here — each CLI carries its own auth (the user logs in once with that CLI).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache

from intelligence.provider_config import get_active

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3


class LLMUnavailable(Exception):
    """Raised by run_strict() when the synthesis CLI could not produce a response.
    Carries the same {status, detail, source} shape as get_component_status()."""

    def __init__(self, detail: str, source: str):
        self.status = "unavailable"
        self.detail = detail
        self.source = source
        super().__init__(detail)


# Status of the most recent run() call, per the Stage 0 truth-layer contract:
# {status: "ok"|"degraded"|"unavailable", detail: str, source: "claude"|"codex"|"gemini"|"skipped"}
_last_status: dict = {"status": "unavailable", "detail": "no synthesis call made yet", "source": "skipped"}


def get_component_status() -> dict:
    """Status of the most recent run()/run_strict() call, per the shared status contract."""
    return dict(_last_status)

# Per-backend metadata: the CLI binary to resolve and a human label.
BACKENDS = {
    "claude": {"bin": "claude", "label": "Claude"},
    "codex":  {"bin": "codex",  "label": "Codex"},
    "gemini": {"bin": "gemini", "label": "Gemini"},
}


@lru_cache(maxsize=8)
def _which(binary: str) -> str | None:
    """Resolve a CLI on PATH. On Windows npm shims are .cmd; shutil.which finds them."""
    return shutil.which(binary)


def active_backend() -> str:
    b = get_active().get("synthesis", {}).get("backend", "claude")
    return b if b in BACKENDS else "claude"


def active_label() -> str:
    return BACKENDS[active_backend()]["label"]


def _model_for(role: str) -> str:
    synth = get_active().get("synthesis", {})
    return synth.get("deep_model", "") if role == "deep" else synth.get("fast_model", "")


def detect() -> dict:
    """{backend: {installed: bool, label: str, bin: str|None}} for the setup modal."""
    out = {}
    for name, meta in BACKENDS.items():
        path = _which(meta["bin"])
        out[name] = {"installed": path is not None, "label": meta["label"], "path": path}
    return out


# ------------------------------------------------------------------ #
# Per-backend runners — each returns response text ('' on failure)    #
# ------------------------------------------------------------------ #
def _run_claude(prompt: str, model: str, system: str, timeout: int) -> str:
    # Prompt on stdin (not argv): it carries non-ASCII (₹, →, •) and can be long;
    # routing through the Windows claude.CMD shim as an argument mangles it.
    cmd = [_which("claude") or "claude", "-p"]
    if model:
        cmd += ["--model", model]
    effort = settings.CLAUDE_CLI_EFFORT
    if effort:
        cmd += ["--effort", effort]
    if system:
        cmd += ["--append-system-prompt", system]  # claude -p uses this, NOT --system
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode == 0:
        return (result.stdout or "").strip()
    log.warning("claude CLI error: %s", (result.stderr or "")[:300])
    return ""


def _run_codex(prompt: str, model: str, system: str, timeout: int) -> str:
    # Codex exec has no system flag, so fold system into the prompt. The final
    # answer is written to a file via --output-last-message (stdout carries event noise).
    binary = _which("codex") or "codex"
    full = f"{system}\n\n{prompt}" if system else prompt
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False, encoding="utf-8") as tf:
        out_path = tf.name
    try:
        cmd = [binary, "exec", "--skip-git-repo-check", "--color", "never",
               "--output-last-message", out_path]
        if model:
            cmd += ["-m", model]
        result = subprocess.run(cmd, input=full, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        if result.returncode != 0:
            log.warning("codex CLI error: %s", (result.stderr or "")[:300])
        with open(out_path, encoding="utf-8") as f:
            text = f.read().strip()
        return text or (result.stdout or "").strip()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _run_gemini(prompt: str, model: str, system: str, timeout: int) -> str:
    # Gemini CLI: prompt on stdin, optional -m model. No separate system flag in the
    # non-interactive path, so fold system in. (Verify flags against the installed CLI.)
    binary = _which("gemini") or "gemini"
    full = f"{system}\n\n{prompt}" if system else prompt
    cmd = [binary]
    if model:
        cmd += ["-m", model]
    result = subprocess.run(cmd, input=full, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode == 0:
        return (result.stdout or "").strip()
    log.warning("gemini CLI error: %s", (result.stderr or "")[:300])
    return ""


_RUNNERS = {"claude": _run_claude, "codex": _run_codex, "gemini": _run_gemini}


def verify_backend(backend: str, timeout: int = 45) -> tuple[bool, str]:
    """Prove a specific backend is installed AND authenticated by running one tiny prompt.
    Used by the setup modal's Verify button before the choice is saved."""
    meta = BACKENDS.get(backend)
    if not meta:
        return False, f"unknown backend '{backend}'"
    if _which(meta["bin"]) is None:
        return False, f"{meta['bin']} CLI not found on PATH — install it first"
    try:
        out = _RUNNERS[backend]("Reply with exactly the word: OK", "", "You are terse.", timeout)
        return (bool(out), (out[:120] if out else "no output (CLI may not be logged in)"))
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:160]


def run(prompt: str, role: str = "fast", system: str = "",
        retries: int = DEFAULT_RETRIES, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Run the prompt through the active synthesis CLI. role selects the model tier
    ("fast" for per-signal review, "deep" for EOD/calibration/weekend). Returns the
    response text, or '' if the CLI is missing/failing (caller treats '' as no-op).

    Also records an explicit status in get_component_status() so callers that want
    to know WHY synthesis was skipped (rather than silently treating '' as success)
    can check it. Use run_strict() to get a raised LLMUnavailable instead.
    """
    global _last_status
    backend = active_backend()
    binary = BACKENDS[backend]["bin"]
    if _which(binary) is None:
        detail = f"{binary} CLI not found on PATH"
        log.warning("Synthesis CLI '%s' not found on PATH — skipping synthesis", binary)
        _last_status = {"status": "unavailable", "detail": detail, "source": "skipped"}
        return ""

    model = _model_for(role)
    runner = _RUNNERS[backend]

    last_detail = "unknown failure"
    for attempt in range(retries):
        try:
            out = runner(prompt, model, system, timeout)
            if out:
                _last_status = {"status": "ok", "detail": f"{backend} responded", "source": backend}
                return out
            last_detail = f"{backend} returned empty response"
        except subprocess.TimeoutExpired:
            last_detail = f"{backend} CLI timeout after {timeout}s (attempt {attempt + 1}/{retries})"
            log.warning("%s CLI timeout (attempt %d)", backend, attempt + 1)
        except FileNotFoundError:
            last_detail = f"{backend} CLI vanished from PATH"
            log.warning("%s CLI vanished from PATH", backend)
            _last_status = {"status": "unavailable", "detail": last_detail, "source": "skipped"}
            return ""
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    _last_status = {"status": "unavailable", "detail": f"{last_detail} — retries exhausted", "source": backend}
    return ""


def run_strict(prompt: str, role: str = "fast", system: str = "",
                retries: int = DEFAULT_RETRIES, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Same as run(), but raises LLMUnavailable instead of returning '' on failure.
    Prefer this in new call sites that must not silently proceed as if synthesis ran.
    """
    out = run(prompt, role=role, system=system, retries=retries, timeout=timeout)
    if not out:
        status = get_component_status()
        raise LLMUnavailable(status["detail"], status["source"])
    return out
