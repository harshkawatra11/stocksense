"""
Protected-path policy — the single most important file in the Foreman
(docs/19-foreman.md).

An autonomous harness is only as trustworthy as its verifier, and the
verifier must sit outside the agent's reach. This project already lived
the failure mode once, by human hand: CRITICAL-2, gate thresholds fitted
to the data they judged, in the same session that documented that exact
risk as the deepest one in the design (docs/10-evaluation.md sec 17).
An agent with write access to gate.py reproduces that failure reliably,
unattended, and reports success -- because loosening a constraint is
usually the shortest path to green.

The fix is not "ask the agent nicely not to." It is a literal allowlist
checked before any write, so touching a protected path is not an error
the agent can retry past -- it is a ROUTING decision: the goal branches
to a human-reviewed PR and halts.

This file protects itself. So does the CI workflow that verifies it. An
agent that can edit its own constraints has none.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

# Patterns are matched against a POSIX-style path relative to the repo
# root (forward slashes, regardless of OS) via fnmatch, so "*" matches
# within a path segment and this stays readable as a literal list.
PROTECTED_PATTERNS: tuple[str, ...] = (
    "src/stocksense/evaluation/gate.py",
    "src/stocksense/evaluation/walkforward.py",
    "src/stocksense/execution/cost_model.py",
    "tests/unit/test_leakage.py",
    "tests/unit/test_determinism.py",
    "tests/unit/test_gate.py",
    "research/*preregistration*.md",
    "src/stocksense/evaluation/attempts.py",
    "src/stocksense/foreman/policy.py",
    ".github/workflows/*",
)


def _normalize(path: str | Path) -> str:
    """Relative, forward-slash, repo-root-anchored form. Accepts both
    absolute paths (from a tool that resolved one) and repo-relative
    paths (from a planner-emitted patch) so callers don't have to know
    which form they were handed."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(_repo_root())
        except ValueError:
            pass  # outside the repo entirely; fall through and let is_protected() decide
    normalized = p.as_posix()
    # AUDIT FIX: str.lstrip strips any of the given CHARACTERS, not a
    # literal prefix -- "./" strips leading dots too, which silently ate
    # the "." off ".github/workflows/..." and made the CI-workflow
    # protection pattern never match. Strip only an actual "./" prefix.
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _repo_root() -> Path:
    # foreman/policy.py -> foreman -> stocksense -> src -> REPO_ROOT
    return Path(__file__).resolve().parents[3]


def is_protected(path: str | Path) -> bool:
    """True if `path` matches any protected pattern. Matching is
    intentionally permissive (fnmatch, not exact-equality) — a new file
    added under research/ containing "preregistration" in its name
    should be caught even though it didn't exist when this list was
    written, since the RISK (a criteria file getting quietly edited) is
    about the file's role, not its exact name."""
    normalized = _normalize(path)
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in PROTECTED_PATTERNS)


def check_patch(file_paths: list[str | Path]) -> list[str]:
    """Returns the subset of file_paths that are protected. Empty list
    means the patch is clear to proceed; callers must check this BEFORE
    applying any write, not after."""
    return [str(p) for p in file_paths if is_protected(p)]


class ProtectedPathError(Exception):
    """Raised by tools that attempt a write despite check_patch() having
    flagged the path — a defense-in-depth backstop, not the primary
    control (the primary control is the executor checking check_patch()
    before ever calling a write tool)."""

    def __init__(self, paths: list[str]):
        self.paths = paths
        super().__init__(f"refused to write to protected path(s): {paths}")
