"""
Code tools (docs/19-foreman.md). write_patch is the one tool in this
module that can modify the working tree, and it is the enforcement
point for foreman.policy: every write is checked against the protected
list BEFORE touching disk, not after. This is the primary control;
policy.ProtectedPathError is the backstop if a caller bypasses this
function and writes to Path directly, which no Foreman code should ever
do outside this module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from stocksense.core.config import REPO_ROOT
from stocksense.foreman.policy import ProtectedPathError, check_patch
from stocksense.foreman.tools.base import ActionClass, Tool, ToolResult, registry


def read_file(path: str, max_bytes: int = 200_000) -> ToolResult:
    p = (REPO_ROOT / path).resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return ToolResult(ok=False, output="", error=f"path escapes repo root: {path}")
    if not p.exists():
        return ToolResult(ok=False, output="", error=f"file not found: {path}")
    content = p.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    return ToolResult(ok=True, output=content, data={"path": path, "truncated": len(content) >= max_bytes})


def search_code(pattern: str, glob: str = "*.py") -> ToolResult:
    """Simple recursive grep over src/ and tests/, avoiding a dependency
    on ripgrep being installed in whatever environment the Foreman runs
    in. Adequate for the Foreman's own use (finding where a symbol is
    defined/used); not a replacement for the Grep tool a human session uses."""
    import fnmatch
    import re

    matches = []
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return ToolResult(ok=False, output="", error=f"invalid regex: {e}")

    for base in (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "research"):
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file() or not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(f"{f.relative_to(REPO_ROOT).as_posix()}:{i}: {line.strip()[:200]}")

    output = "\n".join(matches[:200])
    return ToolResult(ok=True, output=output, data={"n_matches": len(matches)})


def write_patch(file_path: str, content: str) -> ToolResult:
    """Writes `content` to `file_path` (repo-relative), full-file
    replacement (not a diff applier — the planner is responsible for
    producing complete file contents, matching how this Foreman
    generates output). Refuses protected paths unconditionally."""
    blocked = check_patch([file_path])
    if blocked:
        return ToolResult(ok=False, output="", error=f"protected path, routed to PR instead: {blocked}", data={"blocked_paths": blocked})

    p = (REPO_ROOT / file_path).resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return ToolResult(ok=False, output="", error=f"path escapes repo root: {file_path}")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ToolResult(ok=True, output=f"wrote {file_path} ({len(content)} bytes)", data={"path": file_path})


def run_tests(target: str = "tests/unit", timeout_s: int = 300) -> ToolResult:
    proc = subprocess.run(
        ["python", "-m", "pytest", target, "-q"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=timeout_s,
    )
    ok = proc.returncode == 0
    return ToolResult(ok=ok, output=(proc.stdout + proc.stderr)[-8000:], data={"returncode": proc.returncode})


def run_lint(target: str = "src") -> ToolResult:
    """Best-effort: if ruff isn't installed, report that rather than
    failing the whole verification chain over a missing dev tool."""
    try:
        proc = subprocess.run(
            ["python", "-m", "ruff", "check", target],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        )
    except FileNotFoundError:
        return ToolResult(ok=True, output="ruff not installed, skipping lint", data={"skipped": True})
    return ToolResult(ok=proc.returncode == 0, output=(proc.stdout + proc.stderr)[-4000:], data={"returncode": proc.returncode})


registry.register(Tool("read_file", ActionClass.READ, read_file, "Read a repo-relative file's contents."))
registry.register(Tool("search_code", ActionClass.READ, search_code, "Regex-search src/tests/research for a pattern."))
registry.register(Tool(
    "write_patch", ActionClass.WRITE, write_patch,
    "Write full contents to a repo-relative file. Refuses protected paths.",
    touches_paths=lambda file_path, content: [file_path],
))
registry.register(Tool("run_tests", ActionClass.EXEC, run_tests, "Run pytest against a target path."))
registry.register(Tool("run_lint", ActionClass.EXEC, run_lint, "Run ruff against a target path, if installed."))
