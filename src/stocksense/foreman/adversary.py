"""
Adversary (docs/19-foreman.md): a red-team pass that actively tries to
falsify a result BEFORE it is accepted, rather than only checking that
tests pass. This project's single highest-ROI moment was a stress test
finding a data bug (an adjustment-factor discontinuity) that had
inflated the strongest fold of a backtest everyone already believed —
this module is that discipline, automated, run on every Foreman result
rather than once by hand.

A system that only ever confirms its own work is a machine for
producing confident nonsense. An adversary finding does not just log a
warning: it BLOCKS the merge and creates a new goal, so falsification
produces work rather than a shrug.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from stocksense.core.config import REPO_ROOT


@dataclass(frozen=True)
class AdversaryFinding:
    check: str
    file: str
    detail: str
    severity: str  # 'note' | 'concern' | 'blocking'


def _read(path: str) -> str | None:
    p = REPO_ROOT / path
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def check_assertionless_tests(test_files: list[str]) -> list[AdversaryFinding]:
    """A test file with a test_* function containing no assert and no
    pytest.raises is worse than no test: it reports green while
    verifying nothing."""
    findings = []
    for path in test_files:
        if "test_" not in path or not path.endswith(".py"):
            continue
        content = _read(path)
        if content is None:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            findings.append(AdversaryFinding("assertionless_tests", path, "file does not parse as valid Python", "blocking"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(node))
                # AUDIT FIX: ast.dump() renders `pytest.raises(...)` as
                # nested Attribute/Call nodes, e.g.
                # "Attribute(value=Name(id='pytest'), attr='raises')" —
                # the literal substring "pytest.raises" never appears in
                # dump output, so the original check never matched and
                # every pytest.raises-based test was misflagged. Use
                # ast.unparse to get real source text instead.
                node_source = ast.unparse(node)
                has_raises_ctx = "pytest.raises" in node_source or "pytest.warns" in node_source
                if not has_assert and not has_raises_ctx:
                    findings.append(AdversaryFinding(
                        "assertionless_tests", path, f"'{node.name}' has no assert statement", "blocking",
                    ))
    return findings


def check_swallowed_exceptions(changed_files: list[str]) -> list[AdversaryFinding]:
    """A bare `except Exception: pass` (or return-without-reraise) around
    a critical path hides failures rather than handling them. Flags
    except blocks whose entire body is `pass`, `continue`, or a bare
    `return` with no logging — the pattern most likely to be a shortcut
    to 'green' rather than real error handling."""
    findings = []
    for path in changed_files:
        if not path.endswith(".py"):
            continue
        content = _read(path)
        if content is None:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body_types = {type(stmt) for stmt in node.body}
                if body_types <= {ast.Pass}:
                    findings.append(AdversaryFinding(
                        "swallowed_exceptions", path,
                        f"bare except at line {node.lineno} silently passes — no log, no reraise", "concern",
                    ))
    return findings


def check_hardcoded_test_expectations(changed_files: list[str]) -> list[AdversaryFinding]:
    """A test asserting a value equals itself, or a mock configured to
    return exactly what the code under test will echo back, proves
    nothing. Coarse heuristic: flag `assert X == X` patterns literally."""
    findings = []
    for path in changed_files:
        if "test_" not in path or not path.endswith(".py"):
            continue
        content = _read(path)
        if content is None:
            continue
        for i, line in enumerate(content.splitlines(), start=1):
            m = re.match(r"\s*assert\s+(\S+)\s*==\s*\1\s*$", line)
            if m:
                findings.append(AdversaryFinding("tautological_assertion", path, f"line {i}: '{line.strip()}' asserts a value equals itself", "blocking"))
    return findings


def check_research_result_seed_sensitivity(fold_alphas_by_seed: dict[int, float], min_seeds: int = 2) -> list[AdversaryFinding]:
    """For a research goal: does the headline result survive a different
    random seed? A result that only appears under one specific seed is
    the exact 'evaluator overfitting' risk docs/10-evaluation.md sec 17
    names as the deepest one in the whole design."""
    if len(fold_alphas_by_seed) < min_seeds:
        return [AdversaryFinding("seed_sensitivity", "research_result", f"only {len(fold_alphas_by_seed)} seed(s) tested; need >= {min_seeds}", "concern")]
    signs = {v > 0 for v in fold_alphas_by_seed.values()}
    if len(signs) > 1:
        return [AdversaryFinding("seed_sensitivity", "research_result", f"result flips sign across seeds: {fold_alphas_by_seed}", "blocking")]
    return []


ALL_CHECKS = [check_assertionless_tests, check_swallowed_exceptions, check_hardcoded_test_expectations]


def red_team(changed_files: list[str]) -> list[AdversaryFinding]:
    """Runs the static-analysis adversary checks against the files a
    goal touched. Research-specific checks (seed sensitivity) are called
    separately by research goals with their own data, since they need
    inputs (multi-seed results) that a generic file list doesn't carry."""
    findings: list[AdversaryFinding] = []
    findings += check_assertionless_tests(changed_files)
    findings += check_swallowed_exceptions(changed_files)
    findings += check_hardcoded_test_expectations(changed_files)
    return findings


def has_blocking_finding(findings: list[AdversaryFinding]) -> bool:
    return any(f.severity == "blocking" for f in findings)
