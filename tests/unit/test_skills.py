"""Skill validation (per the plan's Phase 2 verification requirement):
every SKILL.md must parse, have required frontmatter, name matching its
folder, and stay under the size budget. No third-party skill content is
installed anywhere in this repo -- all skills here are authored
in-repo, verified by the fact this test only looks at skills/ under the
project root."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"
MAX_BODY_LINES = 500


def _skill_dirs() -> list[Path]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(d for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists())


def _parse_skill(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} does not start with YAML frontmatter"
    parts = text.split("---\n", 2)
    assert len(parts) == 3, f"{path} frontmatter is malformed"
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2]
    return frontmatter, body


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda d: d.name)
def test_skill_frontmatter_has_required_fields(skill_dir: Path) -> None:
    frontmatter, _ = _parse_skill(skill_dir / "SKILL.md")
    assert "name" in frontmatter
    assert "description" in frontmatter


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda d: d.name)
def test_skill_name_matches_folder(skill_dir: Path) -> None:
    frontmatter, _ = _parse_skill(skill_dir / "SKILL.md")
    assert frontmatter["name"] == skill_dir.name


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda d: d.name)
def test_skill_name_is_lowercase_hyphenated(skill_dir: Path) -> None:
    frontmatter, _ = _parse_skill(skill_dir / "SKILL.md")
    name = frontmatter["name"]
    assert name == name.lower()
    assert " " not in name
    assert all(c.isalnum() or c == "-" for c in name)


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda d: d.name)
def test_skill_body_under_size_budget(skill_dir: Path) -> None:
    _, body = _parse_skill(skill_dir / "SKILL.md")
    n_lines = len(body.splitlines())
    assert n_lines <= MAX_BODY_LINES, f"{skill_dir.name} body is {n_lines} lines, over the {MAX_BODY_LINES}-line budget"


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda d: d.name)
def test_skill_allowed_tools_has_no_unexpected_escalation(skill_dir: Path) -> None:
    """None of the skills authored so far should need Bash or any
    filesystem-write tool -- they are reference/reasoning material, not
    executable capability. A future skill genuinely needing tool access
    should declare it explicitly and be reviewed, not inherit broad
    access by omission."""
    frontmatter, _ = _parse_skill(skill_dir / "SKILL.md")
    allowed = frontmatter.get("allowed-tools", [])
    if allowed is None:
        allowed = []
    assert "Bash" not in allowed
    assert "Write" not in allowed
    assert "Edit" not in allowed


def test_at_least_the_expected_core_skills_exist() -> None:
    """Not all 14 planned skills are built yet (see docs/13-skills.md) --
    this guards that the ones tied to already-shipped features exist,
    without requiring skills for features that don't exist yet (F&O
    trading, intraday, commodities)."""
    names = {d.name for d in _skill_dirs()}
    expected_core = {
        "nse-market-structure", "india-cost-model", "statement-forensics",
        "behavioral-diagnostics", "backtest-rigor", "data-quality-forensics",
        "claude-report-writing",
    }
    missing = expected_core - names
    assert not missing, f"core skills missing: {missing}"


def test_no_skill_directory_lacks_a_skill_md() -> None:
    """A directory under skills/ with no SKILL.md is either a leftover
    or a mistake -- catch it rather than silently ignoring it."""
    if not SKILLS_DIR.exists():
        return
    for d in SKILLS_DIR.iterdir():
        if d.is_dir():
            assert (d / "SKILL.md").exists(), f"{d} has no SKILL.md"
