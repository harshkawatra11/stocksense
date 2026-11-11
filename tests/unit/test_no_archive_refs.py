from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
NEEDLE = "_archive_v1"


def find_violations(root: Path) -> list[str]:
    """`file:line: text` for every .py source line under root that contains the archive folder name."""
    out: list[str] = []
    for p in sorted(root.rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if NEEDLE in line:
                out.append(f"{p}:{i}: {line.strip()}")
    return out


def test_no_archive_refs_in_src():
    bad = find_violations(SRC)
    assert not bad, "old-system state must never be referenced:\n" + "\n".join(bad)


def test_checker_flags_archive_reference(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("ok = 1\np = root / '_archive_v1' / 'x'\n", encoding="utf-8")
    bad = find_violations(tmp_path)
    assert len(bad) == 1 and "x.py:2" in bad[0]
