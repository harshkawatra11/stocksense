from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "stocksense"
PATTERN = re.compile(r"\b(datetime\.now|datetime\.utcnow|date\.today|time\.time|time\.sleep)\s*\(")


def find_violations(root: Path, exclude: Path | None = None) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*.py")):
        if exclude is not None and p.resolve() == exclude.resolve():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                out.append(f"{p}:{i}: {line.strip()}")
    return out


def test_no_naked_now_in_src():
    bad = find_violations(SRC, exclude=SRC / "core" / "clock.py")
    assert not bad, "use a Clock instead:\n" + "\n".join(bad)


def test_checker_flags_naked_now(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("import datetime\nx = datetime.now()\n", encoding="utf-8")
    bad = find_violations(tmp_path)
    assert len(bad) == 1 and "x.py:2" in bad[0]
    f.write_text("y = time.sleep (3)\nz = date.today()\n", encoding="utf-8")
    assert len(find_violations(tmp_path)) == 2
