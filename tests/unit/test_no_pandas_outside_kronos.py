from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
ALLOWED = "stocksense/models/kronos_model.py"
PATTERN = re.compile(r"^\s*(import pandas|from pandas)")


def find_violations(root: Path, allowed: str = ALLOWED) -> list[str]:
    """`file:line: text` for pandas imports in any .py under root except the allowed relative path."""
    out: list[str] = []
    for p in sorted(root.rglob("*.py")):
        if p.relative_to(root).as_posix().endswith(allowed):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                out.append(f"{p}:{i}: {line.strip()}")
    return out


def test_no_pandas_outside_kronos_model():
    bad = find_violations(SRC)
    assert not bad, "pandas is only allowed in models/kronos_model.py:\n" + "\n".join(bad)


def test_checker_flags_pandas_but_allows_kronos_model(tmp_path):
    pkg = tmp_path / "stocksense"
    (pkg / "models").mkdir(parents=True)
    (pkg / "models" / "kronos_model.py").write_text("import pandas as pd\n", encoding="utf-8")
    assert find_violations(tmp_path) == []
    (pkg / "other.py").write_text(
        "x = 1\n    import pandas as pd\nfrom pandas import DataFrame\n", encoding="utf-8"
    )
    bad = find_violations(tmp_path)
    assert len(bad) == 2 and "other.py:2" in bad[0] and "other.py:3" in bad[1]
    (pkg / "models" / "gbm.py").write_text("import pandas\n", encoding="utf-8")
    assert len(find_violations(tmp_path)) == 3
