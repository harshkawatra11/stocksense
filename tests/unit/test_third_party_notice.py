from __future__ import annotations

from pathlib import Path


def test_market_lab_notice() -> None:
    root = Path(__file__).resolve().parents[2] / "third_party" / "market-lab"
    assert "MIT License" in (root / "LICENSE").read_text(encoding="utf-8")
    assert "ba3f771" in (root / "NOTICE.md").read_text(encoding="utf-8")
