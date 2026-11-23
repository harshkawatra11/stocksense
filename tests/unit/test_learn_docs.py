from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.k151_sections import K151_SECTIONS, STATUS

LEARN = Path(__file__).resolve().parents[2] / "docs" / "learn"

NOTES = [
    "00-data-reality",
    "01-order-flow",
    "02-delta-and-cvd",
    "03-footprint",
    "04-big-trades-and-absorption",
    "05-heatmap-and-liquidity",
    "06-vwap",
    "07-volume-profile",
    "08-market-profile-and-amt",
    "09-time-scales",
    "10-options-positioning-and-gex",
    "11-oi-buildup",
    "12-delivery-fii-dii-participants",
    "13-preopen-0900",
    "14-tier-list-assessment",
    "15-151-strategies-map",
    "16-smart-money-concepts",
    "17-macro-trading-india",
]
HEADINGS = [
    "What it is",
    "How it is computed",
    "What it says about flow and positioning",
    "What NSE data allows and how StockSense computes it",
    "Evidence",
    "How traders misuse it",
    "Where it appears in the dashboard",
    "Sources",
]


@pytest.mark.parametrize("n", NOTES)
def test_note_shape(n: str) -> None:
    t = (LEARN / f"{n}.md").read_text(encoding="utf-8")
    assert re.findall(r"^## (.+)$", t, re.M) == HEADINGS
    assert "http" in t.split("## Sources", 1)[1]


def test_151_table_complete() -> None:
    t = (LEARN / "15-151-strategies-map.md").read_text(encoding="utf-8")
    rows = [
        ln
        for ln in t.splitlines()
        if ln.startswith("| ") and not ln.startswith("| §") and not ln.startswith("|---")
    ]
    secs = [r.split("|")[1].strip() for r in rows]
    assert sorted(secs) == sorted(K151_SECTIONS) and len(secs) == len(set(secs))
    for r in rows:
        cells = r.split("|")
        status = re.match(r"\w+", cells[5].strip()).group(0)
        assert status in STATUS, r
        if status in ("implemented", "adapted"):
            assert cells[6].strip().startswith("k151_"), r


def test_151_glossary_notes() -> None:
    t = (LEARN / "18-151-glossary-notes.md").read_text(encoding="utf-8")
    rows = [
        ln
        for ln in t.splitlines()
        if ln.startswith("| ") and not ln.startswith("| Term") and not ln.startswith("|---")
    ]
    assert len(rows) >= 40


def test_readme_lists_every_note() -> None:
    t = (LEARN / "README.md").read_text(encoding="utf-8")
    for n in [*NOTES, "18-151-glossary-notes"]:
        assert n in t
