from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PATTERN = re.compile(
    r"placeorder|modifyorder|cancelorder|place_order|modify_order|cancel_order"
    r"|/order/place|/order/modify|/order/cancel|/v1/placeOrder|gttCreateRule",
    re.IGNORECASE,
)


def find_violations(root: Path) -> list[str]:
    """`file:line: text` for every source line mentioning an order-placement API."""
    out: list[str] = []
    for p in sorted(root.rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                out.append(f"{p}:{i}: {line.strip()}")
    return out


def test_no_order_placement_code_in_src():
    bad = find_violations(SRC)
    assert not bad, "order placement is out of scope (paper only):\n" + "\n".join(bad)


def test_checker_flags_order_calls(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("api.placeOrder(x)\nurl = '/order/Place'\nok = 1\n", encoding="utf-8")
    bad = find_violations(tmp_path)
    assert len(bad) == 2 and "x.py:1" in bad[0] and "x.py:2" in bad[1]
    f.write_text("a = cancel_order(1)\nb = gttCreateRule()\nc = ModifyOrder()\n", encoding="utf-8")
    assert len(find_violations(tmp_path)) == 3
