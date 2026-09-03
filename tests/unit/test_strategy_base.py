"""The Strategy protocol's one enforced rule: no hypothesis, no strategy."""

from __future__ import annotations

import pytest

from stocksense.strategies.base import Action, Trigger, require_hypothesis


def test_require_hypothesis_accepts_a_real_string():
    assert require_hypothesis("overnight order imbalance reverts") == "overnight order imbalance reverts"


def test_require_hypothesis_rejects_empty_string():
    with pytest.raises(ValueError):
        require_hypothesis("")


def test_require_hypothesis_rejects_whitespace_only():
    with pytest.raises(ValueError):
        require_hypothesis("   ")


def test_trigger_is_dated_and_symbol_scoped():
    from datetime import date

    t = Trigger(date=date(2026, 1, 5), symbol="RELIANCE", kind="overnight_gap")
    assert t.symbol == "RELIANCE"
    assert t.kind == "overnight_gap"


def test_action_side_and_qty_weight():
    from datetime import date

    a = Action(date=date(2026, 1, 5), symbol="RELIANCE", side=1, qty_weight=0.2)
    assert a.side == 1
    assert a.qty_weight == 0.2
