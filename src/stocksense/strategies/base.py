"""The strategy protocol every family implements.

Adopts the gs-quant `Strategy = f(Triggers -> Actions)` pattern -- a proven
institutional shape for "run thousands of configurations against a common
engine" -- without depending on gs-quant itself.

`hypothesis` is required and enforced, not decorative: a strategy with no
stated economic mechanism is exactly how the previous builds' search found
noise instead of alpha (Bailey & Lopez de Prado's expected-max-Sharpe result
in evaluation/robustness.py is what that noise looks like at scale). Every
family in this codebase must be able to answer "why would this work" before
it is allowed to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class Trigger:
    """A dated, symbol-scoped condition firing at a point in the strategy's
    decision clock -- the unit `Strategy.triggers` emits."""

    date: date
    symbol: str
    kind: str


@dataclass(frozen=True)
class Action:
    """One order-shaped intent -- the unit `Strategy.actions` emits. `side`
    is +1 (buy/long) or -1 (sell/short); `qty_weight` is a fraction of the
    position's allocated capital, not a share count -- share counts are a
    sizing-time concern (simulation/sizing.py), not a strategy-time one."""

    date: date
    symbol: str
    side: int
    qty_weight: float


class Strategy(Protocol):
    """Every family (search/space.py sweeps `params`) must implement this."""

    params: dict
    hypothesis: str

    def triggers(self, state: object) -> list[Trigger]: ...

    def actions(self, trigger: Trigger, state: object) -> list[Action]: ...


def require_hypothesis(hypothesis: str) -> str:
    """Validation used by every family's config `__post_init__`. A blank or
    missing economic story is rejected at construction time, not at review
    time -- the whole point of requiring one is that it cannot be skipped."""
    if not hypothesis or not hypothesis.strip():
        raise ValueError("Strategy.hypothesis is required: state the economic mechanism")
    return hypothesis
