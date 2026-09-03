"""Wires a strategy's daily net returns to the promotion gate.

Family-agnostic on purpose: every family in strategies/ ends up producing a
Series of date -> net daily return (e.g. strategies.overnight_reversal.
daily_pnl), and this is the one place that turns that into the
`fold_alpha_net: list[float | None]` shape evaluation.gate.evaluate_gate
consumes, over the folds evaluation.walkforward.make_folds produced.
"""

from __future__ import annotations

from stocksense.evaluation.walkforward import Fold


def fold_alpha_net(daily_net_returns, folds: list[Fold]) -> list[float | None]:
    """Mean daily net return over each fold's test_dates.

    A fold whose test_dates carry no strategy return at all (the signal
    never fired in that window) is None -- dropped by evaluate_gate, never
    treated as a zero, which would be a false claim that the book was flat.
    Dates within a fold that simply have no return (no trade that day) are
    silently excluded from the mean rather than padded with zeros, since a
    strategy with rare, event-conditioned trades is expected to be silent
    on most days by design (Q3.5 -- the 5-minute clock evaluates constantly,
    trades rarely).
    """
    result: list[float | None] = []
    for fold in folds:
        present = daily_net_returns.reindex(fold.test_dates).dropna()
        result.append(float(present.mean()) if not present.empty else None)
    return result
