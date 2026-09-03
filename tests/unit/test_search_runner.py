"""Aggregate a strategy's daily net returns into the per-fold alpha list that
evaluation.gate.evaluate_gate consumes -- the piece that wires a strategy's
output to the existing, PROTECTED statistical machinery.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stocksense.evaluation.walkforward import Fold
from stocksense.search.runner import fold_alpha_net


def _fold(fold_id: int, test_dates: list[date]) -> Fold:
    return Fold(fold_id=fold_id, test_group_ids=(fold_id,), train_dates=[], test_dates=test_dates)


def test_fold_alpha_net_is_the_mean_daily_return_over_test_dates():
    daily = pd.Series(
        {date(2026, 1, 5): 0.01, date(2026, 1, 6): 0.03, date(2026, 1, 7): -0.02},
    )
    folds = [_fold(0, [date(2026, 1, 5), date(2026, 1, 6)])]

    result = fold_alpha_net(daily, folds)

    assert result == [pytest.approx(0.02)]


def test_fold_alpha_net_is_none_when_no_test_date_has_a_return():
    daily = pd.Series({date(2026, 1, 5): 0.01})
    folds = [_fold(0, [date(2026, 1, 6), date(2026, 1, 7)])]

    assert fold_alpha_net(daily, folds) == [None]


def test_fold_alpha_net_ignores_missing_dates_within_a_fold():
    # the strategy fired on only one of the fold's two test dates -- that
    # date still contributes, the missing one is simply absent, not zero.
    daily = pd.Series({date(2026, 1, 5): 0.04})
    folds = [_fold(0, [date(2026, 1, 5), date(2026, 1, 6)])]

    assert fold_alpha_net(daily, folds) == [pytest.approx(0.04)]


def test_fold_alpha_net_returns_one_entry_per_fold_in_order():
    daily = pd.Series({date(2026, 1, 5): 0.01, date(2026, 1, 10): 0.05})
    folds = [_fold(0, [date(2026, 1, 5)]), _fold(1, [date(2026, 1, 10)])]

    assert fold_alpha_net(daily, folds) == [pytest.approx(0.01), pytest.approx(0.05)]
