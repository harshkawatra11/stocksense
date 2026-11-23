from __future__ import annotations

import numpy as np
import pytest

from stocksense.marketlab import npx

nan = np.nan


def eq(a, b, tol=1e-9):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, equal_nan=True
    )


def test_ema_hand_values():
    eq(npx.ema([1, 2, 3], 2), [1, 1 + (2 / 3) * 1, (1 / 3) * (1 + 2 / 3) + (2 / 3) * 3])
    eq(npx.ema([1, 2, 3], 2), [1, 1.6666666667, 2.5555555556], 1e-9)


def test_rma_hand_values():
    eq(npx.rma([1, 2, 3], 2), [1, 1.5, 2.25])


def test_ewm_nan_gap_matches_pandas_rule():
    # alpha .5, values 1, NaN, 3: old weight decays to .25, so (.25*1 + .5*3) / (.25 + .5)
    eq(npx.ewm_alpha([1, nan, 3], 0.5), [1, 1, (0.25 * 1 + 0.5 * 3) / 0.75])
    eq(npx.ewm_alpha([nan, 2, 4], 0.5), [nan, 2, 3])


def test_roll_std_and_nan_window():
    eq(npx.roll([1, 2, 3, 4], 2, "std"), [nan, 0.7071067812, 0.7071067812, 0.7071067812])
    eq(npx.roll([1, nan, 3], 2, "max"), [nan, nan, nan])
    eq(npx.roll([1, 2, 3, 4], 3, "mean"), [nan, nan, 2, 3])
    eq(npx.roll([1, 2], 5, "sum"), [nan, nan])
    with pytest.raises(ValueError):
        npx.roll([1, 2, 3], 2, "median")


def test_shift_pct_change_ffill_factorize():
    eq(npx.shift([1, 2, 3], 1), [nan, 1, 2])
    eq(npx.pct_change([100, 110, 99], 1), [nan, 0.1, -0.1])
    eq(npx.ffill([nan, 1, nan, 2]), [nan, 1, 1, 2])
    assert list(npx.factorize(np.array(["b", "b", "a", "c"]))) == [0, 0, 1, 2]


def test_group_transform():
    day = np.array([0, 0, 1, 1])
    eq(npx.group_transform([1, 2, 3, 4], day, "cumsum"), [1, 3, 3, 7])
    eq(npx.group_transform([1, 3, 2, 5], day, "cummax"), [1, 3, 2, 5])
    eq(npx.group_transform([4, 3, 9, 5], day, "cummin"), [4, 3, 9, 5])
    eq(npx.group_transform([4, 3, 9, 5], day, "first"), [4, 4, 9, 9])
    eq(npx.group_transform([4, 3, 9, 5], day, "max"), [4, 4, 9, 9])
    eq(npx.group_transform([4, 3, 9, 5], day, "min"), [3, 3, 5, 5])


def test_rvol_same_bar_hand_case():
    # 7 days x 2 bars; volume 10 everywhere except day 6 bar 0 = 20.
    day = np.repeat(np.arange(7), 2)
    k = np.tile([0, 1], 7)
    v = np.full(14, 10.0)
    v[12] = 20.0
    r = npx.rvol_same_bar(v, day, k)
    assert np.all(np.isnan(r[:10]))  # days 0-4: fewer than 5 earlier days
    eq(r[10:12], [1.0, 1.0])  # day 5 against its 5 earlier days
    eq(r[12], 2.0)  # 20 / 10
    eq(r[13], 1.5)  # cumulative 30 against a same-bar baseline of 20
