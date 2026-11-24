from __future__ import annotations

import numpy as np
import pytest

from stocksense.strategies import exits as X


def daily(o, h, l, c, atr=1.0, sma5=None, ema20=None):  # noqa: E741
    n = len(o)
    return {
        "o": np.array(o, float),
        "h": np.array(h, float),
        "l": np.array(l, float),
        "c": np.array(c, float),
        "atr": np.full(n, atr),
        "n": n,
        "sma5": np.full(n, 0.0) if sma5 is None else np.array(sma5, float),
        "ema20": np.full(n, 0.0) if ema20 is None else np.array(ema20, float),
    }


def test_target_fills_at_the_target_or_the_gap_open_after_entry_day():
    f = daily([100, 103], [104, 111], [99, 102], [103, 110])
    assert X.follow_1d(f, 0, 95.0, "rr2_20") == (1, 110.0, "target")
    f = daily([100, 112], [104, 115], [99, 111], [103, 114])  # opens above the target: fills at the open
    assert X.follow_1d(f, 0, 95.0, "rr2_20") == (1, 112.0, "target")


def test_target_on_the_entry_day_fills_at_the_target():
    f = daily([100, 103], [111, 104], [99, 102], [110, 103])
    assert X.follow_1d(f, 0, 95.0, "rr2_20") == (0, 110.0, "target")


def test_gap_through_the_stop_fills_at_the_open():
    f = daily([100, 94], [104, 95], [99, 90], [103, 92])
    assert X.follow_1d(f, 0, 95.0, "rr2_20") == (1, 94.0, "gap_stop")


def test_stop_wins_when_one_bar_touches_stop_and_target():
    f = daily([100], [112], [94], [100])
    assert X.follow_1d(f, 0, 95.0, "rr2_20") == (0, 95.0, "stop")


def test_trail3_moves_to_break_even_then_trails_three_atr_under_the_peak():
    f = daily([100, 105, 105], [107, 106, 106], [99, 102.9, 103.5], [106, 104, 105])
    # day 0: peak 106 >= 105, so the stop becomes max(100, 106 - 3) = 103; day 1 low 102.9 hits it
    assert X.follow_1d(f, 0, 95.0, "trail3") == (1, 103.0, "stop")


def test_time_exit_at_the_close_of_the_last_day():
    n = 12
    f = daily([100] * n, [101] * n, [99] * n, [100 + i * 0.01 for i in range(n)])
    j, px, why = X.follow_1d(f, 0, 90.0, "time5")
    assert (j, why) == (4, "time") and px == pytest.approx(100.04)
    assert X.follow_1d(f, 0, 90.0, "time10")[0] == 9


def test_mr5_exits_when_the_close_is_above_the_five_day_mean():
    f = daily([100] * 4, [101] * 4, [99] * 4, [99, 99.5, 101, 102], sma5=[100, 100, 100, 100])
    assert X.follow_1d(f, 0, 90.0, "mr5") == (2, 101.0, "mr5")


def test_ema20_exit_never_on_the_entry_day():
    f = daily([100] * 4, [101] * 4, [99] * 4, [98, 99, 97, 96], ema20=[100, 100, 100, 100])
    assert X.follow_1d(f, 0, 90.0, "ema20") == (1, 99.0, "ema20")


def test_exit_lists():
    assert X.EXITS_1D == ("trail3", "ema20", "rr2_20", "rr3_40", "time5", "time10", "mr5")
    assert X.EXITS_INTRA == ("rr1", "rr15", "rr2", "rr3", "be2", "trail", "scale", "eod")


def intra(h, l, c, atr=1.0):  # noqa: E741
    n = len(h)
    return {
        "h": np.array(h, float),
        "l": np.array(l, float),
        "c": np.array(c, float),
        "o": np.array(c, float),
        "atr": np.full(n, atr),
        "n": n,
    }


def test_intraday_rr_and_eod():
    f = intra([103, 111], [99, 100], [102, 110])
    out = X.follow_intra(f, 0, 95.0, 100.0, 5.0, 1)
    assert out["rr2"] == 2 and out["rr3"] == (110 - 100) / 5 and out["rr1"] == 1 and out["eod"] == 2.0
    assert out["mfe"] == pytest.approx(2.2)


def test_intraday_stop_first_gives_minus_one_r():
    f = intra([103, 111], [94, 100], [102, 110])
    out = X.follow_intra(f, 0, 95.0, 100.0, 5.0, 1)
    assert out["rr2"] == -1 and out["eod"] == -1 and out["be2"] == -1 and out["scale"] == -1


def test_intraday_break_even_and_scale_after_one_r():
    # bar 0 reaches 1R (105); bar 1 comes back to entry (low 100 <= e) so be2 and the scale second half are 0
    f = intra([106, 104, 120], [99, 100, 101], [105, 102, 118])
    out = X.follow_intra(f, 0, 95.0, 100.0, 5.0, 2)
    assert out["be2"] == 0.0 and out["scale"] == 0.5


def test_entry_window_and_square_off():
    mins = np.array([555, 560, 870, 875, 912, 555])
    f = {"tf": 5, "mins": mins, "first": np.array([True, False, False, False, False, True]), "n": 6}
    entry_ok, exit_idx = X.entry_window(f)
    assert entry_ok.tolist() == [False, True, True, False, False, False]
    assert exit_idx.tolist() == [3, 5]  # 912 + 5 > 915, so day one last allowed bar is index 3 (875)


def test_trend_mask_kinds():
    f = {
        "c": np.array([2.0, 0.0]),
        "vwap": np.array([1.0, 1.0]),
        "ema21": np.array([2.0, 2.0]),
        "ema50": np.array([1.0, 1.0]),
        "dup": np.array([True, True]),
        "ix_up": np.array([False, True]),
    }
    assert X.trend_mask(f, "I").tolist() == [True, False]
    assert X.trend_mask(f, "D").tolist() == [True, True]
    assert X.trend_mask(f, "DI").tolist() == [True, False]
    assert X.trend_mask(f, "DIX").tolist() == [False, False]
    with pytest.raises(ValueError):
        X.trend_mask(f, "Z")
