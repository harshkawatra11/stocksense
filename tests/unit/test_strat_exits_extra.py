"""M5.3 extra exits: hand-built fills of vwap_touch, va_edge, ib_ext, poc_touch, ll10_exit and the rest."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.strategies import exits as X
from tests.unit.strat_helpers import day_rows, feats
from tests.unit.test_strat_tiers import DAY1_ROWS, IB_CLOSES, mirror_rows


def session(o, h, low, c, vwap=100.0):
    """A one-session intraday dict of 5-minute bars from 09:15 with a constant VWAP."""
    n = len(o)
    return {
        "o": np.array(o, float),
        "h": np.array(h, float),
        "l": np.array(low, float),
        "c": np.array(c, float),
        "vwap": np.full(n, vwap),
        "first": np.r_[True, np.zeros(n - 1, bool)],
        "mins": 555 + 5 * np.arange(n),
        "day": np.zeros(n, int),
        "tf": 5,
        "n": n,
    }


def _vwap_day(**over):
    base = {
        "o": [99, 98, 97, 97.5, 98, 99, 99.5, 99.8],
        "h": [99.5, 98.5, 98, 99.5, 99.9, 100.3, 99.9, 99.9],
        "low": [98.5, 97.5, 96.5, 97, 97.5, 98.5, 99, 99.2],
        "c": [98, 97.5, 97.5, 98, 99, 99.5, 99.8, 99.7],
    }
    base.update(over)
    return session(**base)


def test_vwap_touch_fills_at_the_previous_bar_vwap_when_the_high_reaches_it():
    assert X.vwap_touch(_vwap_day(), 2, 95.0, "long") == (5, 100.0, "target")


def test_vwap_touch_gap_through_the_target_fills_at_the_open():
    o = [99, 98, 97, 97.5, 98, 100.6, 99.5, 99.8]
    assert X.vwap_touch(_vwap_day(o=o), 2, 95.0, "long") == (5, 100.6, "target")


def test_vwap_touch_stop_wins_over_target_on_the_same_bar_and_gap_stop_fills_at_the_open():
    low = [98.5, 97.5, 96.5, 97, 94.9, 98.5, 99, 99.2]
    h = [99.5, 98.5, 98, 99.5, 100.2, 100.3, 99.9, 99.9]
    assert X.vwap_touch(_vwap_day(low=low, h=h), 2, 95.0, "long") == (4, 95.0, "stop")
    o = [99, 98, 97, 97.5, 94.0, 99, 99.5, 99.8]
    assert X.vwap_touch(_vwap_day(o=o, low=low), 2, 95.0, "long") == (4, 94.0, "gap_stop")


def test_vwap_touch_ends_at_the_last_bar_when_nothing_is_touched():
    h = [99.5, 98.5, 98, 99.5, 99.9, 99.9, 99.9, 99.9]
    assert X.vwap_touch(_vwap_day(h=h), 2, 95.0, "long") == (7, 99.7, "eod")


def test_vwap_touch_never_uses_the_vwap_of_its_own_bar():
    f = _vwap_day()
    f["vwap"] = np.array([100.0, 100, 100, 100, 100, 150, 100, 100])  # bar 5's own VWAP is far above its high
    assert X.vwap_touch(f, 2, 95.0, "long")[0] == 5  # the previous bar's 100 is what counts
    f["vwap"] = np.array([150.0, 150, 150, 150, 150, 100, 100, 100])
    assert X.vwap_touch(f, 2, 95.0, "long")[0] == 7  # a high VWAP in the past is the target: not reached


def test_a_bad_side_is_rejected():
    with pytest.raises(ValueError):
        X.vwap_touch(_vwap_day(), 2, 95.0, "up")


# ---------------------------------------------------------------------------- levels-based exits
def _day2(closes):
    return [DAY1_ROWS, day_rows(closes, 1000.0, 0.1, o0=closes[0])]


def test_poc_touch_exits_at_the_previous_sessions_poc():
    f = feats(_day2([100.0] * 5 + [101.0, 102.0, 103.0, 103.6, 103.0]))
    lv = X.levels(f)
    assert lv["ppoc"][80] == pytest.approx(102.575) and lv["pvah"][80] == pytest.approx(103.4)
    assert X.poc_touch(f, 80, 99.0, "long") == (82, pytest.approx(102.575), "target")


def test_va_edge_exits_at_the_previous_value_area_high():
    f = feats(_day2([100.0] * 5 + [101.0, 102.0, 103.0, 103.6, 103.0]))
    assert X.va_edge(f, 80, 99.0, "long") == (83, pytest.approx(103.4), "target")


def test_ib_extension_exit_is_ibh_plus_one_ib_range():
    closes = [*IB_CLOSES, 103.0, 103.5, 103.2]
    f = feats([day_rows(closes)])
    lv = X.levels(f)
    assert lv["ibh"][14] == pytest.approx(101.9) and lv["ibl"][14] == pytest.approx(100.4)
    assert np.isnan(lv["ibh"][11])  # the first hour is not complete before 10:15
    assert X.ib_ext(f, 14, 101.15, "long") == (16, pytest.approx(103.4), "target")


def test_levels_are_the_same_numbers_in_the_mirror_frame():
    closes = [*IB_CLOSES, 103.0, 103.5, 103.2]
    f = feats([day_rows(closes)])
    flipped = [mirror_rows(day_rows(closes))]
    fm = feats(flipped, sign=-1)  # the real chart is upside down; the trade frame is the original minus 200
    assert X.ib_ext(fm, 14, 101.15 - 200, "short") == (16, pytest.approx(103.4 - 200), "target")
    assert X.levels(fm)["ibh"][14] == pytest.approx(X.levels(f)["ibh"][14] - 200)
    d1 = feats(_day2([100.0] * 10))
    d1m = feats([mirror_rows(d) for d in _day2([100.0] * 10)], sign=-1)
    for key in ("pvah", "pval", "ppoc"):
        assert X.levels(d1m)[key][80] == pytest.approx(X.levels(d1)[key][80] - 200, abs=0.2)


def test_level_cache_is_dropped_when_the_dict_is_cut():
    f = feats([day_rows([*IB_CLOSES, 103.0])])
    X.levels(f)
    g = {k: (v[:13] if isinstance(v, np.ndarray) and v.shape[:1] == (f["n"],) else v) for k, v in f.items()}
    g["n"] = 13
    assert len(X.levels(g)["ibh"]) == 13


# ---------------------------------------------------------------------------- daily exits
def daily(o, low, c, ll10=90.0):
    n = len(o)
    return {
        "o": np.array(o, float),
        "l": np.array(low, float),
        "c": np.array(c, float),
        "ll10": np.full(n, ll10),
        "n": n,
    }


def test_ll10_exit_leaves_at_the_close_under_the_previous_ten_day_low():
    f = daily(
        [100, 101, 102, 101, 96, 92, 88, 87],
        [99, 100, 101, 100, 94, 90, 86, 85],
        [101, 102, 101, 95, 92, 89, 87, 86],
    )
    assert X.ll10_exit(f, 1, 85.0, "long") == (5, 89.0, "ll10")


def test_ll10_exit_stop_gap_and_time():
    f = daily([100, 101, 102, 101, 96], [99, 100, 101, 100, 94], [101, 102, 101, 100, 99])
    assert X.ll10_exit(f, 0, 94.5, "long") == (4, 94.5, "stop")
    f2 = daily([100, 101, 102, 80, 96], [99, 100, 101, 79, 94], [101, 102, 101, 100, 99])
    assert X.ll10_exit(f2, 0, 85.0, "long") == (3, 80.0, "gap_stop")
    f3 = daily([100, 101, 102], [99, 100, 101], [101, 102, 101])
    assert X.ll10_exit(f3, 0, 50.0, "long") == (2, 101.0, "time")


def test_ll10_exit_does_not_use_todays_low_in_the_ten_day_low():
    f = daily(
        [100, 100, 100], [99, 99, 99], [100, 99, 99], ll10=99.0
    )  # a close at its own low is not "under" it
    assert X.ll10_exit(f, 0, 50.0, "long")[2] == "time"


def test_next_open_buys_the_close_and_sells_the_next_open():
    f = daily([100, 101, 103], [99, 100, 102], [101, 102, 104])
    assert X.next_open(f, 1, 0.0, "long") == (2, 103.0, "next_open")
    assert X.next_open(f, 2, 0.0, "long") == (2, 104.0, "no_next")


def test_state_exit_leaves_at_the_open_after_the_condition_close():
    f = daily([100, 101, 102, 103, 104], [99, 100, 101, 102, 103], [101, 102, 103, 104, 105])
    f["exit_cond"] = np.array([False, False, True, False, False])
    assert X.state_exit(f, 0, float("nan"), "long") == (3, 103.0, "state")
    f["exit_cond"] = np.array([False, False, False, False, True])
    assert X.state_exit(f, 0, float("nan"), "long") == (4, 105.0, "time")  # no next bar: the close
    f["exit_cond"] = np.zeros(5, bool)
    assert X.state_exit(f, 0, float("nan"), "long") == (4, 105.0, "time")


def test_state_exit_honours_an_explicit_stop_only():
    f = daily([100, 101, 102, 103, 104], [99, 98, 101, 102, 103], [101, 102, 103, 104, 105])
    f["exit_cond"] = np.array([False, False, True, False, False])
    assert X.state_exit(f, 0, 98.5, "long") == (1, 98.5, "stop")
    assert X.state_exit(f, 0, float("nan"), "long")[2] == "state"


def test_extra_exit_registry():
    assert set(X.EXITS_EXTRA) == {
        "vwap_touch",
        "va_edge",
        "ib_ext",
        "poc_touch",
        "ll10_exit",
        "next_open",
        "state_exit",
    }
