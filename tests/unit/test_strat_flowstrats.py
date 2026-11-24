"""M5.5 flow strategies: each rule fires at the expected bar on a constructed series, negative cases stay
silent, and every signal / position is unchanged when the later rows are removed."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from stocksense.strategies import base
from stocksense.strategies import flowstrats as FS
from tests.unit.strat_helpers import assert_past_only, day_rows, feats, fired, make_bars
from tests.unit.test_strat_tiers import mirror_rows


def dfeat(n: int, **arrays: Any) -> dict[str, Any]:
    """A minimal daily dict of length n with flat price, ATR 2, and the given arrays."""
    f: dict[str, Any] = {"n": n, "c": np.full(n, 100.0), "l": np.full(n, 99.0), "atr": np.full(n, 2.0)}
    for k, v in arrays.items():
        f[k] = np.asarray(v, dtype=object if k == "oi_label" else float)
    return f


# ------------------------------------------------------------------------------------ registry
def test_registry_holds_every_flow_strategy():
    base.load_library()
    want = {
        "flow_oi_long_buildup": ("m-swing", "1d", "ready"),
        "flow_short_covering_rev": ("m-swing", "1d", "ready"),
        "flow_deliv_spike": ("m-swing", "1d", "ready"),
        "flow_fii_index": ("m-index", "1d", "ready"),
        "flow_gex_regime": ("m-index", "5m", "await_data"),
        "flow_preopen_imb": ("m-intra", "5m", "await_data"),
        "flow_cvd_trend": ("m-intra", "5m", "ready"),
    }
    for key, (fam, tf, status) in want.items():
        s = base.REGISTRY[key]
        assert (s.family, s.timeframe, s.status, s.source, s.kind) == (fam, tf, status, "flow", s.kind), key
        assert s.default_exit in s.exits and s.description, key
    assert base.REGISTRY["flow_oi_long_buildup"].exits == ("rr2_20", "time5", "time10", "trail3")
    assert base.REGISTRY["flow_short_covering_rev"].exits == ("time5", "rr2_20")
    assert base.REGISTRY["flow_deliv_spike"].exits == ("time10", "rr2_20", "trail3")
    assert base.REGISTRY["flow_fii_index"].timing is not None


# ------------------------------------------------------------------------------------ attach
def test_attach_daily_flow_aligns_by_date_and_defaults_missing_days():
    dates = np.array(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], dtype="datetime64[D]")
    oi = pl.DataFrame(
        {
            "date": [date(2026, 1, 6), date(2026, 1, 8)],
            "quad_label": ["long buildup", None],
            "oi_chg_pct": [6.0, None],
            "price_chg_pct": [2.0, 1.0],
        }
    )
    dv = pl.DataFrame({"date": [date(2026, 1, 5), date(2026, 1, 7)], "deliv_spike": [True, False]})
    f = FS.attach_daily_flow({"n": 4, "date": dates}, oi, dv)
    assert f["oi_label"].tolist() == ["", "long buildup", "", ""]
    assert np.isnan(f["oi_chg_pct"][[0, 2, 3]]).all() and f["oi_chg_pct"][1] == 6.0
    assert f["price_chg_pct"][3] == 1.0
    assert f["deliv_spike"].tolist() == [True, False, False, False]
    g = FS.attach_daily_flow({"n": 4, "date": dates}, None, None)  # nothing stored: every rule stays silent
    assert not FS.flow_oi_long_buildup({**dfeat(4), **g})[0].any()


def test_attach_preopen_aligns_by_date():
    f = feats([day_rows([100.0] * 3), day_rows([100.0] * 3)])
    pre = pl.DataFrame({"date": [date(2026, 1, 6)], "imb": [0.5], "gap": [0.4]})
    FS.attach_preopen(f, pre)
    assert (
        np.isnan(f["pre_imb"][:3]).all()
        and (f["pre_imb"][3:] == 0.5).all()
        and (f["pre_gap"][3:] == 0.4).all()
    )


# ------------------------------------------------------------------------------------ OI long buildup
def test_oi_long_buildup_fires_on_label_with_oi_over_5_and_price_over_1():
    label = ["long buildup", "long buildup", "long buildup", "short buildup", "long buildup"]
    f = dfeat(5, oi_chg_pct=[6, 4.9, 7, 8, 9], price_chg_pct=[2, 2, 1.5, 3, 0.9])
    f["oi_label"] = np.array(label, dtype=object)
    sig, stop = FS.flow_oi_long_buildup(f)
    assert fired(sig) == [0, 2]
    assert stop[0] == 96.0
    assert_past_only(FS.flow_oi_long_buildup, f, [1, 2, 3, 5])


def test_oi_long_buildup_silent_without_data_or_on_the_mirror_labels():
    f = dfeat(3)
    assert not FS.flow_oi_long_buildup(f)[0].any()  # nothing attached
    f.update(
        oi_label=np.array(["short buildup"] * 3, dtype=object),
        oi_chg_pct=np.full(3, 9.0),
        price_chg_pct=np.full(3, 3.0),
    )
    assert not FS.flow_oi_long_buildup(f)[0].any()


# ------------------------------------------------------------------------------------ short covering
def _cover_f(label="short covering"):
    n = 14
    c = np.array([100.0] * 10 + [90.0, 95.0, 96.0, 97.0])
    f = dfeat(n)
    f.update(c=c, l=c - 1, oi_label=np.array([label] * n, dtype=object))
    return f


def test_short_covering_reversal_needs_a_fall_of_more_than_8_percent_over_ten_days():
    f = _cover_f()
    sig, stop = FS.flow_short_covering_rev(f)
    assert fired(sig) == [10]
    assert stop[10] == pytest.approx(89.0 - 0.3 * 2.0)  # the 10-day low less 0.3 ATR
    assert_past_only(FS.flow_short_covering_rev, f, [9, 10, 11, 14])


def test_short_covering_silent_for_other_labels():
    assert not FS.flow_short_covering_rev(_cover_f("long unwinding"))[0].any()
    assert not FS.flow_short_covering_rev(_cover_f("short buildup"))[0].any()


# ------------------------------------------------------------------------------------ delivery spike
def test_delivery_spike_needs_the_close_above_its_fifty_day_mean():
    f = dfeat(4, sma50=[100, 100, 100, 100])
    f["c"] = np.array([101.0, 99.0, 101.0, 101.0])
    f["deliv_spike"] = np.array([True, True, False, True])
    sig, stop = FS.flow_deliv_spike(f)
    assert fired(sig) == [0, 3]
    assert stop[0] == pytest.approx(101.0 - 5.0)
    assert_past_only(FS.flow_deliv_spike, f, [1, 2, 4])
    del f["deliv_spike"]
    assert not FS.flow_deliv_spike(f)[0].any()


# ------------------------------------------------------------------------------------ FII index
def _fii(chg5):
    return pl.DataFrame(
        {"date": [date(2026, 1, 5) + timedelta(days=i) for i in range(len(chg5))], "fii_ratio_chg5": chg5}
    )


def test_fii_index_holds_five_sessions_after_a_rise_over_0_05():
    chg = [0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.0, 0.0, 0.0]
    pos = FS.fii_index_position(_fii(chg))["position"].to_list()
    assert pos == [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  # decided at t, held through the next five sessions
    assert FS.fii_index_position(_fii(chg[:9]))["position"].to_list() == pos[:9]  # past-only
    assert FS.fii_index_position(_fii([0.05, None, -0.2]))["position"].to_list() == [
        0,
        0,
        0,
    ]  # 0.05 is not over 0.05


# ------------------------------------------------------------------------------------ GEX regime
def _gex_day(first12_step, rest=101.2):
    closes = [100.0 + first12_step * (i + 1) for i in range(12)] + [rest] * 63
    return make_bars([day_rows(closes, 1000.0, 0.1, o0=100.0)])


def test_negative_gex_follows_the_first_hour_from_1015_to_1515():
    up = FS.gex_day_trade(_gex_day(0.1), -5.0)
    assert up == {"regime": "negative", "side": 1, "entry_i": 12, "exit_i": 71}
    dn = FS.gex_day_trade(_gex_day(-0.1, rest=98.8), -5.0)
    assert dn["side"] == -1 and dn["entry_i"] == 12
    assert FS.gex_day_trade(_gex_day(0.0, rest=100.0), -5.0) is None  # no direction


def _flat_with_spike(spike):
    closes = [100.0] * 10 + [spike] + [100.0] * 64
    return make_bars([day_rows(closes, 1000.0, 0.1, o0=100.0)])


def test_positive_gex_fades_the_first_close_beyond_one_sigma_back_to_vwap():
    short = FS.gex_day_trade(_flat_with_spike(100.5), 3.0)
    assert (short["regime"], short["side"], short["entry_i"], short["target"]) == ("positive", -1, 11, "vwap")
    long_ = FS.gex_day_trade(_flat_with_spike(99.5), 3.0)
    assert (long_["side"], long_["entry_i"]) == (1, 11)
    assert FS.gex_day_trade(_flat_with_spike(100.0), 3.0) is None  # nothing stretched


def test_gex_trade_is_past_only_and_needs_a_known_regime():
    day = _flat_with_spike(100.5)
    full = FS.gex_day_trade(day, 3.0)
    part = FS.gex_day_trade(day.take(slice(0, 12)), 3.0)
    assert (part["side"], part["entry_i"]) == (full["side"], full["entry_i"])
    up = _gex_day(0.1)
    part = FS.gex_day_trade(up.take(slice(0, 13)), -5.0)
    assert (part["side"], part["entry_i"]) == (1, 12)
    assert FS.gex_day_trade(up, float("nan")) is None
    assert FS.gex_day_trade(up.take(slice(0, 8)), -5.0) is None


def test_gex_timing_is_empty_without_index_bars(monkeypatch):
    monkeypatch.setattr("stocksense.strategies.data.intraday_frame", lambda *a, **k: None)
    out = FS.timing_gex(None, date(2026, 1, 5), date(2026, 1, 6))
    assert out.height == 0 and out.columns == ["date", "position"]


# ------------------------------------------------------------------------------------ pre-open
def _preopen_f(imb, gap, flip=False):
    days = [day_rows([100.0, 100.1, 100.2], 1000.0, 0.1) for _ in range(3)]
    f = feats([mirror_rows(d) for d in days], sign=-1) if flip else feats(days)
    pre = pl.DataFrame(
        {"date": [date(2026, 1, 5) + timedelta(days=i) for i in range(3)], "imb": imb, "gap": gap}
    )
    return FS.attach_preopen(f, pre)


def test_preopen_imbalance_follows_when_gap_agrees():
    f = _preopen_f([0.5, 0.5, 0.2], [0.4, -0.1, 0.4])  # day 1: gap disagrees; day 2: imbalance too small
    sig, stop = FS.flow_preopen_imb(f)
    assert fired(sig) == [0]
    assert stop[0] == pytest.approx(f["c"][0] - 2 * f["atr"][0])
    assert_past_only(FS.flow_preopen_imb, f, [1, 3, 6])


def test_preopen_imbalance_short_runs_on_the_mirrored_chart():
    f = _preopen_f([-0.5, 0.5, -0.5], [-0.4, 0.4, 0.4], flip=True)  # real selling pressure with a gap down
    assert fired(FS.flow_preopen_imb(f)[0]) == [0]
    assert not FS.flow_preopen_imb(_preopen_f([-0.5] * 3, [-0.4] * 3))[0].any()  # a long never trades selling


# ------------------------------------------------------------------------------------ CVD trend
def _cvd_days(rise=0.3):
    day1 = day_rows([100.1 if i % 2 == 0 else 100.0 for i in range(75)], 1000.0, 0.02)
    closes, c = [], 100.0
    for i in range(75):
        c = c + rise if i < 12 else (100.1 + 0.3 * 12 * (rise != 0)) - (0.1 if i % 2 else 0.0)
        closes.append(c)
    return [day1, day_rows(closes, 1000.0, 0.02, o0=100.0)]


def test_cvd_slope_is_positive_after_buying_and_near_zero_when_balanced():
    f = feats(_cvd_days())
    s = FS.cvd_slope_open(f)
    assert np.isnan(s[75 + 10]) and np.isfinite(s[75 + 11])  # known from the 10:10 bar on
    assert s[75 + 11] == pytest.approx(s[75 + 40]) and s[75 + 11] > 0.8
    assert abs(FS.cvd_slope_open(feats(_cvd_days(rise=0.0)))[75 + 11]) < 0.2


def test_cross_rank_orders_the_universe():
    assert FS.cross_rank(np.array([3.0, 1.0, 2.0, np.nan])).tolist()[:3] == [1.0, 0.0, 0.5]
    assert np.isnan(FS.cross_rank(np.array([3.0, 1.0, 2.0, np.nan]))[3])
    r = FS.cross_rank(np.arange(11.0))
    assert (r >= 0.9).sum() == 2  # the top decile of 11 names is 2 names


def test_cvd_trend_fires_at_1015_for_the_top_decile_only():
    f = feats(_cvd_days())
    rank = np.full(f["n"], np.nan)
    rank[75 + 11] = 0.95
    rank[75 + 10] = 0.99  # wrong bar
    f["cvd_rank"] = rank
    sig, stop = FS.flow_cvd_trend(f)
    assert fired(sig) == [75 + 11]
    assert stop[86] == pytest.approx(f["lod"][86])
    assert_past_only(FS.flow_cvd_trend, f, [80, 86, 87, 100])
    rank[75 + 11] = 0.85
    assert not FS.flow_cvd_trend({**f, "cvd_rank": rank})[0].any()
    assert not FS.flow_cvd_trend(feats(_cvd_days()))[0].any()  # no rank attached


def test_cvd_trend_short_is_the_same_rule_on_the_mirrored_chart():
    days = [mirror_rows(d) for d in _cvd_days()]  # real: heavy first-hour selling
    f = feats(days, sign=-1)
    rank = np.full(f["n"], np.nan)
    rank[86] = 0.95  # in the trade's frame this stock is in the top decile (the real bottom decile)
    f["cvd_rank"] = rank
    assert fired(FS.flow_cvd_trend(f)[0]) == [86]
