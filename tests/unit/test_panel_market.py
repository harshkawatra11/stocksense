from __future__ import annotations

import statistics

import polars as pl
import pytest

import stocksense.panel as panel
from stocksense.core.errors import DataNotAvailable
from stocksense.data import io
from stocksense.panel.market import build_market
from tests.fixtures.handmade import eq_rows, weekdays, write_raw
from tests.fixtures.synth import _settings_for, make_market, make_tiny_market, write_market


def _scaled(settings, **kw):
    return settings.model_copy(update={"universe": settings.universe.model_copy(update=kw)})


_SMALL = {
    "min_history_sessions": 2,
    "min_traded_of_last_60": 1,
    "min_close": 1.0,
    "min_med_turnover_inr": 1.0,
    "u200_min_med_turnover_inr": 1.0,
    "u500_size": 3,
    "u500_exit_rank": 5,
    "u200_size": 2,
    "u200_exit_rank": 3,
}


def test_requires_universe(tmp_data_root):
    with pytest.raises(DataNotAvailable):
        build_market(tmp_data_root)


def test_tiny_market_by_hand(tmp_path):
    write_market(make_tiny_market(), tmp_path)
    s = _scaled(_settings_for(tmp_path), **_SMALL)
    panel.build_all(s)
    mk = io.read(s, "panel_market").collect().sort("date")
    sess = make_tiny_market().sessions
    # eligible from row index 2, so members (previous-day u500) exist from session 3: 9 rows
    assert mk["date"].to_list() == sess[3:]
    r = mk.row(0, named=True)
    # session 3: AAA 102 -> 104, BBB 51 -> 52, CCC 205 -> 203; opens equal the previous close, so co = 0
    rets = [104 / 102 - 1, 52 / 51 - 1, 203 / 205 - 1]
    assert r["ew500_ret"] == pytest.approx(sum(rets) / 3)
    assert r["ew500_oc"] == pytest.approx(sum(rets) / 3)
    assert r["ew500_co"] == pytest.approx(0.0, abs=1e-12)
    assert r["xs_dispersion"] == pytest.approx(statistics.stdev(rets))
    assert r["n_u500"] == 3
    # u200_size = 2: members are the two highest-turnover stocks (CCC then AAA) on the previous session
    top2 = [203 / 205 - 1, 104 / 102 - 1]
    assert r["ew200_ret"] == pytest.approx(sum(top2) / 2)
    # no index data and fewer than 200 rows per stock
    assert r["nifty50_ret"] is None and r["nifty500_ret"] is None and r["breadth_200"] is None
    assert mk["n_u500"].dtype == pl.Int32


def test_previous_day_membership_only(tmp_data_root):
    d = weekdays(30)
    base = [100.0] * 30
    a = eq_rows("AAA", d, base)
    b = eq_rows("BBB", d, base)
    # C lists at session 10 and is eligible from its 3rd row (min_history 2): session 12
    c_close = [100.0] * 20
    c_close[2] = 110.0  # +10% on the day it first becomes eligible (session 12)
    c_close[3] = 110.0
    c = eq_rows("CCC", d[10:], c_close)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, c]))
    s = _scaled(tmp_data_root, **_SMALL)
    panel.build_all(s)
    mk = io.read(s, "panel_market").collect().sort("date")
    row = {r["date"]: r for r in mk.iter_rows(named=True)}
    # on session 12 CCC is a member of u500 (own-day) but not previous-day: it does not count
    assert row[d[12]]["n_u500"] == 2
    assert row[d[12]]["ew500_ret"] == pytest.approx(0.0)
    # session 13: CCC counted (member on 12). Its return on 13 is 0 (110 -> 110): mean of 0,0,0
    assert row[d[13]]["n_u500"] == 3
    assert row[d[13]]["ew500_ret"] == pytest.approx(0.0)


def test_bigmove_rows_ignored_and_means_skip_nulls(tmp_data_root):
    d = weekdays(30)
    a = eq_rows("AAA", d, [100.0] * 30)
    b = eq_rows("BBB", d, [100.0] * 30)
    up = [100.0] * 20 + [150.0] * 10  # +50% on session 20: |ret| > 0.40 -> flag_bigmove
    c = eq_rows("CCC", d, up)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, c]))
    s = _scaled(tmp_data_root, **_SMALL)
    panel.build_all(s)
    daily = io.read(s, "panel_daily").collect()
    assert daily.filter(pl.col("flag_bigmove")).height == 1
    mk = io.read(s, "panel_market").collect()
    r = mk.filter(pl.col("date") == d[20]).row(0, named=True)
    assert r["n_u500"] == 3  # still a member, just excluded from the means
    assert r["ew500_ret"] == pytest.approx(0.0)  # mean of AAA and BBB only
    assert r["xs_dispersion"] == pytest.approx(0.0)


def test_index_returns(tmp_data_root):
    d = weekdays(30)
    a = eq_rows("AAA", d, [100.0] * 30)
    b = eq_rows("BBB", d, [100.0] * 30)
    cl = [1000.0 + 10 * i for i in range(10, 30)]
    idx = pl.DataFrame(
        {
            "index_name": ["NIFTY 50"] * 20,
            "date": d[10:],
            "open": cl,
            "high": cl,
            "low": cl,
            "close": cl,
        }
    )
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b]), index_daily=idx)
    s = _scaled(tmp_data_root, **_SMALL)
    panel.build_all(s)
    mk = io.read(s, "panel_market").collect()
    row = {r["date"]: r for r in mk.iter_rows(named=True)}
    # index data starts at session 10: null on 10 (no previous), 1110/1100 - 1 on 11, null before
    assert row[d[5]]["nifty50_ret"] is None
    assert row[d[10]]["nifty50_ret"] is None
    assert row[d[11]]["nifty50_ret"] == pytest.approx(1110 / 1100 - 1)
    assert row[d[29]]["nifty500_ret"] is None


def test_synth_market_breadth_and_shape(tmp_path):
    m = make_market()
    write_market(m, tmp_path)
    s = _scaled(_settings_for(tmp_path), min_med_turnover_inr=1.0, u200_min_med_turnover_inr=1.0)
    panel.build_all(s)
    mk = io.read(s, "panel_market").collect().sort("date")
    assert mk.columns == [
        "date",
        "ew500_ret",
        "ew500_co",
        "ew500_oc",
        "ew200_ret",
        "nifty50_ret",
        "nifty500_ret",
        "n_u500",
        "breadth_200",
        "xs_dispersion",
    ]
    assert mk["date"].is_unique().all()
    # eligible from session 252, members from 253: sessions 253..600
    assert mk["date"].min() == m.sessions[253]
    assert mk["breadth_200"].null_count() == 0
    assert mk["breadth_200"].min() >= 0.0 and mk["breadth_200"].max() <= 1.0
    assert 0.2 < mk["breadth_200"].mean() < 0.8
    assert mk["n_u500"].max() <= 40
    # the equal-weight market return averages the planted market factor (daily mean 0.0004)
    assert abs(mk["ew500_ret"].mean()) < 0.003
    # overnight + intraday compound to roughly the close-to-close return on average
    assert abs((mk["ew500_co"] + mk["ew500_oc"] - mk["ew500_ret"]).mean()) < 0.001


def test_build_all_writes_everything(tmp_path):
    write_market(make_market(n=10, n_sessions=300), tmp_path)
    s = _settings_for(tmp_path)
    panel.build_all(s)
    for name in ("security_master", "panel_daily", "adj_events", "panel_market"):
        assert io.list_partitions(s, name), name
