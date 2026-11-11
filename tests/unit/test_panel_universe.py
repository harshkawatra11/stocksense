from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from stocksense.data import io
from stocksense.panel.returns import build_daily_panel
from stocksense.panel.security_master import build_security_master
from stocksense.panel.universe import build_universe
from tests.fixtures.handmade import eq_rows, weekdays, write_raw
from tests.fixtures.synth import _settings_for, make_market, write_market


def _scaled(settings, **kw):
    """Settings with universe parameters overridden (tests need small universes)."""
    return settings.model_copy(update={"universe": settings.universe.model_copy(update=kw)})


def _run(settings):
    build_security_master(settings)
    build_daily_panel(settings)
    build_universe(settings)
    return io.read(settings, "panel_universe").collect()


def _members(u: pl.DataFrame, daily: pl.DataFrame, col: str = "u500") -> set[tuple[str, date]]:
    j = u.filter(pl.col(col)).join(daily.select("sid", "date", "symbol"), on=["sid", "date"])
    return set(zip(j["symbol"].to_list(), j["date"].to_list(), strict=True))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_universe")
    m = make_market()
    write_market(m, root)
    s = _scaled(
        _settings_for(root),
        u500_size=15,
        u500_exit_rank=20,
        u200_size=6,
        u200_exit_rank=8,
        min_med_turnover_inr=1.0,
        u200_min_med_turnover_inr=1.0,
    )
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    return s, m, u, daily


def _sym_dates(synth, symbol: str, col: str = "u500") -> list[date]:
    _, _, u, daily = synth
    j = u.filter(pl.col(col)).join(daily.select("sid", "date", "symbol"), on=["sid", "date"])
    return sorted(j.filter(pl.col("symbol") == symbol)["date"].to_list())


def test_schema_and_key(synth):
    _, _, u, _ = synth
    assert u.columns == ["date", "sid", "turnover_rank", "med_turn_60", "u500", "u200", "in_fno"]
    assert u.select("date", "sid").is_duplicated().sum() == 0
    assert u["turnover_rank"].min() == 1
    # ranks are contiguous 1..n within a date
    g = u.group_by("date").agg(pl.col("turnover_rank").max().alias("mx"), pl.len().alias("n"))
    assert (g["mx"] == g["n"]).all()


def test_etf_never_member(synth):
    _, _, u, daily = synth
    etf = daily.filter(pl.col("symbol") == "NIFTYBEES")["sid"][0]
    assert u.filter(pl.col("sid") == etf).height == 0


def test_low_price_never_member(synth):
    _, _, u, daily = synth
    sid = daily.filter(pl.col("symbol") == "S21")["sid"][0]
    assert u.filter(pl.col("sid") == sid).height == 0


def test_new_listing_not_before_252_rows(synth):
    _, m, u, daily = synth
    sid = daily.filter(pl.col("symbol") == "S19")["sid"][0]
    first = u.filter(pl.col("sid") == sid)["date"].min()
    # S19 lists at session 350; 252 prior rows -> row index 252 is session 602 > 600: never eligible
    assert first is None
    # a stock listed at 0 first appears at session 252 (rows 0..251 are the prior rows)
    sid0 = daily.filter(pl.col("symbol") == "S00")["sid"][0]
    assert u.filter(pl.col("sid") == sid0)["date"].min() >= m.sessions[252]


def test_delisted_member_before_delisting(synth):
    _, m, u, daily = synth
    dates = _sym_dates(synth, "S11", "u500") or _sym_dates(synth, "S11", "u200")
    eligible = u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"]).filter(
        pl.col("symbol") == "S11"
    )
    assert eligible.height > 0
    assert eligible["date"].max() <= m.sessions[400]
    _ = dates


def test_be_series_days_not_eligible(synth):
    _, m, u, daily = synth
    s23 = daily.filter(pl.col("symbol") == "S23")
    in_u = u.join(s23.select("sid", "date", "series"), on=["sid", "date"])
    assert in_u.height > 0
    assert (in_u["series"] == "EQ").all()
    be_days = set(s23.filter(pl.col("series") == "BE")["date"].to_list())
    assert be_days and not (be_days & set(in_u["date"].to_list()))


def test_in_fno_proxy_before_fno_data(synth):
    _, _, u, _ = synth
    # synthetic dates are 2018-2020: in_fno = rank <= 150, true for every row (at most 40 stocks)
    assert u["in_fno"].all()


def test_truncation_membership_identical(tmp_path):
    full_root = tmp_path / "full"
    m = make_market()
    write_market(m, full_root)
    sf = _scaled(
        _settings_for(full_root),
        u500_size=15,
        u500_exit_rank=20,
        u200_size=6,
        u200_exit_rank=8,
        min_med_turnover_inr=1.0,
        u200_min_med_turnover_inr=1.0,
    )
    uf = _run(sf)
    df = io.read(sf, "panel_daily").collect()
    assert uf.height > 0
    for k, cut_idx in enumerate([300, 350, 400, 450, 550]):
        cut = m.sessions[cut_idx]
        import dataclasses

        mc = dataclasses.replace(
            m,
            eq=m.eq.filter(pl.col("date") <= cut),
            delivery=m.delivery.filter(pl.col("date") <= cut),
            corporate_actions=m.corporate_actions.filter(pl.col("ex_date") <= cut),
            symbol_changes=m.symbol_changes.filter(pl.col("change_date") <= cut),
        )
        root = tmp_path / f"cut{k}"
        write_market(mc, root)
        sc = _scaled(
            _settings_for(root),
            u500_size=15,
            u500_exit_rank=20,
            u200_size=6,
            u200_exit_rank=8,
            min_med_turnover_inr=1.0,
            u200_min_med_turnover_inr=1.0,
        )
        uc = _run(sc)
        dc = io.read(sc, "panel_daily").collect()
        for col in ("u500", "u200"):
            a = {x for x in _members(uf, df, col) if x[1] <= cut}
            b = _members(uc, dc, col)
            assert a == b, (cut_idx, col, len(a), len(b))
            assert len(a) > 0 or cut_idx < 300
        # ranks too, not just membership
        ra = uf.join(df.select("sid", "date", "symbol"), on=["sid", "date"]).filter(pl.col("date") <= cut)
        rb = uc.join(dc.select("sid", "date", "symbol"), on=["sid", "date"])
        assert sorted(zip(ra["symbol"], ra["date"], ra["turnover_rank"], strict=True)) == sorted(
            zip(rb["symbol"], rb["date"], rb["turnover_rank"], strict=True)
        )


# ---- hand-made: hysteresis, traded-days rule, F&O carry ----------------------------------------


def _hand_settings(tmp_data_root, **kw):
    base = {
        "u500_size": 5,
        "u500_exit_rank": 8,
        "u200_size": 2,
        "u200_exit_rank": 4,
        "min_med_turnover_inr": 1.0,
        "u200_min_med_turnover_inr": 1.0,
        "min_close": 1.0,
    }
    base.update(kw)
    return _scaled(tmp_data_root, **base)


def test_hysteresis_oscillation(tmp_data_root):
    n = 600
    d = weekdays(n)
    frames = []
    # T01..T11 fixed volumes 11000, 10000, ..., 1000 (price 100)
    for k in range(1, 12):
        frames.append(eq_rows(f"T{k:02d}", d, [100.0] * n, volume=1000 * (12 - k)))
    # X: rank 5 (vol 7500) until 399? schedule below
    vx = [7500] * 300 + [4500] * 100 + [2500] * 100 + [4500] * 100
    frames.append(eq_rows("XXX", d, [100.0] * n, volume=vx))
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat(frames))
    s = _hand_settings(tmp_data_root)
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    x = (
        u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"])
        .filter(pl.col("symbol") == "XXX")
        .sort("date")
    )
    by = dict(zip(x["date"], zip(x["turnover_rank"], x["u500"], strict=True), strict=True))
    # rank 5 at the start (above T05..: 11000, 10000, 9000, 8000 < 7500? ranks: 4 stocks above) -> enters
    assert by[d[260]] == (5, True)
    # at session ~399 the 60-row median is 4500 -> rank 8 (<= exit 8): stays
    assert by[d[399]][0] == 8 and by[d[399]][1] is True
    # volume 2500 for 100 sessions: rank 10 > 8 -> leaves
    assert by[d[499]][0] == 10 and by[d[499]][1] is False
    # back to rank 8: within the exit band but not a member and not <= 5 -> stays out
    assert by[d[599]][0] == 8 and by[d[599]][1] is False
    # the always-top stock is a u200 member throughout
    t01 = u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"]).filter(pl.col("symbol") == "T01")
    assert t01["u200"].all()


def test_u200_requires_turnover_floor(tmp_data_root):
    d = weekdays(300)
    a = eq_rows("AAA", d, [100.0] * 300, volume=1000)  # turnover 1e5
    b = eq_rows("BBB", d, [100.0] * 300, volume=1)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b]))
    s = _hand_settings(tmp_data_root, u200_min_med_turnover_inr=5.0e4, min_med_turnover_inr=1.0)
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    j = u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"])
    assert j.filter(pl.col("symbol") == "AAA")["u200"].all()
    assert not j.filter(pl.col("symbol") == "BBB")["u200"].any()
    assert j.filter(pl.col("symbol") == "BBB")["u500"].all()  # still a u500 member (rank 2 <= 5)


def test_traded_57_of_60_rule(tmp_data_root):
    d = weekdays(420)
    ok = eq_rows("OKK", d, [100.0] * 420)
    miss = [i for i in range(420) if not (300 <= i <= 303)]  # 4 missing sessions
    m = eq_rows("MMM", [d[i] for i in miss], [100.0] * len(miss))
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([ok, m]))
    s = _hand_settings(tmp_data_root, min_history_sessions=5)
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    mm = (
        u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"])
        .filter(pl.col("symbol") == "MMM")["date"]
        .to_list()
    )
    gone = [d[i] for i in range(304, 360)]
    assert not (set(mm) & set(gone))  # 4 missing in the window -> 56 < 57 until session 360
    assert d[360] in mm and d[299] in mm and d[304] not in mm


def test_stale_in_last_20_sessions_not_eligible(tmp_data_root):
    d = weekdays(330)
    ok = eq_rows("OKK", d, [100.0] * 330)
    # a 7-session gap makes the first row after it flag_stale; keep traded-60 satisfied with a low bar
    idx = [i for i in range(330) if not (200 <= i <= 206)]
    st = eq_rows("STL", [d[i] for i in idx], [100.0] * len(idx))
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([ok, st]))
    s = _hand_settings(tmp_data_root, min_history_sessions=5, min_traded_of_last_60=50)
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    j = u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"]).filter(pl.col("symbol") == "STL")
    dates = set(j["date"].to_list())
    # stale row is session 207; excluded for sessions 207..226 (20 sessions), back at 227
    assert not any(d[i] in dates for i in range(207, 227))
    assert d[227] in dates and d[206 - 7] in dates


def test_etf_name_regex_excludes(tmp_data_root):
    d = weekdays(300)
    a = eq_rows("AAA", d, [100.0] * 300)
    b = eq_rows("GOLDX", d, [100.0] * 300)  # symbol matches GOLD
    c = eq_rows("CCC", d, [100.0] * 300, name="Liquid Fund Units")  # name matches LIQUID
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, c]))
    s = _hand_settings(tmp_data_root, min_history_sessions=5)
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    syms = set(u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"])["symbol"].to_list())
    assert syms == {"AAA"}


def test_in_fno_from_futures_with_carry(tmp_data_root):
    d = weekdays(40, start=date(2024, 7, 8))
    a = eq_rows("AAA", d, [100.0] * 40)
    b = eq_rows("BBB", d, [100.0] * 40)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b]))
    have = [i for i in range(40) if i < 10 or i >= 16]  # no F&O file for sessions 10..15
    fo_rows = []
    for i in have:
        fo_rows.append(("AAA", "STF", d[i]))
        fo_rows.append(("BBB", "STO", d[i]))  # option, not a future
    fo = pl.DataFrame(
        {
            "symbol": [r[0] for r in fo_rows],
            "instrument_type": [r[1] for r in fo_rows],
            "expiry_date": [date(2024, 12, 26)] * len(fo_rows),
            "date": [r[2] for r in fo_rows],
            "close": [1.0] * len(fo_rows),
            "strike_price": [None] * len(fo_rows),
            "option_type": [None] * len(fo_rows),
        },
        schema_overrides={"strike_price": pl.Float64, "option_type": pl.Utf8},
    )
    write_raw(tmp_data_root, fo_bhavcopy=fo)
    s = _hand_settings(tmp_data_root, min_history_sessions=5, min_traded_of_last_60=1)
    u = _run(s)
    daily = io.read(s, "panel_daily").collect()
    j = u.join(daily.select("sid", "date", "symbol"), on=["sid", "date"])
    fa = dict(
        zip(
            *[j.filter(pl.col("symbol") == "AAA").sort("date")[c].to_list() for c in ("date", "in_fno")],
            strict=True,
        )
    )
    assert fa[d[9]] is True
    # missing F&O data for 6 sessions: carry forward the last known set for 5 of them
    assert all(fa[d[i]] is True for i in range(10, 15))
    assert fa[d[15]] is False
    assert fa[d[16]] is True
    assert not j.filter(pl.col("symbol") == "BBB")["in_fno"].any()
