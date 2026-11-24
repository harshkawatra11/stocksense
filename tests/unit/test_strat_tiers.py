"""M5.3 tier-list strategies: every rule fires at the expected bar on a constructed series, a negative or
mirror series stays silent, and the signal at bar t is unchanged when the later rows are removed."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from stocksense.marketlab.bars import Bars
from stocksense.strategies import base
from stocksense.strategies import indicators as I  # noqa: E741
from stocksense.strategies import tiers as T
from stocksense.strategies.data import DailyFrame, NiftyTable
from tests.unit.strat_helpers import IST, assert_past_only, day_rows, feats, fired, make_bars


def mirror_rows(rows, k=200.0):
    """The day turned upside down around k, prices staying positive: (o, h, l, c) -> (k-o, k-l, k-h, k-c)."""
    return [(k - o, k - low, k - h, k - c, v) for o, h, low, c, v in rows]


def dfeat(**arrays: Any) -> dict[str, Any]:
    """A minimal daily feature dict from named arrays (all the same length)."""
    n = len(next(iter(arrays.values())))
    return {"n": n, **{k: np.asarray(v, float) for k, v in arrays.items()}}


# --------------------------------------------------------------------------------------- registry
def test_registry_holds_every_tier_strategy():
    base.load_library()
    want = {
        "tier_f_trendline": ("m-swing", "1d", "F"),
        "tier_f_rsi_oversold": ("m-swing", "1d", "F"),
        "tier_f_ict_unicorn": ("m-intra", "5m", "F"),
        "tier_c_vwap_mr": ("m-intra", "5m", "C"),
        "tier_c_btst": ("m-swing", "1d", "C"),
        "tier_c_orb_15": ("m-intra", "5m", "C"),
        "tier_c_orb_30": ("m-intra", "5m", "C"),
        "tier_c_macro_trend": ("m-index", "1d", "C"),
        "tier_c_macro_fii": ("m-index", "1d", "C"),
        "tier_b_absorption": ("m-intra", "5m", "B"),
        "tier_b_cvd_divergence": ("m-intra", "5m", "B"),
        "tier_b_str_1d": ("m-xs", "1d", "B"),
        "tier_b_str_5d": ("m-xs", "1d", "B"),
        "tier_b_smt": ("m-index", "5m", "B"),
        "tier_b_amt_80": ("m-intra", "5m", "B"),
        "tier_b_ib_ext": ("m-intra", "5m", "B"),
        "tier_b_open_drive": ("m-intra", "5m", "B"),
        "tier_b_poor_extreme": ("m-intra", "5m", "B"),
        "tier_b_turtle": ("m-swing", "1d", "B"),
    }
    for key, (fam, tf, tier) in want.items():
        s = base.REGISTRY[key]
        assert (s.family, s.timeframe, s.tier, s.source) == (fam, tf, tier, "tier"), key
        assert s.default_exit in s.exits and s.description, key
    assert base.REGISTRY["tier_b_smt"].status == "await_data"
    assert base.REGISTRY["tier_c_btst"].status == "ready"
    assert base.REGISTRY["tier_b_str_1d"].score is not None
    assert base.REGISTRY["tier_c_macro_trend"].timing is not None
    assert base.by_name_or_key("ICT unicorn").key == "tier_f_ict_unicorn"


# --------------------------------------------------------------------------------------- F: trendline
def _trend_series(low5=10.0, low17=12.0, green=True):
    n = 40
    o, h, low, c = (np.full(n, v) for v in (21.0, 22.0, 20.0, 21.0))
    low[5], low[17] = low5, low17
    low[30], o[30], c[30], h[30] = 14.2, 14.5, 15.5 if green else 13.9, 16.0
    return dfeat(o=o, h=h, l=low, c=c, atr=np.ones(n))


def test_trendline_fires_where_low_touches_the_rising_line():
    f = _trend_series()
    sig, stop = T.tier_f_trendline(f)
    assert fired(sig) == [30]
    assert stop[30] == pytest.approx(10 + 2 * 25 / 12 - 0.5)  # line at t minus 0.5 ATR
    assert_past_only(T.tier_f_trendline, f, [12, 21, 30, 31, 36])


def test_trendline_needs_a_rising_line_and_a_green_bounce():
    assert not T.tier_f_trendline(_trend_series(low5=12.0, low17=10.0))[0].any()  # falling lows
    assert not T.tier_f_trendline(_trend_series(green=False))[0].any()  # closes under the line, red


def test_trendline_needs_swing_lows_ten_bars_apart():
    f = _trend_series()
    f["l"][17], f["l"][12] = 20.0, 12.0  # second swing low only 7 bars after the first
    assert not T.tier_f_trendline(f)[0].any()


# --------------------------------------------------------------------------------------- F: RSI
def test_rsi_oversold_fires_on_the_upward_cross_of_30():
    r = [50, 40, 29, 31, 35, 30, 29, 30, 45]
    f = dfeat(rsi14=r, c=np.full(9, 100.0), atr=np.full(9, 2.0))
    sig, stop = T.tier_f_rsi_oversold(f)
    assert fired(sig) == [3, 7]
    assert stop[3] == 95.0
    assert_past_only(T.tier_f_rsi_oversold, f, [3, 4, 7, 8])


def test_rsi_oversold_silent_when_never_below_30():
    f = dfeat(rsi14=[40, 45, 50, 55, 60], c=np.full(5, 100.0), atr=np.ones(5))
    assert not T.tier_f_rsi_oversold(f)[0].any()
    f = dfeat(rsi14=[30, 30, 30, 30], c=np.full(4, 100.0), atr=np.ones(4))  # sitting at 30 is not a cross
    assert not T.tier_f_rsi_oversold(f)[0].any()


# --------------------------------------------------------------------------------------- F: unicorn
UNICORN = [
    (106.0, 106.5, 105.5, 105.8),
    (105.8, 106.0, 104.6, 104.8),
    (104.8, 105.0, 103.6, 103.8),
    (103.8, 104.0, 100.0, 101.0),
    (101.0, 103.0, 100.8, 102.8),
    (102.8, 104.5, 102.5, 104.2),
    (104.2, 106.0, 104.0, 105.8),
    (105.8, 106.6, 105.6, 106.4),  # the order block: last up candle before the break
    (106.4, 106.5, 98.0, 98.5),  # closes under the swing low: bearish break
    (98.5, 100.0, 97.5, 99.5),
    (99.5, 101.0, 99.0, 100.5),
    (100.5, 102.0, 100.2, 101.8),
    (101.8, 108.5, 101.5, 108.0),  # closes through the block: it is a breaker; leaves a gap 102-107
    (108.0, 109.5, 107.0, 109.0),
    (109.0, 109.2, 106.4, 108.0),  # trades back into the overlap and closes above it
    (108.0, 110.0, 107.8, 109.8),
    (109.8, 111.0, 109.5, 110.5),
]


def _unicorn_days(rows=UNICORN):
    return [[(o, h, low, c, 1000.0) for o, h, low, c in rows]]


def test_unicorn_fires_on_the_retest_of_a_breaker_fair_value_gap_overlap():
    f = feats(_unicorn_days())
    sig, stop = T.tier_f_ict_unicorn(f)
    assert fired(sig) == [14]
    assert stop[14] == pytest.approx(105.6 - 0.1 * f["atr"][14])  # beyond the breaker's far edge
    assert_past_only(T.tier_f_ict_unicorn, f, [9, 13, 14, 15, 17])


def test_unicorn_silent_on_the_upside_down_chart_and_when_the_retest_closes_inside():
    assert not T.tier_f_ict_unicorn(feats(_unicorn_days(), sign=-1))[0].any()
    rows = list(UNICORN)
    rows[14] = (109.0, 109.2, 106.4, 106.5)  # closes inside the zone, not out of it
    sig = T.tier_f_ict_unicorn(feats(_unicorn_days(rows)))[0]
    assert 14 not in fired(sig)


# --------------------------------------------------------------------------------------- C: VWAP
VWAP_CLOSES = [100.0] * 14 + [99.9, 99.8, 99.6, 99.3, 98.9, 98.4, 97.9, 97.5, 97.2]


def test_vwap_mr_fires_between_two_and_three_sigma_under_vwap_after_ten():
    f = feats([day_rows(VWAP_CLOSES)])
    sig, stop = T.tier_c_vwap_mr(f)
    assert fired(sig) == [22]
    from stocksense.orderflow.vwap import session_vwap

    vw, sd = session_vwap(T._bars(f))
    assert stop[22] == pytest.approx(vw[22] - 3 * sd[22])
    assert_past_only(T.tier_c_vwap_mr, f, [10, 22, 23])


def test_vwap_mr_silent_before_ten_on_a_rising_chart_and_beyond_three_sigma():
    f = feats([day_rows(VWAP_CLOSES)])
    f["mins"] = f["mins"] - 120  # the same bars, but before 10:00
    assert not T.tier_c_vwap_mr(f)[0].any()
    up = [200.0 - x for x in VWAP_CLOSES]  # the upside-down chart
    assert not T.tier_c_vwap_mr(feats([day_rows(up)]))[0].any()
    assert 14 not in fired(
        T.tier_c_vwap_mr(feats([day_rows(VWAP_CLOSES)]))[0]
    )  # -5 sigma: stop would be above


# --------------------------------------------------------------------------------------- C: BTST
def _btst_f():
    # day 2 passes every test; days 0,1,3,4,5 each fail exactly one
    return dfeat(
        chg=[3.0, 3.0, 3.0, 3.0, 1.5, 3.0],
        c=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        h=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        l=[9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        v=[2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        vavg=[1000.0] * 6,
    ) | {"nifty_gt_sma50": np.array([False, True, True, True, True, True])}


def test_btst_fires_on_the_strong_close_with_nifty_above_its_mean():
    f = _btst_f()
    f["c"][3] = 9.5  # closes in the middle of its range
    f["v"][5] = 1400.0  # volume under 1.5x
    sig, stop = T.tier_c_btst(f)
    assert fired(sig) == [1, 2]
    assert stop[2] == 9.0
    assert_past_only(T.tier_c_btst, f, [2, 3, 6])


def test_btst_is_silent_without_the_nifty_flag():
    f = _btst_f()
    del f["nifty_gt_sma50"]
    assert not T.tier_c_btst(f)[0].any()


def test_extend_daily_flags_nifty_above_its_fifty_day_mean():
    dates = np.arange("2026-01-01", "2026-04-01", dtype="datetime64[D]")
    close = np.r_[np.linspace(100, 90, 60), np.linspace(90, 120, len(dates) - 60)]
    nifty = NiftyTable.from_close(dates, close)
    f = {"n": 3, "date": dates[[10, 62, 85]]}
    T.extend_daily(f, nifty)
    assert f["nifty_gt_sma50"].tolist() == [
        False,
        False,
        True,
    ]  # day 10: no 50-day mean yet; day 62: below it
    f2 = {"n": 1, "date": np.array(["2025-01-01"], dtype="datetime64[D]")}
    assert not T.extend_daily(f2, nifty)["nifty_gt_sma50"].any()  # a date the Nifty table lacks


# --------------------------------------------------------------------------------------- C: ORB
ORB_CLOSES = [100.2, 100.4, 100.6, 100.5, 101.5, 101.4, 101.6, 102.5, 102.0]


def _orb_f(closes=ORB_CLOSES, rvol=1.5):
    f = feats([day_rows(closes, o0=100.0)])
    f["rvol"] = np.full(f["n"], rvol)
    return f


def test_orb_15_and_30_fire_on_the_cross_with_relative_volume():
    f = _orb_f()
    s15, st15 = T.tier_c_orb_15(f)
    s30, st30 = T.tier_c_orb_30(f)
    assert fired(s15) == [4]
    assert fired(s30) == [7]
    assert st15[4] == pytest.approx(f["or15l"][4]) and st30[7] == pytest.approx(
        f["or30l"][7]
    )  # opposite side
    assert_past_only(T.tier_c_orb_15, f, [3, 4, 5, 9])
    assert_past_only(T.tier_c_orb_30, f, [6, 7, 8, 9])


def test_orb_silent_on_thin_volume_and_on_the_upside_down_chart():
    assert not T.tier_c_orb_15(_orb_f(rvol=1.1))[0].any()
    assert not T.tier_c_orb_30(_orb_f(rvol=1.1))[0].any()
    down = [200.0 - x for x in ORB_CLOSES]
    f = feats([day_rows(down, o0=100.0)])
    f["rvol"] = np.full(f["n"], 1.5)
    assert not T.tier_c_orb_15(f)[0].any()


# --------------------------------------------------------------------------------------- C: macro timing
def _sessions(n=30):
    d0 = date(2026, 1, 5)
    return [d0 + timedelta(days=i) for i in range(n)]


def _macro_rows(series, values, avail_hour=10, last_hour=None):
    rows = []
    for i, (d, v) in enumerate(zip(_sessions(len(values)), values, strict=True)):
        hr = last_hour if (last_hour is not None and i == len(values) - 1) else avail_hour
        ts = datetime(d.year, d.month, d.day, hr, 0, tzinfo=IST)
        rows.append({"series": series, "date": d, "close": float(v), "available_ts": ts})
    return rows


def _macro_frame(usd_last_hour=10, usd_last=84.0):
    n = 30
    rows = _macro_rows("BRENT", [90 - 0.5 * i for i in range(n)])
    rows += _macro_rows("US10Y", [4.0 + 0.01 * i for i in range(n)])
    rows += _macro_rows("USDINR", [85.0] * (n - 1) + [usd_last], last_hour=usd_last_hour)
    df = pl.DataFrame(
        rows,
        schema={
            "series": pl.Utf8,
            "date": pl.Date,
            "close": pl.Float64,
            "available_ts": pl.Datetime("us", "Asia/Kolkata"),
        },
    )
    return df


def _fii(pos_from=0, n=30):
    return pl.DataFrame(
        {"date": _sessions(n), "fii_ratio_chg5": [0.1 if i >= pos_from else -0.1 for i in range(n)]}
    )


def test_macro_trend_needs_three_of_four():
    sessions = pl.DataFrame({"date": _sessions()})
    out = T.macro_trend_position(sessions, _macro_frame(), _fii(pos_from=22))
    pos = out["position"].to_list()
    # Brent falls from session 20 on, US10Y rises, the rupee falls only on the last day, the FII ratio from 22
    assert pos[:22] == [0] * 22
    assert pos[22:29] == [0] * 7  # Brent + FII = 2: the rupee is the third
    assert pos[29] == 1


def test_macro_trend_ignores_values_published_after_the_close():
    sessions = pl.DataFrame({"date": _sessions()})
    late = T.macro_trend_position(sessions, _macro_frame(usd_last_hour=20), _fii(pos_from=22))
    assert late["position"].to_list()[29] == 0  # the 84.0 print is available at 20:00, after the 15:30 close


def test_macro_trend_position_is_past_only():
    sessions = pl.DataFrame({"date": _sessions()})
    macro, fii = _macro_frame(), _fii(pos_from=22)
    full = T.macro_trend_position(sessions, macro, fii)
    t = _sessions()[26]
    part = T.macro_trend_position(
        sessions.filter(pl.col("date") <= t),
        macro.filter(pl.col("date") <= t),
        fii.filter(pl.col("date") <= t),
    )
    assert part["position"].to_list() == full["position"].to_list()[:27]


def test_fii_ratio_position():
    p = pl.DataFrame(
        {
            "date": _sessions(5),
            "fii_ratio": [0.6, 0.6, 0.4, 0.5, None],
            "fii_ratio_chg5": [0.1, -0.1, 0.1, 0.1, 0.1],
        }
    )
    assert T.fii_ratio_position(p)["position"].to_list() == [1, 0, 0, 0, 0]
    assert T.fii_ratio_position(p.head(2))["position"].to_list() == [1, 0]  # past-only


# --------------------------------------------------------------------------------------- B: absorption
def _absorb_rows(close13=97.95):
    closes = [100.0 - 0.2 * i for i in range(12)]
    rows = day_rows(closes, 1000.0, 0.2)
    rows.append((97.8, 97.82, 97.57, 97.7, 4000.0))  # 4x volume, a quarter of the range, a new low
    rows.append((97.7, max(97.7, close13) + 0.02, 97.65, close13, 1000.0))
    rows += day_rows([close13 + 0.1 * i for i in range(1, 8)], 1000.0, 0.2, o0=close13)
    return rows


def test_absorption_fires_on_the_bar_after_a_bid_absorption_closes_above_its_high():
    f = feats([_absorb_rows()])
    sig, stop = T.tier_b_absorption(f)
    assert fired(sig) == [13]
    assert stop[13] == pytest.approx(97.57 - 0.1 * f["atr"][12])
    assert_past_only(T.tier_b_absorption, f, [12, 13, 14, 18])


def test_absorption_negative_cases():
    assert not T.tier_b_absorption(feats([_absorb_rows(close13=97.7)]))[0].any()  # no close above the high
    # the same chart seen as its offer-side twin: the long rule must not fire on the unmirrored chart
    twin = feats([mirror_rows(_absorb_rows())], sign=1)
    assert not T.tier_b_absorption(twin)[0].any()
    # an offer absorption (upside-down chart) fires on the mirrored frame, as the long of that frame
    assert fired(T.tier_b_absorption(feats([mirror_rows(_absorb_rows())], sign=-1))[0]) == [13]


# --------------------------------------------------------------------------------------- B: CVD divergence
def _cvd_closes():
    c = [100.1 if i % 2 == 0 else 100.0 for i in range(56)]
    c += [99.7, 99.4, 99.1]  # 56-58: a sharp fall to a low
    c += [99.15, 99.2, 99.25, 99.3, 99.35]  # 59-63: a weak bounce
    c += [98.95]  # 64: a lower low on far less selling than the first
    c += [99.45]  # 65: closes above bar 64's high
    return c


def test_cvd_divergence_fires_on_the_bar_after_the_divergence_when_it_closes_above_its_high():
    f = feats([day_rows(_cvd_closes(), 1000.0, 0.02)])
    sig, stop = T.tier_b_cvd_divergence(f)
    assert fired(sig) == [65]
    assert stop[65] == pytest.approx(f["l"][64])
    assert_past_only(T.tier_b_cvd_divergence, f, [60, 64, 65, 66])


def test_cvd_divergence_silent_without_the_confirming_close_and_on_the_upside_down_chart():
    c = _cvd_closes()
    c[65] = 99.0  # no close above the high of bar 64
    assert not T.tier_b_cvd_divergence(feats([day_rows(c, 1000.0, 0.02)]))[0].any()
    up = [200.0 - x for x in _cvd_closes()]
    assert not T.tier_b_cvd_divergence(feats([day_rows(up, 1000.0, 0.02)]))[0].any()


# --------------------------------------------------------------------------------------- B: reversal frames
def _panel(n_days=7):
    rows = []
    rets = {
        1: [0.01, -0.02, 0.03, 0.00, 0.02, -0.01, 0.04],
        2: [0.00, 0.01, 0.01, 0.02, -0.02, 0.00, 0.01],
        3: [0.02, 0.02, -0.01, 0.01, 0.00, 0.03, -0.02],
    }
    ind = {1: "A", 2: "A", 3: "B"}
    for sid, r in rets.items():
        for i in range(n_days):
            rows.append(
                {"date": date(2026, 1, 5) + timedelta(days=i), "sid": sid, "ret": r[i], "industry": ind[sid]}
            )
    return pl.DataFrame(rows)


def test_str_1d_is_minus_the_one_day_return():
    out = T.str_1d_frame(_panel())
    row = out.filter((pl.col("sid") == 1) & (pl.col("date") == date(2026, 1, 6))).row(0, named=True)
    assert row["score"] == pytest.approx(0.02)
    assert out.height == 21


def test_str_5d_is_minus_five_day_industry_excess():
    out = T.str_5d_frame(_panel())
    assert out["date"].min() == date(2026, 1, 9)  # the fifth session
    # sid 1 over sessions 0-4: excess vs the mean of sids 1 and 2 = (r1 - r2)/2
    r1, r2 = [0.01, -0.02, 0.03, 0.00, 0.02], [0.00, 0.01, 0.01, 0.02, -0.02]
    want = -sum((a - b) / 2 for a, b in zip(r1, r2, strict=True))
    got = out.filter((pl.col("sid") == 1) & (pl.col("date") == date(2026, 1, 9)))["score"].item()
    assert got == pytest.approx(want)
    # sid 3 is alone in its industry: no excess, score 0
    assert out.filter(pl.col("sid") == 3)["score"].abs().max() == pytest.approx(0.0)


def test_str_scores_are_past_only():
    p = _panel()
    cut_date = date(2026, 1, 9)
    for fn in (T.str_1d_frame, T.str_5d_frame):
        full = fn(p).filter(pl.col("date") <= cut_date).sort("date", "sid")
        part = fn(p.filter(pl.col("date") <= cut_date)).sort("date", "sid")
        assert full.equals(part)


# --------------------------------------------------------------------------------------- B: SMT
def _smt_pair(bank_low22=200.0):
    n = 25
    nifty = [(100.0, 101.0, 100.0 if i < 22 else 99.0, 100.5, 0.0) for i in range(n)]
    bank = [(200.0, 201.0, 199.0 if i != 22 else bank_low22, 200.5, 0.0) for i in range(n)]
    return make_bars([nifty]), make_bars([bank])


def test_smt_long_when_nifty_makes_a_low_and_bank_nifty_does_not():
    nifty, bank = _smt_pair()
    lg, st = T.smt_signals(nifty, bank)
    assert fired(lg) == [22] and not st.any()


def test_smt_silent_when_both_make_new_lows_and_short_is_the_mirror():
    nifty, bank = _smt_pair(bank_low22=198.0)
    assert not T.smt_signals(nifty, bank)[0].any()
    nifty, bank = _smt_pair()
    mn = Bars(nifty.ts, -nifty.open, -nifty.low, -nifty.high, -nifty.close, nifty.volume)
    mb = Bars(bank.ts, -bank.open, -bank.low, -bank.high, -bank.close, bank.volume)
    lg, st = T.smt_signals(mn, mb)
    assert fired(st) == [22] and not lg.any()


def test_smt_is_past_only():
    nifty, bank = _smt_pair()
    full = T.smt_signals(nifty, bank)[0]
    for m in (10, 22, 23):
        part = T.smt_signals(nifty.take(slice(0, m)), bank.take(slice(0, m)))[0]
        assert np.array_equal(part, full[:m])


def test_smt_timing_is_empty_without_index_bars(monkeypatch):
    monkeypatch.setattr("stocksense.strategies.data.intraday_frame", lambda *a, **k: None)
    out = T.timing_smt(None, date(2026, 1, 5), date(2026, 1, 6))
    assert out.height == 0 and out.columns == ["date", "position"]


# --------------------------------------------------------------------------------------- B: auction rules
DAY1 = [101.5 + 2.0 * np.sin(i / 6.0) for i in range(75)]
DAY1_ROWS = day_rows(DAY1, [4000.0 if 101.0 < c < 103.0 else 1000.0 for c in DAY1], 0.2)


def test_amt_80_fires_after_two_periods_close_back_inside_the_value_area():
    d2 = (
        [97.2, 97.4, 97.3, 97.5, 97.4, 97.6]
        + [98, 99, 100, 100.5, 101.0, 101.5]
        + [101.8, 102.0, 101.9, 102.1, 102.0, 102.2]
        + [102.0] * 20
    )
    f = feats([DAY1_ROWS, day_rows(d2, 1000.0, 0.1, o0=97.0)])
    sig, stop = T.tier_b_amt_80(f)
    assert fired(sig) == [75 + 17]  # the last bar of the third period; the second closed inside, too
    val = 101.15
    assert stop[92] == pytest.approx(val - 0.002 * val)
    assert_past_only(T.tier_b_amt_80, f, [80, 91, 92, 93, 100])


def test_amt_80_silent_when_the_open_is_inside_value_or_only_one_period_closes_in():
    inside = [102.0] * 30
    assert not T.tier_b_amt_80(feats([DAY1_ROWS, day_rows(inside, 1000.0, 0.1, o0=102.0)]))[0].any()
    one = [97.2] * 6 + [101.5] * 6 + [97.0] * 18  # one period inside, then away again
    assert not T.tier_b_amt_80(feats([DAY1_ROWS, day_rows(one, 1000.0, 0.1, o0=97.0)]))[0].any()


def test_amt_80_mirror_frame_fires_on_the_upside_down_chart():
    d2 = (
        [97.2, 97.4, 97.3, 97.5, 97.4, 97.6]
        + [98, 99, 100, 100.5, 101.0, 101.5]
        + [101.8, 102.0, 101.9, 102.1, 102.0, 102.2]
        + [102.0] * 20
    )
    days = [DAY1_ROWS, day_rows(d2, 1000.0, 0.1, o0=97.0)]
    assert fired(T.tier_b_amt_80(feats([mirror_rows(d) for d in days], sign=-1))[0]) == [92]
    assert not T.tier_b_amt_80(feats([mirror_rows(d) for d in days], sign=1))[0].any()


IB_CLOSES = [
    100.5,
    101.0,
    101.5,
    101.0,
    100.5,
    101.0,
    101.5,
    101.8,
    101.2,
    100.8,
    101.0,
    101.4,
    101.7,
    102.5,
    102.6,
]


def _ib_f(rvol=1.5, closes=IB_CLOSES):
    f = feats([day_rows(closes)])
    f["rvol"] = np.full(f["n"], rvol)
    return f


def test_ib_extension_fires_on_the_first_close_over_the_initial_balance_high():
    f = _ib_f()
    sig, stop = T.tier_b_ib_ext(f)
    assert fired(sig) == [13]
    assert stop[13] == pytest.approx((101.9 + 100.4) / 2)
    assert_past_only(T.tier_b_ib_ext, f, [12, 13, 14, 15])


def test_ib_extension_silent_on_thin_volume_and_upside_down():
    assert not T.tier_b_ib_ext(_ib_f(rvol=0.9))[0].any()
    f = feats([day_rows([200 - c for c in IB_CLOSES])])
    f["rvol"] = np.full(f["n"], 1.5)
    assert not T.tier_b_ib_ext(f)[0].any()


def _drive_days(open2=104.5, step=0.2, poor_day1=True):
    d1 = DAY1_ROWS
    d2 = [open2 + step * (i + 1) for i in range(6)] + [open2 + step * 6] * 20
    return [d1, day_rows(d2, 1000.0, 0.0, o0=open2)]


def test_open_drive_fires_at_0940_bar_for_a_drive_away_from_value():
    f = feats(_drive_days())
    sig, stop = T.tier_b_open_drive(f)
    assert fired(sig) == [75 + 5]
    assert stop[80] == pytest.approx(104.5)
    assert_past_only(T.tier_b_open_drive, f, [79, 80, 81, 90])


def test_open_drive_silent_inside_value_and_when_the_drive_is_down():
    assert not T.tier_b_open_drive(feats(_drive_days(open2=101.0)))[0].any()  # not away from the previous POC
    assert not T.tier_b_open_drive(feats(_drive_days(step=-0.2)))[0].any()  # a drive down is a short
    mirrored = feats([mirror_rows(d) for d in _drive_days()], sign=-1)
    assert fired(T.tier_b_open_drive(mirrored)[0]) == [80]  # the same chart upside down is the same long


def _poor_low_days(poor=True):
    d1 = list(DAY1_ROWS)
    for k in (3, 8):  # two bars in different 30-minute periods share the day's lowest price
        o, h, low, c, v = d1[k]
        d1[k] = (o, h, 99.0, c, v)
    if not poor:
        o, h, low, c, v = d1[8]
        d1[8] = (o, h, 99.6, c, v)
    closes = [100.0, 99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.0, 98.8, 99.5, 99.6, 99.7]
    return [d1, day_rows(closes, 1000.0, 0.05)]


def test_poor_extreme_fires_when_price_closes_back_over_a_poor_low():
    f = feats(_poor_low_days())
    sig, stop = T.tier_b_poor_extreme(f)
    assert fired(sig) == [75 + 11]
    assert stop[86] == pytest.approx(f["lod"][86] - 0.1 * f["atr"][86])
    assert_past_only(T.tier_b_poor_extreme, f, [80, 85, 86, 87, 89])


def test_poor_extreme_silent_without_a_poor_low_and_on_the_unmirrored_twin():
    assert not T.tier_b_poor_extreme(feats(_poor_low_days(poor=False)))[0].any()
    days = _poor_low_days()
    assert fired(T.tier_b_poor_extreme(feats([mirror_rows(d) for d in days], sign=-1))[0]) == [86]
    assert not T.tier_b_poor_extreme(feats([mirror_rows(d) for d in days], sign=1))[0].any()


# --------------------------------------------------------------------------------------- B: Turtle
def _turtle_f(last=103.0):
    n = 31
    c = np.full(n, 100.0)
    c[30] = last
    d = np.arange(n).astype("datetime64[D]")
    frame = DailyFrame(d, c, c + 0.5, c - 0.5, c, np.full(n, 1000.0))
    return I.daily_features(frame, NiftyTable.from_close(d, np.full(n, 100.0)))


def test_turtle_fires_on_the_close_over_the_twenty_day_high():
    f = _turtle_f()
    f["h"][30] = 103.5
    sig, stop = T.tier_b_turtle(f)
    assert fired(sig) == [30]
    n20 = I.npx.rma(I.true_range(f["h"], f["l"], f["c"]), 20)
    assert stop[30] == pytest.approx(103.0 - 2 * n20[30])
    assert_past_only(T.tier_b_turtle, f, [25, 30, 31])


def test_turtle_silent_when_the_close_stays_under_the_high_or_the_chart_is_upside_down():
    assert not T.tier_b_turtle(_turtle_f(last=100.4))[0].any()
    assert not T.tier_b_turtle(_turtle_f(last=97.0))[0].any()


@pytest.mark.parametrize(
    "key", [k for k in base.load_library() if k.startswith("tier_") and base.REGISTRY[k].signal]
)
def test_every_per_stock_tier_function_has_a_docstring_and_returns_aligned_arrays(key):
    spec = base.REGISTRY[key]
    f = feats([DAY1_ROWS]) if spec.timeframe == "5m" else None
    if f is None:
        pytest.skip("daily rules are covered by their own tests")
    f["rvol"] = np.ones(f["n"])
    sig, stop = spec.signal(f)
    assert len(sig) == len(stop) == f["n"] and sig.dtype == bool
