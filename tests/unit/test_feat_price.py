"""T4.3a/T4.3b: price features (momentum, trend, reversal, price action), each hand-computed."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.features import price
from stocksense.features.registry import load_all

REG = load_all()


def _df(sid=None, sx=None, **cols) -> pl.DataFrame:
    """A mini panel sorted by sid, date; `sx` defaults to 0..n-1; every value column Float64."""
    n = len(next(iter(cols.values())))
    data = {
        "sid": sid or [1] * n,
        "sx": sx or list(range(n)),
        "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
    }
    df = pl.DataFrame(data, schema_overrides={"sid": pl.Int32, "sx": pl.Int64})
    return df.with_columns(*[pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()])


def _run(df: pl.DataFrame, beta_w: int = 63, **exprs: pl.Expr) -> pl.DataFrame:
    st = price._stage(df.lazy(), beta_w=beta_w)
    return st.select("date", "sid", *[e.alias(k) for k, e in exprs.items()]).collect()


def _ret(la: list[float]) -> list[float]:
    return [math.expm1(x) for x in la]


def _close(vals: list[float], **kw) -> pl.DataFrame:
    return _df(adj_close=vals, adj_high=vals, adj_low=vals, **kw)


# ---- T4.3a ------------------------------------------------------------------------------------


def test_mom_sums_the_adjusted_log_returns():
    la = [0.1, 0.2, -0.1, 0.3, 0.0, 0.1]
    out = _run(_df(ret=_ret(la)), m=price.mom(3))["m"].to_list()
    # windows of 3: .1+.2-.1 = .2 ; .2-.1+.3 = .4 ; -.1+.3+0 = .2 ; .3+0+.1 = .4
    assert out[:2] == [None, None]
    assert out[2:] == pytest.approx([0.2, 0.4, 0.2, 0.4])


def test_mom_skip_and_accel():
    la = [0.1, 0.2, -0.1, 0.3, 0.0, 0.1]
    out = _run(_df(ret=_ret(la)), s=price.mom_skip(4, 1), a=price.mom_accel(2, 4))
    s = out["s"].to_list()
    # sum of 4 minus last 1: row 3: (.1+.2-.1+.3) - .3 = .2 ; row 4: (.2-.1+.3+0) - 0 = .4
    # row 5: (-.1+.3+0+.1) - .1 = .2
    assert s[3:] == pytest.approx([0.2, 0.4, 0.2])
    a = out["a"].to_list()
    # accel(2,4) = mom2 - (mom4 - mom2); row 3: mom2 = .3+(-.1) = .2, mom4 = .5 -> .2-(.5-.2) = -.1
    assert a[3] == pytest.approx(-0.1)
    # row 5: mom2 = 0+.1 = .1, mom4 = -.1+.3+0+.1 = .3 -> .1-(.3-.1) = -.1
    assert a[5] == pytest.approx(-0.1)


def test_52_week_proximity():
    high = [10.0, 12.0, 11.0, 9.0, 8.0, 7.0]
    low = [8.0, 9.0, 9.0, 7.0, 6.0, 5.0]
    close = [9.0, 11.0, 10.0, 8.0, 7.0, 6.5]
    df = _df(adj_close=close, adj_high=high, adj_low=low)
    out = _run(df, hi=price.hi_prox(3), lo=price.lo_prox(3))
    # 3-row max of high: 12, 12, 11, 9 -> close/max: 10/12, 8/12, 7/11, 6.5/9
    assert out["hi"].to_list()[2:] == pytest.approx([10 / 12, 8 / 12, 7 / 11, 6.5 / 9])
    # 3-row min of low: 8, 7, 6, 5 -> 10/8, 8/7, 7/6, 6.5/5
    assert out["lo"].to_list()[2:] == pytest.approx([1.25, 8 / 7, 7 / 6, 1.3])


def test_days_since_high_counts_rows_since_the_window_max_latest_tie():
    high = [10.0, 12.0, 11.0, 9.0, 13.0, 13.0, 7.0]
    out = _run(_close(high), d=price.days_since_hi(3))["d"].to_list()
    # windows: [10,12,11] max 1 row ago; [12,11,9] 2 ago; [11,9,13] 0; [9,13,13] tie -> latest: 0;
    # [13,13,7] latest 13 is 1 row ago. Divided by the window length 3.
    assert out[:2] == [None, None]
    assert out[2:] == pytest.approx([1 / 3, 2 / 3, 0.0, 0.0, 1 / 3])


def test_days_since_high_is_null_when_the_window_has_a_null():
    high = [10.0, None, 11.0, 9.0, 8.0]
    df = _df(adj_close=[1.0] * 5, adj_high=high, adj_low=[1.0] * 5)
    out = _run(df, d=price.days_since_hi(3))["d"].to_list()
    # last window [11,9,8]: max 2 rows ago -> 2/3 ; windows holding the null are null
    assert out[:4] == [None] * 4
    assert out[4] == pytest.approx(2 / 3)


def test_ma_gap():
    close = [10.0, 11.0, 12.0, 9.0, 12.0, 15.0]
    out = _run(_close(close), g=price.ma_gap(3))["g"].to_list()
    # means of 3: 11, 32/3, 11, 12 -> 12/11-1, 9/(32/3)-1 = -0.15625, 12/11-1, 15/12-1 = .25
    assert out[2:] == pytest.approx([12 / 11 - 1, -0.15625, 12 / 11 - 1, 0.25])


def test_trend_r2_is_signed_by_the_slope():
    y = [1.0, 3.0, 2.0, 8.0]
    out = _run(_close([math.exp(v) for v in y]), t=price.trend_r2(3))["t"].to_list()
    # window [1,3,2]: slope .5, corr .5 -> r2 .25 ; window [3,2,8]: cov 2.5, var(y) 93/9 -> r2 6.25/(93/9)
    assert out[2] == pytest.approx(0.25)
    assert out[3] == pytest.approx(6.25 / (93 / 9))
    # falling line [3,2,0]: cov -1.5 (slope -1.5), var(y) 42/18 -> r2 2.25/(42/18), negative sign
    o2 = _run(_close([math.exp(v) for v in (3.0, 2.0, 0.0)]), t=price.trend_r2(3))["t"].to_list()
    assert o2[2] == pytest.approx(-2.25 / (42 / 18))


def test_resid_mom_uses_the_previous_rows_beta():
    la = [0.2, 0.4, 0.6, 0.5, 0.2, 0.7]
    lm = [0.1, 0.2, 0.3, 0.1, 0.2, 0.4]
    out = _run(_df(ret=_ret(la), lm=lm), beta_w=3, r=price.resid_mom(2))["r"].to_list()
    # beta_3 on rows 0-2 = 2 (la = 2 lm), rows 1-3 = .5, rows 2-4 = .5; each is used one row later.
    # e3 = .5 - 2(.1) = .3 ; e4 = .2 - .5(.2) = .1 ; e5 = .7 - .5(.4) = .5
    # row 4: sum(.3,.1) = .4, std = .2/sqrt2 ; row 5: sum(.1,.5) = .6, std = .4/sqrt2
    assert out[:4] == [None] * 4
    assert out[4] == pytest.approx(0.4 / (0.2 / math.sqrt(2)))
    assert out[5] == pytest.approx(0.6 / (0.4 / math.sqrt(2)))


def test_resid_mom_beta_ignores_the_current_row():
    la = [0.2, 0.4, 0.6, 0.5, 0.2, 0.7]
    lm = [0.1, 0.2, 0.3, 0.1, 0.2, 0.4]
    a = _run(_df(ret=_ret(la), lm=lm), beta_w=3, b=pl.col("beta"))["b"].to_list()
    b = _run(_df(ret=_ret(la), lm=[*lm[:-1], 0.9]), beta_w=3, b=pl.col("beta"))["b"].to_list()
    assert a[5] == pytest.approx(b[5]) and a[3] == pytest.approx(2.0)


def _long_panel(n=300, seed=3):
    rng = np.random.default_rng(seed)
    la = rng.normal(0.0005, 0.02, n)
    px = 100 * np.exp(np.cumsum(la))
    df = _df(ret=_ret(list(la)), adj_close=list(px), adj_high=list(px * 1.01), adj_low=list(px * 0.99))
    return la, px, df


def test_registered_windows_match_an_independent_numpy_computation():
    la, px, df = _long_panel()
    names = [
        "mom_5",
        "mom_21",
        "mom_63",
        "mom_126",
        "mom_252_skip21",
        "mom_accel",
        "hi52_prox",
        "lo52_prox",
        "days_since_hi52",
        "ma_gap_20",
        "ma_gap_50",
        "ma_gap_200",
    ]
    o = price.compute(df.lazy(), names).collect()
    t = 299
    assert o["mom_5"][t] == pytest.approx(la[t - 4 : t + 1].sum())
    assert o["mom_21"][t] == pytest.approx(la[t - 20 : t + 1].sum())
    assert o["mom_63"][t] == pytest.approx(la[t - 62 : t + 1].sum())
    assert o["mom_126"][t] == pytest.approx(la[t - 125 : t + 1].sum())
    assert o["mom_252_skip21"][t] == pytest.approx(la[t - 251 : t + 1].sum() - la[t - 20 : t + 1].sum())
    m63, m126 = la[t - 62 : t + 1].sum(), la[t - 125 : t + 1].sum()
    assert o["mom_accel"][t] == pytest.approx(m63 - (m126 - m63))
    assert o["hi52_prox"][t] == pytest.approx(px[t] / (px[t - 251 : t + 1] * 1.01).max())
    assert o["lo52_prox"][t] == pytest.approx(px[t] / (px[t - 251 : t + 1] * 0.99).min())
    assert o["days_since_hi52"][t] == pytest.approx((251 - int(np.argmax(px[t - 251 : t + 1]))) / 252)
    for w, nm in ((20, "ma_gap_20"), (50, "ma_gap_50"), (200, "ma_gap_200")):
        assert o[nm][t] == pytest.approx(px[t] / px[t - w + 1 : t + 1].mean() - 1)
    assert o["mom_252_skip21"][250] is None and o["mom_252_skip21"][251] is not None


def test_trend_r2_63_on_an_exact_exponential_is_one():
    px = [100 * 1.01**k for k in range(80)]
    out = price.compute(_close(px, ret=[0.01] * 80).lazy(), ["trend_r2_63"]).collect()["trend_r2_63"]
    assert out[79] == pytest.approx(1.0)


def test_t43a_registered_with_labels():
    names = [
        "mom_5",
        "mom_21",
        "mom_63",
        "mom_126",
        "mom_252_skip21",
        "mom_accel",
        "hi52_prox",
        "lo52_prox",
        "days_since_hi52",
        "ma_gap_20",
        "ma_gap_50",
        "ma_gap_200",
        "trend_r2_63",
        "resid_mom_126",
    ]
    for n in names:
        assert n in REG, n
        assert REG[n].label and REG[n].group == "price" and REG[n].kind == "xs"


# ---- T4.3b ------------------------------------------------------------------------------------


def test_ret_1_is_the_adjusted_log_return():
    out = _run(_df(ret=[0.1, -0.5, None]), r=price.ret_1())["r"].to_list()
    assert out == [pytest.approx(math.log(1.1)), pytest.approx(math.log(0.5)), None]


def test_rsi_wilder_by_hand_n3():
    close = [10.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0]
    out = _run(_close(close), r=price.rsi(3))["r"].to_list()
    # deltas from row 1: +1 -1 +2 +1 -1 +2 -> gains 1 0 2 1 0 2 ; losses 0 1 0 0 1 0
    # row 3 (seed = simple mean of the first 3): avg gain 1, avg loss 1/3 -> RS 3 -> RSI 75
    # row 4: gain (1*2+1)/3 = 1, loss (1/3*2+0)/3 = 2/9 -> RS 4.5 -> 100 - 100/5.5 = 81.8182
    # row 5: gain (1*2+0)/3 = 2/3, loss (2/9*2+1)/3 = 13/27 -> RS 18/13 -> 100 - 1300/31 = 58.0645
    # row 6: gain (2/3*2+2)/3 = 10/9, loss (13/27*2+0)/3 = 26/81 -> RS 45/13 -> 100 - 1300/58 = 77.5862
    assert out[:3] == [None, None, None]
    assert out[3:] == pytest.approx([75.0, 100 - 100 / 5.5, 100 - 1300 / 31, 100 - 1300 / 58])


def test_rsi_14_matches_a_20_row_hand_recursion():
    close = [
        44.0,
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.42,
        45.84,
        46.08,
        45.89,
        46.03,
        45.61,
        46.28,
        46.28,
        46.00,
        46.03,
        46.41,
        46.22,
    ]
    out = _run(_close(close), r=price.rsi(14))["r"].to_list()
    d = [close[i] - close[i - 1] for i in range(1, 20)]  # 19 changes
    g = [max(x, 0) for x in d]
    lo = [max(-x, 0) for x in d]
    ag, al = sum(g[:14]) / 14, sum(lo[:14]) / 14  # Wilder seed: simple mean of the first 14
    expect = {14: 100 - 100 / (1 + ag / al)}
    for k in range(14, 19):  # changes 15..19 -> rows 15..19
        ag, al = (ag * 13 + g[k]) / 14, (al * 13 + lo[k]) / 14
        expect[k + 1] = 100 - 100 / (1 + ag / al)
    assert out[:14] == [None] * 14
    for row, v in expect.items():
        assert out[row] == pytest.approx(v)


def test_rsi_extremes():
    up = _run(_close([1.0, 2.0, 3.0, 4.0, 5.0]), r=price.rsi(3))["r"].to_list()
    assert up[3:] == [100.0, 100.0]  # no losses -> 100
    flat = _run(_close([5.0] * 5), r=price.rsi(3))["r"].to_list()
    assert flat[3:] == [50.0, 50.0]  # no movement -> neutral 50


def test_bb_z():
    out = _run(_close([10.0, 12.0, 11.0, 15.0]), z=price.bb_z(3))["z"].to_list()
    # row 2: mean 11, std 1 -> 0 ; row 3: [12,11,15] mean 38/3, var (13/3) -> (15-38/3)/sqrt(13/3)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx((7 / 3) / math.sqrt(13 / 3))


def test_max_min_ret():
    la = [0.1, -0.2, 0.3, 0.0, -0.1]
    out = _run(_df(ret=_ret(la)), hi=price.max_ret(3), lo=price.min_ret(3))
    # windows [.1,-.2,.3] [-.2,.3,0] [.3,0,-.1]
    assert out["hi"].to_list()[2:] == pytest.approx([0.3, 0.3, 0.3])
    assert out["lo"].to_list()[2:] == pytest.approx([-0.2, -0.2, -0.1])


def test_close_loc_and_its_mean():
    high = [12.0, 10.0, 9.0, 10.0]
    low = [8.0, 10.0, 5.0, 6.0]
    close = [11.0, 10.0, 5.0, 9.0]
    df = _df(adj_close=close, adj_high=high, adj_low=low)
    out = _run(df, c=price.close_loc(), m=price.close_loc_mean(2))
    # (11-8)/4 = .75 ; high == low -> null ; (5-5)/4 = 0 ; (9-6)/4 = .75
    assert out["c"].to_list() == [pytest.approx(0.75), None, pytest.approx(0.0), pytest.approx(0.75)]
    # 2-row mean: null while a null is in the window; rows 2-3: (0 + .75)/2 = .375
    assert out["m"].to_list() == [None, None, None, pytest.approx(0.375)]


def test_range_pct_and_nr7():
    close = [10.0] * 5
    high = [10.0, 10.0, 10.0, 10.0, 10.0]
    low = [5.0, 6.0, 4.0, 7.0, 6.5]  # range 5, 4, 6, 3, 3.5 -> range_pct .5 .4 .6 .3 .35
    df = _df(adj_close=close, adj_high=high, adj_low=low)
    out = _run(df, r=price.range_pct(), n=price.nr7(3))
    assert out["r"].to_list() == pytest.approx([0.5, 0.4, 0.6, 0.3, 0.35])
    # 3-row min of range_pct: .4 -> row2 .6 is not it (0); row3 .3 is the min (1); row4 .35 vs min .3 (0)
    assert out["n"].to_list() == [None, None, 0.0, 1.0, 0.0]


def test_inside_day():
    high = [10.0, 11.0, 10.5, 10.8, 12.0, 12.0]
    low = [5.0, 4.0, 4.5, 4.2, 3.0, 3.5]
    out = _run(_df(adj_close=high, adj_high=high, adj_low=low), i=price.inside_day())["i"].to_list()
    # row1: 11 > 10 -> 0 ; row2: 10.5<=11 and 4.5>=4 -> 1 ; row3: 10.8 > 10.5 -> 0 ; row4: 0 ;
    # row5: 12<=12 and 3.5>=3 -> 1
    assert out == [None, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_streaks_reset_cap_and_zero():
    la = [0.1, 0.2, -0.1, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1]
    out = _run(_df(ret=_ret(la)), u=price.streak(True, 3), d=price.streak(False, 3))
    # up: 1 2 0 1 0 1 2 3 3 (zero return ends a streak; capped at 3)
    assert out["u"].to_list() == [1, 2, 0, 1, 0, 1, 2, 3, 3]
    # down: only row 2 is a down day
    assert out["d"].to_list() == [0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_streak_is_per_sid():
    df = _df(sid=[1, 1, 2, 2], sx=[0, 1, 0, 1], ret=_ret([0.1, 0.1, 0.1, -0.1]))
    out = _run(df, u=price.streak(True, 10))["u"].to_list()
    assert out == [1, 2, 1, 0]  # sid 2 starts again at 1


def test_dist_round():
    df = _df(close=[123.0, 98.0, 1050.0, 5.0, 100.0])
    out = _run(df, d=price.dist_round())["d"].to_list()
    # 123: u=100 -> nearest 100 -> 23/123 ; 98: u=10 -> 100 -> 2/98 ; 1050: u=1000 -> 1000 -> 50/1050 ;
    # 5: u=1 -> 5 -> 0 ; 100 -> 0
    assert out == pytest.approx([23 / 123, 2 / 98, 50 / 1050, 0.0, 0.0])


def test_t43b_registered_and_seed_ret_1_gone():
    for n in [
        "ret_1",
        "rsi_14",
        "bb_z_20",
        "max_ret_21",
        "min_ret_21",
        "close_loc",
        "close_loc_5",
        "range_pct",
        "nr7",
        "inside_day",
        "up_streak",
        "down_streak",
        "dist_round",
    ]:
        assert REG[n].label and REG[n].group == "price", n
    assert REG["ret_1"].fn.__module__ == "stocksense.features.price"
