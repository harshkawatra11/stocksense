"""PLAN-M M5.4a: cross-sectional scores of 151 Trading Strategies (hand numbers, truncation, loader)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from stocksense.data import io
from stocksense.strategies import base
from stocksense.strategies import k151 as K
from tests.fixtures.synth import _frame
from tests.unit.test_k151_stock import bdays


def walk(t=1500, n=24, seed=11):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.02, (t, n))
    return 100 * np.exp(np.cumsum(r, axis=0))


def rand_wide(t=1500, n=24, seed=11):
    p = walk(t, n, seed)
    rng = np.random.default_rng(seed + 1)
    ret = np.vstack([np.full((1, n), np.nan), p[1:] / p[:-1] - 1])
    fno = np.zeros((t, n), bool)
    fno[:, :14] = True
    return K.Wide.make(
        p,
        dates=bdays(t),
        fno=fno,
        industry=np.arange(n) % 4,
        mkt=np.nan_to_num(np.nanmean(ret, axis=1)),
        med_turn=rng.uniform(1e7, 1e9, (t, n)),
        volume=rng.uniform(1e5, 2e5, (t, n)),
        oi=rng.uniform(1e5, 2e5, (t, n)),
        phi=rng.uniform(0.99, 1.01, (t, n)),
        call_iv=rng.uniform(0.2, 0.4, (t, n)),
        put_iv=rng.uniform(0.2, 0.4, (t, n)),
    )


# ---- hand numbers ------------------------------------------------------------------------------------------


def test_momentum_cum_is_p_21_over_p_252_minus_one():
    t = 300
    p = np.full((t, 1), 110.0)
    p[t - 1 - 252, 0], p[t - 1 - 21, 0] = 100.0, 120.0
    assert K.f_mom_cum(K.Wide.make(p))[-1, 0] == pytest.approx(0.20)


def test_momentum_riskadj_is_mean_over_sample_std_of_the_twelve_months():
    t = 300
    r = {m: (0.1 if m % 2 else 0.0) for m in range(1, 13)}  # monthly returns of formation months 1..12
    p = np.full((t, 1), 100.0)
    level = 100.0
    for m in range(12, 0, -1):  # build from the oldest knot (month 13 start) to the newest
        level *= 1 + r[m]
        p[t - 1 - 21 * m, 0] = level
    p[t - 1 - 21 * 13, 0] = 100.0
    want = 0.05 / np.sqrt(12 * 0.0025 / 11)
    assert K.f_mom_riskadj(K.Wide.make(p))[-1, 0] == pytest.approx(want, rel=1e-9)


def test_low_vol_uses_the_sample_std_with_ddof_one():
    sd = K.roll_std(np.array([0.01, -0.01, 0.01, -0.01]), 4)[-1]
    assert sd == pytest.approx(0.011547005, rel=1e-6)
    w = K.Wide.make(100 * np.cumprod(1 + np.tile([0.01, -0.01], 200))[:, None])
    assert K.f_low_vol(w)[-1, 0] < 0 and K.f_low_vol_126(w)[-1, 0] < 0


def test_skewness_of_one_spike_is_three_quarters():
    assert K.roll_skew(np.array([1.0, 0, 0, 0]), 4)[-1] == pytest.approx(0.75)
    assert K.roll_skew(np.array([1.0, -1, 1, -1]), 4)[-1] == pytest.approx(0.0, abs=1e-12)


def test_demeaned_rank_average_of_two_factors():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[3.0, 1.0, 2.0]])
    out = K.demeaned_rank_mean([a, b], np.ones((1, 3), bool))
    assert out[0].tolist() == [0.0, -0.5, 0.5]  # (-1+1)/2, (0-1)/2, (1+0)/2


def test_multi_cluster_demeaning_and_the_other_bucket():
    r = np.array([[0.02, 0.00, 0.05, 0.01, 0.03]])
    ok = np.ones((1, 5), bool)
    grp = np.array([0, 0, 1, 1, 1])
    got = r - K.cluster_wmean(r, None, ok, grp)  # the 2-member industry is merged into OTHER (alone)
    assert np.allclose(got[0], [0.01, -0.01, 0.02, -0.02, 0.0])
    r2 = np.array([[0.02, 0.00, 0.10]])  # a small industry and an unclassified name share OTHER
    got2 = r2 - K.cluster_wmean(r2, None, np.ones((1, 3), bool), np.array([0, 0, -1]))
    assert np.allclose(got2[0], [-0.02, -0.04, 0.06])


def test_weighted_regression_hand_case_and_equal_weight_limit():
    r = np.array([[0.02, 0.00]])
    ok = np.ones((1, 2), bool)
    grp = np.array([0, 0])
    z = np.array([[1.0, 4.0]])
    mean = K.cluster_wmean(r, z, ok, grp, min_size=2)
    assert mean[0, 0] == pytest.approx(0.004)
    eps = r - mean
    assert np.allclose(eps[0], [0.016, -0.004])
    rt = z * eps
    assert np.allclose(rt[0], [0.016, -0.016]) and rt.sum() == pytest.approx(0.0, abs=1e-15)  # eq. 318
    zc = np.full((1, 2), 3.0)
    assert np.allclose(zc * (r - K.cluster_wmean(r, zc, ok, grp, 2)), 3.0 * (r - r.mean()))


def test_statarb_weights_are_dollar_neutral_and_match_the_hand_case():
    assert K.statarb_weights(np.array([1.0, 3.0]), np.array([1.0, 1.0])).tolist() == [-1.0, 1.0]
    rng = np.random.default_rng(0)
    w = K.statarb_weights(rng.normal(size=50), rng.uniform(1e-4, 4e-4, 50))
    assert abs(w.sum()) < 1e-9 * np.abs(w).max()


def test_friday_anchor_holds_a_score_from_the_last_session_before_the_week():
    days = "2024-03-21 2024-03-22 2024-03-25 2024-03-26 2024-03-27 2024-03-28 2024-04-01 2024-04-02"
    d = np.array(days.split(), dtype="datetime64[D]")  # Fri 03-29 had no session
    a = K.friday_anchor_rows(d)
    assert a.tolist() == [-1, 1, 1, 1, 1, 1, 5, 5]  # Mon 04-01 holds Thu 03-28: Good Friday had no session


def test_lt_value_uses_the_window_5_5_to_4_5_years_back():
    t = 1500
    p = np.full((t, 1), 100.0)
    p[t - 1 - 1386 : t - 1134 + 1, 0] = 150.0  # exactly the paper's window
    p[t - 1, 0] = 120.0
    assert K.f_lt_value(K.Wide.make(p))[-1, 0] == pytest.approx(150.0 / 120.0)


def test_ou_score_matches_a_direct_ar1_fit_and_is_nan_without_mean_reversion():
    rng = np.random.default_rng(2)
    x = np.zeros(600)
    for i in range(1, 600):
        x[i] = 0.2 + 0.9 * x[i - 1] + 0.1 * rng.normal()
    price = np.exp(x)[:, None]
    s = K.ou_score(np.log(price))[-1, 0]
    win = x[-252:]
    b, c = np.polyfit(win[:-1], win[1:], 1)
    e = win[1:] - (c + b * win[:-1])
    want = (c / (1 - b) - win[-1]) / (e.std(ddof=2) / np.sqrt(1 - b * b))
    assert 0 < b < 1 and s == pytest.approx(want, rel=1e-5)
    explosive = (0.5 * 1.03 ** np.arange(600) + 0.01 * rng.normal(size=600))[
        :, None
    ]  # b > 1: not mean reverting
    assert np.isnan(K.ou_score(explosive)[-1, 0])


def test_voltgt_scale_uses_only_returns_before_the_day_and_never_levers():
    r = np.tile([0.01, -0.01], 300)
    s = K.voltgt_scale(r, 0.12)
    ann = np.std(r[-253:-1], ddof=1) * np.sqrt(252)
    assert s[-1] == pytest.approx(0.12 / ann, rel=1e-6) and s[-1] < 1
    r2 = r.copy()
    r2[-1] = 5.0  # today's return does not move today's scale
    assert K.voltgt_scale(r2, 0.12)[-1] == s[-1]
    assert K.voltgt_scale(r * 0.01, 0.12)[-1] == 1.0  # a quiet book is not levered


def test_bernoulli_naive_bayes_learns_a_keyword():
    x = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 0]], float)
    y = np.array([1, 1, 1, -1, -1, -1])
    m = K.bernoulli_nb_fit(x, y)
    p = K.bernoulli_nb_predict(m, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert p[0] > 0.5 > p[1]


def test_nb_news_waits_for_data(tmp_data_root):
    spec = base.load_library()["k151_nb_news"]
    out = spec.score(tmp_data_root, date(2026, 1, 1), date(2026, 2, 1))
    assert out.height == 0 and out.columns == ["date", "sid", "score"]


def test_near_next_ratio_skips_expiring_contracts():
    fo = pl.DataFrame(
        {
            "date": [date(2024, 8, 1)] * 3,
            "symbol": ["AAA"] * 3,
            "expiry_date": [date(2024, 8, 2), date(2024, 8, 29), date(2024, 9, 26)],
            "settle_price": [100.0, 101.0, 102.0],
        }
    )
    out = K.near_next_ratio(fo)  # the 08-02 contract has 1 day left (< MIN_DAYS): near = 08-29
    assert out["phi"].to_list() == [pytest.approx(101.0 / 102.0)]


def test_implied_vol_score_is_change_in_call_iv_minus_change_in_put_iv():
    c = np.full((30, 1), 0.30)
    p = np.full((30, 1), 0.30)
    c[-1], p[-1] = 0.36, 0.33
    assert K.delta_iv_score(c, p)[-1, 0] == pytest.approx(0.03)


# ---- truncation: every score at t is unchanged when later rows are removed ---------------------------------

_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
CUT = 1400


def both(key):
    if key not in _CACHE:
        w = rand_wide()
        fn = K.XS[key][0]
        _CACHE[key] = (fn(w), fn(w.truncate(CUT)))
    return _CACHE[key]


@pytest.mark.parametrize("key", list(K.XS))
def test_score_at_t_does_not_change_when_later_rows_are_removed(key):
    full, cut = both(key)
    a = full[:CUT]
    assert a.shape == cut.shape
    assert np.array_equal(np.isnan(a), np.isnan(cut)), key
    ok = np.isfinite(cut)
    assert np.allclose(a[ok], cut[ok], rtol=1e-9, atol=1e-12), key
    assert ok[-60:].any(), f"{key}: no finite value near the cut, the check would be vacuous"


def test_every_score_key_is_registered_with_its_section_in_the_reference():
    reg = base.load_library()
    for key, (_, _, name, sec, _) in K.XS.items():
        s = reg[key]
        assert s.family == "m-xs" and s.kind == "xs_factor" and s.score is not None and s.name == name
        assert f"sec {sec}" in s.reference
    assert reg["k151_nb_news"].family == "m-xs"


# ---- loader: stored panel to a (date, sid, score) frame -----------------------------------------


def test_loader_builds_the_score_frame_and_respects_end(tmp_data_root):
    from stocksense.data.catalog import get_spec

    s = tmp_data_root
    t, n = 700, 10
    dates = bdays(t)
    ds = [date.fromisoformat(str(d)) for d in dates]
    p = walk(t, n, 3)
    pd_rows = {
        c: [] for c in ("sid", "date", "symbol", "series", "close", "adj_close", "adj_open", "adj_volume")
    }
    pd_rows.update({"turnover_inr": [], "industry": []})
    for j in range(n):
        for i in range(t):
            for k, v in (
                ("sid", j + 1), ("date", ds[i]), ("symbol", f"S{j:02d}"), ("series", "EQ"),
                ("close", float(p[i, j])), ("adj_close", float(p[i, j])), ("adj_open", float(p[i, j])),
                ("adj_volume", 1e5), ("turnover_inr", float(p[i, j]) * 1e5), ("industry", f"IND{j % 2}"),
            ):  # fmt: skip
                pd_rows[k].append(v)
    io.upsert(s, "panel_daily", _frame(get_spec("panel_daily").schema, pd_rows))
    uni = {
        "date": [d for d in ds for _ in range(n)],
        "sid": [j + 1 for _ in ds for j in range(n)],
        "med_turn_60": [1e7 * (j + 1) for _ in ds for j in range(n)],
        "u500": [True] * (t * n),
        "u200": [True] * (t * n),
        "in_fno": [True] * (t * n),
    }
    io.upsert(s, "panel_universe", _frame(get_spec("panel_universe").schema, uni))
    mk = {"date": ds, "ew500_ret": [0.0004] * t}
    io.upsert(s, "panel_market", _frame(get_spec("panel_market").schema, mk))

    reg = base.load_library()
    d1, d2 = ds[400], ds[650]
    for key in ("k151_mom_cum", "k151_low_vol", "k151_mr_multi", "k151_fut_trend"):
        a = reg[key].score(s, d1, d2)
        b = reg[key].score(s, d1, ds[500])
        assert a.columns == ["date", "sid", "score"] and a.height > 0, key
        assert a["date"].max() <= d2 and a["date"].min() >= d1
        assert b["date"].max() <= ds[500]
        joined = b.join(a, on=["date", "sid"], suffix="_full")
        assert joined.height == b.height, key  # past-only: the shorter run is a subset ...
        assert np.allclose(joined["score"].to_numpy(), joined["score_full"].to_numpy(), atol=1e-6), key
