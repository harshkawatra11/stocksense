"""PLAN-M M5.4a: the per-stock 1d rules of 151 Trading Strategies (hand numbers, state exits, KNN, HP)."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.marketlab import npx
from stocksense.strategies import base
from stocksense.strategies import exits as X
from stocksense.strategies import indicators as I  # noqa: E741
from stocksense.strategies import k151 as K


def feat(c, h=None, l=None, o=None, v=None, dates=None):  # noqa: E741
    c = np.array(c, float)
    n = len(c)
    h = c + 0.5 if h is None else np.array(h, float)
    l = c - 0.5 if l is None else np.array(l, float)  # noqa: E741
    o = c.copy() if o is None else np.array(o, float)
    f = {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "v": np.full(n, 1000.0) if v is None else np.array(v, float),
        "n": n,
        "atr": npx.rma(I.true_range(h, l, c), 14),
    }
    if dates is not None:
        f["date"] = dates
    return f


def sma(c, k):
    return np.array([np.mean(c[i - k + 1 : i + 1]) if i >= k - 1 else np.nan for i in range(len(c))])


def ma2_series():
    return [100.0] * 20 + [101.0 + k for k in range(10)] + [110.0 - 2 * (k + 1) for k in range(10)]


# ---- moving averages (sec 3.11-3.13) ---------------------------------------------------------------------


def test_ma2_enters_and_exits_at_the_hand_listed_days():
    f = feat(ma2_series())
    sig, stop = K.s_ma2(f)
    # SMA10 = 105.5 > SMA30 = 101.83 on day 29; day 37: 102.7 > 102.1; day 38: 101.0 < 101.83
    assert np.flatnonzero(sig).tolist() == list(range(29, 38))
    assert np.flatnonzero(K.x_ma2(f)).tolist() == [38, 39]
    assert K.follow_rule(f, "k151_ma2", 30, stop[29]) == (39, 90.0, "state")  # out at the open after day 38


def test_ma2_stop_adds_the_two_percent_drop_below_the_previous_close():
    c = [100.0] * 30 + [101.0 + k for k in range(10)] + [100.0, 101.0]
    f = feat(c)
    assert not K.x_ma2(f)[40]  # the averages have not crossed back
    assert K.x_ma2_stop(f)[40]  # 100 < 0.98 * 110
    assert not K.x_ma2_stop(f)[39]


def test_ma3_and_ma1_follow_their_equations_against_a_plain_loop():
    rng = np.random.default_rng(4)
    c = 100 + np.cumsum(rng.normal(0, 1, 90))
    f = feat(c)
    s3, s10, s21 = sma(c, 3), sma(c, 10), sma(c, 21)
    want = np.array([bool(a > b > d) for a, b, d in zip(s3, s10, s21, strict=True)])  # NaN compares False
    assert (K.s_ma3(f)[0] == want).all()
    assert (K.x_ma3(f) == (np.isfinite(s10) & (s3 <= s10))).all()
    s50 = sma(c[:90], 50)
    assert (K.s_ma1_sma(f)[0] == (c > s50)).all() and (K.x_ma1_sma(f) == (c < s50)).all()
    e50 = npx.ema(c, 50)
    assert (K.s_ma1_ema(f)[0] == (c > e50)).all() and (K.x_ma1_ema(f) == (c < e50)).all()


def test_ma_rules_never_hit_test_their_display_stop():
    c = [100.0] * 20 + [101.0 + k for k in range(10)] + [60.0, 59.0]
    f = feat(c)
    sig, stop = K.s_ma2(f)
    # the crash below the display stop is ignored: the exit is the cross, one open later
    j, px, why = K.follow_rule(f, "k151_ma2", 30, stop[29])
    assert why in ("state", "end") and j >= 31


# ---- pivots, channel -----------------------------------------------------------------------------------


def test_pivot_levels_hand_numbers():
    cc, r, s = K.pivot_levels(np.array([110.0, 1]), np.array([100.0, 1]), np.array([105.0, 1]))
    assert (cc[1], r[1], s[1]) == (105.0, 110.0, 100.0)  # C = 105, R = 2C - L, S = 2C - H


def test_pivots_target_fills_at_the_resistance():
    f = feat(
        [105, 107, 109, 108],
        h=[110, 108, 111, 109],
        l=[100, 106, 107, 106],
        o=[104, 105, 107.5, 109],
    )
    sig, stop = K.s_pivots(f)
    assert sig[1] and stop[1] == 100.0  # P 107 > C 105; protective stop at S
    assert K.follow_rule(f, "k151_pivots", 2, stop[1]) == (2, 110.0, "target")


def test_pivots_flip_closes_the_long_at_the_next_open():
    f = feat(
        [105, 107, 104, 103],
        h=[110, 108, 106, 105],
        l=[100, 106, 103, 102],
        o=[104, 105, 106, 103.5],
    )
    sig, stop = K.s_pivots(f)
    # signal day 1, entry day 2; close(2) = 104 < C(2) = (108 + 106 + 107)/3 = 107 -> out at day 3's open
    assert K.follow_rule(f, "k151_pivots", 2, stop[1]) == (3, 103.5, "state")


def channel_mr_series():
    c = [100.0, 102.0] * 10 + [101.0] + [102.0, 100.0] * 5
    h, l = [x + 0.5 for x in c], [x - 0.5 for x in c]  # noqa: E741
    l[20] = 99.0
    h[21] = 102.5
    return feat(c, h=h, l=l)


def test_channel_floor_touch_then_ceiling_touch_is_one_trade():
    f = channel_mr_series()
    sig, stop = K.s_channel_mr(f)
    assert np.flatnonzero(sig).tolist() == [20]  # low 99 <= B_down 100, close 101 holds above it
    assert K.t_channel_mr(f)[20] == 102.0  # B_up = max of the previous 20 closes
    assert K.follow_rule(f, "k151_channel_mr", 21, stop[20]) == (21, 102.0, "target")


def test_channel_breakout_needs_volume_and_exits_below_the_floor():
    c = [100.0, 101.0] * 13 + [105.0, 104.0, 103.0, 102.0, 99.0]
    v = [1000.0] * 26 + [2000.0] + [1000.0] * 4
    f = feat(c, v=v)
    sig, stop = K.s_channel_bo(f)
    assert np.flatnonzero(sig).tolist() == [26]
    assert stop[26] == 100.0  # B_down
    assert np.flatnonzero(K.x_channel_bo(f)).tolist() == [30]  # 99 < B_down
    low_vol = feat(c, v=[1000.0] * 31)
    assert not K.s_channel_bo(low_vol)[0].any()


# ---- follow_state ----------------------------------------------------------------------------------------


def test_follow_state_stop_target_and_end():
    f = feat([100, 100, 100, 100], h=[101, 101, 101, 101], l=[99, 99, 99, 99])
    ex = np.array([False, False, True, False])
    assert X.follow_state(f, 0, 90.0, ex) == (3, 100.0, "state")
    assert X.follow_state(f, 0, 99.5, ex) == (0, 99.5, "stop")
    assert X.follow_state(f, 0, 90.0, ex, target=100.5) == (0, 100.5, "target")
    last = np.array([False, False, False, True])
    assert X.follow_state(f, 0, 90.0, last) == (3, 100.0, "end")
    assert X.follow_state(f, 0, 90.0, np.zeros(4, bool), max_days=2) == (1, 100.0, "time")


# ---- KNN (sec 3.17) --------------------------------------------------------------------------------------


def toy_periodic(reps=40, period=20, seed=2):
    rng = np.random.default_rng(seed)
    p = rng.uniform(100, 110, period)
    v = rng.uniform(1000, 2000, period)
    return np.tile(p, reps), np.tile(v, reps)


def test_knn_recovers_a_repeating_label_with_k_equal_one():
    p, v = toy_periodic()
    yh = K.knn_predict(p, v, horizon=10, t_star=100, k=1, p_w=(5, 10, 20), v_w=(5, 20))
    y = np.r_[p[10:] / p[:-10] - 1, [np.nan] * 10]
    rows = np.arange(150, 780)
    assert np.isfinite(yh[rows]).all()
    assert np.allclose(yh[rows], y[rows], atol=1e-12)


def test_knn_neighbours_are_purged_of_labels_not_yet_known():
    rng = np.random.default_rng(8)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900)))
    v = rng.uniform(1e5, 2e5, 900)
    yh, nb = K.knn_predict(p, v, horizon=10, t_star=200, k=5, return_idx=True)
    rows = np.flatnonzero(np.isfinite(yh))
    assert len(rows) > 300
    for t in rows[::17]:
        assert nb[t].max() <= t - 10  # t' + 10 <= t
        assert nb[t].min() >= t - 10 - 200 + 1  # inside the 200 most recent
    assert K.knn_predict(p, v, t_star=200).shape == (900,)


def test_knn_value_at_t_is_unchanged_when_later_rows_are_removed():
    rng = np.random.default_rng(9)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
    v = rng.uniform(1e5, 2e5, 800)
    full = K.knn_predict(p, v, t_star=150)
    cut = K.knn_predict(p[:600], v[:600], t_star=150)
    assert np.array_equal(np.isnan(full[:600]), np.isnan(cut))
    assert np.allclose(full[:600][np.isfinite(cut)], cut[np.isfinite(cut)], rtol=0, atol=1e-13)


def test_knn_rule_uses_the_one_percent_and_zero_thresholds(monkeypatch):
    yh = np.array([np.nan, 0.02, 0.011, 0.005, -0.001, 0.0])
    monkeypatch.setattr(K, "knn_predict", lambda *a, **k: yh)
    f = feat([100.0] * 6)
    assert K.s_knn(f)[0].tolist() == [False, True, True, False, False, False]
    assert K.x_knn(f).tolist() == [False, False, False, False, True, True]


# ---- Hodrick-Prescott (sec 8.1), one-sided -----------------------------------------------------------------


def test_hp_weights_preserve_a_linear_trend_and_match_a_direct_solve():
    m = 50
    w = K.hp_last_weights(m)
    y = 2.0 + 0.5 * np.arange(m)
    assert w.sum() == pytest.approx(1.0) and w @ y == pytest.approx(y[-1], abs=1e-9)
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=m))
    d = np.zeros((m - 2, m))
    for i in range(m - 2):
        d[i, i : i + 3] = (1, -2, 1)
    tau = np.linalg.solve(np.eye(m) + K.HP_LAMBDA * d.T @ d, y)
    assert w @ y == pytest.approx(tau[-1], abs=1e-8)


def test_hp_one_sided_value_is_unchanged_by_later_months():
    rng = np.random.default_rng(3)
    y = 100 + np.cumsum(rng.normal(0, 1, 90))
    full = K.hp_oneside_series(y)
    assert np.isnan(full[:35]).all() and np.isfinite(full[35:]).all()
    for k in (40, 60, 80):
        assert K.hp_oneside_series(y[: k + 1])[k] == full[k]


def bdays(n, start="2010-01-04"):
    d = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(2 * n, "D"))
    return d[(d.astype("int64") + 3) % 7 < 5][:n]


def test_hp_ma_state_uses_only_months_completed_before_t():
    dates = bdays(1700)
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(dates))))
    state, valid = K.hp_ma_state(dates, c)
    assert valid.sum() > 500
    mon = dates.astype("datetime64[M]")
    same_month = np.r_[False, mon[1:] == mon[:-1]]
    assert (
        state[same_month] == np.r_[False, state[:-1]][same_month]
    ).all()  # moves only at a month's first session
    for cut in (900, 1234, 1500):
        s2, v2 = K.hp_ma_state(dates[:cut], c[:cut])
        assert (s2 == state[:cut]).all() and (v2 == valid[:cut]).all()


def test_hp_ma_signal_and_exit_are_complements_when_valid():
    dates = bdays(1700)
    rng = np.random.default_rng(6)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates))))
    f = feat(c, dates=dates)
    sig, _ = K.s_hp_ma(f)
    ex = K.x_hp_ma(f)
    assert not (sig & ex).any()
    assert (sig | ex).sum() == K.hp_ma_state(dates, c)[1].sum()


# ---- registry ---------------------------------------------------------------------------------------------


def test_every_stock_rule_is_registered_with_a_state_exit_and_its_section():
    reg = base.load_library()
    for key, (name, fn, _kind, sec) in K.STOCK_RULES.items():
        s = reg[key]
        assert s.family == "m-swing" and s.exits == (X.EXIT_STATE,) and s.signal is fn
        assert s.name == name and f"sec {sec}" in s.reference and s.source == "k151"
        assert key in K.STATE_RULES
