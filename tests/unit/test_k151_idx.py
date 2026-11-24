"""PLAN-M M5.4b: index rotation, multi-asset, vol target and macro rules of 151 Trading Strategies."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stocksense.strategies import base
from stocksense.strategies import k151_idx as I  # noqa: E741
from tests.unit.test_k151_stock import bdays


def synth_panel(t=1400, seed=3):
    rng = np.random.default_rng(seed)
    a = len(I.ASSETS)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (t, a)), axis=0))
    o = c * np.exp(rng.normal(0, 0.002, (t, a)))
    return I.Panel(
        bdays(t), I.ASSETS, o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c,
        rng.uniform(10, 30, t), 70 + np.cumsum(rng.normal(0, 0.05, t)), 3 + np.cumsum(rng.normal(0, 0.01, t)),
    )  # fmt: skip


def trunc(p, n):
    return I.Panel(
        p.dates[:n],
        p.names,
        p.open[:n],
        p.high[:n],
        p.low[:n],
        p.close[:n],
        p.vix[:n],
        p.usdinr[:n],
        p.us10y[:n],
    )


def test_hand_numbers():
    assert I.ibs(np.array([110.0]), np.array([100.0]), np.array([105.0]))[0] == 0.5
    assert np.allclose(I.trend_weights(np.array([0.2, 0.1]), np.array([0.2, 0.1])), [1 / 3, 2 / 3])
    assert I.vol_target_weight(30.0) == pytest.approx(0.5) and I.vol_target_weight(10.0) == 1.0
    v = np.array([30.0, 31.5, 36.0])  # w = 0.5, 0.476 (|dw|/w = 0.048 < 0.10: held), 0.4167 (0.167: re-set)
    assert np.allclose(I.vol_target(v), [0.5, 0.5, 0.15 / 0.36])
    assert I.barbell_weight_long(6.0, 20.0, 10.5) == pytest.approx(4.5 / 14)
    f, b = I.cash_and_carry(110.0, 100.0, 365.0, r=0.065)
    assert f == pytest.approx(100 * np.exp(0.065)) and b == pytest.approx((110 - f) / 100)


def test_top_three_of_five_indices_each_month():
    dates = bdays(120)
    t = np.arange(120)[:, None]
    p = 100 * (1 + 0.001 * np.arange(5))[None, :] ** t
    w = I.sector_mom(p, dates, n=20)
    assert np.allclose(w[-1], [0, 0, 1 / 3, 1 / 3, 1 / 3]) and (w[:20] == 0).all()
    ma = I.sector_mom_ma(p, dates, n=20, ma=10)
    assert np.allclose(ma[-1], w[-1])  # all above their SMA
    down = p[::-1].copy()  # falling prices: no pick is above its SMA, so the slots are cash
    assert I.sector_mom_ma(down, dates, n=20, ma=10)[-1].sum() == 0


def test_dual_momentum_goes_to_gold_below_the_200_day_mean():
    dates = bdays(500)
    t = np.arange(500)
    sect = 100 * (1 + 0.001 * np.arange(4))[None, :] ** t[:, None]
    bench = np.where(t < 300, 100.0 + t * 0.2, 160.0 - (t - 300) * 0.4)
    w = I.sector_dual_mom(sect, bench, np.full(500, 5000.0), dates, n=60)
    assert w[250, :-1].sum() == pytest.approx(1.0) and w[250, -1] == 0
    assert w[-1, -1] == 1.0 and w[-1, :-1].sum() == 0


def test_announcement_days_use_the_session_after_a_fomc_date():
    dates = np.array(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"], dtype="datetime64[D]"
    )
    ev = {"fomc": [date(2024, 1, 31)], "rbi_mpc": [date(2024, 2, 5)]}
    assert I.announcement_days(dates, ev).tolist() == [False, False, True, False, True]


def test_macro_events_file_is_pinned():
    ev = I.load_events()
    assert (
        len(ev["fomc"]) == 109 and ev["fomc"][0] == date(2012, 1, 25) and ev["fomc"][-1] == date(2025, 9, 17)
    )
    assert ev["fomc"] == sorted(set(ev["fomc"])) and ev["rbi_mpc"] == []
    import tomllib

    raw = tomllib.loads(I.EVENTS_PATH.read_text(encoding="utf-8"))
    assert (
        raw["fomc"]["source_url"].startswith("https://www.federalreserve.gov")
        and raw["rbi_mpc"]["source_url"]
    )


def test_index_book_hand_case_and_costs():
    open_ = np.array([[100.0], [110.0], [121.0], [133.1]])
    w = np.array([[1.0], [1.0], [1.0], [1.0]])  # decided at each close
    out = I.index_book(open_, w, bdays(4), cost1=0.001)
    # held weight on day s = decided on s-1; return dated s = open[s+1]/open[s] - 1
    assert out["ret_gross"].to_list() == pytest.approx([0.0, 0.1, 0.1])
    assert out["net"].to_list() == pytest.approx([0.0, 0.1 - 0.001, 0.1])  # the entry is paid on day 1
    assert out["gross"].to_list() == [0.0, 1.0, 1.0]


def test_book_charges_turnover_on_both_sides():
    open_ = np.full((5, 1), 100.0)
    w = np.array([[0.0], [1.0], [0.0], [0.0], [0.0]])
    out = I.index_book(open_, w, bdays(5), cost1=0.002)
    assert out["net"].to_list() == pytest.approx([0.0, 0.0, -0.002, -0.002])  # in on day 2, out on day 3


def test_macro_mom_respects_the_three_month_minimum_hold():
    p = synth_panel()
    w = I.macro_mom(p.col(I.MAIN), p.usdinr, p.us10y, p.dates)
    starts = np.flatnonzero(I.month_starts(p.dates))
    changes = [t for t in starts[1:] if w[t] != w[t - 1]]
    assert np.diff([starts.tolist().index(t) for t in changes]).min() >= 3 if len(changes) > 1 else True


@pytest.mark.parametrize("key", [k for k in I.INDEX_RULES if k not in ("k151_hedging_pressure",)])
def test_weights_at_t_do_not_change_when_later_rows_are_removed(key):
    p = synth_panel()
    ev = {"fomc": [date(2016, 3, 16), date(2017, 6, 14)], "rbi_mpc": [date(2016, 4, 5), date(2018, 2, 7)]}
    fn = I.INDEX_RULES[key][1]
    full, cut = fn(p, events=ev), fn(trunc(p, 1200), events=ev)
    assert np.allclose(full[:1200], cut, atol=1e-12), key
    assert np.abs(cut[-300:]).sum() > 0 or key in ("k151_announcement",), key


def test_hedging_pressure_rule_and_truncation():
    rng = np.random.default_rng(1)
    n = 400
    fl, fs = rng.uniform(1e4, 2e4, n), rng.uniform(1e4, 2e4, n)
    cl, cs = rng.uniform(1e4, 2e4, n), rng.uniform(1e4, 2e4, n)
    a = I.hedging_pressure_long(fl, fs, cl, cs)
    b = I.hedging_pressure_long(fl[:300], fs[:300], cl[:300], cs[:300])
    assert (a[:300] == b).all() and a.any() and not a[:125].any()


def test_every_index_key_is_registered():
    reg = base.load_library()
    for key, (name, _fn, sec, _sd, _d) in I.INDEX_RULES.items():
        s = reg[key]
        assert (
            s.family == "m-index" and s.timing is not None and s.name == name and f"sec {sec}" in s.reference
        )
    assert "k151_cash_carry" in reg and "k151_gold_diversify" in reg
