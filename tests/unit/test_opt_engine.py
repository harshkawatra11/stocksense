"""M5.4c: run_combo on a synthetic F&O panel, option costs, and the VRP condition."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.features.derivs import R_FREE, bs_price
from stocksense.strategies.costs import LegFill, load_deriv_costs, options_cost_frac
from stocksense.strategies.k151_opt import COMBOS, realised_vol_pct, vrp_condition
from stocksense.strategies.options_bt import EntryRule, Leg, monthly_expiries, run_combo, terminal_pnl

E1, E2, E3 = date(2026, 1, 29), date(2026, 2, 26), date(2026, 3, 26)
STRIKES = np.arange(80.0, 125.0, 1.0)


def _spot(d: date) -> float:
    return 100.0 + 0.15 * (d - date(2026, 1, 1)).days + 3 * np.sin((d - date(2026, 1, 1)).days / 4.0)


def _fo() -> pl.DataFrame:
    rows = []
    d = date(2026, 1, 20)
    days = []
    while d <= E3:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    for d in days:
        s = _spot(d)
        for e in (E1, E2, E3, date(2026, 4, 30)):
            if e < d:
                continue
            t = (e - d).days / 365
            rows.append(
                ("NIFTY", "IDF", e, 0.0, "", d, s * (1 + 0.006 * (e - d).days / 30) if e > d else s, s)
            )
            for k in STRIKES:
                for typ in ("CE", "PE"):
                    px = (
                        bs_price(s, float(k), t, R_FREE, 0.2, typ == "CE") if e > d else 9999.0
                    )  # garbage on expiry day
                    rows.append(("NIFTY", "IDO", e, float(k), typ, d, max(px, 0.05), s))
    return pl.DataFrame(
        rows,
        schema=[
            "symbol",
            "instrument_type",
            "expiry_date",
            "strike_price",
            "option_type",
            "date",
            "settle_price",
            "underlying_price",
        ],
        orient="row",
    )


FO = _fo()
RULE = EntryRule()


def test_monthly_expiries():
    assert monthly_expiries(FO)[:3] == [E1, E2, E3]


def test_straddle_cycle_equals_closed_form_and_daily_sums():
    legs = (Leg("CE", -1), Leg("PE", -1))
    res = run_combo(legs, RULE, None, "NIFTY", fo=FO, lot=65.0)
    assert res.cycles.height == 2  # E2 and E3 (the first expiry only opens the first cycle)
    c = res.cycles.row(0, named=True)
    assert c["entry_date"] == date(2026, 1, 30) and c["exit_date"] == E2
    s0, st = c["spot_entry"], c["spot_exit"]
    k = float(STRIKES[np.argmin(np.abs(STRIKES - s0))])
    t = (E2 - c["entry_date"]).days / 365
    ent = (bs_price(s0, k, t, R_FREE, 0.2, True), bs_price(s0, k, t, R_FREE, 0.2, False))
    want = terminal_pnl(legs, (k, k), ent, st) / s0
    assert c["gross"] == pytest.approx(want)  # expiry-day rows (garbage 9999) were not used
    assert c["net"] == pytest.approx(c["gross"] - c["cost"]) and c["cost"] > 0
    day = res.daily.filter((pl.col("date") >= c["entry_date"]) & (pl.col("date") <= c["exit_date"]))
    assert day["ret"].sum() == pytest.approx(c["net"])


def test_covered_call_uses_future_and_calendar_uses_next_expiry():
    cc = run_combo((Leg("FUT", 1), Leg("CE", -1, 1)), RULE, None, "NIFTY", fo=FO, lot=65.0)
    c = cc.cycles.row(0, named=True)
    assert c["gross"] != 0.0
    cal = run_combo((Leg("CE", -1), Leg("CE", 1, expiry="next")), RULE, None, "NIFTY", fo=FO, lot=65.0)
    assert cal.cycles.height == 2  # E2 and E3 cycles; the April expiry is the next month of E3


def test_condition_false_means_flat_and_hedge_runs():
    legs = (Leg("CE", -1), Leg("PE", -1))
    flat = run_combo(legs, EntryRule(condition=lambda d: False), None, "NIFTY", fo=FO, lot=65.0)
    assert flat.cycles.height == 0
    plain = run_combo(legs, RULE, None, "NIFTY", fo=FO, lot=65.0).cycles
    dh = run_combo(legs, EntryRule(delta_hedge=True), None, "NIFTY", fo=FO, lot=65.0).cycles
    assert dh.height == plain.height
    assert dh["cost"][0] > plain["cost"][0]  # futures hedge costs are included
    assert dh["gross"][0] != plain["gross"][0]


def test_options_cost_frac_hand_case():
    c = load_deriv_costs()
    lot, spot = 65.0, 20000.0
    legs = [LegFill("CE", -1, 1.0, 100.0, 0.0)]  # short, expires worthless
    o = c.options
    v = 100.0 * lot
    exch, sebi = v * o["exchange"], v * o["sebi"]
    want = (20.0 + v * o["stt_sell_premium"] + exch + sebi + o["gst"] * (20.0 + exch + sebi)) / (spot * lot)
    assert options_cost_frac(legs, spot, None, lot) == pytest.approx(want)
    long_itm = [LegFill("CE", 1, 1.0, 100.0, 50.0)]
    extra = options_cost_frac(long_itm, spot, None, lot) - options_cost_frac(
        [LegFill("CE", 1, 1.0, 100.0, 0.0)], spot, None, lot
    )
    assert extra == pytest.approx(50.0 * lot * o["stt_exercise"] / (spot * lot))


def test_realised_vol_and_vrp_condition_truncation():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    dates = np.array([date(2026, 1, 1) + timedelta(days=i) for i in range(60)])
    assert realised_vol_pct(close, 10) is None
    rv = realised_vol_pct(close, 40)
    assert rv == pytest.approx(np.std(np.diff(np.log(close[19:41])), ddof=1) * np.sqrt(252) * 100)
    vix = {d: 30.0 for d in dates}
    full = [vrp_condition(dates, close, vix)(d) for d in dates[:41]]
    trunc = [vrp_condition(dates[:41], close[:41], vix)(d) for d in dates[:41]]
    assert full == trunc  # a decision at d never uses later data
    assert vrp_condition(dates, close, {**vix, dates[40]: 5.0})(dates[40]) is False
    assert vrp_condition(dates, close, vix)(dates[40]) is True


def test_every_combo_runs_on_the_synthetic_panel():
    n_ok = 0
    for suffix, _sec, _eq, legs in COMBOS:
        res = run_combo(legs, RULE, None, "NIFTY", fo=FO, lot=65.0)
        assert res.cycles.height >= 1, suffix
        n_ok += 1
    assert n_ok == 57
