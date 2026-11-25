"""M5.7: the strategy book simulators (strategies/portfolio.py) on hand-built bars."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import ProductCosts
from stocksense.costs.india import trade_cost
from stocksense.costs.slippage import SlippageTable
from stocksense.strategies import portfolio as P
from stocksense.strategies.data import DailyFrame
from stocksense.strategies.explore import SymCtx

N = 14
START = date(2026, 1, 5)


def _sessions(n: int = N) -> list[date]:
    out, d = [], START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


SESSIONS = _sessions()
CAL = TradingCalendar.from_dates(SESSIONS)
ZERO = ProductCosts(
    stt_buy=0.0, stt_sell=0.0, stamp_buy=0.0, exchange=0.0, sebi=0.0, gst=0.0, dp_per_scrip_sell_inr=0.0,
    brokerage_rate=0.0, brokerage_cap_inr=0.0, brokerage_min_inr=0.0,
)  # fmt: skip


def zero_cost_settings(s):
    return s.model_copy(update={"costs": s.costs.model_copy(update={"delivery": ZERO, "intraday": ZERO})})


def view(sym="AAA", **over: dict[int, float]) -> P.SymView:
    """Flat 100 bars (high 101, low 99) with per-index overrides like o={6: 100}."""
    a = {k: np.full(N, v) for k, v in (("o", 100.0), ("h", 101.0), ("l", 99.0), ("c", 100.0))}
    for k, d in over.items():
        for i, v in d.items():
            a[k][i] = v
    fr = DailyFrame(
        np.array(SESSIONS, dtype="datetime64[D]"), a["o"], a["h"], a["l"], a["c"], np.full(N, 1e6)
    )
    ctx = SymCtx(sym, fr, np.full(N, 100.0), np.full(N, 0.02), np.full(N, 5e8))
    slim = {**a, "atr": np.full(N, 2.0), "sma5": np.full(N, 100.0), "ema20": np.full(N, 100.0), "n": N}
    return P.SymView(ctx, slim)


def cand(sym="AAA", sig=4, exit_key="rr2_20", stop=95.0, priority=1.0) -> P.Candidate:
    return P.Candidate(sym, 1, SESSIONS[sig], "t_hand", exit_key, priority, stop)


def panel(settings, **views) -> P.PanelView:
    return P.PanelView(views, SlippageTable.load_latest(settings))


def sim(settings, cands, pnl, cfg=None, start=0, end=N - 1):
    by: dict[date, list[P.Candidate]] = {}
    for c in cands:
        by.setdefault(c.t_sig, []).append(c)
    cfg = cfg or P.SimConfig(100_000.0, 2, slip_scale=0.0)
    return P.simulate_daily(by, pnl, cfg, settings, CAL, SESSIONS[start], SESSIONS[end])


def test_hand_case_five_percent_on_the_exit_day(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    # entry at the open of day 5 (100), stop 95, 2R target 110 touched on day 6
    out = sim(s, [cand()], panel(s, AAA=view(h={6: 111.0}, c={6: 105.0})))
    nets = dict(zip(out["date"].to_list(), out["net"].to_list(), strict=True))
    assert nets[SESSIONS[6]] == pytest.approx(0.05)  # 500 shares x (110 - 100) on a 100,000 book
    assert all(abs(v) < 1e-12 for d, v in nets.items() if d != SESSIONS[6])
    day6 = out.filter(pl.col("date") == SESSIONS[6]).row(0, named=True)
    assert day6["n_pos"] == 0 and day6["gross"] == 0.0
    assert out.filter(pl.col("date") == SESSIONS[5])["n_pos"][0] == 1


def test_costs_and_slippage_lower_the_return(tmp_data_root):
    v = view(h={6: 111.0}, c={6: 105.0})
    free = sim(zero_cost_settings(tmp_data_root), [cand()], panel(tmp_data_root, AAA=v))["net"].sum()
    real = sim(tmp_data_root, [cand()], panel(tmp_data_root, AAA=v), P.SimConfig(100_000.0, 2))["net"].sum()
    assert real < free - 0.001
    half = trade_cost("BUY", "delivery", 50_000.0, tmp_data_root.costs).total
    assert half > 0  # the statutory charge exists, so the drop is not only slippage


def test_a_gap_through_the_stop_fills_at_the_open(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    v = view(o={7: 90.0}, h={7: 91.0}, l={7: 88.0}, c={7: 89.0})
    out = sim(s, [cand()], panel(s, AAA=v))
    day7 = out.filter(pl.col("date") == SESSIONS[7])["net"][0]
    assert day7 == pytest.approx(500 * (90 - 100) / 100_000)  # out at the gap open of 90, not at the stop


def test_slots_limit_entries_and_priority_decides(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    pv = panel(s, AAA=view("AAA"), BBB=view("BBB"), CCC=view("CCC"))
    cs = [cand("AAA", priority=1.0), cand("BBB", priority=3.0), cand("CCC", priority=2.0)]
    out = sim(s, cs, pv)
    assert out.filter(pl.col("date") == SESSIONS[5])["n_pos"][0] == 2  # two slots
    recs = P.run_daily(
        {SESSIONS[4]: cs}, pv, P.SimConfig(100_000.0, 2, slip_scale=0.0), s, CAL, SESSIONS[0], SESSIONS[N - 1]
    ).trades
    assert {r["sym"] for r in recs} == {"BBB", "CCC"}  # AAA had the lowest priority


def test_a_stock_already_held_is_not_bought_twice(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    pv = panel(s, AAA=view())
    out = sim(s, [cand(sig=4), cand(sig=5)], pv, P.SimConfig(100_000.0, 3, slip_scale=0.0))
    assert out["n_pos"].max() == 1


def test_bad_risk_is_skipped_like_the_engine(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    pv = panel(s, AAA=view())
    assert sim(s, [cand(stop=70.0)], pv)["n_pos"].max() == 0  # 30% risk
    assert sim(s, [cand(stop=101.0)], pv)["n_pos"].max() == 0  # stop above the open


def test_nothing_after_the_end_date_is_read(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    a = view(h={9: 111.0})  # the target would be hit on day 9
    b = view(h={9: 111.0}, c={9: 400.0}, o={9: 400.0}, l={9: 399.0})  # wild after the cut
    cut = 7
    ra = sim(s, [cand(exit_key="rr3_40", stop=95.0)], panel(s, AAA=a), end=cut)
    rb = sim(s, [cand(exit_key="rr3_40", stop=95.0)], panel(s, AAA=b), end=cut)
    assert ra["net"].to_list() == rb["net"].to_list()
    assert ra["n_pos"][-1] == 0  # the open trade is closed at the last bar of the window


def test_state_exit_and_overnight_exit(tmp_data_root):
    s = zero_cost_settings(tmp_data_root)
    v = view(c={8: 100.0})
    v.slim["exit_state"] = np.arange(N) == 7  # the condition is true at the close of day 7
    v.slim["state_tgt"] = None
    v.slim["use_stop"] = False
    v.slim["o"][8] = 104.0
    v.ctx.frame.open[8] = 104.0
    out = sim(s, [cand(exit_key="state_exit")], panel(s, AAA=v))
    assert out.filter(pl.col("date") == SESSIONS[8])["net"][0] == pytest.approx(500 * 4 / 100_000)
    w = view()
    w.ctx.frame.open[6] = 103.0
    w.slim["o"][6] = 103.0
    out2 = sim(s, [cand(exit_key="next_open", stop=95.0)], panel(s, AAA=w))
    assert out2.filter(pl.col("date") == SESSIONS[6])["net"][0] == pytest.approx(500 * 3 / 100_000)


def test_timing_enters_at_the_open_and_pays_per_change(tmp_data_root):
    bars = pl.DataFrame(
        {
            "date": SESSIONS[:6],
            "open": [100.0, 100.0, 102.0, 102.0, 104.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    pos = pl.DataFrame({"date": SESSIONS[:6], "position": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]})
    s = zero_cost_settings(tmp_data_root)
    out = P.simulate_timing(pos, bars, P.SimConfig(100_000.0, slip_scale=0.0), s)
    r = dict(zip(out["date"].to_list(), out["net"].to_list(), strict=True))
    assert r[SESSIONS[1]] == pytest.approx(101 / 100 - 1)  # in at the open (100), close 101
    assert r[SESSIONS[2]] == pytest.approx(102 / 101 - 1)  # held: close to close
    assert r[SESSIONS[3]] == pytest.approx(0.0)  # decided out at the day-2 close: overnight to the open, flat
    out2 = P.simulate_timing(pos, bars, P.SimConfig(100_000.0, slip_scale=1.0), s)
    assert out2["costs_frac"].sum() == pytest.approx(2 * 2e-4)  # two changes of 1.0 at 2 bps
