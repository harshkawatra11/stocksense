from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.costs.india import trade_cost
from stocksense.costs.slippage import SlippageTable, slip_frac
from stocksense.data import io
from stocksense.strategies import base, explore
from stocksense.strategies.data import DailyFrame, NiftyTable
from stocksense.strategies.exits import EXITS_1D

N = 40
DATES = np.datetime64("2026-01-01") + np.arange(N)
NOTIONAL = 100_000.0


def hand_frame(**over: dict[int, float]) -> DailyFrame:
    """Flat 100 bars (high 101, low 99) with per-index overrides like o={6: 100}."""
    arr = {k: np.full(N, v) for k, v in (("o", 100.0), ("h", 101.0), ("l", 99.0), ("c", 100.0))}
    for k, d in over.items():
        for i, v in d.items():
            arr[k][i] = v
    return DailyFrame(DATES, arr["o"], arr["h"], arr["l"], arr["c"], np.full(N, 1e6))


def hand_spec(signal_at: list[int], stop: float = 95.0, key: str = "t_hand") -> base.StrategySpec:
    def sig(f):
        s = np.zeros(f["n"], bool)
        s[signal_at] = True
        return s, np.full(f["n"], stop)

    return base.StrategySpec(
        key=key,
        name="Hand " + key,
        source="tier",
        tier="none",
        family="m-swing",
        timeframe="1d",
        kind="trend",
        direction="long",
        exits=EXITS_1D,
        default_exit="rr2_20",
        signal=sig,
    )


def hand_ctx(frame: DailyFrame) -> explore.SymCtx:
    n = len(frame)
    return explore.SymCtx("HAND", frame, np.full(n, 100.0), np.full(n, 0.02), np.full(n, 5e8))


def run(frame, spec, exit_key="rr2_20", settings=None, clock=None, **kw):
    nifty = NiftyTable.from_close(frame.date, frame.close)
    return explore.run_symbol_1d(
        "HAND", spec, exit_key, settings, clock, warmup=0, ctx=hand_ctx(frame), nifty=nifty, **kw
    )


def ep(i: int) -> int:
    return int(DATES[i].astype("datetime64[s]").astype(np.int64))


def test_entry_is_next_open_and_target_is_two_r(tmp_data_root, fixed_clock):
    f = hand_frame(o={6: 100.0}, h={7: 111.0})
    (t,) = run(f, hand_spec([5]), settings=tmp_data_root, clock=fixed_clock)
    assert (t.t_sig, t.t_in, t.t_out) == (ep(5), ep(6), ep(7))
    assert (t.p_in, t.stop, t.target, t.p_out) == (100.0, 95.0, 110.0, 110.0)
    assert (t.status, t.reason, t.days, t.side) == ("CLOSED", "target", 2, 1)
    assert t.gross_pct == pytest.approx(10.0)
    assert pytest.approx((0.10 - t.cost_pct / 100) * 100 / 5) == t.R
    assert t.net_pct == pytest.approx(t.gross_pct - t.cost_pct)


def test_signal_close_is_not_the_entry_price(tmp_data_root, fixed_clock):
    f = hand_frame(c={5: 90.0}, o={6: 100.0})  # close 90 at the signal, next open 100
    (t,) = run(f, hand_spec([5]), settings=tmp_data_root, clock=fixed_clock)
    assert t.p_in == 100.0 and t.t_in == ep(6)


def test_gap_through_the_stop_fills_at_the_open(tmp_data_root, fixed_clock):
    f = hand_frame(o={6: 100.0, 7: 90.0}, h={7: 91.0}, l={7: 88.0}, c={7: 89.0})
    (t,) = run(f, hand_spec([5]), settings=tmp_data_root, clock=fixed_clock)
    assert (t.reason, t.p_out, t.t_out) == ("gap_stop", 90.0, ep(7))
    assert t.gross_pct == pytest.approx(-10.0) and t.R < -2


def test_one_position_at_a_time_and_bad_risk_is_skipped(tmp_data_root, fixed_clock):
    f = hand_frame(o={6: 100.0}, h={7: 111.0})
    trades = run(f, hand_spec([5, 6, 7, 12]), settings=tmp_data_root, clock=fixed_clock)
    assert [t.t_sig for t in trades] == [ep(5), ep(12)]  # 6 and 7 were inside the first trade
    assert run(f, hand_spec([5], stop=70.0), settings=tmp_data_root, clock=fixed_clock) == []  # risk 30%
    assert (
        run(f, hand_spec([5], stop=101.0), settings=tmp_data_root, clock=fixed_clock) == []
    )  # stop above open


def test_a_signal_on_the_last_bar_is_a_signal_with_no_exit(tmp_data_root, fixed_clock):
    f = hand_frame(c={N - 1: 104.0})
    (t,) = run(f, hand_spec([N - 1]), settings=tmp_data_root, clock=fixed_clock)
    assert t.status == "SIGNAL" and t.p_in == 104.0 and t.t_in == ep(N - 1) + 86400
    assert t.target == pytest.approx(104.0 + 2 * 9.0) and t.t_out is None and t.R is None


def test_unfinished_time_exit_is_open_not_closed(tmp_data_root, fixed_clock):
    f = hand_frame(o={N - 4: 100.0})
    (t,) = run(f, hand_spec([N - 5]), exit_key="time10", settings=tmp_data_root, clock=fixed_clock)
    assert t.status == "OPEN" and t.reason == "open" and t.t_out == ep(N - 1) and t.days == 4


def test_cost_is_statutory_plus_slippage_on_both_sides(tmp_data_root, fixed_clock):
    f = hand_frame(o={6: 100.0}, h={7: 111.0})
    (t,) = run(f, hand_spec([5]), settings=tmp_data_root, clock=fixed_clock, notional=NOTIONAL)
    slip = SlippageTable.load_latest(tmp_data_root)
    asof = date(2026, 1, 6)
    stat = (
        trade_cost("BUY", "delivery", NOTIONAL, tmp_data_root.costs).total
        + trade_cost("SELL", "delivery", NOTIONAL, tmp_data_root.costs).total
    ) / NOTIONAL
    want = (
        stat
        + slip_frac(100, 0.02, NOTIONAL, 5e8, "open", slip, asof)
        + slip_frac(100, 0.02, NOTIONAL, 5e8, "now", slip, asof)
    )
    assert t.cost_pct / 100 == pytest.approx(want, rel=1e-12)


def test_only_completed_bars_are_used(tmp_data_root):
    days = [date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)]
    rows = [(1, "AAA", d, "EQ", 5, 10.0, 11.0, 9.0, 10.5, 200.0) for d in days]
    cols = [
        "sid",
        "symbol",
        "date",
        "series",
        "volume",
        *[f"adj_{k}" for k in ("open", "high", "low", "close")],
    ]
    io.upsert(tmp_data_root, "panel_daily", pl.DataFrame(rows, schema=[*cols, "adj_volume"], orient="row"))
    cal = TradingCalendar.from_dates(days)
    clock = FixedClock(datetime(2026, 11, 19, 11, 0, tzinfo=IST))  # the 19th is still trading
    spec = hand_spec([])

    def everything(f):
        return np.ones(f["n"], bool), f["c"] - 1.0

    spec = base.StrategySpec(**{**spec.__dict__, "signal": everything})
    nifty = NiftyTable.from_close(np.array(days, dtype="datetime64[D]"), np.array([1.0, 2.0, 3.0, 4.0]))
    trades = explore.run_symbol_1d("AAA", spec, "time5", tmp_data_root, clock, warmup=0, nifty=nifty, cal=cal)
    assert trades and max(t.t_sig for t in trades) == int(np.datetime64("2026-11-18", "s").astype(np.int64))
    assert trades[-1].status == "SIGNAL"
    assert all(t.t_in <= int(np.datetime64("2026-11-19", "s").astype(np.int64)) for t in trades)


def test_expected_max_sharpe_null():
    assert explore.expected_max_sharpe_null(1, 252) == pytest.approx(0.0, abs=1e-9)
    vals = [explore.expected_max_sharpe_null(n, 252) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals) and vals[0] > 0
    assert explore.expected_max_sharpe_null(100, 1000) < explore.expected_max_sharpe_null(100, 250)


def test_summarize_hand_list():
    def tr(t_in, r, pct, days=3, status="CLOSED"):
        return explore.Trade(
            "X",
            "k",
            "e",
            1,
            t_in,
            t_in,
            100.0,
            95.0,
            None,
            t_in,
            100.0,
            "stop",
            r,
            pct,
            0.1,
            pct,
            days,
            status,
            True,
            None,
        )

    d = lambda y, m: int(np.datetime64(f"{y}-{m:02d}-10", "s").astype(np.int64))  # noqa: E731
    ts = [
        tr(d(2024, 1), 2.0, 10.0),
        tr(d(2024, 2), -1.0, -5.0),
        tr(d(2024, 9), -1.0, -5.0),
        tr(d(2024, 10), -1.0, -5.0),
        tr(d(2024, 11), 1.0, 5.0),
        tr(d(2024, 12), 0.5, 2.0, status="OPEN"),
    ]
    s = explore.summarize(ts, date(2024, 7, 1))
    assert (s["n"], s["win"], s["totR"], s["avgR"]) == (5, 40.0, 0.0, 0.0)
    assert s["pf"] == pytest.approx(3.0 / 3.0) and s["max_consec_losses"] == 3 and s["avg_hold"] == 3
    assert (
        s["is"] == {"n": 2, "avgR": 0.5} and s["oos"]["n"] == 3 and s["oos"]["avgR"] == pytest.approx(-1 / 3)
    )
    assert explore.summarize([])["n"] == 0


def test_trades_for_shape_and_cache(tmp_data_root, fixed_clock, monkeypatch):
    spec = hand_spec([5, 20], key="t_hand")
    monkeypatch.setattr(base, "REGISTRY", {spec.key: spec})
    monkeypatch.setattr(base, "load_library", lambda: base.REGISTRY)
    n = 300  # trades_for needs the 260-bar history
    dates = np.datetime64("2025-01-01") + np.arange(n)
    f = DailyFrame(
        dates, np.full(n, 100.0), np.full(n, 101.0), np.full(n, 99.0), np.full(n, 100.0), np.full(n, 1e6)
    )
    ctx = explore.make_ctx("HAND", f, np.full(n, 100.0))

    def sig(ff):
        s = np.zeros(ff["n"], bool)
        s[[240, 299]] = True
        return s, np.full(ff["n"], 95.0)

    spec2 = base.StrategySpec(**{**spec.__dict__, "signal": sig})
    monkeypatch.setattr(base, "REGISTRY", {spec2.key: spec2})
    nifty = NiftyTable.from_close(dates, np.full(n, 100.0))
    out = explore.trades_for("HAND", "Hand t_hand", None, tmp_data_root, fixed_clock, ctx=ctx, nifty=nifty)
    assert out["exploratory"] is True and out["gated"] is None and out["tf"] == "1d"
    assert "costs_model" in out and out["strat"] == "t_hand" and out["exit"] == "rr2_20"
    assert {"t_in", "p_in", "stop", "target", "status"} <= set(out["trades"][0])
    assert [t["status"] for t in out["trades"]] == ["CLOSED", "SIGNAL"] or out["trades"][-1][
        "status"
    ] == "SIGNAL"
    assert out["summary"]["n"] == 1 and out["exit_name"] == "2R or 20 days"
    cached = list((tmp_data_root.paths.data_root / "lab" / "explore_cache" / "HAND").glob("*.json"))
    assert len(cached) == 1
    again = explore.trades_for("HAND", "t_hand", "rr2_20", tmp_data_root, fixed_clock, ctx=ctx, nifty=nifty)
    assert again == out
    with pytest.raises(KeyError):
        explore.trades_for("HAND", "nope", None, tmp_data_root, fixed_clock, ctx=ctx, nifty=nifty)
    with pytest.raises(ValueError):
        explore.trades_for("HAND", "t_hand", "bogus", tmp_data_root, fixed_clock, ctx=ctx, nifty=nifty)
    thin = explore.make_ctx("HAND", hand_frame())
    assert explore.trades_for("HAND", "t_hand", None, tmp_data_root, fixed_clock, ctx=thin)["trades"] == []


def test_matrix_rows_cover_every_allowed_exit(tmp_data_root, fixed_clock, monkeypatch):
    a, b = hand_spec([], key="t_a"), hand_spec([], key="t_b")
    b = base.StrategySpec(**{**b.__dict__, "exits": ("trail3", "time5")})
    monkeypatch.setattr(base, "REGISTRY", {"t_a": a, "t_b": b})
    monkeypatch.setattr(base, "load_library", lambda: base.REGISTRY)
    n = 300
    dates = np.datetime64("2025-01-01") + np.arange(n)
    f = DailyFrame(
        dates, np.full(n, 100.0), np.full(n, 101.0), np.full(n, 99.0), np.full(n, 100.0), np.full(n, 1e6)
    )
    m = explore.matrix(
        "HAND",
        tmp_data_root,
        fixed_clock,
        ctx=explore.make_ctx("HAND", f),
        nifty=NiftyTable.from_close(dates, f.close),
    )
    assert len(m["rows"]) == len(EXITS_1D) + 2 == m["n_tests"]
    assert m["expected_max_sharpe_null"] > 0
    assert {
        "strat",
        "name",
        "tier",
        "source",
        "exit",
        "n",
        "win",
        "avgR",
        "totR",
        "pct",
        "is_avgR",
        "oos_avgR",
        "gated",
    } == set(m["rows"][0])


def test_hold_days_match_the_exit_module():
    from stocksense.strategies import exits

    assert explore.HOLD_DAYS == exits._MAX_DAYS
