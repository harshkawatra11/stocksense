"""M5.7: the gated strategy runner (validation/strategy_eval.py) on a synthetic market. No real data, no real gate run."""

# ruff: noqa: E501
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.state import State
from stocksense.costs.slippage import SlippageTable
from stocksense.strategies import base, explore, portfolio
from stocksense.strategies.data import DailyFrame, NiftyTable
from stocksense.strategies.exits import EXITS_1D
from stocksense.strategies.explore import Trade, make_ctx
from stocksense.validation import strategy_eval as se
from stocksense.validation.splits import Fold
from stocksense.validation.trials import n_trials

N_STOCKS, T = 60, 520
FIRST = date(2022, 1, 3)


def make_sessions(n: int = T) -> list[date]:
    out, d = [], FIRST
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


SESSIONS = make_sessions()
CAL = TradingCalendar.from_dates(SESSIONS)
PROTO = {
    "reference_capital": 500000,
    "capital_grid": [100000, 500000],
    "slots": 10,
    "research_end": date(2023, 12, 31),
}
FOLDS = [
    Fold("2023H1", CAL.prev_session(date(2023, 1, 2)), date(2023, 1, 2), date(2023, 6, 30)),
    Fold("2023H2", date(2023, 6, 30), date(2023, 7, 3), date(2023, 12, 29)),
]


def section(**kw) -> dict:
    s = {
        "name": "t", "source": "tier", "tier": "none", "family": "m-swing", "timeframe": "1d", "kind": "trend",
        "direction": "long", "exits": ["rr2_20", "time5", "time10"], "default_exit": "time10", "params": {},
        "strength": "signal", "status": "ready", "eval": "swing", "target": "y_c2o_10", "benchmark": "ew500",
        "hypothesis": "", "reference": "",
    }  # fmt: skip
    s.update(kw)
    return s


def build_market(planted: float = 0.0, seed: int = 7) -> tuple[se.MarketData, dict[float, np.ndarray]]:
    """60 stocks, 520 sessions; the return of day t+1 gets planted * 0.02 * alpha[t]."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.008, T)
    beta = rng.uniform(0.6, 1.4, N_STOCKS)
    e = rng.normal(0, 0.015, (N_STOCKS, T))
    alpha = rng.normal(0, 1, (N_STOCKS, T))
    gap = rng.normal(0, 0.003, (N_STOCKS, T))
    r = beta[:, None] * m[None, :] + e
    r[:, 1:] += planted * 0.02 * alpha[:, :-1]
    close = 100 * np.cumprod(1 + r, axis=1)
    prev = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    open_ = prev * (1 + gap)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, (N_STOCKS, T))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, (N_STOCKS, T))))
    dates = np.array(SESSIONS, dtype="datetime64[D]")
    ctxs, sids, member, alpha_by = {}, {}, {}, {}
    for i in range(N_STOCKS):
        sym = f"S{i:02d}"
        fr = DailyFrame(dates, open_[i], hi[i], lo[i], close[i], np.full(T, 1e6))
        ctxs[sym] = make_ctx(sym, fr)
        sids[sym], member[sym] = i + 1, np.ones(T, bool)
        alpha_by[round(float(close[i, 5]), 9)] = alpha[i]
    idx = 100 * np.cumprod(1 + m)
    nifty = NiftyTable.from_close(dates, idx)
    bars = pl.DataFrame({"date": SESSIONS, "open": idx * (1 - 0.001), "close": idx})
    bench = pl.DataFrame({"date": SESSIONS, "ew500_ret": r.mean(axis=0)})
    fwd = np.full((N_STOCKS, T), np.nan)
    fwd[:, :-10] = close[:, 10:] / close[:, :-10] - 1
    oc = close / open_ - 1
    rows = [
        (SESSIONS[t], i + 1, fwd[i, t], oc[i, t])
        for i in range(N_STOCKS)
        for t in range(T)
        if np.isfinite(fwd[i, t])
    ]
    labels = pl.DataFrame(rows, schema=["date", "sid", "y_c2o_10", "y_oc"], orient="row")
    return se.MarketData(
        ctxs, sids, member, nifty, bars, bench, labels, SlippageTable.__new__(SlippageTable)
    ), alpha_by


@pytest.fixture
def state(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    return st


def planted_spec(alpha_by, key="t_planted", prob=None) -> base.StrategySpec:
    def sig(f):
        a = alpha_by[round(float(f["c"][5]), 9)]
        if prob is not None:  # a random signal, seeded by the stock
            s = np.random.default_rng(int(abs(f["c"][5]) * 1e6) % 2**31).random(len(a)) < prob
        else:
            s = a > 1.0
        return s, f["l"] - 1.0 * f["atr"]

    return base.StrategySpec(
        key=key, name="Spec " + key, source="tier", tier="none", family="m-swing", timeframe="1d", kind="trend",
        direction="long", exits=tuple(EXITS_1D), default_exit="time10", signal=sig,
    )  # fmt: skip


@pytest.fixture(scope="module")
def planted_market():
    return build_market(planted=0.5)


def _run(key, spec, sec, market, settings, state, clock, **kw):
    return se.run_strategy(
        key,
        settings,
        state,
        CAL,
        clock,
        market=market,
        folds=FOLDS,
        spec=spec,
        section=sec,
        proto=PROTO,
        **kw,
    )


def _market_with_slip(market, settings):
    market.slip = SlippageTable.load_latest(settings)
    return market


def test_planted_alpha_has_positive_ic_and_beats_a_random_signal(
    tmp_data_root, state, fixed_clock, planted_market
):
    market, alpha_by = planted_market
    market = _market_with_slip(market, tmp_data_root)
    out = _run("t_planted", planted_spec(alpha_by), section(), market, tmp_data_root, state, fixed_clock)
    (tr,) = out["trials"]
    ev = json.loads((se.eval_dir(tmp_data_root, tr["trial_id"]) / "eval.json").read_text(encoding="utf-8"))
    assert ev["ic"]["mean_ic"] > 0.03 and ev["strategy"] == "t_planted" and ev["verdict"] in ("PASS", "FAIL")
    rnd = _run(
        "t_random",
        planted_spec(alpha_by, "t_random", prob=0.03),
        section(),
        market,
        tmp_data_root,
        state,
        fixed_clock,
    )
    rev = json.loads(
        (se.eval_dir(tmp_data_root, rnd["trials"][0]["trial_id"]) / "eval.json").read_text("utf-8")
    )
    assert abs(rev["ic"]["mean_ic"]) < 0.02
    assert ev["net_sharpe"] > rev["net_sharpe"]
    assert set(ev["net_sharpe_by_capital"]) == {"100000", "500000"} and ev["trades"] > 30
    assert (
        ev["gate"]["lines"][7]["passed"] is False
    )  # fewer than 10 finished trials in the family: PBO cannot pass
    daily = pl.read_parquet(se.eval_dir(tmp_data_root, tr["trial_id"]) / "daily.parquet")
    assert {"date", "ret", "net", "net_2x", "benchmark", "gross"} <= set(daily.columns)
    assert daily["date"].min() >= FOLDS[0].test_start and daily["date"].max() <= FOLDS[-1].test_end


def test_exit_selection_never_reads_test_fold_data(
    tmp_data_root, state, fixed_clock, planted_market, monkeypatch
):
    market, alpha_by = planted_market
    market = _market_with_slip(market, tmp_data_root)
    calls: list[tuple[date, date]] = []
    real = portfolio.run_daily

    def spy(by, panel, cfg, settings, cal, start, end, exit_key=None):
        calls.append((start, end))
        return real(by, panel, cfg, settings, cal, start, end, exit_key=exit_key)

    monkeypatch.setattr(portfolio, "run_daily", spy)
    _run("t_planted", planted_spec(alpha_by), section(), market, tmp_data_root, state, fixed_clock)
    inner_ends = {CAL.shift(f.train_end, -se.EMBARGO_DAILY) for f in FOLDS}
    final = (FOLDS[0].test_start, FOLDS[-1].test_end)
    inner = [c for c in calls if c[1] in inner_ends]
    assert inner, "exit selection ran no inner simulation"
    for _, end in inner:
        assert end < min(f.test_start for f in FOLDS if CAL.shift(f.train_end, -se.EMBARGO_DAILY) == end)
        assert end <= CAL.shift(FOLDS[0].train_end, -se.EMBARGO_DAILY) or end < FOLDS[1].test_start
    assert all(c[1] in inner_ends or c == final for c in calls)  # every other call is a final simulation
    first_fold_inner = [c for c in inner if c[1] == CAL.shift(FOLDS[0].train_end, -se.EMBARGO_DAILY)]
    assert all(c[1] < FOLDS[0].test_start - timedelta(days=5) for c in first_fold_inner)


def test_select_exit_uses_only_the_inner_window_and_the_trade_floor():
    seen: list[date] = []

    def sim(ex: str, inner_end: date):
        seen.append(inner_end)
        net = np.full(50, 0.001 if ex == "b" else 0.0005) + np.arange(50) % 3 * 1e-4
        return pl.DataFrame({"net": net}), (40 if ex != "c" else 5)

    f = FOLDS[1]
    assert se.select_exit(("a", "b", "c"), "a", f, CAL, 5, sim) == "b"
    assert set(seen) == {CAL.shift(f.train_end, -5)} and seen[0] < f.test_start
    few = se.select_exit(("a", "b"), "dflt", f, CAL, 5, lambda e, d: (pl.DataFrame({"net": [0.0] * 5}), 3))
    assert few == "dflt"  # under 30 inner trades: the default exit


def test_trial_is_registered_before_eval_json_is_written(
    tmp_data_root, state, fixed_clock, planted_market, monkeypatch
):
    market, alpha_by = planted_market
    market = _market_with_slip(market, tmp_data_root)
    order: list[str] = []
    real_reg, real_write = se.register_trial, se._write_json
    monkeypatch.setattr(
        se, "register_trial", lambda *a, **k: (order.append("register"), real_reg(*a, **k))[1]
    )
    monkeypatch.setattr(
        se, "_write_json", lambda p, d: (order.append(f"write:{p.name}"), real_write(p, d))[1]
    )
    before = n_trials(state, "m-swing")
    _run("t_planted", planted_spec(alpha_by), section(), market, tmp_data_root, state, fixed_clock)
    assert order[0] == "register" and order.index("register") < order.index("write:eval.json")
    assert n_trials(state, "m-swing") == before + 1
    con = state.connect()
    try:
        st = con.execute("SELECT status, created_at FROM trials WHERE family='m-swing'").fetchone()
    finally:
        con.close()
    assert st[0] == "done"


def test_resume_skips_a_finished_strategy(tmp_data_root, state, fixed_clock, planted_market, monkeypatch):
    market, alpha_by = planted_market
    market = _market_with_slip(market, tmp_data_root)
    spec = planted_spec(alpha_by)
    first = _run("t_planted", spec, section(), market, tmp_data_root, state, fixed_clock)
    p = se.eval_dir(tmp_data_root, first["trials"][0]["trial_id"]) / "eval.json"
    mtime, n = p.stat().st_mtime_ns, n_trials(state)
    monkeypatch.setattr(portfolio, "run_daily", lambda *a, **k: pytest.fail("a finished strategy was re-run"))
    again = _run("t_planted", spec, section(), market, tmp_data_root, state, fixed_clock)
    assert again["trials"][0]["skipped"] == "done" and p.stat().st_mtime_ns == mtime
    assert n_trials(state) == n  # registering the same spec again adds nothing


def test_await_data_and_informational_strategies_are_skipped_without_a_trial(
    tmp_data_root, state, fixed_clock
):
    before = n_trials(state)
    for sec in (section(status="await_data"), section(eval="informational")):
        out = se.run_strategy(
            "t_x", tmp_data_root, state, CAL, fixed_clock, spec=planted_spec({}), section=sec, proto=PROTO
        )
        assert out["trials"] == [] and out["skipped"]
    assert n_trials(state) == before


def test_block_ic_and_month_folds():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=210)
    nxt = 0.5 * pos + rng.normal(size=210)
    good = se.block_ic(pos, nxt)
    assert good["mean_ic"] > 0.2 and good["ic_tstat"] > 3 and good["n_blocks"] == 10
    assert se.block_ic(np.ones(100), nxt[:100])["mean_ic"] == 0.0  # a constant position has no IC
    cal = TradingCalendar.from_dates(
        [
            date(2024, 9, 2) + timedelta(days=i)
            for i in range(500)
            if (date(2024, 9, 2) + timedelta(days=i)).weekday() < 5
        ]
    )
    mf = se.month_folds(cal)
    assert [f.name for f in mf][0] == "2024-10" and [f.name for f in mf][-1] == "2025-09" and len(mf) == 12
    assert mf[0].train_end == cal.prev_session(mf[0].test_start) and mf[-1].test_end <= date(2025, 9, 30)


def test_xs_strategy_goes_through_the_backend_and_the_gate(
    tmp_data_root, state, fixed_clock, planted_market, monkeypatch
):
    market, _ = planted_market
    market = _market_with_slip(market, tmp_data_root)
    days = [d for d in SESSIONS if FOLDS[0].test_start <= d <= FOLDS[-1].test_end]
    rng = np.random.default_rng(3)
    daily = pl.DataFrame(
        {
            "date": days, "net": rng.normal(0.0008, 0.01, len(days)), "net_2x": rng.normal(0.0006, 0.01, len(days)),
            "gross": [0.9] * len(days), "net_100000": [0.0] * len(days), "net_500000": [0.0] * len(days),
        }
    )  # fmt: skip
    seen = {}

    def fake(spec, trial_id, settings, st, cal, start, end, clock):
        seen["trial"] = trial_id
        return daily, {"mean_ic": 0.05, "ic_tstat": 4.0}, {"s1_trial": "g2-s1-x"}

    monkeypatch.setattr(se, "_xs_backend", fake)
    spec = base.StrategySpec(
        key="t_xs", name="t xs", source="tier", tier="none", family="m-xs", timeframe="1d", kind="xs_factor",
        direction="long", exits=(), default_exit="", signal=None, score=lambda s, a, b: pl.DataFrame(),
    )  # fmt: skip
    sec = section(family="m-xs", kind="xs_factor", eval="xs", exits=[], default_exit="")
    out = se.run_strategy(
        "t_xs",
        tmp_data_root,
        state,
        CAL,
        fixed_clock,
        market=market,
        folds=FOLDS,
        spec=spec,
        section=sec,
        proto=PROTO,
    )
    tid = out["trials"][0]["trial_id"]
    ev = json.loads((se.eval_dir(tmp_data_root, tid) / "eval.json").read_text("utf-8"))
    assert seen["trial"] == tid and ev["ic"] == {"mean_ic": 0.05, "ic_tstat": 4.0}
    assert (
        ev["gate"]["lines"][0]["passed"] and ev["gate"]["lines"][1]["passed"]
    )  # the IC lines read the backend's IC
    assert ev["family"] == "m-xs" and set(ev["net_sharpe_by_capital"]) == {"100000", "500000"}


def test_option_trial_has_a_cash_benchmark_and_doubled_charges(
    tmp_data_root, state, fixed_clock, planted_market
):
    market, _ = planted_market
    market = _market_with_slip(market, tmp_data_root)
    days = [d for d in SESSIONS if FOLDS[0].test_start <= d <= FOLDS[-1].test_end]
    ret = np.where(np.arange(len(days)) % 21 == 20, -0.01, 0.0004)
    cost = np.where(np.arange(len(days)) % 21 == 20, 0.002, 0.0)

    def with_cost(settings, start=None, end=None):
        return pl.DataFrame({"date": days, "ret": ret, "cost": cost})

    def timing(settings, start=None, end=None):
        return with_cost(settings, start, end).select("date", "ret")

    timing.with_cost = with_cost
    spec = base.StrategySpec(
        key="t_opt", name="t opt", source="k151", tier="none", family="m-opt", timeframe="1d", kind="options",
        direction="both", exits=("expiry",), default_exit="expiry", signal=None, timing=timing,
    )  # fmt: skip
    sec = section(
        family="m-opt",
        kind="options",
        eval="options",
        exits=["expiry"],
        default_exit="expiry",
        benchmark="cash",
        target="index",
    )
    out = se.run_strategy(
        "t_opt",
        tmp_data_root,
        state,
        CAL,
        fixed_clock,
        market=market,
        folds=FOLDS,
        spec=spec,
        section=sec,
        proto=PROTO,
    )
    tid = out["trials"][0]["trial_id"]
    daily = pl.read_parquet(se.eval_dir(tmp_data_root, tid) / "daily.parquet").sort("date")
    assert daily["benchmark"].abs().sum() == 0.0
    assert daily["net_2x"].to_numpy() == pytest.approx(ret - cost)
    ev = json.loads((se.eval_dir(tmp_data_root, tid) / "eval.json").read_text("utf-8"))
    assert ev["ic"] == {"mean_ic": 0.0, "ic_tstat": 0.0} and ev["gate"]["lines"][0]["passed"] is False


def test_intraday_trial_per_trend_with_the_monthly_protocol(tmp_data_root, state, fixed_clock, monkeypatch):
    sess = [d for d in (date(2024, 9, 13) + timedelta(days=i) for i in range(400)) if d.weekday() < 5]
    cal = TradingCalendar.from_dates(sess)
    folds = se.month_folds(cal)[:3]
    syms = [f"S{i:02d}" for i in range(55)]
    members = {d: frozenset(syms) for d in sess}
    rng = np.random.default_rng(5)
    rows = [(d, i + 1, rng.normal(0, 0.01)) for d in sess for i in range(55)]
    market = se.MarketData(
        {}, {s: i + 1 for i, s in enumerate(syms)}, {}, NiftyTable.from_close(np.array(sess, dtype="datetime64[D]"), np.arange(len(sess)) + 100.0),
        pl.DataFrame({"date": sess, "open": 1.0, "close": 1.0}), pl.DataFrame({"date": sess, "ew500_ret": 0.0}),
        pl.DataFrame(rows, schema=["date", "sid", "y_oc"], orient="row").with_columns(pl.lit(0.0).alias("y_c2o_10")),
        SlippageTable.load_latest(tmp_data_root), members,
    )  # fmt: skip

    def trade(sym, d, win):
        t_in = int(datetime(d.year, d.month, d.day, 0, 0, tzinfo=UTC).timestamp()) + 3600  # UTC date = d
        return Trade(
            sym=sym, strat="t_intra", exit="rr2", side=1, t_sig=t_in - 300, t_in=t_in, p_in=100.0, stop=99.0,
            target=102.0, t_out=t_in + 3600, p_out=102.0 if win else 99.0, reason="target" if win else "stop",
            R=2.0 if win else -1.0, gross_pct=2.0, cost_pct=0.1, net_pct=1.9, days=None, status="CLOSED", mkt_up=True, rs63=None,
        )  # fmt: skip

    monkeypatch.setattr(explore, "intra_prep", lambda sym, settings, a, b, daily=None: object())
    monkeypatch.setattr(
        portfolio, "intra_trades",
        lambda ip, spec, ex, trend, notional: [trade(s, d, (i + ord(trend[0])) % 3 != 0) for d in sess[:300] for i, s in enumerate(syms[:5])],
    )  # fmt: skip
    spec = base.StrategySpec(
        key="t_intra", name="t intra", source="tier", tier="none", family="m-intra", timeframe="5m", kind="breakout",
        direction="long", exits=("rr1", "rr2"), default_exit="rr2", signal=lambda f: (np.zeros(1, bool), np.zeros(1)),
    )  # fmt: skip
    sec = section(
        family="m-intra",
        timeframe="5m",
        eval="intra",
        exits=["rr1", "rr2"],
        default_exit="rr2",
        target="y_oc",
        trends=["D", "I", "DI"],
    )
    proto = {**PROTO, "research_end": date(2025, 9, 30)}
    out = se.run_strategy(
        "t_intra",
        tmp_data_root,
        state,
        cal,
        fixed_clock,
        market=market,
        folds=folds,
        spec=spec,
        section=sec,
        proto=proto,
    )
    assert [t["trend"] for t in out["trials"]] == ["D", "I", "DI"] and len(
        {t["trial_id"] for t in out["trials"]}
    ) == 3
    ev = json.loads(
        (se.eval_dir(tmp_data_root, out["trials"][0]["trial_id"]) / "eval.json").read_text("utf-8")
    )
    assert ev["trades"] > 0 and set(ev["chosen_exits"]) == {f.name for f in folds} and ev["target"] == "y_oc"
    assert ev["first_date"] >= folds[0].test_start.isoformat()


def test_table_lists_the_rows(tmp_data_root, state, fixed_clock, planted_market):
    market, alpha_by = planted_market
    market = _market_with_slip(market, tmp_data_root)
    _run("t_planted", planted_spec(alpha_by), section(), market, tmp_data_root, state, fixed_clock)
    rows = se.load_evals(tmp_data_root)
    assert len(rows) == 1
    text = se.table(rows)
    assert rows[0]["trial_id"] in text and rows[0]["verdict"] in text


def test_cli_strat_table_and_unknown_family(tmp_data_root):
    from typer.testing import CliRunner

    from stocksense.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["research", "strat-table"], obj=tmp_data_root)
    assert res.exit_code == 0 and "no evaluated strategies" in res.output
    bad = runner.invoke(app, ["research", "strat-run", "--family", "m-nope"], obj=tmp_data_root)
    assert bad.exit_code == 2 and "unknown family" in bad.output


def test_run_family_rejects_keys_of_another_family(tmp_data_root, state, fixed_clock):
    with pytest.raises(KeyError):
        se.run_family("m-xs", tmp_data_root, state, CAL, fixed_clock, keys=["ml_swing_vcp"])
    with pytest.raises(KeyError):
        se.run_family("m-swing", tmp_data_root, state, CAL, fixed_clock, keys=["no_such_key"])


def test_run_family_skips_waiting_strategies_and_lists_the_rest(tmp_data_root, state, fixed_clock):
    out = se.run_family(
        "m-xs", tmp_data_root, state, CAL, fixed_clock, keys=["k151_nb_news", "k151_implied_vol"]
    )
    assert [r["skipped"] for r in out] == ["await_data", "await_data"] and all(r["trials"] == [] for r in out)
    assert n_trials(state, "m-xs") == 0  # nothing registered for a strategy that cannot run
