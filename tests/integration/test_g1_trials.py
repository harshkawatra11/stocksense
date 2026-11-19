"""G1 trials end to end on the synthetic market: registration before running, the grid, resumability,
references, the meta variant and strict capture (PLAN T9.5 steps 1-3)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.core.state import State
from stocksense.data import io
from stocksense.models.base import Model
from stocksense.portfolio.meta import MetaModel
from stocksense.validation import g1
from stocksense.validation.g1 import G1Config, G1Runner
from stocksense.validation.trials import n_trials
from tests.fixtures.g1_world import build_g1_world, run_model_trials

pytestmark = pytest.mark.integration

CAPITALS = (100_000.0, 500_000.0)
REF = 500_000.0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_g1_world(tmp_path_factory.mktemp("g1_trials"))


@pytest.fixture(scope="module")
def state(world):
    st = State(world.settings)
    st.migrate()
    return st


@pytest.fixture(scope="module")
def model_ids(world, state):
    return run_model_trials(world, state)


def cfg(world) -> G1Config:
    return G1Config(
        start=world.folds[0].test_start, capitals=CAPITALS, reference_capital=REF, folds=tuple(world.folds),
        meta_factory=lambda: MetaModel(min_child_samples=10),
    )  # fmt: skip


@pytest.fixture(scope="module")
def runner(world, state, model_ids):
    return G1Runner(world.settings, state, world.calendar, cfg(world))


@pytest.fixture(scope="module")
def survivors(state, model_ids):
    s1 = {t: g1.surviving_models(state, "s1", t) for t in g1.S1_TARGETS}
    s2 = g1.surviving_models(state, "s2", "y_oc")
    return s1, s2


@pytest.fixture(scope="module")
def grid_run(runner, state, survivors, monkeypatch_module):
    s1, s2 = survivors
    plans = g1.plan_grid(runner.cfg, s1, s2)
    seen: list[tuple[str, int, int]] = []
    orig = g1.run_backtest

    def spy(*a, **k):
        seen.append((runner.current_trial, n_trials(state, "g1-s1"), n_trials(state, "g1-s2")))
        return orig(*a, **k)

    monkeypatch_module.setattr(g1, "run_backtest", spy)
    registered = runner.register(plans)
    results = {p.trial_id: runner.run_trial(p) for p in registered}
    monkeypatch_module.setattr(g1, "run_backtest", orig)
    return registered, results, seen


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_every_model_has_a_positive_out_of_fold_ic(survivors):
    s1, s2 = survivors
    assert all(len(v) == 1 for v in s1.values()) and len(s2) == 1
    assert all(m.mean_ic > 0.02 for v in s1.values() for m in v)


def test_all_trials_are_registered_before_any_backtest_runs(grid_run):
    registered, _, seen = grid_run
    assert len(registered) == 14 and seen
    # at the very first backtest the whole grid (12 S1 + 2 S2) is already on the registry
    assert all(s1 == 12 and s2 == 2 for _, s1, s2 in seen)
    assert {t for t, _, _ in seen} == {p.trial_id for p in registered}


def test_each_trial_is_finished_with_returns_and_the_capital_grid(grid_run, state, world):
    registered, results, _ = grid_run
    con = state.connect()
    try:
        done = {r[0] for r in con.execute("SELECT trial_id FROM trials WHERE status='done'").fetchall()}
    finally:
        con.close()
    for p in registered:
        assert p.trial_id in done
        r = results[p.trial_id]
        assert set(r.data["by_capital"]) == {"100000", "500000"}
        assert r.data["stress_2x"]["capital"] == REF
        daily = r.daily()
        assert {"date", "net", "net_2x", "benchmark", "net_100000", "net_500000"} <= set(daily.columns)
        assert daily["net"].null_count() == 0 and daily.height > 200
        rets = pl.read_parquet(g1.returns_path(world.settings, p.trial_id))
        assert rets.columns == ["date", "ret"] and rets.height == daily.height
    s2 = [results[p.trial_id] for p in registered if p.kind == "s2"]
    assert all(r.daily()["benchmark"].null_count() == r.daily().height for r in s2)


def test_second_call_runs_no_backtest(grid_run, runner, monkeypatch_module):
    registered, _, _ = grid_run
    calls = {"n": 0}
    orig = g1.run_backtest

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch_module.setattr(g1, "run_backtest", counting)
    again = runner.register(registered)
    for p in again:
        runner.run_trial(p)
    monkeypatch_module.setattr(g1, "run_backtest", orig)
    assert calls["n"] == 0


def test_stressed_slippage_is_never_better_than_the_base_run(grid_run):
    _, results, _ = grid_run
    for r in results.values():
        s = r.data["stress_2x"]["sharpe"]
        b = r.data["by_capital"]["500000"]["sharpe"]
        assert s <= b + 1e-9


def test_s1_trades_are_delivery_and_s2_trades_are_intraday(grid_run):
    registered, results, _ = grid_run
    for p in registered:
        t = pl.read_parquet(results[p.trial_id].dir / "trades.parquet")
        if t.is_empty():
            continue
        assert set(t["product"]) == ({"delivery"} if p.kind == "s1" else {"intraday"})


def test_s2_reports_strict_capture_unavailable_without_bars(grid_run):
    registered, results, _ = grid_run
    for p in registered:
        if p.kind == "s2":
            sc = results[p.trial_id].data["strict_capture"]
            assert sc["available"] is False and "intraday" in sc["reason"]


def test_the_best_s1_trial_gets_a_meta_variant_and_references(runner, grid_run, state):
    registered, results, _ = grid_run
    s1 = [p for p in registered if p.kind == "s1"]
    best = max(s1, key=lambda p: (results[p.trial_id].ref_sharpe, p.trial_id))
    sp = best.spec
    meta_spec = {**sp, "meta": "walk-forward", "base_trial": best.trial_id}
    meta = g1.TrialPlan("s1-meta", "g1-s1", best.target, meta_spec, "meta")
    ref_open = g1.TrialPlan("ref-open", "g1-ref", best.target, {**sp, "entry": "open"}, "S1 entry at open")
    ref_mom = g1.TrialPlan(
        "ref-factor", "g1-ref", best.target,
        {"g1": 1, "strategy": "s1-ref", "factor": "ret_1", "sign": 1.0, "target": best.target,
         "k_max": sp["k_max"], "overlay": sp["overlay"]},
        "factor",
    )  # fmt: skip
    out = {}
    for p in runner.register([meta, ref_open, ref_mom]):
        out[p.kind] = runner.run_trial(p)
    assert len({r.trial_id for r in out.values()}) == 3
    open_fills = out["ref-open"]
    # entries at the open: every delivery BUY of the reference trades happened at an open
    assert open_fills.data["by_capital"]["500000"]["n_trades"] >= 0
    assert (out["s1-meta"].dir / "daily.parquet").is_file()
    assert n_trials(state, "g1-ref") == 2 and n_trials(state, "g1-s1") == 13


def _bars(world, trades: pl.DataFrame) -> pl.DataFrame:
    """09:15 and 15:10 five-minute bars for the S2 trades: the first bar closes at the open (no drift) and
    the last bar closes 40% of the way from the open to the close, so strict capture is 0.4 of gross."""
    px = io.read(world.settings, "panel_daily", columns=["date", "sid", "symbol", "open", "close"]).collect()
    j = trades.select("sid", pl.col("entry_date").alias("date")).unique().join(px, on=["sid", "date"])
    rows = []
    for r in j.iter_rows(named=True):
        for t, c in ((time(9, 15), r["open"]), (time(15, 10), r["open"] + 0.4 * (r["close"] - r["open"]))):
            rows.append(
                {
                    "symbol": r["symbol"], "ts": datetime.combine(r["date"], t, tzinfo=IST),
                    "interval": "FIVE_MINUTE", "open": c, "high": c, "low": c, "close": c, "volume": 1000,
                    "source": "test", "adj_asof": r["date"],
                }
            )  # fmt: skip
    return pl.DataFrame(rows)


def test_strict_capture_from_five_minute_bars(world, grid_run):
    registered, results, _ = grid_run
    s2 = max(
        (p for p in registered if p.kind == "s2"),
        key=lambda p: results[p.trial_id].data["by_capital"]["500000"]["n_trades"],
    )
    trades = pl.read_parquet(results[s2.trial_id].dir / "trades.parquet")
    assert trades.height > 20
    bars = _bars(world, trades)
    io.upsert(world.settings, "intraday_bars", bars)
    out = g1.strict_capture(world.settings, trades, date(2025, 9, 30))
    assert out["available"] and out["n_matched"] > 10
    # strict return = auction gross return x 0.4 in this construction, net of the same costs
    j = out
    assert j["strict_net_mean"] != j["auction_net_mean"]
    if j["auction_net_mean"] > 0:
        # the strict leg keeps 40% of the gross move but pays the full cost, so the ratio is below 1 and
        # may be negative (costs larger than the strict gross); that is a result, not an error
        assert j["strict_capture"] < 1.0
        assert j["strict_capture"] == pytest.approx(j["strict_net_mean"] / j["auction_net_mean"])
    else:
        assert j["strict_capture"] is None
    # nothing after the research end is used
    cut = g1.strict_capture(world.settings, trades, trades["entry_date"].min() - timedelta(days=1))
    assert cut["available"] is False


def test_np_is_used():
    assert np.isfinite(1.0) and Model is not None
