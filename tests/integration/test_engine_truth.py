from __future__ import annotations

import math
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.backtest.engine import (
    Clairvoyant,
    EqualWeightUniverse,
    PanelData,
    RandomPolicy,
    c2o_forward_labels,
    run_backtest,
)
from stocksense.backtest.metrics import perf_stats
from stocksense.core.calendar import TradingCalendar
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.panel.returns import build_daily_panel
from stocksense.panel.security_master import build_security_master
from tests.fixtures.engine_panel import settings, zero_slippage
from tests.fixtures.synth import _settings_for, make_market, write_market

pytestmark = pytest.mark.integration

CAPITAL = 2_500_000.0
H = 5


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_engine")
    m = make_market(planted_ic=0.10)
    write_market(m, root)
    s = _settings_for(root)
    build_security_master(s)
    build_daily_panel(s)
    daily = io.read(s, "panel_daily").collect().sort("sid", "date")
    cal = TradingCalendar.from_dates(m.sessions)
    return s, m, daily, cal, PanelData.from_frame(daily)


def test_clairvoyant_sharpe_and_random_policy_match_expectation(world):
    s, m, daily, cal, panel = world
    sessions = m.sessions
    slip = SlippageTable.load_latest(s)
    labels = c2o_forward_labels(daily, sessions, H)
    # clairvoyant: the true top-5 of the 5-session close-to-open label, every day
    t0 = time.perf_counter()
    clair = run_backtest(
        panel, cal, Clairvoyant(labels, k=5, h=H), sessions[0], sessions[-1], CAPITAL, s.costs, slip, s.risk
    )
    runtime = time.perf_counter() - t0
    sr = perf_stats(clair.daily["day_ret"].to_numpy(), None, dates=clair.daily["date"].to_list())["sharpe"]
    print(f"\nengine runtime {runtime:.2f}s; clairvoyant net Sharpe {sr:.2f}")
    assert sr > 5.0

    # random policy: net mean daily return vs (labels' mean sleeve return x invested fraction) - cost drag
    rnd = run_backtest(
        panel,
        cal,
        RandomPolicy(seed=11, k=10, h=H),
        sessions[0],
        sessions[-1],
        CAPITAL,
        s.costs,
        slip,
        s.risk,
    )
    d = rnd.daily
    net = d["day_ret"].to_numpy()
    # a sleeve of fraction f earns the mean label R once; one new sleeve starts per session, so the expected
    # gross daily return is mean(f) * R where f = value bought per session / equity; the measured cost drag
    # is the fees plus slippage paid per day over equity
    bought = rnd.fills.filter(pl.col("side") == "BUY").group_by("date").agg(pl.col("value_inr").sum())
    f = d.join(bought, on="date", how="left").with_columns(
        (pl.col("value_inr").fill_null(0.0) / pl.col("equity")).alias("f")
    )["f"]
    r_bar = float(labels["label"].mean())
    drag = float(((d["costs_inr"] + d["slip_inr"]) / d["equity"]).mean())
    expected = float(f.mean()) * r_bar - drag
    se = float(np.std(net, ddof=1) / math.sqrt(len(net)))
    got = float(net.mean())
    print(
        f"random policy mean daily net {got:.6f} vs expected {expected:.6f} (1 SE {se:.6f}); drag {drag:.6f}"
    )
    assert abs(got - expected) <= se


def test_runtime_scales_for_long_panels():
    """16 years x ~20 positions finishes well inside the 60 s budget (here: 12 years x 150 names)."""
    s = settings()
    rng = np.random.default_rng(3)
    n_days, n_stocks = 3000, 150
    ds, d = [], date(2012, 1, 2)
    while len(ds) < n_days:
        if d.weekday() < 5:
            ds.append(d)
        d += timedelta(days=1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (n_days, n_stocks)), axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.003, (n_days, n_stocks)))
    prev = np.vstack([open_[:1], close[:-1]])
    df = pl.DataFrame(
        {
            "sid": np.tile(np.arange(n_stocks, dtype=np.int32), n_days),
            "date": np.repeat(np.array(ds, dtype="datetime64[D]"), n_stocks),
            "open": open_.ravel(),
            "high": np.maximum(open_, close).ravel() + 0.5,
            "low": np.minimum(open_, close).ravel() - 0.5,
            "close": close.ravel(),
            "prev_close": prev.ravel(),
            "turnover_inr": np.full(n_days * n_stocks, 5e8),
            "gap_factor": np.ones(n_days * n_stocks),
            "div_ret": np.zeros(n_days * n_stocks),
            "ret": (close / prev - 1).ravel(),
            "ret_co": (open_ / prev - 1).ravel(),
            "flag_one_price": np.zeros(n_days * n_stocks, dtype=bool),
            "flag_locked_up": np.zeros(n_days * n_stocks, dtype=bool),
            "flag_locked_dn": np.zeros(n_days * n_stocks, dtype=bool),
        }
    )
    t0 = time.perf_counter()
    panel = PanelData.from_frame(df)
    res = run_backtest(
        panel,
        TradingCalendar.from_dates(ds),
        EqualWeightUniverse(rebalance_every=5, max_names=20, band=0.02),
        ds[0], ds[-1], 500_000.0, s.costs, zero_slippage(), s.risk, debug=True,
    )  # fmt: skip
    elapsed = time.perf_counter() - t0
    print(f"\n{n_days} sessions x {n_stocks} names: {elapsed:.1f}s, {res.fills.height} fills")
    assert elapsed < 60.0
    assert res.daily["n_pos"].max() >= 15
