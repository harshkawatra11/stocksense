"""S1Policy through the real engine on the synthetic market (planted_ic = 0.15, Rs 25 lakh)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from stocksense.backtest.engine import PanelData, c2o_forward_labels, run_backtest
from stocksense.core.calendar import TradingCalendar
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.models.base import ModelSpec
from stocksense.panel.returns import build_daily_panel
from stocksense.panel.security_master import build_security_master
from stocksense.portfolio.construct import DayScores, EnginePanelAccess, S1Policy
from tests.fixtures.synth import _settings_for, make_market, write_market

pytestmark = pytest.mark.integration

CAPITAL = 2_500_000.0
H = 5


class OracleScores:
    """Scores built from the realised h-session close-to-open label plus noise (an oracle-ish source)."""

    def __init__(self, labels: pl.DataFrame, noise: float, seed: int = 5) -> None:
        rng = np.random.default_rng(seed)
        sd = float(labels["label"].std())
        self._by: dict = {}
        for d, g in labels.group_by("date", maintain_order=True):
            sids = g["sid"].to_list()
            x = g["label"].to_numpy() + rng.normal(0.0, noise * sd, len(sids))
            r = (np.argsort(np.argsort(x)) + 0.5) / len(x) - 0.5
            self._by[d[0]] = DayScores(
                dict(zip(sids, r, strict=True)), dict(zip(sids, 0.12 * r, strict=True))
            )

    def scores(self, d):
        return self._by.get(d)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1_e2e")
    m = make_market(planted_ic=0.15)
    write_market(m, root)
    s = _settings_for(root)
    build_security_master(s)
    build_daily_panel(s)
    daily = io.read(s, "panel_daily").collect().sort("sid", "date")
    cal = TradingCalendar.from_dates(m.sessions)
    panel = PanelData.from_frame(daily)
    industry = {
        int(r["sid"]): r["industry"] for r in daily.group_by("sid").agg(pl.col("industry").last()).to_dicts()
    }
    labels = c2o_forward_labels(daily, m.sessions, H)
    return s, m, cal, panel, industry, labels


def run(world, *, hysteresis: bool, noise: float = 3.0, stop: bool = True):
    s, m, cal, panel, industry, labels = world
    slip = SlippageTable.load_latest(s)
    ps = (
        s
        if stop
        else s.model_copy(
            update={"risk": s.risk.model_copy(update={"book_dd_half": 1.0, "book_dd_stop": 1.0})}
        )
    )
    pol = S1Policy(
        ModelSpec(family="s1-gbm", target=f"y_c2o_{H}"), ps, OracleScores(labels, noise),
        EnginePanelAccess(panel, cal, industry=industry), s.costs, slip, k_max=10, hysteresis=hysteresis,
    )  # fmt: skip
    res = run_backtest(
        panel, cal, pol, m.sessions[0], m.sessions[-1], CAPITAL, s.costs, slip, s.risk, debug=True
    )
    return res, pol


def test_s1_policy_runs_clean_and_earns_with_a_strong_signal(world):
    res, pol = run(world, hysteresis=True)
    d = res.daily
    s = res.summary
    print(
        f"\nhysteresis: final NAV {s['final_nav']:.2f}, sharpe {s['sharpe']:.2f}, trades {s['n_trades']}, "
        f"mean turnover {s['mean_turnover']:.4f}, cost drag {s['cost_drag_frac_of_capital']:.4f}"
    )
    assert d.height == len(world[1].sessions)
    assert (
        (d["cash_frac"] >= -1e-9).all()
        and d["n_pos"].max() <= 12
        and d["n_pos"].mean() <= 10
        and d["gross"].max() <= 1.05
    )
    assert (res.trades["qty"] > 0).all() and (res.trades["product"] == "delivery").all()
    assert set(res.trades["reason"]) <= {
        "entry",
        "horizon",
        "rank_decay",
        "stop",
        "universe_exit",
        "delisted",
    }
    assert s["n_trades"] > 50
    assert s["final_nav"] > 100.0  # the planted signal pays more than the costs
    # holdings are bought at the close and sold at the open
    assert set(res.fills.filter(pl.col("side") == "BUY")["when"]) == {"close"}
    # sells go at the open; the close-time sells are the engine's retries of the part of a sale that the
    # 1%-of-ADV participation cap left unfilled on thin synthetic names
    assert res.fills.filter(pl.col("side") == "SELL")["when"].is_in(["open", "close"]).all()
    assert pol.entry_log and {"score", "score_rank", "vol_21", "gross_target"} <= set(pol.entry_log[0])


def test_hysteresis_lowers_turnover_against_naive_daily_rebalancing(world):
    # the book-drawdown stop is switched off in both runs so that only the exit rules differ
    keep, _ = run(world, hysteresis=True, stop=False)
    naive, _ = run(world, hysteresis=False, stop=False)
    print(
        f"\nturnover: hysteresis {keep.summary['mean_turnover']:.4f} ({keep.summary['n_trades']} trades), "
        f"naive {naive.summary['mean_turnover']:.4f} ({naive.summary['n_trades']} trades)"
    )
    assert keep.summary["mean_turnover"] < naive.summary["mean_turnover"]
    assert keep.summary["n_trades"] < naive.summary["n_trades"]
