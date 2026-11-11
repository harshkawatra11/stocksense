"""S2Policy through the real engine on the synthetic market (oracle-ish open->close scores, Rs 25 lakh)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from stocksense.backtest.engine import PanelData, run_backtest
from stocksense.core.calendar import TradingCalendar
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.panel.returns import build_daily_panel
from stocksense.panel.security_master import build_security_master
from stocksense.portfolio.construct import DayScores, EnginePanelAccess, S2Policy
from tests.fixtures.synth import _settings_for, make_market, write_market

pytestmark = pytest.mark.integration


class OracleOC:
    """Per trade date d: scores from the realised open->close return of d plus noise."""

    def __init__(self, daily: pl.DataFrame, noise: float, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        oc = daily.filter(pl.col("flag_one_price").not_()).select(
            "date", "sid", (pl.col("close") / pl.col("open") - 1.0).alias("oc")
        )
        sd = float(oc["oc"].std())
        self._by: dict = {}
        for d, g in oc.group_by("date", maintain_order=True):
            x = g["oc"].to_numpy() + rng.normal(0.0, noise * sd, g.height)
            r = (np.argsort(np.argsort(x)) + 0.5) / len(x) - 0.5
            sids = g["sid"].to_list()
            self._by[d[0]] = DayScores(
                dict(zip(sids, r, strict=True)), dict(zip(sids, 0.03 * r, strict=True))
            )

    def scores(self, d):
        return self._by.get(d)


def test_s2_policy_runs_clean_through_the_engine():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    m = make_market(planted_ic=0.15)
    write_market(m, root)
    s = _settings_for(root)
    build_security_master(s)
    build_daily_panel(s)
    daily = io.read(s, "panel_daily").collect().sort("sid", "date")
    cal = TradingCalendar.from_dates(m.sessions)
    panel = PanelData.from_frame(daily)
    slip = SlippageTable.load_latest(s)
    sids = sorted(daily["sid"].unique().to_list())
    fno = {x for x in sids if x % 2 == 0}
    pol = S2Policy(s, OracleOC(daily, noise=1.0), EnginePanelAccess(panel, cal, fno=fno), slip, k_max=6)
    res = run_backtest(
        panel, cal, pol, m.sessions[0], m.sessions[-1], 2_500_000.0, s.costs, slip, s.risk, debug=True
    )
    fills = res.fills
    print(
        f"\nS2: final NAV {res.summary['final_nav']:.2f}, trades {res.summary['n_trades']}, "
        f"shorts {res.trades.filter(pl.col('direction') == 'short').height}"
    )
    assert res.summary["n_trades"] > 100
    assert (res.daily["n_pos"] == 0).all()  # everything squared off every day
    assert set(res.trades["product"]) == {"intraday"}
    shorts = res.trades.filter(pl.col("direction") == "short")
    assert shorts.height > 0 and set(shorts["sid"].to_list()) <= fno  # F&O-only shorts
    assert set(fills.filter(pl.col("reason").is_in(["s2_long", "s2_short"]))["when"]) == {"open"}
    assert res.summary["final_nav"] > 100.0
