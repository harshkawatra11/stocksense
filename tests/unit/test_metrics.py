from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.backtest.metrics import ic_stats, newey_west_tstat, perf_stats, trade_stats, yearly_table

R = np.array([0.01, -0.02, 0.03, 0.00, -0.01, 0.02])
D6 = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]


def test_perf_stats_hand_computed():
    # mean 0.03/6 = 0.005; deviations .005,-.025,.025,-.005,-.015,.015 -> sum sq 1.75e-3
    # var(ddof=1) = 3.5e-4, std = 0.0187083; sharpe = 0.005/0.0187083*sqrt(252) = sqrt(18) = 4.24264
    s = perf_stats(R, None, dates=D6)
    assert s["sharpe"] == pytest.approx(4.24264, abs=1e-4)
    assert s["vol"] == pytest.approx(0.0187083 * math.sqrt(252), abs=1e-5)  # 0.29698
    # prod(1+r) = 1.0294850412; cagr = 1.0294850412 ** (252/6) - 1
    assert s["cagr"] == pytest.approx(1.0294850412**42 - 1, abs=1e-9)
    # downside: min(r,0)^2 = 4e-4, 1e-4 -> mean 8.3333e-5 -> dd 0.00912871; sortino = 0.005/dd*sqrt(252)
    assert s["sortino"] == pytest.approx(0.547723 * math.sqrt(252), abs=1e-3)  # 8.6946
    # curve 1, 1.01, 0.9898, 1.019494, ...: max dd = 1 - 0.9898/1.01 = 0.02
    assert s["max_drawdown"] == pytest.approx(0.02, abs=1e-9)
    assert (s["dd_start"], s["dd_trough"], s["dd_recovery"]) == (D6[0], D6[1], D6[2])
    assert s["calmar"] == pytest.approx(s["cagr"] / 0.02)
    assert s["hit_rate_days"] == pytest.approx(0.5)  # 3 of 6 days positive (the 0.00 day is not)
    # third central moments cancel -> skew 0; kurtosis m4/m2^2 = 1.47292e-7 / 8.50694e-8
    assert s["skew"] == pytest.approx(0.0, abs=1e-9)
    assert s["kurtosis"] == pytest.approx(1.73143, abs=1e-4)


def test_zero_std_sharpe_is_zero():
    assert perf_stats(np.array([0.01, 0.01, 0.01]), None)["sharpe"] == 0.0


def test_drawdown_unrecovered():
    # curve 1, 1.10, 0.99, 0.891: trough after last day, never recovers
    s = perf_stats(np.array([0.10, -0.10, -0.10]), None, dates=D6[:3])
    assert s["max_drawdown"] == pytest.approx(1 - 0.891 / 1.10, abs=1e-9)  # 0.19
    assert (s["dd_start"], s["dd_trough"], s["dd_recovery"]) == (D6[0], D6[2], None)


def test_newey_west_hand_example():
    # x = 1,2,3,4: mean 2.5, deviations -1.5,-.5,.5,1.5
    # g0 = 1.25, g1 = 0.3125, g2 = -0.375, g3 = -0.5625; weights (L=5) 5/6, 4/6, 3/6
    # lrv = 1.25 + 2 * (0.260417 - 0.25 - 0.28125) = 0.708333; se = sqrt(0.708333/4) = 0.420827; t = 5.9409
    t = newey_west_tstat(np.array([1.0, 2.0, 3.0, 4.0]), lags=5)
    assert t == pytest.approx(5.9409, abs=1e-3)


def test_newey_west_matches_direct_formula():
    rng = np.random.default_rng(3)
    x = rng.normal(0.1, 1.0, 20)
    n, L = 20, 5
    d = x - x.mean()
    lrv = float(d @ d) / n
    for lag in range(1, L + 1):
        g = float(d[lag:] @ d[:-lag]) / n
        lrv += 2 * (1 - lag / (L + 1)) * g
    expect = x.mean() / math.sqrt(lrv / n)
    assert newey_west_tstat(x, lags=5) == pytest.approx(expect, rel=1e-12)


def test_excess_tstat_and_positive_years():
    # excess vs benchmark: r - b = x (1,2,3,4)/100 -> t as above
    r = np.array([0.011, 0.022, 0.033, 0.044])
    b = np.array([0.001, 0.002, 0.003, 0.004])
    s = perf_stats(r, b)
    assert s["excess_tstat"] == pytest.approx(5.9409, abs=1e-3)
    # years: 2023 days +, + ; 2024 days -, - ; compounded 2023 > 0, 2024 < 0 -> 0.5
    ds = [date(2023, 6, 1), date(2023, 6, 2), date(2024, 6, 3), date(2024, 6, 4)]
    s2 = perf_stats(np.array([0.01, 0.02, -0.01, -0.02]), None, dates=ds)
    assert s2["frac_positive_years"] == pytest.approx(0.5)


def _frame() -> pl.DataFrame:
    rows = []
    # four dates with 50 names: label = +score, +score, -score, +score -> IC 1, 1, -1, 1
    spec = [(date(2020, 3, 2), 1), (date(2020, 3, 3), 1), (date(2021, 3, 2), -1), (date(2021, 3, 3), 1)]
    for d, sgn in spec:
        for i in range(1, 51):
            rows.append({"date": d, "sid": i, "score": float(i), "label": sgn * float(i)})
    for i in range(10):  # a date with only 10 pairs is skipped
        rows.append({"date": date(2021, 3, 4), "sid": i, "score": float(i), "label": float(i)})
    return pl.DataFrame(rows)


def test_ic_stats():
    s = ic_stats(_frame(), "label")
    # IC = [1, 1, -1, 1]: mean 0.5, std(ddof=1) = sqrt((3*0.25 + 2.25)/3) = 1, t = 0.5/1*sqrt(4) = 1.0
    assert s["n_dates"] == 4
    assert s["mean_ic"] == pytest.approx(0.5)
    assert s["ic_std"] == pytest.approx(1.0)
    assert s["ic_tstat"] == pytest.approx(1.0)
    # by year: 2020 -> mean(1, 1) = 1; 2021 -> mean(-1, 1) = 0
    assert s["ic_by_year"] == {2020: pytest.approx(1.0), 2021: pytest.approx(0.0)}
    # decile k (1..10) holds scores 5k-4..5k, label mean 5k-2 on +dates; -(5k-2) on the minus date
    # average over 4 dates: (3 - 1)/4 * (5k - 2) = 0.5 * (5k - 2): decile 1 -> 1.5, decile 10 -> 24
    assert s["decile_mean_label"][0] == pytest.approx(1.5)
    assert s["decile_mean_label"][9] == pytest.approx(24.0)
    assert s["monotonicity"] == pytest.approx(1.0)


def test_trade_stats():
    trades = pl.DataFrame(
        {
            "ret": [0.02, -0.01, 0.03, -0.02],
            "hold_sessions": [5, 5, 10, 4],
            "cost_frac": [0.002, 0.002, 0.003, 0.001],
        }
    )
    rejects = pl.DataFrame({"reason": ["locked_up_open", "locked_up_open", "one_price"]})
    s = trade_stats(trades, rejects)
    # wins .02,.03 -> avg .025; losses -.01,-.02 -> avg -.015; PF = .05/.03 = 1.66667
    assert s["count"] == 4
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["avg_win"] == pytest.approx(0.025)
    assert s["avg_loss"] == pytest.approx(-0.015)
    assert s["profit_factor"] == pytest.approx(5 / 3)
    assert s["avg_hold_sessions"] == pytest.approx(6.0)
    assert s["avg_cost_frac"] == pytest.approx(0.002)
    assert s["rejects_by_reason"] == {"locked_up_open": 2, "one_price": 1}


def test_trade_stats_empty():
    s = trade_stats(
        pl.DataFrame(
            {"ret": [], "hold_sessions": [], "cost_frac": []},
            schema_overrides={"ret": pl.Float64, "hold_sessions": pl.Float64, "cost_frac": pl.Float64},
        )
    )
    assert s["count"] == 0 and s["rejects_by_reason"] == {}


def test_yearly_table():
    daily = pl.DataFrame(
        {
            "date": [date(2023, 1, 2), date(2023, 1, 3), date(2024, 1, 2), date(2024, 1, 3)],
            "ret": [0.01, 0.02, -0.01, -0.02],
        }
    )
    t = yearly_table(daily)
    # 2023: 1.01*1.02 - 1 = 0.0302; 2024: 0.99*0.98 - 1 = -0.0298
    assert t["year"].to_list() == [2023, 2024]
    assert t["ret"].to_list() == [pytest.approx(0.0302), pytest.approx(-0.0298)]
    assert t["n"].to_list() == [2, 2]
    # 2023 sharpe: mean .015 / std(ddof=1) .0070711 * sqrt(252) = 33.676
    assert t["sharpe"][0] == pytest.approx(0.015 / 0.00707107 * math.sqrt(252), rel=1e-4)
