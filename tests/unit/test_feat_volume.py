"""T4.3e: volume and liquidity features, each hand-computed."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.features import volume
from stocksense.features.registry import load_all

REG = load_all()


def _df(**cols) -> pl.DataFrame:
    n = len(next(iter(cols.values())))
    df = pl.DataFrame(
        {
            "sid": [1] * n,
            "sx": list(range(n)),
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        },
        schema_overrides={"sid": pl.Int32, "sx": pl.Int64},
    )
    return df.with_columns(*[pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()])


def _run(df: pl.DataFrame, **exprs: pl.Expr) -> pl.DataFrame:
    return volume._stage(df.lazy()).select(*[e.alias(k) for k, e in exprs.items()]).collect()


def test_log_turnover_and_size_proxy():
    df = _df(turnover_inr=[100.0, 200.0, 300.0, 400.0])
    out = _run(df, a=volume.log_turn(3))["a"].to_list()
    # 3-row means: 200, 300
    assert out[:2] == [None, None]
    assert out[2:] == pytest.approx([math.log(200), math.log(300)])


def test_log_turnover_is_null_for_a_zero_mean():
    out = _run(_df(turnover_inr=[0.0, 0.0, 0.0]), a=volume.log_turn(3))["a"].to_list()
    assert out == [None, None, None]


def test_amihud():
    df = _df(ret=[0.01, -0.02, 0.03], turnover_inr=[1e7, 2e7, 1.5e7])
    out = _run(df, a=volume.amihud(2))["a"].to_list()
    # |ret| per crore: .01/1 = .01 ; .02/2 = .01 ; .03/1.5 = .02 -> 2-row means .01 and .015
    assert out[0] is None
    assert out[1:] == pytest.approx([0.01, 0.015])


def test_amihud_zero_turnover_is_null():
    out = _run(_df(ret=[0.01, 0.02], turnover_inr=[0.0, 1e7]), a=volume.amihud(2))["a"].to_list()
    assert out == [None, None]


def test_turn_shock_uses_the_previous_rows_average():
    out = _run(_df(turnover_inr=[10.0, 20.0, 30.0, 40.0, 100.0]), t=volume.turn_shock(3))["t"].to_list()
    # mean3 at row 2 = 20, row 3 = 30; shifted one row: row 3 uses 20 -> 40/20 ; row 4 uses 30 -> 100/30
    assert out[:3] == [None, None, None]
    assert out[3:] == pytest.approx([2.0, 100 / 30])


def test_vol_shock():
    out = _run(_df(adj_volume=[10.0, 20.0, 30.0, 40.0, 50.0]), v=volume.vol_shock(2, 3))["v"].to_list()
    # row 3: mean2 = 35, previous-row mean3 = (10+20+30)/3 = 20 -> 1.75 ; row 4: 45 / 30 = 1.5
    assert out[3:] == pytest.approx([1.75, 1.5])


def test_turn_trend():
    out = _run(_df(turnover_inr=[10.0, 20.0, 30.0, 40.0]), t=volume.turn_trend(2, 3))["t"].to_list()
    # row 3: mean2 = 35, mean3 = 30
    assert out[3] == pytest.approx(35 / 30)


def test_avg_trade_size_rel():
    df = _df(turnover_inr=[100.0, 200.0, 300.0, 400.0], n_trades=[10.0, 10.0, 20.0, None])
    out = _run(df, a=volume.avg_trade_size_rel(2))["a"].to_list()
    # trade size 10, 20, 15, null ; 2-row mean at row 1 = 15 ; row 2: 15 / 15 = 1 ; row 3 null
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(1.0)
    assert out[3] is None


def test_avg_trade_size_zero_trades_is_null():
    df = _df(turnover_inr=[100.0, 100.0, 100.0], n_trades=[10.0, 0.0, 10.0])
    assert _run(df, a=volume.avg_trade_size_rel(2))["a"].to_list() == [None, None, None]


def test_zero_ret_frac():
    out = _run(_df(ret=[0.0, 0.1, 0.0, 0.0, 0.2, 0.3]), z=volume.zero_ret_frac(3))["z"].to_list()
    # flags 1 0 1 1 0 0 -> windows 2/3, 2/3, 2/3, 1/3
    assert out[2:] == pytest.approx([2 / 3, 2 / 3, 2 / 3, 1 / 3])


def test_pv_corr():
    la = [0.1, 0.2, 0.4, 0.1]
    df = _df(ret=[math.expm1(x) for x in la], turnover_inr=[math.exp(x) for x in (1, 2, 4, 3)])
    out = _run(df, p=volume.pv_corr(3))["p"].to_list()
    # row 2: la = ln(turnover) / 10 -> corr 1 ; row 3: la [.2,.4,.1], ln turnover [2,4,3]:
    # cov sum .2, sq la 7/150, sq x 2 -> .2 / sqrt(14/150)
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(0.2 / math.sqrt(14 / 150))


def test_registered_features():
    names = [
        "log_turn_21", "amihud_21", "turn_shock", "vol_shock_5", "turn_trend", "avg_trade_size_rel",
        "zero_ret_frac_63", "size_proxy", "pv_corr_21",
    ]  # fmt: skip
    for n in names:
        assert REG[n].label and REG[n].group == "volume" and REG[n].kind == "xs", n
