"""T4.3f: delivery features (null where the delivery data does not exist), each hand-computed."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.features import delivery
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
    return delivery._stage(df.lazy()).select(*[e.alias(k) for k, e in exprs.items()]).collect()


PCT = [40.0, 50.0, 60.0, 70.0, 40.0, 50.0]


def test_deliv_pct_passes_through_with_nulls():
    out = _run(_df(deliv_pct=[40.0, None, 60.0]), p=delivery.deliv_pct())["p"].to_list()
    assert out == [40.0, None, 60.0]


def test_deliv_z_uses_the_previous_window():
    out = _run(_df(deliv_pct=PCT), z=delivery.deliv_z(3))["z"].to_list()
    # mean3 / std3 at row 2: 50 / 10 ; row 3 (window 50,60,70): 60 / 10. Shifted one row:
    # row 3: (70 - 50)/10 = 2 ; row 4: (40 - 60)/10 = -2
    assert out[:3] == [None, None, None]
    assert out[3] == pytest.approx(2.0)
    assert out[4] == pytest.approx(-2.0)


def test_deliv_z_is_null_without_delivery_data():
    out = _run(_df(deliv_pct=[None] * 6), z=delivery.deliv_z(3))["z"].to_list()
    assert out == [None] * 6


def test_deliv_val_shock():
    df = _df(deliv_qty=[10.0, 20.0, 30.0, 40.0], close=[1.0, 1.0, 2.0, 2.0])
    out = _run(df, s=delivery.deliv_val_shock(2))["s"].to_list()
    # delivered value 10, 20, 60, 80 ; 2-row means 15 (row 1), 40 (row 2) used one row later:
    # row 2: 60/15 = 4 ; row 3: 80/40 = 2
    assert out[:2] == [None, None]
    assert out[2:] == pytest.approx([4.0, 2.0])


def test_deliv_up_signs_the_z_by_the_days_move():
    ret = [0.0, 0.0, 0.0, -0.05, 0.02, 0.0]
    out = _run(_df(deliv_pct=PCT, ret=ret), u=delivery.deliv_up(3))["u"].to_list()
    # z = 2 (row 3, down day -> -2), -2 (row 4, up day -> -2), row 5 flat -> 0
    assert out[3] == pytest.approx(-2.0)
    assert out[4] == pytest.approx(-2.0)
    assert out[5] == pytest.approx(0.0)


def test_deliv_trend():
    out = _run(_df(deliv_pct=[10.0, 20.0, 30.0, 40.0]), t=delivery.deliv_trend(2, 3))["t"].to_list()
    # row 3: mean2 = 35, mean3 = 30
    assert out[3] == pytest.approx(35 / 30)


def test_deliv_z_mean():
    out = _run(_df(deliv_pct=PCT), m=delivery.deliv_z_mean(2, 3))["m"].to_list()
    # z: row 3 = 2, row 4 = -2, row 5 = (50 - 56.667)/15.275 = -0.43644 (window 60,70,40)
    z5 = (50 - 170 / 3) / math.sqrt(700 / 3)
    assert out[:4] == [None] * 4
    assert out[4] == pytest.approx(0.0)
    assert out[5] == pytest.approx((-2 + z5) / 2)


def test_registered_features():
    names = ["deliv_pct", "deliv_z", "deliv_val_shock", "deliv_up", "deliv_trend", "deliv_z_5"]
    for n in names:
        assert REG[n].label and REG[n].group == "delivery" and REG[n].kind == "xs", n
