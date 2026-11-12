"""T4.3c: overnight / intraday split features, each hand-computed."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.features import overnight
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


def _r(la: list[float]) -> list[float]:
    return [math.expm1(x) for x in la]


def _run(df: pl.DataFrame, **exprs: pl.Expr) -> pl.DataFrame:
    return overnight._stage(df.lazy()).select(*[e.alias(k) for k, e in exprs.items()]).collect()


LCO = [0.1, 0.2, -0.1, 0.3, 0.0, 0.1]


def test_one_day_values_are_log_returns():
    loc = [0.05, -0.05, 0.0, 0.1, 0.2, 0.3]
    df = _df(ret=_r(LCO), ret_co=_r(LCO), ret_oc=_r(loc))
    out = _run(df, a=overnight.co_1(), b=overnight.oc_1())
    assert out["a"].to_list() == pytest.approx(LCO)
    assert out["b"].to_list() == pytest.approx(loc)


def test_rolling_sums():
    df = _df(ret=_r(LCO), ret_co=_r(LCO), ret_oc=_r(LCO))
    out = _run(df, c=overnight.co_sum(3), o=overnight.oc_sum(3))
    # windows of 3: .1+.2-.1 = .2 ; .2-.1+.3 = .4 ; -.1+.3+0 = .2 ; .3+0+.1 = .4
    assert out["c"].to_list()[:2] == [None, None]
    assert out["c"].to_list()[2:] == pytest.approx([0.2, 0.4, 0.2, 0.4])
    assert out["o"].to_list()[2:] == pytest.approx([0.2, 0.4, 0.2, 0.4])


def test_co_minus_oc():
    lco = [0.1, 0.2, -0.1, 0.3]
    loc = [0.05, 0.05, 0.1, -0.1]
    out = _run(_df(ret=_r(lco), ret_co=_r(lco), ret_oc=_r(loc)), d=overnight.co_minus_oc(3))["d"].to_list()
    # row 2: (.1+.2-.1) - (.05+.05+.1) = 0 ; row 3: (.2-.1+.3) - (.05+.1-.1) = .4 - .05 = .35
    assert out[2:] == pytest.approx([0.0, 0.35])


def test_co_vol_share():
    lco = [0.1, 0.2, -0.1]
    loc = [0.05, 0.05, 0.1]
    out = _run(_df(ret=_r(lco), ret_co=_r(lco), ret_oc=_r(loc)), s=overnight.co_vol_share(3))["s"].to_list()
    # var(lco) = 7/300 ; var(loc) = 1/1200 -> share (28/1200) / (29/1200) = 28/29
    assert out[2] == pytest.approx(28 / 29)


def test_gap_abs_z_uses_yesterdays_volatility():
    la = [0.1, 0.2, -0.1, 0.3, 0.0]
    out = _run(_df(ret=_r(la), ret_co=_r(la), ret_oc=_r(la)), z=overnight.gap_abs_z(3))["z"].to_list()
    # std3 of la at row 2 = sqrt(7/300) = .152753, at row 3 (window .2,-.1,.3) = sqrt(.043333) = .208167
    # row 3: |.3| / std3(row 2) ; row 4: |0| / std3(row 3) = 0 ; row 2 has no previous std -> null
    assert out[:3] == [None, None, None]
    assert out[3] == pytest.approx(0.3 / math.sqrt(7 / 300))
    assert out[4] == pytest.approx(0.0)


def test_gap_fill():
    lco = [0.1, -0.2, 0.3, 0.0]
    loc = [0.05, 0.1, -0.2, 0.4]
    out = _run(_df(ret=_r(lco), ret_co=_r(lco), ret_oc=_r(loc)), g=overnight.gap_fill(3))["g"].to_list()
    # x = -sign(lco) * loc = [-.05, .1, .2, 0 (no gap)] ; 3-row means: .25/3 and .3/3
    assert out[2:] == pytest.approx([0.25 / 3, 0.1])


def test_co_hit():
    lco = [0.1, 0.2, -0.1, 0.3, 0.0]
    out = _run(_df(ret=_r(lco), ret_co=_r(lco), ret_oc=_r(lco)), h=overnight.co_hit(3))["h"].to_list()
    # up flags 1 1 0 1 0 (zero is not up) ; windows: 2/3, 2/3, 1/3
    assert out[2:] == pytest.approx([2 / 3, 2 / 3, 1 / 3])


def test_registered_features():
    names = [
        "co_1", "co_sum_5", "co_sum_21", "oc_1", "oc_sum_5", "oc_sum_21", "co_minus_oc_63",
        "co_vol_share_63", "gap_abs_z", "gap_fill_5", "co_hit_21",
    ]  # fmt: skip
    for n in names:
        assert REG[n].label and REG[n].group == "overnight" and REG[n].kind == "xs", n
