"""T4.3d: risk features, each hand-computed."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.features import risk
from stocksense.features.registry import load_all

REG = load_all()
LN2 = math.log(2.0)


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


def _run(df: pl.DataFrame, beta_w: int = 126, **exprs: pl.Expr) -> pl.DataFrame:
    return risk._stage(df.lazy(), beta_w=beta_w).select(*[e.alias(k) for k, e in exprs.items()]).collect()


LA = [0.1, 0.2, -0.1, 0.3, 0.0, 0.1]


def test_vol_is_sample_std_of_log_returns():
    out = _run(_df(ret=_r(LA)), v=risk.vol(3))["v"].to_list()
    # row 2: [.1,.2,-.1] -> var 7/300 ; row 3: [.2,-.1,.3] -> var .043333
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(math.sqrt(7 / 300))
    assert out[3] == pytest.approx(math.sqrt(0.13 / 3))


def test_vol_ratio():
    out = _run(_df(ret=_r(LA)), q=risk.vol_ratio(2, 4))["q"].to_list()
    # row 5: std2 of [0,.1] = .1/sqrt2 ; std4 of [-.1,.3,0,.1] = sqrt(.0875/3)
    assert out[5] == pytest.approx((0.1 / math.sqrt(2)) / math.sqrt(0.0875 / 3))
    assert out[2] is None  # the 4-row window is not full yet


def test_parkinson():
    lnhl = [0.1, 0.2, 0.3]
    low = [10.0] * 3
    high = [10.0 * math.exp(x) for x in lnhl]
    df = _df(adj_high=high, adj_low=low, adj_open=low, adj_close=low)
    out = _run(df, p=risk.parkinson(2))["p"].to_list()
    # row 1: mean(.1^2, .2^2) = .025 -> sqrt(.025 / (4 ln 2)) ; row 2: mean(.04, .09) = .065
    assert out[0] is None
    assert out[1] == pytest.approx(math.sqrt(0.025 / (4 * LN2)))
    assert out[2] == pytest.approx(math.sqrt(0.065 / (4 * LN2)))


def test_garman_klass_and_clip():
    lnhl = [0.1, 0.2, 0.0]
    lnco = [0.05, -0.1, 0.1]
    low = [10.0] * 3
    df = _df(
        adj_high=[10.0 * math.exp(x) for x in lnhl],
        adj_low=low,
        adj_open=low,
        adj_close=[10.0 * math.exp(x) for x in lnco],
    )
    out = _run(df, g=risk.garman_klass(2))["g"].to_list()
    k = 2 * LN2 - 1
    t0 = 0.5 * 0.1**2 - k * 0.05**2  # .005 - .386294 * .0025
    t1 = 0.5 * 0.2**2 - k * 0.1**2  # .02 - .386294 * .01
    t2 = 0.5 * 0.0**2 - k * 0.1**2  # negative
    assert out[1] == pytest.approx(math.sqrt((t0 + t1) / 2))
    # row 2: mean(t1, t2) = (.016137 - .003863)/2 > 0
    assert out[2] == pytest.approx(math.sqrt((t1 + t2) / 2))
    # a window whose mean is negative is clipped at 0
    neg = _df(adj_high=low, adj_low=low, adj_open=low, adj_close=[11.0, 11.0, 11.0])
    assert _run(neg, g=risk.garman_klass(2))["g"].to_list()[1:] == [0.0, 0.0]


def test_downside_vol():
    la = [0.1, -0.2, 0.3, -0.1]
    out = _run(_df(ret=_r(la)), d=risk.downside_vol(3))["d"].to_list()
    # squares of min(la, 0): 0, .04, 0, .01 -> row 2: sqrt(.04/3) ; row 3: sqrt(.05/3)
    assert out[2] == pytest.approx(math.sqrt(0.04 / 3))
    assert out[3] == pytest.approx(math.sqrt(0.05 / 3))


def test_skew_and_kurtosis_sample_estimators():
    la = [1.0, 2.0, 4.0, 9.0]  # treat as log returns via ret = expm1
    out = _run(_df(ret=_r(la)), s=risk.skew(4), k=risk.kurt(4))
    # mean 4, deviations -3 -2 0 5 : sum sq 38 (s^2 = 38/3), sum cubes 90, sum 4th powers 722
    # G1 = n/((n-1)(n-2)) * 90 / s^3 = (4/6) * 90 / (38/3)^1.5
    assert out["s"][3] == pytest.approx((4 / 6) * 90 / (38 / 3) ** 1.5)
    # G2 = n(n+1)/((n-1)(n-2)(n-3)) * 722/s^4 - 3 (n-1)^2/((n-2)(n-3)) = (20/6)(4.5) - 13.5 = 1.5
    assert out["k"][3] == pytest.approx(1.5)
    assert out["s"][2] is None


def test_atr_14_pct_wilder_by_hand_n3():
    high = [10.0, 11.0, 12.0, 13.0, 12.0]
    low = [8.0, 9.0, 9.0, 10.0, 9.0]
    close = [9.0, 10.0, 11.0, 12.0, 11.0]
    out = _run(_df(adj_high=high, adj_low=low, adj_close=close), a=risk.atr_pct(3))["a"].to_list()
    # true ranges from row 1: max(h-l, |h-prev c|, |l-prev c|) = 2, 3, 3, 3
    # row 3: seed = (2+3+3)/3 = 8/3 -> /12 ; row 4: (8/3*2 + 3)/3 = 25/9 -> /11
    assert out[:3] == [None, None, None]
    assert out[3] == pytest.approx((8 / 3) / 12)
    assert out[4] == pytest.approx((25 / 9) / 11)


MKT_LA = [0.2, 0.4, 0.6, 0.5, 0.2, 0.7]
MKT_LM = [0.1, 0.2, 0.3, 0.1, 0.2, 0.4]


def test_beta_and_correlation_with_the_market():
    df = _df(ret=_r(MKT_LA), lm=MKT_LM)
    out = _run(df, beta_w=3, b=risk.beta(3), c=risk.corr_mkt(3))
    # rows 0-2: la = 2 lm -> beta 2, corr 1 ; rows 1-3: cov .005, var .01 -> .5 ; rows 2-4 .5 ;
    # rows 3-5: cov .021667, var .023333 -> 13/14
    assert out["b"].to_list()[2:] == pytest.approx([2.0, 0.5, 0.5, 13 / 14])
    # rows 1-3: la devs (-.1, .1, 0), lm devs (0, .1, -.1): corr = .01 / .02 = .5
    assert out["c"][2] == pytest.approx(1.0) and out["c"][3] == pytest.approx(0.5)


def test_idio_vol_uses_the_previous_rows_beta():
    out = _run(_df(ret=_r(MKT_LA), lm=MKT_LM), beta_w=3, i=risk.idio_vol(2))["i"].to_list()
    # e3 = .5 - 2(.1) = .3 ; e4 = .2 - .5(.2) = .1 ; e5 = .7 - .5(.4) = .5
    # std([.3,.1]) = .2/sqrt2 ; std([.1,.5]) = .4/sqrt2
    assert out[:4] == [None] * 4
    assert out[4] == pytest.approx(0.2 / math.sqrt(2))
    assert out[5] == pytest.approx(0.4 / math.sqrt(2))


def test_registered_features():
    names = [
        "vol_21", "vol_63", "vol_ratio", "parkinson_21", "gk_21", "downside_vol_63", "skew_63", "kurt_63",
        "atr_14_pct", "beta_126", "idio_vol_63", "corr_mkt_63",
    ]  # fmt: skip
    for n in names:
        assert REG[n].label and REG[n].group == "risk" and REG[n].kind == "xs", n
    assert REG["vol_21"].fn.__module__ == "stocksense.features.risk"
