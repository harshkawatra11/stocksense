from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from stocksense.data import io
from stocksense.features.base import FeatureContext
from stocksense.features.derivs import R_FREE, bs_price, implied_vol, max_pain, oi_price_quad
from stocksense.features.registry import load_all
from tests.fixtures.feat58 import write_env
from tests.fixtures.handmade import weekdays

REG = load_all()
START = date(2024, 7, 8)
E1 = date(2024, 9, 26)
E2 = date(2024, 10, 31)
N = 27
STRIKES = [90.0 + 5 * k for k in range(0, 11)]  # 90 .. 140


def col(s, name, as_of=None) -> pl.DataFrame:
    spec = REG[name]
    return spec.fn(FeatureContext(s, as_of)).collect().sort("sid", "date")


def test_bs_price_and_implied_vol_round_trip():
    # at-the-money call, S = K = 100, T = 0.5, r = 6.5%, sigma = 0.3
    p = bs_price(100.0, 100.0, 0.5, R_FREE, 0.3, True)
    assert p == pytest.approx(10.0078, abs=1e-3)  # d1=0.2593, d2=0.0472: 60.23 - 100*e^-0.0325*0.5188
    assert implied_vol(p, 100.0, 100.0, 0.5, True) == pytest.approx(0.3, abs=1e-6)
    q = bs_price(100.0, 105.0, 0.25, R_FREE, 0.45, False)
    assert implied_vol(q, 100.0, 105.0, 0.25, False) == pytest.approx(0.45, abs=1e-6)


def test_implied_vol_returns_none_when_unusable():
    assert implied_vol(0.4, 100.0, 100.0, 0.1, True) is None  # settle below 0.5
    assert implied_vol(5.0, 120.0, 100.0, 0.1, True) is None  # intrinsic 20 exceeds the price
    assert implied_vol(60.0, 100.0, 100.0, 0.1, True) is None  # no volatility in [0.01, 5] gives 60
    assert implied_vol(2.0, 100.0, 100.0, 0.0, True) is None  # no time left


def test_max_pain_by_hand():
    strikes = np.array([100.0, 110.0, 120.0])
    calls = np.array([0.0, 10.0, 0.0])  # 10 call contracts at 110
    puts = np.array([0.0, 10.0, 0.0])  # 10 put contracts at 110
    assert max_pain(strikes, calls, puts) == 110.0
    # 5 calls at 100, 10 puts at 120: expiring at 100 pays 200, at 110 pays 50 + 100, at 120 pays 100
    assert max_pain(strikes, np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0])) == 120.0
    # tie between the lowest strikes picks the lowest
    assert max_pain(strikes, np.zeros(3), np.zeros(3)) == 100.0


def test_oi_price_quad_table():
    df = pl.DataFrame(
        {"p": [0.1, -0.1, 0.1, -0.1, 0.0, 0.1, None], "o": [5.0, 5.0, -5.0, -5.0, 5.0, 0.0, 5.0]}
    )
    out = df.select(oi_price_quad(pl.col("p"), pl.col("o")).alias("q"))["q"].to_list()
    assert out == [1.0, -1.0, 0.5, -0.5, None, None, None]


def _fo_rows(sym, dates, closes, iv_c, iv_p, oi_put0=200.0, oi_call=100.0):
    rows = []
    for i, d in enumerate(dates):
        spot = closes[i]
        for exp, oi_f in ((E1, 1000.0 + 10 * i), (E2, 500.0)):
            rows.append((sym, "STF", exp, None, None, d, spot + 1.0, oi_f))
        t = (E1 - d).days / 365.0
        for k in STRIKES:
            sc = iv_c(i)
            sp = iv_p(i)
            rows.append((sym, "STO", E1, k, "CE", d, bs_price(spot, k, t, R_FREE, sc, True), oi_call))
            rows.append(
                (sym, "STO", E1, k, "PE", d, bs_price(spot, k, t, R_FREE, sp, False), oi_put0 + 10 * i)
            )
    cols = [
        "symbol",
        "instrument_type",
        "expiry_date",
        "strike_price",
        "option_type",
        "date",
        "settle_price",
        "oi",
    ]
    return pl.DataFrame(
        rows, schema=cols, orient="row", schema_overrides={"strike_price": pl.Float64, "option_type": pl.Utf8}
    )


@pytest.fixture
def world(tmp_data_root):
    s = tmp_data_root
    ds = weekdays(N, START)
    closes = [100.0 + i for i in range(N)]
    write_env(
        s,
        ds,
        {
            "XYZ": {"sid": 1, "close": closes},
            "NOFO": {"sid": 2, "close": [50.0 + 0.5 * i for i in range(N)]},
        },
    )
    fo = _fo_rows("XYZ", ds, closes, lambda i: 0.25 + 0.001 * i, lambda i: 0.30 + 0.001 * i)
    io.upsert(s, "fo_bhavcopy", fo)
    return s, ds, closes


def at(df, name, sid, d):
    return df.filter((pl.col("sid") == sid) & (pl.col("date") == d))[name][0]


def test_basis_and_days_to_expiry(world):
    s, ds, closes = world
    i = 10
    dte = (E1 - ds[i]).days
    assert at(col(s, "days_to_expiry"), "days_to_expiry", 1, ds[i]) == dte
    expect = (1.0 / closes[i]) * 365.0 / dte
    assert at(col(s, "fut_basis"), "fut_basis", 1, ds[i]) == pytest.approx(expect)


def test_near_expiry_needs_three_days_left(tmp_data_root):
    s = tmp_data_root
    ds = weekdays(3, date(2024, 9, 23))  # Mon 23, Tue 24, Wed 25 Sep; E1 = Thu 26 Sep
    write_env(s, ds, {"XYZ": {"sid": 1, "close": [100.0, 100.0, 100.0]}})
    rows = []
    for d in ds:
        for exp, settle in ((E1, 101.0), (E2, 102.0)):
            rows.append(("XYZ", "STF", exp, None, None, d, settle, 10.0))
    cols = [
        "symbol",
        "instrument_type",
        "expiry_date",
        "strike_price",
        "option_type",
        "date",
        "settle_price",
        "oi",
    ]
    fo = pl.DataFrame(
        rows, schema=cols, orient="row", schema_overrides={"strike_price": pl.Float64, "option_type": pl.Utf8}
    )
    io.upsert(s, "fo_bhavcopy", fo)
    out = col(s, "days_to_expiry")
    assert out["days_to_expiry"].to_list() == [3, (E2 - ds[1]).days, (E2 - ds[2]).days]  # 2 days left -> next


def test_oi_features(world):
    s, ds, _ = world
    i = 25
    oi = np.array([1000.0 + 10 * k + 500.0 for k in range(N)])
    exp = (oi[i] - oi[i - 5]) / oi[i - 19 : i + 1].mean()
    assert at(col(s, "fut_oi_chg_5"), "fut_oi_chg_5", 1, ds[i]) == pytest.approx(exp)
    assert at(col(s, "fut_oi_chg_5"), "fut_oi_chg_5", 1, ds[18]) is None  # needs 20 sessions of OI history
    assert at(col(s, "oi_price_quad"), "oi_price_quad", 1, ds[i]) == 1.0  # price up, OI up


def test_pcr(world):
    s, ds, _ = world
    n = len(STRIKES)
    pcr = [n * (200.0 + 10 * k) / (n * 100.0) for k in range(N)]
    assert at(col(s, "pcr_oi"), "pcr_oi", 1, ds[3]) == pytest.approx(pcr[3])
    prev = np.array(pcr[5:25])
    exp = (pcr[25] - prev.mean()) / prev.std(ddof=1)
    assert at(col(s, "pcr_oi_z_20"), "pcr_oi_z_20", 1, ds[25]) == pytest.approx(exp)
    assert at(col(s, "pcr_oi_z_20"), "pcr_oi_z_20", 1, ds[19]) is None


def test_implied_vol_features(world):
    s, ds, closes = world
    i = 8
    atm = col(s, "atm_iv")
    # the call is priced at 0.25 + 0.001 i, the put 0.05 higher: ATM IV is the mean
    assert at(atm, "atm_iv", 1, ds[i]) == pytest.approx(0.275 + 0.001 * i, abs=1e-6)
    assert at(col(s, "skew_put_call"), "skew_put_call", 1, ds[i]) == pytest.approx(0.05, abs=1e-6)
    assert at(col(s, "iv_chg_5"), "iv_chg_5", 1, ds[i]) == pytest.approx(0.005, abs=1e-6)
    assert at(col(s, "iv_chg_5"), "iv_chg_5", 1, ds[4]) is None
    # realised vol over 21 sessions of the (+1 per day) price path
    j = 24
    la = np.diff(np.log(closes))
    rv = la[j - 21 : j].std(ddof=1) * math.sqrt(252)
    assert at(col(s, "iv_minus_rv"), "iv_minus_rv", 1, ds[j]) == pytest.approx(
        0.275 + 0.001 * j - rv, abs=1e-6
    )


def test_max_pain_dist(world):
    s, ds, closes = world
    i = 12
    # uniform call OI and heavier uniform put OI over strikes 90..140
    # because put OI exceeds call OI; recompute with the module's own algorithm
    k = np.array(STRIKES)
    calls = np.full(len(k), 100.0)
    puts = np.full(len(k), 200.0 + 10 * i)
    pain = max_pain(k, calls, puts)
    assert at(col(s, "max_pain_dist"), "max_pain_dist", 1, ds[i]) == pytest.approx(
        (pain - closes[i]) / closes[i]
    )


def test_non_fno_names_and_missing_data_are_null(world, tmp_data_root):
    s, ds, _ = world
    for name in ("fut_basis", "atm_iv", "pcr_oi", "max_pain_dist", "days_to_expiry"):
        assert col(s, name).filter(pl.col("sid") == 2)[name].null_count() == N
    # no fo_bhavcopy at all: null, not an error
    from tests.fixtures.synth import _settings_for

    root = s.paths.data_root.parent / "empty_root"
    root.mkdir()
    s2 = _settings_for(root)
    write_env(s2, ds, {"XYZ": {"sid": 1, "close": [100.0 + i for i in range(N)]}})
    for name in REG:
        if REG[name].group == "derivs":
            out = col(s2, name)
            assert out.height == N and out[name].null_count() == N, name
