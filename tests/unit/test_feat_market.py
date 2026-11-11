from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.data import io
from stocksense.features.base import FeatureContext
from stocksense.features.market import pctile_last
from stocksense.features.registry import load_all
from tests.fixtures.feat58 import write_env
from tests.fixtures.handmade import weekdays

REG = load_all()


def run(s, name, as_of=None) -> pl.DataFrame:
    spec = REG[name]
    return spec.fn(FeatureContext(s, as_of)).collect().sort("date" if spec.kind == "mkt" else ["sid", "date"])


def col(s, name, as_of=None) -> list:
    return run(s, name, as_of)[name].to_list()


def env(s, n=30, rets=None, **mk):
    ds = weekdays(n)
    rets = rets if rets is not None else [0.01 * ((i % 3) - 1) for i in range(n)]
    close = [100.0]
    for r in rets[1:]:
        close.append(close[-1] * (1 + r))
    market = {"date": ds, "ew500_ret": rets, **mk}
    write_env(s, ds, {"A": {"sid": 1, "close": close}}, market=market)
    return ds


def test_pctile_last_by_hand():
    x = np.array([2.0, 1.0, 5.0, 9.0, 5.0])
    out = pctile_last(x, 4)
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx(1.0)  # 9 is the max of [2,1,5,9]: 3 of the 3 others are below
    assert out[4] == pytest.approx((1 + 0.5) / 3)  # window [1,5,9,5]: others 1,5,9 -> 1 below, 1 equal


def test_mkt_returns_by_hand(tmp_data_root):
    rets = [0.0, 0.01, 0.02, -0.01, 0.03, 0.0, 0.01]
    env(tmp_data_root, 7, rets)
    ln = [math.log(1 + r) for r in rets]
    assert col(tmp_data_root, "mkt_ret_1")[1] == pytest.approx(ln[1])
    r5 = col(tmp_data_root, "mkt_ret_5")
    assert r5[3] is None and r5[5] == pytest.approx(sum(ln[1:6])) and r5[6] == pytest.approx(sum(ln[2:7]))
    assert all(x is None for x in col(tmp_data_root, "mkt_vol_21"))


def test_mkt_vol_21_and_ret_21(tmp_data_root):
    rets = [0.0] + [0.01 * ((i % 4) - 1.5) for i in range(1, 25)]
    env(tmp_data_root, 25, rets)
    ln = np.array([math.log(1 + r) for r in rets])
    assert col(tmp_data_root, "mkt_vol_21")[24] == pytest.approx(np.std(ln[4:25], ddof=1))
    assert col(tmp_data_root, "mkt_ret_21")[24] == pytest.approx(ln[4:25].sum())


def test_co_oc_and_breadth(tmp_data_root):
    n = 12
    co = [0.001 * i for i in range(n)]
    oc = [-0.002] * n
    br = [0.1 * (i % 5) for i in range(n)]
    env(tmp_data_root, n, ew500_co=co, ew500_oc=oc, breadth_200=br, xs_dispersion=[0.02] * n)
    c5 = col(tmp_data_root, "mkt_co_5")
    assert c5[4] == pytest.approx(sum(math.log(1 + x) for x in co[0:5]))
    assert col(tmp_data_root, "mkt_oc_5")[4] == pytest.approx(5 * math.log(1 - 0.002))
    assert col(tmp_data_root, "breadth_200") == pytest.approx(br)
    bc = col(tmp_data_root, "breadth_chg_10")
    assert bc[10] == pytest.approx(br[10] - br[0]) and bc[9] is None


def test_xs_disp_ratio(tmp_data_root):
    n = 62
    disp = [0.01] * 61 + [0.03]
    env(tmp_data_root, n, xs_dispersion=disp)
    assert col(tmp_data_root, "xs_dispersion")[61] == pytest.approx(0.03)
    ratio = col(tmp_data_root, "xs_disp_ratio")
    assert ratio[58] is None
    assert ratio[61] == pytest.approx(0.03 / ((0.01 * 59 + 0.03) / 60))


def test_adv_dec_10(tmp_data_root):
    s = tmp_data_root
    ds = weekdays(12)
    a = [100.0 + i for i in range(12)]  # always up
    b = [100.0 - i for i in range(12)]  # always down
    c = [100.0] * 12  # flat
    write_env(s, ds, {"A": {"sid": 1, "close": a}, "B": {"sid": 2, "close": b}, "C": {"sid": 3, "close": c}})
    v = col(s, "adv_dec_10")
    assert v[9] is None  # row 0 has no return, so 10 usable sessions first exist at index 10
    assert v[10] == pytest.approx(0.0)  # 1 advancer, 1 decliner -> 0
    write_env(s, ds, {"A": {"sid": 1, "close": a}, "B": {"sid": 2, "close": c}, "C": {"sid": 3, "close": c}})
    assert col(s, "adv_dec_10")[10] == pytest.approx(1.0)  # 1 advancer, 0 decliners -> 1
    # more decliners than advancers must give a negative number (guards against unsigned wrap-around)
    write_env(s, ds, {"A": {"sid": 1, "close": c}, "B": {"sid": 2, "close": b}, "C": {"sid": 3, "close": b}})
    assert col(s, "adv_dec_10")[10] == pytest.approx(-1.0)


def test_vix_features(tmp_data_root):
    s = tmp_data_root
    n = 300
    ds = env(s, n)
    vix = [10.0 + i * 0.1 for i in range(n)]
    io.upsert(s, "india_vix", pl.DataFrame({"date": ds, "close": vix}))
    assert col(s, "vix")[7] == pytest.approx(vix[7])
    ch = col(s, "vix_chg_5")
    assert ch[4] is None and ch[5] == pytest.approx(math.log(vix[5] / vix[0]))
    pc = col(s, "vix_pctile_252")
    assert pc[250] is None and pc[299] == pytest.approx(1.0)  # strictly rising: every other value is below


def test_nifty_pe_pctile(tmp_data_root):
    s = tmp_data_root
    n = 800
    ds = env(s, n)
    pe = [20.0 + (i % 50) for i in range(n)]
    ix = pl.DataFrame({"index_name": ["NIFTY 50"] * n, "date": ds, "close": [1.0] * n, "pe": pe})
    io.upsert(s, "index_daily", ix)
    out = col(s, "nifty_pe_pctile_756")
    assert out[754] is None and out[799] is not None
    w = np.array(pe[799 - 755 : 800])
    others = w[:-1]
    assert out[799] == pytest.approx(((others < w[-1]).sum() + 0.5 * (others == w[-1]).sum()) / 755)


def _macro(s, series, ds, vals, avail_offset_h):
    rows = [
        (series, d, v, datetime.combine(d, datetime.min.time(), tzinfo=IST) + timedelta(hours=avail_offset_h))
        for d, v in zip(ds, vals, strict=True)
    ]
    io.upsert(
        s, "macro_daily", pl.DataFrame(rows, schema=["series", "date", "close", "available_ts"], orient="row")
    )


def test_macro_respects_available_ts(tmp_data_root):
    s = tmp_data_root
    ds = env(s, 10)
    # USDINR is published at 17:00 IST the same day -> usable on session t
    _macro(s, "USDINR", ds, [80.0 + i for i in range(10)], 17)
    # SPX: the US close of date d reaches us at 05:30 IST on d+1 -> NOT usable on session d
    _macro(s, "SPX", ds, [4000.0 + 10 * i for i in range(10)], 29.5)
    assert col(s, "usdinr_chg_1")[3] == pytest.approx(math.log(83 / 82))
    assert col(s, "usdinr_chg_5")[5] == pytest.approx(math.log(85 / 80))
    spx = col(s, "spx_chg_1")
    # on session index 3 the newest usable SPX is index 2 (index 3 arrives after 20:00); chg = ln(idx2 / idx1)
    assert spx[3] == pytest.approx(math.log(4020 / 4010))
    # session 6 is a Tuesday-ish weekday; newest usable is index 5, five sessions back is index 0
    assert col(s, "spx_chg_5")[6] == pytest.approx(math.log(4050 / 4000))


def test_us10y_is_a_difference(tmp_data_root):
    s = tmp_data_root
    ds = env(s, 8)
    _macro(s, "US10Y", ds, [4.0, 4.1, 4.3, 4.2, 4.4, 4.5, 4.6, 4.8], 17)
    assert col(s, "us10y_chg_1")[2] == pytest.approx(0.2)
    assert col(s, "us10y_chg_5")[5] == pytest.approx(0.5)


def test_fii_dii_flows(tmp_data_root):
    s = tmp_data_root
    ds = env(s, 8)
    rows = []
    for i, d in enumerate(ds):
        rows.append((d, "FII", 0.0, 0.0, 1000.0 * (i + 1)))
        rows.append((d, "DII", 0.0, 0.0, -500.0))
    cols = ["date", "category", "buy_value", "sell_value", "net_value"]
    io.upsert(s, "fii_dii", pl.DataFrame(rows, schema=cols, orient="row"))
    f = col(s, "fii_net_5")
    assert f[3] is None
    assert f[5] == pytest.approx((2000 + 3000 + 4000 + 5000 + 6000) / 1e4)
    assert col(s, "dii_net_5")[4] == pytest.approx(-0.25)


def test_fii_index_futures(tmp_data_root):
    s = tmp_data_root
    ds = env(s, 8)
    rows = []
    for i, d in enumerate(ds):
        rows.append((d, "FII", 100.0 + 10 * i, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        rows.append((d, "Client", 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    cols = [
        "date",
        "client_type",
        "fut_idx_long",
        "fut_idx_short",
        "fut_stk_long",
        "fut_stk_short",
        "opt_idx_call_long",
        "opt_idx_put_long",
        "opt_idx_call_short",
        "opt_idx_put_short",
    ]
    io.upsert(s, "participant_oi", pl.DataFrame(rows, schema=cols, orient="row"))
    assert col(s, "fii_idx_fut_net")[2] == pytest.approx(20 / 220)
    chg = col(s, "fii_idx_fut_chg_5")
    assert chg[5] == pytest.approx(50 / 250 - 0.0)
    assert chg[4] is None


def test_missing_sources_give_all_null(tmp_data_root):
    env(tmp_data_root, 10)
    names = (
        "vix",
        "vix_chg_5",
        "vix_pctile_252",
        "nifty_pe_pctile_756",
        "usdinr_chg_1",
        "spx_chg_5",
        "fii_net_5",
        "dii_net_5",
        "fii_idx_fut_net",
        "fii_idx_fut_chg_5",
    )
    for name in names:
        out = run(tmp_data_root, name)
        assert out.height == 10 and out[name].null_count() == 10, name


def test_industry_features(tmp_data_root):
    s = tmp_data_root
    n = 140
    ds = weekdays(n)
    stocks = {}
    for k in range(6):  # six Banks with different steady drifts
        stocks[f"B{k}"] = {
            "sid": k + 1,
            "close": [100.0 * (1 + 0.001 * (k + 1)) ** i for i in range(n)],
            "industry": "Banks",
        }
    for k in range(3):  # only three Pharma -> too few members
        stocks[f"P{k}"] = {"sid": 10 + k, "close": [100.0 * 1.001**i for i in range(n)], "industry": "Pharma"}
    write_env(s, ds, stocks)
    last = ds[-1]

    def at(name, sid):
        df = run(s, name)
        return df.filter((pl.col("sid") == sid) & (pl.col("date") == last))[name][0]

    m21 = {k: 21 * math.log(1 + 0.001 * (k + 1)) for k in range(6)}
    exp = (sum(m21.values()) - m21[2]) / 5
    assert at("ind_mom_21", 3) == pytest.approx(exp)
    assert at("rel_ind_mom_21", 3) == pytest.approx(m21[2] - exp)
    m126 = {k: 126 * math.log(1 + 0.001 * (k + 1)) for k in range(6)}
    assert at("ind_mom_126", 1) == pytest.approx((sum(m126.values()) - m126[0]) / 5)
    assert at("ind_mom_21", 10) is None  # Pharma has 3 members
    assert at("ind_breadth_50", 1) == pytest.approx(1.0)  # every Bank is above its 50-row mean
    assert at("ind_breadth_50", 10) is None
