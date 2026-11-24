from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.data import io
from stocksense.features.derivs import R_FREE, bs_price, max_pain
from stocksense.flow.options import bs_gamma, build_options_daily, chain_frame, chain_metrics

D = date(2026, 11, 20)
E1 = D + timedelta(days=90)
E2 = D + timedelta(days=120)
E3 = D + timedelta(days=150)


def test_bs_gamma_matches_finite_difference():
    s, k, t, r, sig = 100.0, 100.0, 0.25, 0.065, 0.2
    h = 0.01

    def delta(x: float) -> float:
        return (bs_price(x + h, k, t, r, sig, True) - bs_price(x - h, k, t, r, sig, True)) / (2 * h)

    fd = (delta(s + h) - delta(s - h)) / (2 * h)
    assert bs_gamma(s, k, t, r, sig) == pytest.approx(fd, abs=1e-4)
    assert bs_gamma(s, k, 0.0, r, sig) == 0.0 and bs_gamma(s, k, t, r, 0.0) == 0.0


def _row(exp, k, typ, oi, doi=0, settle=None, spot=100.0, iv=None, d=D, sym="AAA", kind="STO", sigma=0.2):
    t = (exp - d).days / 365
    if settle is None:
        settle = bs_price(spot, k, t, R_FREE, sigma, typ == "CE")
    row = {
        "symbol": sym,
        "instrument_type": kind,
        "expiry_date": exp,
        "strike_price": float(k),
        "option_type": typ,
        "date": d,
        "settle_price": settle,
        "oi": oi,
        "change_in_oi": doi,
        "underlying_price": spot,
    }
    if iv is not None:
        row["iv"] = iv
    return row


def _fo(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"oi": pl.Int64, "change_in_oi": pl.Int64})


def _basic_chain() -> pl.DataFrame:
    rows = []
    for k, c, p, cd, pd_ in [(90, 100, 400, 5, 30), (100, 200, 300, -10, 20), (110, 500, 50, 40, 0)]:
        rows.append(_row(E1, k, "CE", c, cd))
        rows.append(_row(E1, k, "PE", p, pd_))
    return _fo(rows)


def test_chain_frame_nearest_expiries_and_dte():
    rows = [_row(e, 100, "CE", 10) for e in (E3, E1, E2)] + [_row(E1, 100, "PE", 10, sym="BBB")]
    ch = chain_frame(_fo(rows), D, "AAA", n_expiries=2)
    assert sorted(ch["expiry_date"].unique().to_list()) == [E1, E2]
    assert set(ch["dte"].to_list()) == {90, 120}
    assert ch["t"].max() == pytest.approx(120 / 365)
    assert chain_frame(_fo(rows), D, "AAA", n_expiries=1)["expiry_date"].unique().to_list() == [E1]


def test_expiry_day_rows_excluded():
    rows = [_row(D, 100, "CE", 10, settle=100.0), _row(E2, 100, "CE", 10)]
    ch = chain_frame(_fo(rows), D, "AAA")
    assert ch["expiry_date"].unique().to_list() == [E2]
    assert chain_frame(_fo(rows[:1]), D, "AAA").height == 0


def test_chain_frame_uses_only_the_given_date():
    rows = [_row(E1, 100, "CE", 10), _row(E1, 100, "CE", 99, d=D + timedelta(days=1))]
    ch = chain_frame(_fo(rows), D, "AAA")
    assert ch["oi"].to_list() == [10]


def test_pcr_walls_and_max_pain_by_hand():
    ch = chain_frame(_basic_chain(), D, "AAA")
    m = chain_metrics(ch, 100.0)
    assert m["pcr_oi"] == pytest.approx(750 / 800)
    assert m["pcr_doi"] == pytest.approx(50 / 35)
    assert m["call_wall"] == 110.0 and m["put_wall"] == 90.0
    k = np.array([90.0, 100.0, 110.0])
    assert m["max_pain"] == max_pain(k, np.array([100.0, 200, 500]), np.array([400.0, 300, 50]))
    assert m["near_expiry"] == E1
    assert m["atm_iv"] == pytest.approx(0.2, abs=1e-3)
    assert m["skew"] == pytest.approx(0.0, abs=1e-3)


def test_put_over_call_is_one_and_a_half():
    rows = [_row(E1, 100, "CE", 200), _row(E1, 100, "PE", 300)]
    m = chain_metrics(chain_frame(_fo(rows), D, "AAA"), 100.0)
    assert m["pcr_oi"] == pytest.approx(1.5)
    assert m["pcr_doi"] is None  # call change in OI is 0


def test_atm_call_and_put_gex_cancel():
    rows = [_row(E1, 100, "CE", 1000, iv=0.2), _row(E1, 100, "PE", 1000, iv=0.2)]
    m = chain_metrics(chain_frame(_fo(rows), D, "AAA"), 100.0)
    assert m["gex_total_cr"] == pytest.approx(0.0, abs=1e-9)
    s = m["strikes"]
    assert s["gex_cr"].to_list() == [pytest.approx(0.0, abs=1e-9)]


def test_gex_sign_follows_dealer_convention():
    only_calls = _fo([_row(E1, 100, "CE", 1000, iv=0.2)])
    only_puts = _fo([_row(E1, 100, "PE", 1000, iv=0.2)])
    g = bs_gamma(100.0, 100.0, 90 / 365, R_FREE, 0.2)
    want = g * 1000 * 100.0**2 * 0.01 / 1e7
    assert chain_metrics(chain_frame(only_calls, D, "AAA"), 100.0)["gex_total_cr"] == pytest.approx(want)
    assert chain_metrics(chain_frame(only_puts, D, "AAA"), 100.0)["gex_total_cr"] == pytest.approx(-want)


def test_zero_gamma_between_put_and_call_concentrations():
    rows = [_row(E1, 110, "CE", 10_000, iv=0.2), _row(E1, 90, "PE", 10_000, iv=0.2)]
    m = chain_metrics(chain_frame(_fo(rows), D, "AAA"), 100.0)
    assert 90.0 < m["zero_gamma"] < 110.0


def test_zero_gamma_none_without_sign_change():
    m = chain_metrics(chain_frame(_fo([_row(E1, 100, "CE", 1000, iv=0.2)]), D, "AAA"), 100.0)
    assert m["zero_gamma"] is None


def test_short_dated_rows_skipped_for_iv():
    rows = [_row(D + timedelta(days=2), 100, "CE", 1000), _row(D + timedelta(days=2), 100, "PE", 1000)]
    m = chain_metrics(chain_frame(_fo(rows), D, "AAA"), 100.0)
    assert m["atm_iv"] is None and m["gex_total_cr"] == 0.0


def test_strikes_frame_columns():
    m = chain_metrics(chain_frame(_basic_chain(), D, "AAA"), 100.0)
    s = m["strikes"].sort("strike")
    assert s.columns[:4] == ["expiry", "strike", "call_oi", "put_oi"]
    r = s.row(0, named=True)
    assert r["strike"] == 90.0 and r["call_oi"] == 100 and r["put_oi"] == 400
    assert r["call_doi"] == 5 and r["put_doi"] == 30 and r["oi_imbalance"] == 25


def test_truncation_metrics_use_only_date_d():
    later = D + timedelta(days=1)
    extra = _fo([_row(E1, 100, "CE", 777, d=later), _row(E1, 100, "PE", 555, d=later)])
    full = pl.concat([_basic_chain(), extra])
    a = chain_metrics(chain_frame(full.filter(pl.col("date") <= D), D, "AAA"), 100.0)
    b = chain_metrics(chain_frame(full, D, "AAA"), 100.0)
    a.pop("strikes")
    b.pop("strikes")
    assert a == b


def test_build_idempotent(tmp_data_root):
    rows = []
    for d in (D, D + timedelta(days=1)):
        for sym, kind in (("NIFTY", "IDO"), ("AAA", "STO")):
            rows += [
                _row(E1, 100, "CE", 200, d=d, sym=sym, kind=kind),
                _row(E1, 100, "PE", 300, d=d, sym=sym, kind=kind),
            ]
    fo = _fo(rows).with_columns(
        open=pl.lit(1.0), high=pl.lit(1.0), low=pl.lit(1.0), close=pl.lit(1.0), prev_close=pl.lit(1.0)
    )
    io.upsert(tmp_data_root, "fo_bhavcopy", fo)
    a, b = D, D + timedelta(days=1)
    assert build_options_daily(tmp_data_root, a, b) == 4
    first = io.read(tmp_data_root, "flow_options_daily").collect()
    strikes1 = io.read(tmp_data_root, "flow_options_strikes").collect()
    assert build_options_daily(tmp_data_root, a, b) == 4
    assert io.read(tmp_data_root, "flow_options_daily").collect().equals(first)
    assert io.read(tmp_data_root, "flow_options_strikes").collect().equals(strikes1)
    assert first.height == 4 and strikes1.height == 4
    r = first.filter(pl.col("underlying") == "AAA").sort("date").row(0, named=True)
    assert r["pcr_oi"] == pytest.approx(1.5) and r["near_expiry"] == E1 and r["spot"] == 100.0
