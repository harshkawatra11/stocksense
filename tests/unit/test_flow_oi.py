from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.data import io
from stocksense.flow.oi import build_oi_daily, futures_daily

A = date(2026, 11, 26)  # near expiry (Thursday)
B = date(2026, 12, 31)
D20, D23, D24, D25 = date(2026, 11, 20), date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25)


def _cal() -> TradingCalendar:
    d = date(2026, 11, 2)
    days = []
    while d <= date(2026, 12, 31):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return TradingCalendar.from_dates(days)


def _row(d, exp, settle, oi, sym="AAA", under=100.0, kind="STF"):
    return {
        "symbol": sym,
        "instrument_type": kind,
        "expiry_date": exp,
        "date": d,
        "settle_price": settle,
        "oi": oi,
        "underlying_price": under,
    }


def _fo(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"oi": pl.Int64})


def _chain(settle1=100.0, settle2=101.0, d1=D20, d2=D23):
    """Day 1: near OI 100, next 20; day 2: near 110, next 40."""
    return _fo(
        [
            _row(d1, A, settle1, 100),
            _row(d1, B, 105.0, 20),
            _row(d2, A, settle2, 110),
            _row(d2, B, 106.0, 40),
        ]
    )


def test_long_buildup_hand_chain():
    out = futures_daily(_chain(), _cal()).sort("date")
    assert out.height == 2
    r = out.row(1, named=True)
    assert r["fut_oi"] == 150.0
    assert r["oi_chg_pct"] == pytest.approx(25.0)
    assert r["price_chg_pct"] == pytest.approx(1.0)
    assert r["quad"] == 1.0 and r["quad_label"] == "long buildup"
    assert r["fut_close"] == 101.0
    assert r["basis_pct"] == pytest.approx(1.0)
    assert out.row(0, named=True)["oi_chg_pct"] is None


@pytest.mark.parametrize(
    ("settle2", "oi2", "label", "quad"),
    [
        (99.0, 110, "short buildup", -1.0),
        (101.0, 90, "short covering", 0.5),
        (99.0, 90, "long unwinding", -0.5),
    ],
)
def test_quadrants(settle2, oi2, label, quad):
    fo = _fo([_row(D20, A, 100.0, 100), _row(D23, A, settle2, oi2)])
    r = futures_daily(fo, _cal()).sort("date").row(1, named=True)
    assert r["quad_label"] == label and r["quad"] == quad


def test_flat_price_has_no_quad():
    fo = _fo([_row(D20, A, 100.0, 100), _row(D23, A, 100.0, 110)])
    r = futures_daily(fo, _cal()).sort("date").row(1, named=True)
    assert r["quad"] is None and r["quad_label"] is None


def test_rollover_in_last_five_sessions():
    out = futures_daily(_chain(), _cal()).sort("date")
    assert out["rollover_pct"][1] == pytest.approx(40 / 150 * 100)
    assert out["rollover_pct"][0] == pytest.approx(20 / 120 * 100)


def test_rollover_null_far_from_expiry():
    d1, d2 = date(2026, 11, 3), date(2026, 11, 4)
    out = futures_daily(_chain(d1=d1, d2=d2), _cal())
    assert out["rollover_pct"].null_count() == 2


def test_roll_to_next_contract_the_session_before_expiry():
    fo = _fo(
        [
            _row(D24, A, 100.0, 100),
            _row(D24, B, 105.0, 20),
            _row(D25, A, 101.0, 100),
            _row(D25, B, 106.0, 30),
        ]
    )
    out = futures_daily(fo, _cal()).sort("date")
    assert out["fut_close"].to_list() == [100.0, 106.0]
    # the price change is against the same contract (B: 105 -> 106), so no roll jump
    assert out["price_chg_pct"][1] == pytest.approx((106 - 105) / 105 * 100)


def test_basis_and_index_futures():
    fo = _fo([_row(D20, A, 102.0, 10, sym="NIFTY", under=100.0, kind="IDF")])
    assert futures_daily(fo, _cal())["basis_pct"][0] == pytest.approx(2.0)


def test_other_instruments_ignored():
    fo = pl.concat([_chain(), _fo([_row(D23, A, 5.0, 999, kind="STO")])])
    assert futures_daily(fo, _cal()).height == 2


def test_truncation():
    fo = _chain()
    cal = _cal()
    full = futures_daily(fo, cal).sort("date")
    cut = futures_daily(fo.filter(pl.col("date") <= D20), cal).sort("date")
    assert cut.equals(full.head(1))


def test_build_idempotent(tmp_data_root):
    fo = _chain().with_columns(
        strike_price=pl.lit(0.0),
        option_type=pl.lit(""),
        open=pl.lit(1.0),
        high=pl.lit(1.0),
        low=pl.lit(1.0),
        close=pl.lit(1.0),
        prev_close=pl.lit(1.0),
        change_in_oi=pl.lit(0, pl.Int64),
    )
    io.upsert(tmp_data_root, "fo_bhavcopy", fo)
    n1 = build_oi_daily(tmp_data_root, D20, D23, calendar=_cal())
    first = io.read(tmp_data_root, "flow_oi_daily").collect()
    n2 = build_oi_daily(tmp_data_root, D20, D23, calendar=_cal())
    assert n1 == n2 == 2
    assert io.read(tmp_data_root, "flow_oi_daily").collect().equals(first)
