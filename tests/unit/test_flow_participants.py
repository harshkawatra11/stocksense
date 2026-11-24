from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.data import io
from stocksense.flow.participants import build_participants_daily, participants_daily

D0 = date(2026, 11, 2)


def _poi(n: int = 6) -> pl.DataFrame:
    rows = []
    for i in range(n):
        d = D0 + timedelta(days=i)
        for ct in ("FII", "DII", "PRO", "CLIENT"):
            long_ = 100.0 if ct != "FII" else 100.0 + 40.0 * i  # FII ratio 0.5 -> 0.75 by i=5
            rows.append(
                {
                    "date": d,
                    "client_type": ct,
                    "fut_idx_long": long_,
                    "fut_idx_short": 100.0,
                    "opt_idx_call_long": 50.0,
                    "opt_idx_call_short": 20.0,
                    "opt_idx_put_long": 10.0,
                    "opt_idx_put_short": 25.0,
                }
            )
    return pl.DataFrame(rows)


def _fii_dii() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [D0 + timedelta(days=4)] * 2,
            "category": ["FII", "DII"],
            "buy_value": [10.0, 20.0],
            "sell_value": [4.0, 5.0],
            "net_value": [-6.0, 15.0],
        }
    )


def test_ratio_and_changes():
    out = participants_daily(_poi(), _fii_dii()).sort("date")
    assert out.height == 6
    last = out.row(-1, named=True)
    assert last["fii_ratio"] == pytest.approx(300 / 400)
    assert last["fii_ratio_chg5"] == pytest.approx(0.25)
    assert last["fii_ratio_chg1"] == pytest.approx(0.75 - 260 / 360)
    assert out["fii_ratio_chg5"][:5].null_count() == 5
    assert last["dii_ratio"] == pytest.approx(0.5) and last["dii_ratio_chg5"] == pytest.approx(0.0)


def test_option_net_and_cash():
    out = participants_daily(_poi(), _fii_dii()).sort("date")
    last = out.row(-1, named=True)
    assert last["pro_call_net"] == 30.0 and last["pro_put_net"] == -15.0
    cash = out.filter(pl.col("date") == D0 + timedelta(days=4)).row(0, named=True)
    assert cash["fii_cash_net"] == -6.0 and cash["dii_cash_net"] == 15.0
    assert last["fii_cash_net"] is None  # no cash row that day


def test_truncation():
    full = participants_daily(_poi(), _fii_dii()).sort("date")
    for k in (3, 5):
        cut_poi = _poi().filter(pl.col("date") < D0 + timedelta(days=k))
        cut = participants_daily(cut_poi, _fii_dii()).sort("date")
        assert cut.equals(full.head(k))


def test_zero_total_gives_null_ratio():
    p = _poi(1).with_columns(fut_idx_long=pl.lit(0.0), fut_idx_short=pl.lit(0.0))
    assert participants_daily(p, _fii_dii().head(0))["fii_ratio"][0] is None


def test_build_idempotent(tmp_data_root):
    io.upsert(tmp_data_root, "participant_oi", _poi())
    io.upsert(tmp_data_root, "fii_dii", _fii_dii())
    a, b = D0, D0 + timedelta(days=5)
    assert build_participants_daily(tmp_data_root, a, b) == 6
    first = io.read(tmp_data_root, "flow_participants_daily").collect()
    assert build_participants_daily(tmp_data_root, a, b) == 6
    assert io.read(tmp_data_root, "flow_participants_daily").collect().equals(first)
    assert first.height == 6
