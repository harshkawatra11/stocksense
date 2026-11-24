from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.data import io
from stocksense.flow.sector_flow import (
    ValueHistory,
    build_sector_flow_daily,
    flow_dir,
    live_sector_flow,
    rotation,
    sector_flow_daily,
)


def _rows(spec):
    return [{"sym": s, "sector": sec, "value_cr": v, "from_open": fo, "pct": fo} for s, sec, v, fo in spec]


def test_flow_dir_hand_number():
    assert flow_dir([10, 20, 30], [1, -1, 2]) == pytest.approx(20 / 60)
    assert flow_dir([10, None, 30], [1, -1, 2]) == pytest.approx(1.0)  # missing value skipped
    assert flow_dir([], []) is None
    assert flow_dir([0, 0], [1, -1]) is None


def test_rotation_thresholds_at_the_boundaries():
    assert rotation(1.2, 0.5) == "neutral"  # share_ratio must exceed 1.2
    assert rotation(1.2001, 0.2) == "neutral"  # flow_dir must exceed 0.2
    assert rotation(1.2001, 0.2001) == "in"
    assert rotation(1.2001, -0.2) == "neutral"
    assert rotation(1.2001, -0.2001) == "out"
    assert rotation(None, 0.9) == "neutral"


def test_live_sector_flow_shares_breadth_and_hints():
    rows = _rows([("A", "X", 10, 1.0), ("B", "X", 20, -1.0), ("C", "X", 30, 2.0), ("D", "Y", 40, -0.5)])
    out = {
        s["name"]: s
        for s in live_sector_flow(
            rows, {"X": 0.25, "Y": None}, {"A": "long buildup", "B": "short buildup"}, {"C"}
        )
    }
    x = out["X"]
    assert x["flow_dir"] == pytest.approx(1 / 3, abs=1e-4)
    assert x["turnover_share"] == pytest.approx(0.6)
    assert x["share_ratio"] == pytest.approx(0.6 / 0.25) and x["rotation"] == "in"
    assert (x["up"], x["down"]) == (2, 1)
    assert x["oi_quads"] == {"long buildup": 1, "short buildup": 1} and x["deliv_spikes"] == 1
    assert out["Y"]["share_ratio"] is None and out["Y"]["rotation"] == "neutral"


def test_value_history_needs_five_sessions(tmp_path):
    h = ValueHistory(tmp_path)
    for i in range(4):
        h.record(date(2026, 11, 2 + i), 30, {"X": 10.0, "Y": 30.0})
    assert h.mean_share(date(2026, 11, 9), 30) == {}
    h.record(date(2026, 11, 6), 30, {"X": 30.0, "Y": 30.0})
    ms = h.mean_share(date(2026, 11, 9), 32)
    assert ms["X"] == pytest.approx((4 * 0.25 + 0.5) / 5)
    assert h.mean_share(date(2026, 11, 6), 30) == {}  # today's own file is not history


def _panel(n=3):
    d0 = date(2026, 1, 1)
    rows = []
    for i in range(n):
        for sid, sym, ind, t, r in (
            (1, "AAA", "Information Technology", 10e7, 0.01),
            (2, "BBB", "Information Technology", 30e7, -0.01),
            (3, "CCC", "Power", 60e7, 0.02),
        ):
            rows.append(
                {
                    "sid": sid,
                    "date": d0 + timedelta(days=i),
                    "symbol": sym,
                    "industry": ind,
                    "turnover_inr": t,
                    "ret_oc": r,
                }
            )
    return pl.DataFrame(rows)


def test_sector_flow_daily_hand_numbers():
    out = sector_flow_daily(_panel(1), None).filter(pl.col("sector") == "IT").row(0, named=True)
    assert out["value_cr"] == pytest.approx(40.0)
    assert out["flow_dir"] == pytest.approx((10 - 30) / 40)
    assert out["turnover_share"] == pytest.approx(0.4)
    assert (out["up"], out["down"], out["n"]) == (1, 1, 2)


def test_build_sector_flow_idempotent(tmp_data_root):
    p = _panel().with_columns(series=pl.lit("EQ"))
    io.upsert(tmp_data_root, "panel_daily", p)
    u = p.select("date", "sid").with_columns(
        turnover_rank=pl.lit(1, pl.Int32),
        med_turn_60=pl.lit(1.0),
        u500=pl.lit(True),
        u200=pl.lit(True),
        in_fno=pl.lit(True),
    )
    io.upsert(tmp_data_root, "panel_universe", u)
    a, b = date(2026, 1, 1), date(2026, 1, 3)
    n1 = build_sector_flow_daily(tmp_data_root, a, b)
    first = io.read(tmp_data_root, "flow_sector_daily").collect()
    n2 = build_sector_flow_daily(tmp_data_root, a, b)
    second = io.read(tmp_data_root, "flow_sector_daily").collect()
    assert n1 == n2 == first.height == 6 and first.equals(second)
