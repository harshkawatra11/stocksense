from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.data import io
from stocksense.flow.composite import (
    EodFlow,
    attach_flow,
    build_eod,
    combine,
    cvd_slope,
    rvol,
    slope_per_volume,
    stock_card,
    vol_curve,
    zscores,
)
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.store import LabStore


def test_zscore_hand_numbers_and_clip_at_three():
    z = zscores({"a": 1.0, "b": 2.0, "c": 3.0, "d": None})
    assert z["a"] == pytest.approx(-1.2247, abs=1e-4) and z["b"] == 0.0 and z["d"] is None
    vals = {f"s{i}": 0.0 for i in range(14)} | {"big": 1000.0}
    out = zscores(vals)
    assert out["big"] == 3.0  # raw z is 3.74
    assert zscores({"a": 1.0, "b": 2.0}) == {"a": None, "b": None}  # under 3 rows
    assert zscores({"a": 5.0, "b": 5.0, "c": 5.0}) == {"a": None, "b": None, "c": None}  # no spread


def test_missing_component_is_skipped_mean_of_the_three_present():
    score, used = combine({"disp": 1.0, "rvol": None, "oi_quad": 2.0, "depth_imb5": None, "cvd": 3.0})
    assert score == pytest.approx(2.0) and used == ["disp", "oi_quad", "cvd"]
    assert combine({"a": None}) == (None, [])


def test_rvol_interpolates_the_curve():
    curve = {"x": [0.0, 5.0, 10.0], "y": [0.0, 100.0, 300.0]}
    assert rvol(150.0, curve, 5) == pytest.approx(1.5)
    assert rvol(150.0, curve, 7) == pytest.approx(150 / 180)  # 100 + 2/5 * 200
    assert rvol(150.0, None, 5) is None and rvol(None, curve, 5) is None and rvol(5.0, curve, 0) is None


def _bars5(n_sessions, today, bars_per_day=75, vol=100):
    rows = []
    for i in range(n_sessions):
        d = today - timedelta(days=i + 1)
        for k in range(bars_per_day):
            t = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST) + timedelta(minutes=5 * k)
            rows.append({"ts": t, "volume": vol})
    return pl.DataFrame(rows)


def test_vol_curve_needs_ten_sessions_and_accumulates():
    today = date(2026, 11, 20)
    c = vol_curve(_bars5(10, today), today)
    assert c is not None and c["y"][0] == 0 and c["y"][3] == pytest.approx(300.0)
    assert c["y"][75] == pytest.approx(7500.0) and c["x"][75] == 375.0
    assert vol_curve(_bars5(9, today), today) is None
    assert vol_curve(_bars5(10, today, bars_per_day=40), today) is None  # partial days ignored


def test_cvd_slope_hand_number_and_guards():
    assert slope_per_volume(np.arange(30) * 10.0, np.full(30, 100.0)) == pytest.approx(0.1)
    assert slope_per_volume(np.arange(30) * 10.0, np.zeros(30)) is None
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(np.abs(rng.normal(0.1, 0.05, 80)))  # steady buying
    ts = np.arange(80) * 60 + 1_700_000_000
    b = Bars.from_arrays(ts, close, close + 0.1, close - 0.1, close, np.full(80, 1000.0))
    assert cvd_slope(b) > 0
    assert cvd_slope(b.tail(20)) is None


def _store(disp, volumes=None, minute_iso="2026-11-12T10:15:00+05:30"):
    s = LabStore()
    s.tape["stocks"] = {k: {"sym": k, "disp": v, "name": k, "sector": "X"} for k, v in disp.items()}
    s.tape["meta"] = {k: {"volume": (volumes or {}).get(k)} for k in disp}
    s.tape["time"] = minute_iso
    return s


def _eod(settings, oi=None, options=None, deliv=None):
    e = EodFlow(settings)
    e.asof = date(2026, 11, 11)
    e.oi, e.options, e.deliv = oi or {}, options or {}, deliv or {}
    return e


def test_stock_card_signs_and_components_used(tmp_data_root):
    store = _store({"A": -2.0, "B": -1.0, "C": 0.0, "D": 1.0, "E": 2.0})
    eod = _eod(tmp_data_root)
    a, e = stock_card("A", store, eod), stock_card("E", store, eod)
    assert a["components_used"] == ["disp"] and e["components_used"] == ["disp"]
    assert e["flow_score"] > 0 > a["flow_score"]
    assert e["flow_score"] == pytest.approx(-a["flow_score"])
    assert stock_card("ZZZ", store, eod) is None and e["descriptive"] is True
    assert e["options"] is None and e["rvol"] is None


def test_stock_card_uses_every_available_component(tmp_data_root):
    store = _store({"A": 1.0, "B": 0.0, "C": -1.0}, volumes={"A": 300.0, "B": 100.0, "C": 100.0})
    store.flow["vol_curve"] = {s: {"x": [0.0, 60.0], "y": [0.0, 100.0]} for s in "ABC"}
    store.flow["cvd"] = {"A": 0.5, "B": 0.0, "C": -0.5}
    store.flow["preopen"] = {"rows": {"A": {"imb": 0.5}}}
    oi = {
        "A": {"quad": 1.0, "quad_label": "long buildup", "oi_chg_pct": 4.0},
        "B": {"quad": 0.5, "quad_label": "short covering", "oi_chg_pct": -1.0},
        "C": {"quad": -1.0, "quad_label": "short buildup", "oi_chg_pct": 5.0},
    }
    opt = {"A": {"pcr_oi": 1.5, "gex_total_cr": -3.0, "call_wall": 110.0, "put_wall": 90.0}}
    eod = _eod(tmp_data_root, oi=oi, options=opt, deliv={"A": {"deliv_z20": 2.5}})
    card = stock_card("A", store, eod)
    # 10:15 is minute 60; A traded 300 against an expected 100
    assert card["rvol"] == pytest.approx(3.0)
    assert card["components_used"] == ["disp", "rvol", "oi_quad", "cvd_slope_30m"]
    assert card["flow_score"] > 1.0 and card["preopen_imb"] == 0.5
    assert card["oi_quad_label"] == "long buildup" and card["deliv_z20_prev"] == 2.5
    assert card["options"] == {"pcr": 1.5, "gex_sign": -1, "call_wall": 110.0, "put_wall": 90.0}
    assert stock_card("C", store, eod)["flow_score"] < 0


def test_attach_flow_without_engine_gives_null(tmp_data_root):
    store = _store({"A": 1.0})
    rows = [{"sym": "A", "pct": 1.0}]
    assert attach_flow(rows, store, None) == [{"sym": "A", "pct": 1.0, "flow": None}]
    assert "flow" not in rows[0]  # inputs are copied, not mutated
    out = attach_flow(rows, store, _eod(tmp_data_root))
    assert out[0]["flow"]["flow_score"] is None  # one row cannot be z-scored


def _row(sid, sym, d):
    return {
        "sid": sid,
        "date": d,
        "symbol": sym,
        "series": "EQ",
        "industry": "Power",
        "turnover_inr": 1e8,
        "ret_oc": 0.01,
        "open": 10.0,
        "close": 10.1,
        "deliv_qty": 100.0,
        "deliv_pct": 40.0,
    }


def _put_panel(settings, rows):
    p = pl.DataFrame(rows)
    io.upsert(settings, "panel_daily", p)
    u = p.select("date", "sid").with_columns(
        turnover_rank=pl.lit(1, pl.Int32),
        med_turn_60=pl.lit(1.0),
        u500=pl.lit(True),
        u200=pl.lit(True),
        in_fno=pl.lit(True),
    )
    io.upsert(settings, "panel_universe", u)
    io.upsert(settings, "bhavcopy_delivery", p.select("date", "symbol", "series", "deliv_qty", "deliv_pct"))


def _seed_panel(settings):
    d0 = date(2026, 1, 1)
    _put_panel(
        settings,
        [_row(sid, sym, d0 + timedelta(days=i)) for i in range(20) for sid, sym in ((1, "AAA"), (2, "BBB"))],
    )
    return d0 + timedelta(days=19)


def test_build_eod_is_idempotent_and_fills_only_missing_dates(tmp_data_root):
    asof = _seed_panel(tmp_data_root)
    first = build_eod(tmp_data_root, asof)
    assert first["sector"] == 20 and first["delivery"] == 40  # 1 sector x 20 dates; 2 sids x 20
    snap = {n: io.read(tmp_data_root, n).collect() for n in ("flow_sector_daily", "flow_delivery_daily")}
    second = build_eod(tmp_data_root, asof)
    assert second["sector"] == 0 and second["delivery"] == 0 and second["oi"] == 0
    for n, df in snap.items():
        assert io.read(tmp_data_root, n).collect().equals(df)
    extra = date(2026, 1, 21)  # a new source date is picked up alone
    _put_panel(tmp_data_root, [_row(1, "AAA", extra)])
    third = build_eod(tmp_data_root, extra)
    assert third["sector"] == 1 and third["delivery"] == 1


def test_eod_flow_loads_latest_rows_and_reloads_on_new_asof(tmp_data_root):
    asof = _seed_panel(tmp_data_root)
    build_eod(tmp_data_root, asof)
    prev = asof - timedelta(days=1)
    io.upsert(
        tmp_data_root,
        "flow_oi_daily",
        pl.DataFrame(
            {
                "date": [prev, asof],
                "symbol": ["AAA", "AAA"],
                "oi_chg_pct": [1.0, 2.0],
                "quad": [1.0, -1.0],
                "quad_label": ["long buildup", "short buildup"],
            }
        ),
    )
    e = EodFlow(tmp_data_root)
    e.load(asof)
    assert e.oi["AAA"]["quad"] == -1.0 and "AAA" in e.deliv and e.asof == asof
    e.oi["AAA"] = {"marker": 1}
    e.load(asof)
    assert e.oi["AAA"] == {"marker": 1}  # same asof: not reloaded
    e.load(prev)
    assert e.oi["AAA"]["quad"] == 1.0
