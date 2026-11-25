"""PLAN-M M5.4d: the 15-minute quantile ANN of 151 Trading Strategies sec 18.2."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.strategies import base
from stocksense.strategies import k151_ann as A
from tests.unit.test_k151_stock import bdays


def five_min(day_prices):
    rows = []
    for sym, d, closes in day_prices:
        t0 = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
        for i, c in enumerate(closes):
            rows.append((sym, t0 + timedelta(minutes=5 * i), c, c + 1, c - 1, c, 10))
    return pl.DataFrame(rows, schema=["symbol", "ts", "open", "high", "low", "close", "volume"], orient="row")


def test_resample_aligns_to_0915_and_drops_incomplete_buckets():
    import datetime as dt

    out = A.resample_15m(five_min([("X", dt.date(2025, 1, 6), [100, 101, 102, 103, 104])]))
    assert out.height == 1 and out["ts"][0].minute == 15 and out["ts"][0].hour == 9
    assert (out["open"][0], out["high"][0], out["low"][0], out["close"][0], out["volume"][0]) == (
        100,
        103,
        99,
        102,
        30,
    )


def test_ema_of_a_constant_is_the_constant_and_emsd_zero_and_rsi_one():
    r = np.full(60, 0.7)
    for tau in A.TAUS:
        e, s = A.ema_emsd(r, tau)
        assert e[-1] == pytest.approx(0.7) and s[-1] == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(e[tau - 2]) and np.isfinite(e[tau - 1])
    assert A.rsi(r, 12)[-1] == 1.0 and A.rsi(-r, 12)[-1] == 0.0


def test_ema_weights_the_latest_bar_most():
    r = np.array([0.0, 0.0, 0.0, 1.0])
    e, _ = A.ema_emsd(r, 2)  # lambda = 1/3: (1 - l)/(1 - l^2) * (r[b] + l r[b-1]) = 0.75
    assert e[-1] == pytest.approx(0.75)
    r2 = np.array([1.0, 0.0])
    assert A.ema_emsd(r2, 2)[0][-1] == pytest.approx(0.75 / 3)


def test_inputs_use_only_bars_up_to_b():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 400)))
    x, rh = A.ann_inputs(c)
    x2, rh2 = A.ann_inputs(c[:300])
    assert x.shape == (400, A.N_FEATURES)
    ok = np.isfinite(x2).all(axis=1)
    assert (
        ok.sum() > 100
        and np.allclose(x[:300][ok], x2[ok], atol=1e-12)
        and np.allclose(rh[:300][1:], rh2[1:], equal_nan=True)
    )


def test_quantile_classes_cover_the_top_class():
    e = np.array([-1.0, 0.0, 1.0, 2.0])
    assert A.quantile_classes(np.array([-5.0, -0.5, 0.5, 1.5, 9.0]), e).tolist() == [0, 1, 2, 3, 4]


def synth_bars(n_sym=12, n_days=140, seed=2):
    """Bars whose next-bar return has the sign of the current normalised return (planted momentum)."""
    import datetime as dt

    rng = np.random.default_rng(seed)
    days = [d for d in bdays(n_days)]
    rows = []
    for s in range(n_sym):
        price, last = 100.0, 0.0
        for d in days:
            dd = dt.date.fromisoformat(str(d))
            t0 = datetime(dd.year, dd.month, dd.day, 9, 15, tzinfo=IST)
            for k in range(25):
                r = 0.5 * last + rng.normal(0, 0.002)
                o = price
                price = price * (1 + r)
                last = r
                rows.append(
                    (f"S{s}", t0 + timedelta(minutes=15 * k), o, max(o, price), min(o, price), price, 100)
                )
    return pl.DataFrame(rows, schema=["symbol", "ts", "open", "high", "low", "close", "volume"], orient="row")


def test_walk_forward_trades_only_after_the_minimum_training_months_and_earns_on_planted_momentum():
    bars = synth_bars()
    tr = A.walk_forward(bars, min_train_months=3)
    first = bars["ts"].dt.date().min()  # 2010-01: three training months, so the first test month is April
    assert first.month == 1 and tr.height > 100
    assert tr["date"].min() >= first.replace(month=4, day=1)
    assert tr["ret"].mean() > 0
    assert (
        tr["ts"].dt.hour() * 60 + tr["ts"].dt.minute()
    ).max() <= 15 * 60  # signal bar before the 15:00 entry


def test_a_month_is_predicted_from_earlier_months_only():
    bars = synth_bars()
    full = A.walk_forward(bars)
    cutoff = full["date"].sort()[int(full.height * 0.6)]
    part = A.walk_forward(bars.filter(pl.col("ts").dt.date() <= cutoff))
    m = cutoff.replace(day=1)
    a = full.filter(pl.col("date") < m).sort(["symbol", "ts"])
    b = part.filter(pl.col("date") < m).sort(["symbol", "ts"])
    assert a.height > 0 and a.equals(b)


def test_daily_series_uses_23_slots_and_two_x_costs(tmp_data_root):
    import datetime as dt

    ts = datetime(2025, 3, 3, 10, 0, tzinfo=IST)
    trades = pl.DataFrame(
        {"symbol": ["A", "B"], "ts": [ts, ts], "date": [dt.date(2025, 3, 3)] * 2, "ret": [0.01, 0.03]},
        schema={
            "symbol": pl.Utf8,
            "ts": pl.Datetime("us", "Asia/Kolkata"),
            "date": pl.Date,
            "ret": pl.Float64,
        },
    )
    d = A.daily_from_trades(trades, tmp_data_root, 1_000_000)
    assert d["ret_gross"][0] == pytest.approx(0.02 / 23) and d["gross"][0] == pytest.approx(1 / 23)
    assert d["net"][0] < d["ret_gross"][0] and d["net_2x"][0] < d["net"][0]


def test_ann_is_registered():
    s = base.load_library()["k151_ann_15m"]
    assert s.family == "m-intra" and s.timing is not None and "sec 18.2" in s.reference
