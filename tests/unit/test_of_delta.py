"""M4.3 bulk-volume delta, CLV delta, Lee-Ready, CVD and divergence."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab.bars import IST_OFFSET_S, Bars
from stocksense.orderflow.delta import cvd, delta_bvc, delta_clv, delta_tick, divergence

DAY = 20000


def _bars(close: np.ndarray, volume: float = 100.0, start_day: int = DAY) -> Bars:
    n = len(close)
    ts = start_day * 86400 - IST_OFFSET_S + (555 + np.arange(n)) * 60
    return Bars.from_arrays(ts, close, close + 0.05, close - 0.05, close, np.full(n, volume))


def test_zero_change_gives_zero_delta() -> None:
    d = delta_bvc(_bars(np.full(120, 100.0)))
    assert np.all(d == 0.0)


def test_noisy_rise_has_positive_delta_and_constant_step_is_zero() -> None:
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(0.1 + rng.normal(0, 0.05, 200))
    d = delta_bvc(_bars(close))
    assert d[51:].mean() > 0
    assert np.all(d[:51] == 0.0)  # fewer than 50 past changes
    step = 100 + 0.1 * np.arange(200)
    assert np.all(delta_bvc(_bars(step)) == 0.0)


def test_bvc_is_past_only() -> None:
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 150))
    full = delta_bvc(_bars(close))
    for t in (60, 100, 149):
        part = delta_bvc(_bars(close[: t + 1]))
        assert np.allclose(part, full[: t + 1], atol=1e-12, equal_nan=True)


def test_clv() -> None:
    b = Bars.from_arrays([0, 60], [10, 10], [11, 10], [9, 10], [11, 10], [100, 50])
    assert np.allclose(delta_clv(b), [100.0, 0.0])


def test_lee_ready_and_tick_rule() -> None:
    prices = np.array([10.02, 9.98, 10.00, 10.00, 10.01, 10.01])
    sizes = np.array([5.0, 3.0, 4.0, 2.0, 7.0, 1.0])
    bids = np.array([9.99, 9.99, 9.99, 9.99, 10.01, 10.01])
    asks = np.array([10.01, 10.01, 10.01, 10.01, 10.01, 10.01])
    out = delta_tick(prices, sizes, bids, asks)
    # 1: above mid +5; 2: below mid -3; 3: at mid (10.00), prev different price 9.98 -> up -> +4
    # 4: at mid, same price as before -> still compared with 9.98 -> +2
    # 5-6: price equals mid 10.01, previous different price 10.00 -> +7, +1
    assert np.allclose(out, [5, -3, 4, 2, 7, 1])


def test_cvd_resets_per_day() -> None:
    d = np.array([1.0, 2.0, -1.0, 4.0])
    day = np.array([1, 1, 2, 2])
    assert np.allclose(cvd(d, day), [1, 3, -1, 3])


def test_bullish_divergence_index() -> None:
    close = np.r_[np.arange(10.0, 4.5, -0.5), np.full(14, 7.0), 4.5, 6.0, 6.5]
    n = len(close)
    c = np.zeros(n)
    c[:11] = -np.arange(11.0)
    c[11:25] = -10 + 0.5 * (np.arange(11, 25) - 10)
    c[25:] = -5.0
    out = divergence(_bars(close), c, lookback=20)
    assert out[25] == 1
    assert out.sum() == 1


def test_bearish_divergence_mirror() -> None:
    close = np.r_[np.arange(10.0, 15.5, 0.5), np.full(14, 13.0), 15.5, 14.0, 13.5]
    n = len(close)
    c = np.zeros(n)
    c[:11] = np.arange(11.0)
    c[11:25] = 10 - 0.5 * (np.arange(11, 25) - 10)
    c[25:] = 5.0
    out = divergence(_bars(close), c, lookback=20)
    assert out[25] == -1
    assert out.sum() == -1
