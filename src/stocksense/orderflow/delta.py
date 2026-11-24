"""Estimated order-flow delta: bulk-volume classification, CLV, Lee-Ready, session CVD, CVD divergence."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars

_SIGMA_FLOOR = 1e-10  # price-change std below this is treated as zero (constant series)
_TICK_EPS = 1e-9


def delta_bvc(b: Bars, window: int = 50) -> np.ndarray:
    """Bulk-volume delta (shares): V*(2*Phi(dC/sigma)-1); sigma over the previous `window` changes."""
    n = len(b)
    d_c = np.full(n, np.nan)
    if n > 1:
        d_c[1:] = np.diff(b.close)
    sigma = npx.shift(npx.roll(d_c, window, "std"), 1)  # past-only: excludes the current change
    ok = np.isfinite(sigma) & (sigma > _SIGMA_FLOOR) & np.isfinite(d_c)
    z = np.zeros(n)
    z[ok] = d_c[ok] / sigma[ok]
    buy = b.volume * ndtr(z)
    return np.where(ok, 2.0 * buy - b.volume, 0.0)


def delta_clv(b: Bars) -> np.ndarray:
    """Close-location-value delta per bar in shares: V*(2C-H-L)/(H-L), 0 where H == L."""
    rng = b.high - b.low
    safe = np.where(rng > 0, rng, 1.0)
    return np.where(rng > 0, b.volume * (2 * b.close - b.high - b.low) / safe, 0.0)


def delta_tick(prices: np.ndarray, sizes: np.ndarray, bids: np.ndarray, asks: np.ndarray) -> np.ndarray:
    """Lee-Ready signed size per trade: quote-mid test, else tick rule vs the last different price."""
    p = np.asarray(prices, dtype=float)
    s = np.asarray(sizes, dtype=float)
    mid = (np.asarray(bids, dtype=float) + np.asarray(asks, dtype=float)) / 2.0
    out = np.zeros(len(p))
    last_diff = np.nan
    for i in range(len(p)):
        if i > 0 and p[i] != p[i - 1]:
            last_diff = p[i - 1]
        if np.isfinite(mid[i]) and p[i] > mid[i] + _TICK_EPS:
            sign = 1.0
        elif np.isfinite(mid[i]) and p[i] < mid[i] - _TICK_EPS:
            sign = -1.0
        elif np.isfinite(last_diff):
            sign = float(np.sign(p[i] - last_diff))
        else:
            sign = 0.0
        out[i] = sign * s[i]
    return out


def cvd(delta: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Cumulative delta in shares, reset at each new session (`day` is contiguous per session)."""
    return npx.group_transform(np.asarray(delta, dtype=float), np.asarray(day), "cumsum")


def divergence(b: Bars, cvd_: np.ndarray, lookback: int = 20) -> np.ndarray:
    """+1 at a new `lookback` close low with CVD above its value at the prior low, -1 mirror, else 0."""
    n = len(b)
    out = np.zeros(n, dtype=int)
    day = b.day_index()
    c = b.close
    start = 0
    for t in range(n):
        if t > 0 and day[t] != day[t - 1]:
            start = t
        lo = max(t - lookback, start)  # same session only: CVD resets daily
        if lo >= t:
            continue
        w = c[lo:t]
        if c[t] < w.min() and cvd_[t] > cvd_[lo + int(np.argmin(w))]:
            out[t] = 1
        elif c[t] > w.max() and cvd_[t] < cvd_[lo + int(np.argmax(w))]:
            out[t] = -1
    return out
