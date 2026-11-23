"""Numpy equivalents of the pandas operations used by market-lab (pandas is not allowed in src/)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def shift(x: np.ndarray, k: int = 1, fill: float = np.nan) -> np.ndarray:
    """x shifted down by k (k>0) with `fill` in the first k slots, as a float array."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if k <= 0:
        out[:] = x
        return out
    out[:k] = fill
    out[k:] = x[:-k]
    return out


def ewm_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """pandas Series(x).ewm(alpha=alpha, adjust=False).mean() including its NaN-gap weighting.

    After g missing values the next observation enters with weight alpha/(alpha + (1-alpha)^(g+1)), exactly
    as pandas does with ignore_na=False; leading NaNs stay NaN and a NaN position repeats the last value.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    weighted = np.nan
    old_wt = 1.0
    factor = 1.0 - alpha
    for i, cur in enumerate(x):
        observed = not np.isnan(cur)
        if not np.isnan(weighted):
            old_wt *= factor
            if observed:
                if weighted != cur:
                    weighted = (old_wt * weighted + alpha * cur) / (old_wt + alpha)
                old_wt = 1.0
        elif observed:
            weighted = cur
        out[i] = weighted
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """pandas ewm(span=n, adjust=False).mean()."""
    return ewm_alpha(x, 2.0 / (n + 1.0))


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """pandas ewm(alpha=1/n, adjust=False).mean() (Wilder smoothing)."""
    return ewm_alpha(x, 1.0 / n)


def roll(x: np.ndarray, n: int, fn: str) -> np.ndarray:
    """pandas Series(x).rolling(n).<fn>() with min_periods=n.

    fn is mean, std (ddof=1), max, min or sum; the first n-1 values are NaN and any NaN in a window gives NaN.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = sliding_window_view(x, n)
    if fn == "mean":
        r = w.mean(axis=1)
    elif fn == "std":
        r = w.std(axis=1, ddof=1) if n > 1 else np.full(len(w), np.nan)
    elif fn == "max":
        r = w.max(axis=1)
    elif fn == "min":
        r = w.min(axis=1)
    elif fn == "sum":
        r = w.sum(axis=1)
    else:
        raise ValueError(fn)
    out[n - 1 :] = r
    return out


def pct_change(x: np.ndarray, n: int) -> np.ndarray:
    """x / x[t-n] - 1 (pandas pct_change(n) without forward-filling)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / shift(x, n) - 1.0


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (pandas ffill); leading NaNs stay NaN."""
    out = np.asarray(x, dtype=float).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def factorize(values: np.ndarray) -> np.ndarray:
    """pandas.factorize codes: integers in order of first appearance."""
    _, first_idx, inv = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first_idx))
    return order[np.ravel(inv)]


def group_transform(x: np.ndarray, day: np.ndarray, how: str) -> np.ndarray:
    """groupby(day).transform(how) for how in cumsum, cummax, cummin, first, max, min (groups contiguous).

    cumsum skips NaN by adding zero (pandas keeps NaN at that position); market-lab only applies it to
    volume and price x weight, which never contain NaN, so the difference does not matter there.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(x)]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        seg = x[a:b]
        if how == "cumsum":
            out[a:b] = np.nancumsum(seg)
        elif how == "cummax":
            out[a:b] = np.fmax.accumulate(seg)
        elif how == "cummin":
            out[a:b] = np.fmin.accumulate(seg)
        elif how == "first":
            out[a:b] = seg[0]
        elif how == "max":
            out[a:b] = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        elif how == "min":
            out[a:b] = np.nanmin(seg) if np.isfinite(seg).any() else np.nan
        else:
            raise ValueError(how)
    return out


def rvol_same_bar(
    v: np.ndarray, day: np.ndarray, k: np.ndarray, lookback: int = 10, minp: int = 5
) -> np.ndarray:
    """Cumulative volume so far today divided by its mean at the same bar over the previous `lookback` days.

    `day` is a factorized day index and `k` the bar number within the day. Equals market-lab's pivot,
    rolling(lookback, min_periods=minp).mean().shift(1) and lookup; NaN until `minp` earlier days exist.
    """
    cv = group_transform(v, day, "cumsum")
    nd, nk = int(day.max()) + 1, int(k.max()) + 1
    m = np.full((nd, nk), np.nan)
    m[day, k] = cv
    avg = np.full_like(m, np.nan)
    for d in range(nd):
        lo = max(0, d - lookback)
        w = m[lo:d]
        cnt = np.sum(~np.isnan(w), axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(w, axis=0) if len(w) else np.full(nk, np.nan)
        avg[d] = np.where(cnt >= minp, mean, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cv / avg[day, k]
