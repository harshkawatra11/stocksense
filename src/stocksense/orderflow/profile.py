"""Volume profile, TPO (market) profile, Dalton day types and open types, from bars."""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab.bars import Bars

_EPS = 1e-9
_OPEN_MINUTE = 9 * 60 + 15
_N_PERIODS = 13


def bin_size(price: float) -> float:
    """Price-bin width in rupees: max(0.05, round(0.0005 * price, 2))."""
    return max(0.05, round(0.0005 * float(price), 2))


def _edges(low: float, high: float, bin_: float) -> tuple[float, int]:
    lo = float(np.floor(low / bin_ + _EPS) * bin_)
    n = max(1, int(np.ceil((high - lo) / bin_ - _EPS)))
    return lo, n


def _span(low: float, high: float, lo: float, bin_: float, n: int) -> tuple[int, int]:
    """Inclusive bin index range touched by [low, high]; a zero-range bar touches its own bin."""
    i0 = int(np.floor((low - lo) / bin_ + _EPS))
    i1 = int(np.ceil((high - lo) / bin_ - _EPS)) - 1
    i0 = min(max(i0, 0), n - 1)
    return i0, min(max(i1, i0), n - 1)


def _value_area(vol: np.ndarray, poc: int, frac: float = 0.7) -> tuple[int, int]:
    """Grow from the POC bin, adding the larger neighbour (ties go up), until `frac` of the total is held."""
    lo = hi = poc
    acc = float(vol[poc])
    target = frac * float(vol.sum())
    while acc < target - _EPS:
        up = float(vol[hi + 1]) if hi + 1 < len(vol) else -1.0
        dn = float(vol[lo - 1]) if lo - 1 >= 0 else -1.0
        if up < 0 and dn < 0:
            break
        if up >= dn:
            hi += 1
            acc += up
        else:
            lo -= 1
            acc += dn
    return lo, hi


def volume_profile(b: Bars, bin: float | None = None) -> dict[str, Any]:  # noqa: A002
    """Volume per price bin (each bar spread uniformly over its range); POC, VAH, VAL are prices in rupees."""
    bin_ = float(bin) if bin else bin_size(float(np.mean(b.close)))
    lo, n = _edges(float(b.low.min()), float(b.high.max()), bin_)
    vol = np.zeros(n)
    for i in range(len(b)):
        a, z = _span(float(b.low[i]), float(b.high[i]), lo, bin_, n)
        rng = float(b.high[i] - b.low[i])
        if rng <= 0 or a == z:
            vol[a] += b.volume[i]
            continue
        left = lo + np.arange(a, z + 1) * bin_
        ov = np.minimum(left + bin_, b.high[i]) - np.maximum(left, b.low[i])
        ov = np.maximum(ov, 0.0)
        vol[a : z + 1] += b.volume[i] * ov / ov.sum()
    bins = lo + np.arange(n) * bin_
    poc = int(np.argmax(vol))
    v0, v1 = _value_area(vol, poc)
    return {
        "bin": bin_,
        "bins": bins,
        "vol": vol,
        "poc": float(bins[poc] + bin_ / 2),
        "vah": float(bins[v1] + bin_),
        "val": float(bins[v0]),
        "total": float(vol.sum()),
    }


def tpo_profile(b: Bars, bin: float | None = None) -> dict[str, Any]:  # noqa: A002
    """TPO letters per price bin for the last session in `b` (A=09:15, ..., M=15:15); rupees."""
    day = b.day_index()
    b = b.take(day == day[-1])
    bin_ = float(bin) if bin else bin_size(float(np.mean(b.close)))
    lo, n = _edges(float(b.low.min()), float(b.high.max()), bin_)
    period = np.clip((b.minute_of_day() - _OPEN_MINUTE) // 30, 0, _N_PERIODS - 1).astype(int)
    sets: list[set[int]] = [set() for _ in range(n)]
    for i in range(len(b)):
        a, z = _span(float(b.low[i]), float(b.high[i]), lo, bin_, n)
        for k in range(a, z + 1):
            sets[k].add(int(period[i]))
    letters = ["".join(chr(65 + p) for p in sorted(s)) for s in sets]
    counts = np.array([len(s) for s in sets], dtype=float)
    bins = lo + np.arange(n) * bin_
    centers = bins + bin_ / 2
    mid = (float(b.high.max()) + float(b.low.min())) / 2
    top = np.flatnonzero(counts == counts.max())
    poc = int(top[np.argmin(np.abs(centers[top] - mid))])  # nearest the range middle; ties go lowest
    v0, v1 = _value_area(counts, poc)
    multi = counts >= 2
    single = [float(bins[k]) for k in range(n) if counts[k] == 1 and multi[:k].any() and multi[k + 1 :].any()]
    ib = period <= 1
    return {
        "bin": bin_,
        "bins": bins,
        "letters": letters,
        "counts": counts,
        "periods": int(period.max()) + 1,
        "poc_tpo": float(centers[poc]),
        "vah": float(bins[v1] + bin_),
        "val": float(bins[v0]),
        "open": float(b.open[0]),
        "close": float(b.close[-1]),
        "high": float(b.high.max()),
        "low": float(b.low.min()),
        "ib_high": float(b.high[ib].max()),
        "ib_low": float(b.low[ib].min()),
        "single_prints": single,
        "poor_high": bool(counts[-1] >= 2),
        "poor_low": bool(counts[0] >= 2),
    }


def _has_double_distribution(tpo: dict[str, Any]) -> bool:
    """True if a run of >= 2 consecutive single-print bins separates two peaks of >= 3 TPO letters."""
    bin_ = tpo["bin"]
    bins, counts = tpo["bins"], tpo["counts"]
    run: list[int] = []
    for k in [i for i in range(len(bins)) if bins[i] in set(tpo["single_prints"])] + [-1]:
        if run and (k < 0 or abs(bins[k] - bins[run[-1]] - bin_) >= _EPS):
            if len(run) >= 2 and counts[: run[0]].max() >= 3 and counts[run[-1] + 1 :].max() >= 3:
                return True
            run = []
        if k >= 0:
            run.append(k)
    return False


def day_type(tpo: dict[str, Any], atr_d: float) -> str:
    """Dalton day type of a TPO profile."""
    rng = tpo["high"] - tpo["low"]
    ib = tpo["ib_high"] - tpo["ib_low"]
    ext_up = max(0.0, tpo["high"] - tpo["ib_high"])
    ext_dn = max(0.0, tpo["ib_low"] - tpo["low"])
    if _has_double_distribution(tpo):
        return "double_distribution"
    if rng > 0 and rng >= 2 * ib:
        pos = (tpo["close"] - tpo["low"]) / rng
        if (ext_up > ext_dn and pos >= 0.8) or (ext_dn > ext_up and pos <= 0.2):
            return "trend"
    if ext_up > 0 and ext_dn > 0:
        return "neutral"
    if ib < 0.35 * atr_d and ext_up == 0 and ext_dn == 0:
        return "non_trend"
    if rng > 0 and ib >= 0.85 * rng:
        return "normal"
    return "normal_variation"


def open_type(prev_vah: float, prev_val: float, first30: Bars) -> str:
    """Auction open type from the first 30 minutes of bars and the previous session's value area."""
    o = float(first30.open[0])
    thr = 0.003 * o
    hi, lo, cl = first30.high, first30.low, first30.close
    early = first30.ts < first30.ts[0] + 15 * 60
    up_move = float(hi.max()) - o
    dn_move = o - float(lo.min())
    if (float(lo.min()) >= o and up_move > thr) or (float(hi.max()) <= o and dn_move > thr):
        return "open_drive"
    j_dn, j_up = int(np.argmin(lo)), int(np.argmax(hi))
    tested_dn = bool((lo[early] < o).any())
    tested_up = bool((hi[early] > o).any())
    if (tested_dn and up_move > thr and j_up >= j_dn) or (tested_up and dn_move > thr and j_dn >= j_up):
        return "open_test_drive"
    if (up_move > thr and cl[-1] < o) or (dn_move > thr and cl[-1] > o):
        return "open_rejection_reverse"
    if not (np.isnan(prev_vah) or np.isnan(prev_val)) and prev_val <= o <= prev_vah:
        return "open_auction_in_range"
    return "open_auction_out_of_range"
