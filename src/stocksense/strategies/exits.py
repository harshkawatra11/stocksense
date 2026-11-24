"""Ported from market-lab lab/swing.py follow() and lab/engine.py follow(), windows(), trend_mask()
(MIT, ba3f771); changes: numpy only; the daily exit also returns why the trade ended.

Fills: the stop is checked before the target; a gap through the stop fills at the open; a target fills at
max(target, open) after the entry day; time and moving-average exits fill at the close.
"""

from __future__ import annotations

# ruff: noqa: E741 (Market Lab's h, l, c names)
from typing import Any

import numpy as np

EXITS_1D = ("trail3", "ema20", "rr2_20", "rr3_40", "time5", "time10", "mr5")
EXITS_INTRA = ("rr1", "rr15", "rr2", "rr3", "be2", "trail", "scale", "eod")
_MAX_DAYS = {"rr2_20": 20, "rr3_40": 40, "time5": 5, "time10": 10, "mr5": 10, "trail3": 120, "ema20": 120}


def follow_1d(f: dict[str, Any], ie: int, stop: float, mode: str) -> tuple[int, float, str]:
    """Long from day `ie`'s open. Returns (exit day index, exit price, reason) with reason one of
    gap_stop, stop, target, mr5, ema20, time."""
    o, h, l, c, atr = f["o"], f["h"], f["l"], f["c"], f["atr"]  # noqa: E741
    e = o[ie]
    r = e - stop
    n = f["n"]
    tgt = {"rr2_20": e + 2 * r, "rr3_40": e + 3 * r}.get(mode)
    maxd = _MAX_DAYS[mode]
    st, peak = stop, e
    for j in range(ie, min(n, ie + maxd)):
        if j > ie and o[j] <= st:
            return j, float(o[j]), "gap_stop"
        if l[j] <= st:
            return j, float(st), "stop"
        if tgt is not None and h[j] >= tgt:
            return j, float(max(tgt, o[j]) if j > ie else tgt), "target"
        if mode == "mr5" and c[j] > f["sma5"][j]:
            return j, float(c[j]), "mr5"
        if mode == "ema20" and c[j] < f["ema20"][j] and j > ie:
            return j, float(c[j]), "ema20"
        if mode == "trail3":
            peak = max(peak, c[j])
            if peak >= e + r:
                st = max(st, e)
            st = max(st, peak - 3 * atr[j])
    j = min(n - 1, ie + maxd - 1)
    return j, float(c[j]), "time"


def trend_mask(f: dict[str, Any], kind: str) -> np.ndarray:
    """Which bars agree with the trend filter: D (daily), I (intraday), DI, DIX (plus the index)."""
    intra = (f["c"] > f["vwap"]) & (f["ema21"] > f["ema50"])
    if kind == "D":
        return f["dup"]
    if kind == "I":
        return intra
    if kind == "DI":
        return f["dup"] & intra
    if kind == "DIX":
        return f["dup"] & intra & f["ix_up"]
    raise ValueError(kind)


def entry_window(f: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(entry allowed per bar, 09:15 plus one bar to 14:30; last bar of each day closing by 15:15)."""
    tf = f["tf"]
    mins = f["mins"]
    entry_ok = (mins >= 555 + tf) & (mins <= 870)
    starts = np.flatnonzero(f["first"])
    exit_idx = np.maximum.reduceat(np.where(mins + tf <= 915, np.arange(f["n"]), -1), starts)
    return entry_ok, exit_idx


def follow_intra(f: dict[str, Any], i_e: int, s: float, e: float, r: float, xe: int) -> dict[str, float]:
    """Outcome in gross R of one long from bar `i_e` (entry at its open) for every intraday exit rule."""
    H, L, C = f["h"][i_e : xe + 1], f["l"][i_e : xe + 1], f["c"][i_e : xe + 1]
    A = f["atr"][i_e : xe + 1]
    n = len(H)
    hit = s >= L
    ks = int(np.argmax(hit)) if hit.any() else n
    eod = (C[-1] - e) / r
    out: dict[str, float] = {}
    for name, rr in (("rr1", 1), ("rr15", 1.5), ("rr2", 2), ("rr3", 3)):
        th = e + rr * r <= H
        kt = int(np.argmax(th)) if th.any() else n
        out[name] = -1.0 if ks <= kt and ks < n else rr if kt < n else eod
    out["eod"] = -1.0 if ks < n else eod
    # 1R reached -> stop to break-even; then 2R target (be2) or 3R on the second half (scale)
    t1 = e + r <= H
    k1 = int(np.argmax(t1)) if t1.any() else n
    if ks <= k1 and ks < n:
        out["be2"] = -1.0
        out["scale"] = -1.0
    elif k1 >= n:
        out["be2"] = out["scale"] = eod
    else:

        def after(target: float) -> float:
            for j in range(k1, n):
                if j > k1 and L[j] <= e:
                    return 0.0
                if H[j] >= e + target * r:
                    return float(target)
            return max(eod, 0.0) if L[k1:].min() > e else 0.0

        out["be2"] = after(2)
        out["scale"] = 0.5 + 0.5 * after(3)
    # chandelier trail: 2.5 ATR under the highest high, break-even after 1R
    stop, peak = s, e
    res = eod
    for j in range(n):
        if L[j] <= stop:
            res = (min(stop, f["o"][i_e + j]) - e) / r if j > 0 else (stop - e) / r
            break
        peak = max(peak, H[j])
        if peak >= e + r:
            stop = max(stop, e)
        stop = max(stop, peak - 2.5 * A[j])
    out["trail"] = res
    mx = H[: ks + 1].max() if ks < n else H.max()
    out["mfe"] = (mx - e) / r
    return {k: float(v) for k, v in out.items()}
