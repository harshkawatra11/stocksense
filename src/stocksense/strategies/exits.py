"""Ported from market-lab lab/swing.py follow() and lab/engine.py follow(), windows(), trend_mask()
(MIT, ba3f771); changes: numpy only; the daily exit also returns why the trade ended.

Fills: the stop is checked before the target; a gap through the stop fills at the open; a target fills at
max(target, open) after the entry day; time and moving-average exits fill at the close.
"""

from __future__ import annotations

# ruff: noqa: E741 (Market Lab's h, l, c names)
from collections.abc import Callable
from typing import Any

import numpy as np

from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars
from stocksense.orderflow import profile as P

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


EXIT_STATE = "state_exit"  # PLAN-M M5.4: exit is a condition on the closes, not a price or a count of days


def follow_state(
    f: dict[str, Any],
    ie: int,
    stop: float,
    exit_state: np.ndarray,
    target: float = np.nan,
    use_stop: bool = True,
    max_days: int = 250,
) -> tuple[int, float, str]:
    """Long from day `ie`'s open until the exit condition is true at a close, then out at the NEXT open.

    `exit_state[j]` is the exit condition evaluated at the close of day j. A hit-tested `stop` (gap through
    it fills at the open, else at the stop) and a `target` (fills at max(target, open)) are checked first,
    the stop before the target, as in follow_1d. Returns (exit day index, price, reason) with reason one of
    gap_stop, stop, target, state, end (data ends with the condition true) or time (max_days reached).
    """
    o, h, l, c = f["o"], f["h"], f["l"], f["c"]  # noqa: E741
    n = f["n"]
    has_stop = use_stop and bool(np.isfinite(stop))
    has_tgt = bool(np.isfinite(target))
    last = min(n, ie + max_days)
    for j in range(ie, last):
        if has_stop:
            if j > ie and o[j] <= stop:
                return j, float(o[j]), "gap_stop"
            if l[j] <= stop:
                return j, float(stop), "stop"
        if has_tgt and h[j] >= target:
            return j, float(max(target, o[j])), "target"
        if exit_state[j]:
            if j + 1 < n:
                return j + 1, float(o[j + 1]), "state"
            return j, float(c[j]), "end"
    j = last - 1
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


# ---------------------------------------------------------------------------------------------------------
# Extra exits (PLAN-M M5.3). Signature `(f, ie, stop, side) -> (exit_index, price, reason)`.
#
# `f` is the feature dict in the TRADE's frame: a short is run on the mirrored chart (prices negated, high and
# low swapped, f["sign"] == -1), so every function here is written for a long; the returned price is in
# the same frame as `f` (the caller negates it back for a short). `side` is "long" or "short" and only labels
# the trade. `ie` is the entry bar, entered at its OPEN (except `next_open`, entered at the CLOSE of `ie`).
# Every target level is read from bars BEFORE the bar that can touch it, so a fill never uses its own bar.
# ---------------------------------------------------------------------------------------------------------
ExitFn = Callable[[dict[str, Any], int, float, str], tuple[int, float, str]]
_SIDES = ("long", "short")


def _cached(f: dict[str, Any], key: str, fn: Callable[[], Any]) -> Any:
    """Memoise `fn()` inside the feature dict; a copy of the dict cut to a different length recomputes."""
    hit = f.get(key)
    if hit is not None and hit[0] == f["n"]:
        return hit[1]
    val = fn()
    f[key] = (f["n"], val)
    return val


def real_bars(f: dict[str, Any]) -> Bars:
    """The bars in their real orientation (a mirrored feature dict is turned upside down again)."""
    o, h, l, c = f["o"], f["h"], f["l"], f["c"]
    if f.get("sign", 1) == -1:
        o, h, l, c = -o, -l, -h, -c
    return Bars(np.asarray(f["ts"]), o, h, l, c, f["v"])


def levels(f: dict[str, Any]) -> dict[str, Any]:
    """Per-bar session levels in the trade's frame, from past sessions and the current initial balance only.

    Keys: pvah, pval, ppoc (previous session's volume-profile value area and POC), ppoor_lo (previous session
    had a poor low in the trade's frame), ibh, ibl (first-hour range, NaN until 10:15), and `days`, the real
    per-session profile list (a, z, vah, val, poc, high, low, poor_high, poor_low) for the open-type rules.
    """
    return _cached(f, "_levels", lambda: _levels(f))


def _levels(f: dict[str, Any]) -> dict[str, Any]:
    n, sign = f["n"], f.get("sign", 1)
    b = real_bars(f)
    day = b.day_index()
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, n]
    mins = b.minute_of_day()
    pvah, pval, ppoc, ibh, ibl = (np.full(n, np.nan) for _ in range(5))
    poor_lo = np.zeros(n, bool)
    days: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for a, z in zip(starts[:-1], starts[1:], strict=True):
        d = b.take(slice(a, z))
        vp, tpo = P.volume_profile(d), P.tpo_profile(d)
        info = {
            "a": int(a),
            "z": int(z),
            "vah": vp["vah"],
            "val": vp["val"],
            "poc": vp["poc"],
            "high": tpo["high"],
            "low": tpo["low"],
            "poor_high": tpo["poor_high"],
            "poor_low": tpo["poor_low"],
        }
        if prev is not None:
            pvah[a:z], pval[a:z], ppoc[a:z] = prev["vah"], prev["val"], prev["poc"]
            poor_lo[a:z] = prev["poor_low"] if sign == 1 else prev["poor_high"]
        dm = mins[a:z]
        ib = (dm >= 555) & (dm < 615)
        if ib.any():
            late = dm >= 615
            ibh[a:z] = np.where(late, d.high[ib].max(), np.nan)
            ibl[a:z] = np.where(late, d.low[ib].min(), np.nan)
        days.append(info)
        prev = info
    if sign == 1:
        return {
            "pvah": pvah,
            "pval": pval,
            "ppoc": ppoc,
            "ppoor_lo": poor_lo,
            "ibh": ibh,
            "ibl": ibl,
            "days": days,
        }
    return {
        "pvah": -pval,
        "pval": -pvah,
        "ppoc": -ppoc,
        "ppoor_lo": poor_lo,
        "ibh": -ibl,
        "ibl": -ibh,
        "days": days,
    }


def _day_last(f: dict[str, Any], ie: int) -> int:
    """Last bar of the entry's session that closes by 15:15."""
    _, exit_idx = _cached(f, "_win", lambda: entry_window(f))
    return int(exit_idx[f["day"][ie]])


def _walk_to_level(f: dict[str, Any], ie: int, stop: float, tgt: np.ndarray) -> tuple[int, float, str]:
    """Long from bar `ie`'s open to the session's last bar: stop first, then the target, else the close."""
    o, h, l, c = f["o"], f["h"], f["l"], f["c"]
    last = _day_last(f, ie)
    for j in range(ie, last + 1):
        if j > ie and o[j] <= stop:
            return j, float(o[j]), "gap_stop"
        if l[j] <= stop:
            return j, float(stop), "stop"
        t = tgt[j]
        if np.isfinite(t) and h[j] >= t:
            return j, float(max(t, o[j])), "target"
    return last, float(c[last]), "eod"


def _check_side(side: str) -> None:
    if side not in _SIDES:
        raise ValueError(side)


def vwap_touch(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """Exit at the session VWAP of the previous bar the first time the high reaches it, else 15:15."""
    _check_side(side)
    tgt = np.where(f["first"], np.nan, npx.shift(f["vwap"]))
    return _walk_to_level(f, ie, stop, tgt)


def va_edge(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """Exit at the previous session's value-area high (the far edge for a long re-entering from below)."""
    _check_side(side)
    return _walk_to_level(f, ie, stop, levels(f)["pvah"])


def poc_touch(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """Exit at the previous session's volume-profile POC."""
    _check_side(side)
    return _walk_to_level(f, ie, stop, levels(f)["ppoc"])


def ib_ext(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """Exit at the initial-balance high plus one initial-balance range."""
    _check_side(side)
    lv = levels(f)
    return _walk_to_level(f, ie, stop, lv["ibh"] + (lv["ibh"] - lv["ibl"]))


def ll10_exit(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """Daily Turtle exit: the stop, else the first close under the previous 10 days' low (filled there)."""
    _check_side(side)
    o, l, c, n = f["o"], f["l"], f["c"], f["n"]
    low10 = npx.shift(f["ll10"])
    for j in range(ie, n):
        if j > ie and o[j] <= stop:
            return j, float(o[j]), "gap_stop"
        if l[j] <= stop:
            return j, float(stop), "stop"
        if c[j] < low10[j]:
            return j, float(c[j]), "ll10"
    return n - 1, float(c[n - 1]), "time"


def next_open(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """Overnight trade: in at the CLOSE of day `ie`, out at the next open (no stop applies)."""
    _check_side(side)
    if ie + 1 >= f["n"]:
        return ie, float(f["c"][ie]), "no_next"
    return ie + 1, float(f["o"][ie + 1]), "next_open"


def state_exit(f: dict[str, Any], ie: int, stop: float, side: str) -> tuple[int, float, str]:
    """State exit: when `f["exit_cond"]` is true at a close, leave at the next open (stop optional)."""
    _check_side(side)
    o, l, c, n = f["o"], f["l"], f["c"], f["n"]
    cond = np.asarray(f["exit_cond"], bool)
    for j in range(ie, n):
        if np.isfinite(stop):
            if j > ie and o[j] <= stop:
                return j, float(o[j]), "gap_stop"
            if l[j] <= stop:
                return j, float(stop), "stop"
        if cond[j]:
            if j + 1 < n:
                return j + 1, float(o[j + 1]), "state"
            return j, float(c[j]), "time"
    return n - 1, float(c[n - 1]), "time"


EXITS_EXTRA: dict[str, ExitFn] = {
    "vwap_touch": vwap_touch,
    "va_edge": va_edge,
    "ib_ext": ib_ext,
    "poc_touch": poc_touch,
    "ll10_exit": ll10_exit,
    "next_open": next_open,
    "state_exit": state_exit,
}
