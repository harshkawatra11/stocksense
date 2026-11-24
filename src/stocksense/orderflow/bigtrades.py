"""Big-trade proxy: bars whose volume dwarfs the same minute of earlier sessions, plus their aftermath."""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab.bars import Bars

_LOOKBACK_SESSIONS = 20
_HORIZONS_MIN = (5, 15, 30)


def _baselines(b: Bars, hist: Bars, min_sessions: int) -> np.ndarray:
    """Same-minute median volume of the 20 sessions before each bar's day (NaN if too few sessions)."""
    h_day, h_min, h_vol = hist.day_index(), hist.minute_of_day(), hist.volume
    by_min: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for m in np.unique(h_min):
        sel = h_min == m
        order = np.argsort(h_day[sel], kind="stable")
        by_min[int(m)] = (h_day[sel][order], h_vol[sel][order])
    out = np.full(len(b), np.nan)
    b_day, b_min = b.day_index(), b.minute_of_day()
    for i in range(len(b)):
        entry = by_min.get(int(b_min[i]))
        if entry is None:
            continue
        days, vols = entry
        k = int(np.searchsorted(days, b_day[i], side="left"))  # sessions strictly before this bar's day
        past = vols[max(0, k - _LOOKBACK_SESSIONS) : k]
        if len(past) >= min_sessions:
            out[i] = float(np.median(past))
    return out


def big_bars(b: Bars, hist: Bars, mult: float = 5.0, min_sessions: int = 10) -> list[dict[str, Any]]:
    """Bars with volume > mult x same-minute median of the previous 20 sessions, with fractional outcomes.

    ret_5m/15m/30m are close-to-close after the bar; mfe (>= 0) and mae (<= 0) are the best and worst high/low
    over the next 30 minutes relative to the close, in the bar's direction (side = sign(close-open)).
    Outcome fields are None when the full window is not in `b`.
    """
    base = _baselines(b, hist, min_sessions)
    out: list[dict[str, Any]] = []
    for i in np.flatnonzero(np.isfinite(base) & (b.volume > mult * base)):
        i = int(i)
        t, c = int(b.ts[i]), float(b.close[i])
        side = int(np.sign(b.close[i] - b.open[i]))
        rec: dict[str, Any] = {
            "i": i,
            "ts": t,
            "volume": float(b.volume[i]),
            "baseline": float(base[i]),
            "ratio": float(b.volume[i] / base[i]),
            "side": side,
        }
        for m in _HORIZONS_MIN:
            j = int(np.searchsorted(b.ts, t + m * 60))
            ok = j < len(b) and int(b.ts[j]) == t + m * 60
            rec[f"ret_{m}m"] = float(b.close[j] / c - 1.0) if ok else None
        end = int(np.searchsorted(b.ts, t + 30 * 60))
        if end < len(b) and int(b.ts[end]) == t + 30 * 60:
            hi = float(b.high[i + 1 : end + 1].max())
            lo = float(b.low[i + 1 : end + 1].min())
            up, dn = max(hi / c - 1.0, 0.0), min(lo / c - 1.0, 0.0)
            rec["mfe"], rec["mae"] = (up, dn) if side >= 0 else (-dn, -up)
        else:
            rec["mfe"] = rec["mae"] = None
        out.append(rec)
    return out
