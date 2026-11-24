"""Session and anchored VWAP with volume-weighted sigma bands."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars


def _weights(b: Bars) -> tuple[np.ndarray, np.ndarray]:
    tp = (b.high + b.low + b.close) / 3.0
    w = np.where(b.volume > 0, b.volume, 1.0)
    return tp, w


def _running(tp: np.ndarray, w: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cw = npx.group_transform(w, day, "cumsum")
    ctp = npx.group_transform(tp * w, day, "cumsum")
    ctp2 = npx.group_transform(tp * tp * w, day, "cumsum")
    vwap = ctp / cw
    var = np.maximum(ctp2 / cw - vwap * vwap, 0.0)
    return vwap, np.sqrt(var)


def session_vwap(b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """(vwap, sigma) per bar in price units, reset each IST session; weight is 1 where volume is 0."""
    if len(b) == 0:
        return np.empty(0), np.empty(0)
    tp, w = _weights(b)
    return _running(tp, w, b.day_index())


def bands(vwap: np.ndarray, sigma: np.ndarray, ks: tuple[int, ...] = (1, 2, 3)) -> dict[str, np.ndarray]:
    """VWAP +/- k sigma arrays keyed "u<k>" and "l<k>", in price units."""
    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"u{k}"] = vwap + k * sigma
        out[f"l{k}"] = vwap - k * sigma
    return out


def anchored_vwap(b: Bars, anchor_ts: int) -> np.ndarray:
    """VWAP accumulated from the first bar at or after `anchor_ts` (epoch s), NaN before it; never resets."""
    out = np.full(len(b), np.nan)
    sel = b.ts >= anchor_ts
    if not sel.any():
        return out
    tp, w = _weights(b)
    out[sel] = np.cumsum(tp[sel] * w[sel]) / np.cumsum(w[sel])
    return out
