"""Multi-timeframe alignment table: trend, value, value-area and momentum signs on 5m, 15m, 60m, 1d, 1w."""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab import structure
from stocksense.marketlab.bars import Bars
from stocksense.orderflow.profile import volume_profile
from stocksense.orderflow.vwap import session_vwap

TIMEFRAMES = ("5m", "15m", "60m", "1d", "1w")
_OPEN_MINUTE = 9 * 60 + 15
_INTRADAY = {"5m": 5, "15m": 15, "60m": 60}
_ROC_BARS = 10
_SMA_BARS = 20
_VA_BARS = 20
_MIN_VA_BARS = 5


def resample(b: Bars, tf: str) -> Bars:
    """Aggregate bars to "15m"/"60m" (session-aligned from 09:15) or "1w" (Monday-based IST week); OHLCV."""
    if tf in _INTRADAY:
        m = _INTRADAY[tf]
        key = b.day_index() * 1440 + np.maximum(b.minute_of_day() - _OPEN_MINUTE, 0) // m
    elif tf == "1w":
        key = (b.day_index() + 3) // 7
    else:
        raise ValueError(f"cannot resample to {tf}")
    n = len(b)
    if n == 0:
        return b
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    return Bars.from_arrays(
        b.ts[starts],
        b.open[starts],
        np.array([b.high[a:z].max() for a, z in zip(starts, ends, strict=True)]),
        np.array([b.low[a:z].min() for a, z in zip(starts, ends, strict=True)]),
        b.close[ends - 1],
        np.array([b.volume[a:z].sum() for a, z in zip(starts, ends, strict=True)]),
    )


def _sign(x: float) -> int:
    return int(np.sign(x)) if np.isfinite(x) else 0


def _vs_va(b: Bars, tf: str) -> int:
    """+1 above / -1 below the prior value area (previous session; previous 20 bars for 1d/1w), else 0."""
    if tf in _INTRADAY:
        day = b.day_index()
        prev_days = np.unique(day[day < day[-1]])
        if len(prev_days) == 0:
            return 0
        prev = b.take(day == prev_days[-1])
    else:
        prev = b.take(slice(max(0, len(b) - 1 - _VA_BARS), len(b) - 1))
        if len(prev) < _MIN_VA_BARS:
            return 0
    va = volume_profile(prev)
    c = float(b.close[-1])
    return 1 if c > va["vah"] else -1 if c < va["val"] else 0


def _row(tf: str, b: Bars) -> dict[str, Any]:
    close = float(b.close[-1])
    _, trend = structure.structure_events(b, structure.swings(b, 3))
    if tf in _INTRADAY:
        vwap, _ = session_vwap(b)
        vs_value = _sign(close - float(vwap[-1]))
    else:
        vs_value = _sign(close - float(b.close[-_SMA_BARS:].mean())) if len(b) >= _SMA_BARS else 0
    roc = _sign(close / float(b.close[-1 - _ROC_BARS]) - 1.0) if len(b) > _ROC_BARS else 0
    return {"tf": tf, "trend": int(trend), "vs_value": vs_value, "vs_va": _vs_va(b, tf), "roc": roc}


def alignment(sym_bars: dict[str, Bars]) -> dict[str, Any]:
    """Per-timeframe signs (+1/0/-1) and their mean `score`; absent 15m/60m/1w are built from 5m/1d."""
    frames: dict[str, Bars] = {}
    for tf in TIMEFRAMES:
        src = sym_bars.get(tf)
        if src is None and tf in ("15m", "60m") and "5m" in sym_bars:
            src = resample(sym_bars["5m"], tf)
        if src is None and tf == "1w" and "1d" in sym_bars:
            src = resample(sym_bars["1d"], "1w")
        if src is not None and len(src) > 0:
            frames[tf] = src
    rows = [_row(tf, frames[tf]) for tf in TIMEFRAMES if tf in frames]
    signs = [r[k] for r in rows for k in ("trend", "vs_value", "vs_va", "roc")]
    return {"rows": rows, "score": float(np.mean(signs)) if signs else 0.0}
