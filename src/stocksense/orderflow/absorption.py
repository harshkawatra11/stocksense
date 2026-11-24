"""Absorption: heavy volume with little progress at a day extreme, a VWAP band or a prior value-area edge."""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab.bars import Bars

_WINDOW = 20
_MIN_PRIOR = 10
_NEAR = 0.002


def _prior_median(x: np.ndarray, start: int, t: int) -> float:
    """Median of up to 20 bars before t within the session starting at `start`; NaN with fewer than 10."""
    w = x[max(start, t - _WINDOW) : t]
    return float(np.median(w)) if len(w) >= _MIN_PRIOR else float("nan")


def absorption(
    b: Bars,
    vwap: np.ndarray,
    sigma: np.ndarray,
    prev_vah: float | None,
    prev_val: float | None,
) -> list[dict[str, Any]]:
    """Bars with volume >= 3x and range <= 0.5x the prior-bar medians, sitting at a key level.

    Prior bars: up to 20 before it in the same session (min 10). Levels, within 0.2% of the bar:
    session low/high so far, vwap -/+ 1 and 2 sigma, previous VAL/VAH. Side is "bid"
    or "offer"; when both kinds match, the close in the lower/upper half of the bar decides.
    """
    day = b.day_index()
    rng = b.high - b.low
    out: list[dict[str, Any]] = []
    start = 0
    run_lo, run_hi = np.inf, -np.inf
    for t in range(len(b)):
        if t == 0 or day[t] != day[t - 1]:
            start, run_lo, run_hi = t, np.inf, -np.inf
        run_lo, run_hi = min(run_lo, float(b.low[t])), max(run_hi, float(b.high[t]))
        mv, mr = _prior_median(b.volume, start, t), _prior_median(rng, start, t)
        if not (np.isfinite(mv) and mv > 0 and np.isfinite(mr)):
            continue
        if not (b.volume[t] >= 3.0 * mv and rng[t] <= 0.5 * mr):
            continue
        lo_edge, hi_edge = b.low[t] * (1 - _NEAR), b.high[t] * (1 + _NEAR)

        def hit(level: float, lo_edge: float = lo_edge, hi_edge: float = hi_edge) -> bool:
            return bool(np.isfinite(level) and lo_edge <= level <= hi_edge)

        lower: list[tuple[str, float]] = [("day_low", run_lo)]
        upper: list[tuple[str, float]] = [("day_high", run_hi)]
        if np.isfinite(vwap[t]) and np.isfinite(sigma[t]):
            lower += [("vwap-1s", vwap[t] - sigma[t]), ("vwap-2s", vwap[t] - 2 * sigma[t])]
            upper += [("vwap+1s", vwap[t] + sigma[t]), ("vwap+2s", vwap[t] + 2 * sigma[t])]
        if prev_val is not None:
            lower.append(("val", prev_val))
        if prev_vah is not None:
            upper.append(("vah", prev_vah))
        lo_hit = next((n for n, v in lower if hit(float(v))), None)
        hi_hit = next((n for n, v in upper if hit(float(v))), None)
        if lo_hit and hi_hit:
            bid = b.close[t] >= (b.high[t] + b.low[t]) / 2
            lo_hit, hi_hit = (lo_hit, None) if bid else (None, hi_hit)
        if not (lo_hit or hi_hit):
            continue
        out.append(
            {
                "i": t,
                "ts": int(b.ts[t]),
                "side": "bid" if lo_hit else "offer",
                "level": lo_hit or hi_hit,
                "price": float(b.close[t]),
                "volume_ratio": float(b.volume[t] / mv),
                "range_ratio": float(rng[t] / mr) if mr > 0 else 0.0,
            }
        )
    return out
