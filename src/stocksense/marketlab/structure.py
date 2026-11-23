"""Ported from market-lab sm/structure.py (MIT, ba3f771); changes: numpy arrays and epoch-second times
instead of pandas frames and Timestamps; the "already used" test of a swing uses the swing itself; session
levels group by IST day number. Percent and price units are Market Lab's.

Market structure: swings, HH/HL/LH/LL, break of structure (BOS) and change of character (CHoCH),
support/resistance zones, session ranges and volatility regime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars, weekday


def atr(b: Bars, n: int = 14) -> np.ndarray:
    """Wilder average true range (ewm alpha 1/n, adjust=False); the first bar uses its own range."""
    pc = npx.shift(b.close, 1)
    with np.errstate(invalid="ignore"):
        tr = np.nanmax(np.vstack([b.high - b.low, np.abs(b.high - pc), np.abs(b.low - pc)]), axis=0)
    return npx.rma(tr, n)


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI; NaN until the first change and wherever the average loss is zero (as Market Lab)."""
    d = np.diff(close, prepend=np.nan)
    up = npx.rma(np.where(np.isnan(d), np.nan, np.maximum(d, 0.0)), n)
    dn = npx.rma(np.where(np.isnan(d), np.nan, -np.minimum(d, 0.0)), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 - 100 / (1 + up / np.where(dn == 0, np.nan, dn))


def add_indicators(b: Bars) -> dict[str, np.ndarray]:
    """atr, ema20/50/200 and rsi arrays of a bar series."""
    out = {"atr": atr(b), "rsi": rsi(b.close)}
    for n in (20, 50, 200):
        out[f"ema{n}"] = npx.ema(b.close, n)
    return out


def swings(b: Bars, n: int = 3) -> list[dict[str, Any]]:
    """Fractal pivots: a high above the n bars each side, and the mirror for lows.

    `confirmed_at` (epoch s) is the bar n later, when the pivot could first be traded; `ci` its bar index.
    """
    h, lo = b.high, b.low
    out: list[dict[str, Any]] = []
    for i in range(n, len(b) - n):
        if h[i] == h[i - n : i + n + 1].max():
            out.append(
                {
                    "i": i,
                    "time": int(b.ts[i]),
                    "kind": "H",
                    "price": float(h[i]),
                    "confirmed_at": int(b.ts[i + n]),
                    "ci": i + n,
                }
            )
        if lo[i] == lo[i - n : i + n + 1].min():
            out.append(
                {
                    "i": i,
                    "time": int(b.ts[i]),
                    "kind": "L",
                    "price": float(lo[i]),
                    "confirmed_at": int(b.ts[i + n]),
                    "ci": i + n,
                }
            )
    return out


def label_swings(sw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag each swing HH/LH (highs) or HL/LL (lows) against the previous one of its kind."""
    last: dict[str, float | None] = {"H": None, "L": None}
    out = []
    for s in sw:
        k, p = s["kind"], s["price"]
        prev = last[k]
        if prev is None:
            label = k
        elif k == "H":
            label = "HH" if p > prev else "LH"
        else:
            label = "HL" if p > prev else "LL"
        last[k] = p
        out.append({**s, "label": label})
    return out


def structure_events(b: Bars, sw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Walk the closes: above the last confirmed swing high is a bullish break, below the last low bearish.

    Against the running trend it is a CHoCH, with it a BOS; each swing breaks once. Returns (events, trend).
    """
    events: list[dict[str, Any]] = []
    trend = 0
    hi = lo = None
    used_hi = used_lo = None
    order = sorted(range(len(sw)), key=lambda k: sw[k]["ci"])
    j = 0
    for i, t in enumerate(b.ts):
        while j < len(order) and sw[order[j]]["confirmed_at"] <= t:
            r = sw[order[j]]
            if r["kind"] == "H":
                hi = r
            else:
                lo = r
            j += 1
        c = b.close[i]
        if hi is not None and c > hi["price"] and used_hi is not hi:
            events.append(
                {"time": int(t), "type": "CHoCH" if trend < 0 else "BOS", "dir": "bull", "level": hi["price"]}
            )
            trend, used_hi = 1, hi
        elif lo is not None and c < lo["price"] and used_lo is not lo:
            events.append(
                {"time": int(t), "type": "CHoCH" if trend > 0 else "BOS", "dir": "bear", "level": lo["price"]}
            )
            trend, used_lo = -1, lo
    return events, trend


def zones(
    b: Bars, ind: dict[str, np.ndarray], sw: list[dict[str, Any]], max_zones: int = 8
) -> list[dict[str, Any]]:
    """Cluster swing prices within half an ATR of each other into zones, scored by touches and recency."""
    if not sw:
        return []
    a = float(ind["atr"][-1])
    price = float(b.close[-1])
    end = int(b.ts[-1])
    span = float(end - int(b.ts[0])) or 1.0
    pts = sorted(sw, key=lambda r: r["price"])
    clusters: list[list[dict[str, Any]]] = []
    cur = [pts[0]]
    for r in pts[1:]:
        # chain neighbours within half an ATR, but never let one zone grow past one ATR wide
        if r["price"] - cur[-1]["price"] <= a * 0.5 and r["price"] - cur[0]["price"] <= a:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    clusters.append(cur)
    out = []
    for c in clusters:
        ps = [r["price"] for r in c]
        recency = max(1 - (end - r["time"]) / span for r in c)
        score = len(c) + 2 * recency
        mid = float(np.mean(ps))
        out.append(
            {
                "low": round(float(min(ps)), 2),
                "high": round(float(max(ps)), 2),
                "mid": round(mid, 2),
                "touches": len(c),
                "score": round(score, 2),
                "type": "resistance" if mid > price else "support",
                "dist_atr": round((mid - price) / a, 2),
            }
        )
    out = [z for z in out if z["touches"] >= 2]
    out.sort(key=lambda z: (abs(z["dist_atr"]) > 6, -z["score"]))
    return sorted(out[:max_zones], key=lambda z: -z["mid"])


def session_levels(m15: Bars) -> dict[str, dict[str, float]]:
    """Today's opening range and range, yesterday's high/low/close with its CPR, and last week's range.

    Uses IST trading days; weeks end on Friday (Saturday and Sunday count toward the following week).
    """
    if len(m15) == 0:
        return {}
    day = m15.day_index()
    out: dict[str, dict[str, float]] = {}
    mask = day == day[-1]
    th, tl = m15.high[mask], m15.low[mask]
    if len(th):
        out["Opening 15m"] = {"high": round(float(th[:1].max()), 2), "low": round(float(tl[:1].min()), 2)}
        out["Today"] = {"high": round(float(th.max()), 2), "low": round(float(tl.min()), 2)}
    days = np.unique(day)
    d_hi = np.array([m15.high[day == d].max() for d in days])
    d_lo = np.array([m15.low[day == d].min() for d in days])
    d_cl = np.array([m15.close[day == d][-1] for d in days])
    if len(days) > 1:
        yh, yl, yc = float(d_hi[-2]), float(d_lo[-2]), float(d_cl[-2])
        out["Prev day"] = {"high": round(yh, 2), "low": round(yl, 2), "close": round(yc, 2)}
        pv = (yh + yl + yc) / 3
        bc = (yh + yl) / 2
        tc = 2 * pv - bc
        out["CPR"] = {
            "high": round(max(tc, bc), 2),
            "low": round(min(tc, bc), 2),
            "pivot": round(pv, 2),
            "width_pct": round(abs(tc - bc) / pv * 100, 3),
        }
    week = np.array([int(d) + ((4 - weekday(d)) % 7) for d in days])
    wks = np.unique(week)
    w_hi = [float(d_hi[week == w].max()) for w in wks]
    w_lo = [float(d_lo[week == w].min()) for w in wks]
    if len(wks) > 1:
        out["Prev week"] = {"high": round(w_hi[-2], 2), "low": round(w_lo[-2], 2)}
    return out


def regime(b: Bars, ind: dict[str, np.ndarray]) -> dict[str, Any]:
    """Volatility: today's ATR rank over the last year. Trend: price vs the 50/200 EMAs and the 50's slope."""
    a = ind["atr"][~np.isnan(ind["atr"])]
    pct = float(np.mean(a[-260:] < a[-1]) * 100)
    vol = "low" if pct < 30 else "high" if pct > 75 else "normal"
    c, e50, e200 = b.close[-1], ind["ema50"][-1], ind["ema200"][-1]
    slope = (ind["ema50"][-1] / ind["ema50"][-10] - 1) * 100
    if c > e50 > e200 and slope > 0:
        trend = "uptrend"
    elif c < e50 < e200 and slope < 0:
        trend = "downtrend"
    else:
        trend = "range"
    return {
        "volatility": vol,
        "atr_pctile": round(pct),
        "trend": trend,
        "ema50_slope": round(float(slope), 2),
    }


def analyse(b: Bars, swing_n: int = 3) -> dict[str, Any]:
    """Full structure read of one timeframe (Market Lab's `analyse`; `ind` holds the indicator arrays)."""
    ind = add_indicators(b)
    sw = label_swings(swings(b, swing_n))
    ev, trend = structure_events(b, sw)
    bias = {1: "bullish", -1: "bearish", 0: "neutral"}[trend]
    return {
        "ind": ind,
        "swings": sw,
        "events": ev,
        "bias": bias,
        "last_event": dict(ev[-1]) if ev else None,
        "recent_swings": [{"time": s["time"], "label": s["label"], "price": s["price"]} for s in sw[-6:]],
        "zones": zones(b, ind, sw[-120:]),
        "regime": regime(b, ind),
        "price": round(float(b.close[-1]), 2),
        "atr": round(float(ind["atr"][-1]), 2),
        "rsi": round(float(ind["rsi"][-1]), 1),
    }


__all__ = [
    "add_indicators",
    "analyse",
    "atr",
    "label_swings",
    "regime",
    "rsi",
    "session_levels",
    "structure_events",
    "swings",
    "zones",
]
