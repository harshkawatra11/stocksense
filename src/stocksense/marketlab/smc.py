"""Ported from market-lab sm/smc.py (MIT, ba3f771); changes: numpy arrays and epoch-second times
instead of pandas; a daily frame with no earlier day falls back to the intraday previous day, not an error.

Smart-money read of one chart: structure breaks (BOS/CHoCH), order blocks, fair value gaps, liquidity
(previous day high/low, opening range) and their sweeps, VWAP, and the day's events in order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab import structure as S
from stocksense.marketlab.bars import IST_OFFSET_S, Bars, hm


def fvgs(b: Bars, lookback: int = 300) -> list[dict[str, Any]]:
    """Three-candle gaps: bar i's low above bar i-2's high (bullish) or its high below its low (bearish).

    Each is tracked until price trades back into it (touched) or through it (filled).
    """
    d = b.tail(lookback)
    h, lo = d.high, d.low
    out: list[dict[str, Any]] = []
    for i in range(2, len(d)):
        if lo[i] > h[i - 2]:
            out.append({"dir": 1, "lo": float(h[i - 2]), "hi": float(lo[i]), "i": i})
        elif h[i] < lo[i - 2]:
            out.append({"dir": -1, "lo": float(h[i]), "hi": float(lo[i - 2]), "i": i})
    for g in out:
        lo_after, hi_after = lo[g["i"] + 1 :], h[g["i"] + 1 :]
        if g["dir"] > 0:
            g["touched"] = bool((lo_after <= g["hi"]).any())
            g["filled"] = bool((lo_after <= g["lo"]).any())
        else:
            g["touched"] = bool((hi_after >= g["lo"]).any())
            g["filled"] = bool((hi_after >= g["hi"]).any())
        g["time"] = int(d.ts[g["i"]])
        g["status"] = "filled" if g["filled"] else "touched, not filled" if g["touched"] else "untouched"
    return out


def order_blocks(b: Bars, events: list[dict[str, Any]], lookback: int = 300) -> list[dict[str, Any]]:
    """For every structure break, the last opposite-colour candle before the move that broke it.

    Tested once price comes back into it; broken on a close through it.
    """
    d = b.tail(lookback)
    o, h, lo, c = d.open, d.high, d.low, d.close
    idx = {int(t): i for i, t in enumerate(d.ts)}
    out: list[dict[str, Any]] = []
    for e in events:
        k = idx.get(e["time"])
        if k is None:
            continue
        s = 1 if e["dir"] == "bull" else -1
        j = None
        for m in range(k - 1, max(k - 15, 0), -1):  # walk back to the start of the impulse
            if (c[m] < o[m]) if s > 0 else (c[m] > o[m]):
                j = m
                break
        if j is None:
            continue
        ob = {
            "dir": s,
            "lo": float(lo[j]),
            "hi": float(h[j]),
            "time": int(d.ts[j]),
            "formed": int(d.ts[k]),
            "type": e["type"],
        }
        la, ha, ca = lo[k + 1 :], h[k + 1 :], c[k + 1 :]
        ob["tested"] = bool(((la <= ob["hi"]) if s > 0 else (ha >= ob["lo"])).any())
        ob["broken"] = bool(((ca < ob["lo"]) if s > 0 else (ca > ob["hi"])).any())
        out.append(ob)
    return out


def vwap(b: Bars) -> np.ndarray | None:
    """Volume-weighted average price of each IST day; None when the series has no volume (an index)."""
    v = np.nan_to_num(b.volume, nan=0.0)
    if v.sum() == 0:
        return None
    tp = (b.high + b.low + b.close) / 3
    day = b.day_index()
    num = _day_cumsum(tp * v, day)
    den = _day_cumsum(v, day)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.where(den == 0, np.nan, den)


def _day_cumsum(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(x)]
    for a, z in zip(starts[:-1], starts[1:], strict=True):
        out[a:z] = np.cumsum(x[a:z])
    return out


def _fmt(z: dict[str, Any], last: float) -> dict[str, Any]:
    ref = z["hi"] if last > z["hi"] else z["lo"] if last < z["lo"] else last
    out = {
        "dir": z["dir"],
        "lo": round(z["lo"], 2),
        "hi": round(z["hi"], 2),
        "time": int(z["time"]),
        "dist_pct": round((ref / last - 1) * 100, 2),
    }
    out.update({"tested": z["tested"]} if "tested" in z else {"status": z["status"]})
    return out


def read(b: Bars, swing: int = 2, daily: Bars | None = None) -> dict[str, Any] | None:
    """Everything the levels panel shows for one intraday frame (Market Lab's `read`); None under 30 bars."""
    if len(b) < 30:
        return None
    a = S.analyse(b, swing)
    ev = a["events"]
    day = b.day_index()
    today_mask = day == day[-1]
    today = b.take(today_mask)
    days = sorted(set(day.tolist()))
    prev = b.take(day == days[-2]) if len(days) > 1 else None
    last = float(b.close[-1])
    t0 = int(today.ts[0])
    out: dict[str, Any] = {"price": last, "bias": a["bias"], "atr": a["atr"]}
    tev = [e for e in ev if e["time"] >= t0]
    le = ev[-1] if ev else None
    out["structure"] = {
        "last": None
        if le is None
        else {
            "type": le["type"],
            "dir": le["dir"],
            "level": round(float(le["level"]), 2),
            "time": hm(le["time"]),
            "today": bool(le["time"] >= t0),
        },
        "breaks_today": int(len(tev)),
    }
    obs_all = order_blocks(b, ev)
    obs = [x for x in obs_all if not x["broken"]]
    fv_all = fvgs(b)
    fv = [x for x in fv_all if not x["filled"]]

    def near(zs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(zs, key=lambda z: abs((z["lo"] + z["hi"]) / 2 - last))[:3]

    out["order_blocks"] = [_fmt(z, last) for z in near(obs)]
    out["fvgs"] = [_fmt(z, last) for z in near(fv)]
    liq: dict[str, Any] = {}
    prior = None
    if daily is not None and len(daily) >= 2:
        before = np.flatnonzero(daily.day_index() < day[-1])
        if len(before):
            prior = int(before[-1])
    if prior is not None and daily is not None:
        liq["pdh"], liq["pdl"], liq["pdc"] = (
            float(daily.high[prior]),
            float(daily.low[prior]),
            float(daily.close[prior]),
        )
    elif prev is not None and len(prev):
        liq["pdh"], liq["pdl"], liq["pdc"] = (
            float(prev.high.max()),
            float(prev.low.min()),
            float(prev.close[-1]),
        )
    m = today.minute_of_day() - 555
    orng = today.take(m < 15)
    if len(orng):
        liq["or_hi"], liq["or_lo"] = float(orng.high.max()), float(orng.low.min())
    swept = []
    for k, side in (("pdh", 1), ("pdl", -1)):
        if k in liq:
            hit = np.flatnonzero(today.high > liq[k]) if side > 0 else np.flatnonzero(today.low < liq[k])
            if len(hit):
                swept.append(f"{k.upper()} {hm(int(today.ts[hit[0]]))}")
    liq["swept"] = swept
    out["liquidity"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in liq.items()}
    vw = vwap(b)
    if vw is not None:
        out["vwap"] = round(float(vw[-1]), 2)
        up = today.volume[today.close >= today.open].sum()
        tot = today.volume.sum()
        out["buyers_pct"] = round(float(up / tot * 100)) if tot else None
    evs: list[tuple[int, str]] = []
    if "pdc" in liq and len(today):
        o = float(today.open[0])
        gap = (o / liq["pdc"] - 1) * 100
        if abs(gap) >= 0.2:
            fill = (
                np.flatnonzero(today.low <= liq["pdc"])
                if gap > 0
                else np.flatnonzero(today.high >= liq["pdc"])
            )
            evs.append((t0, f"Gap {'up' if gap > 0 else 'down'} {gap:+.1f}%"))
            if len(fill):
                evs.append((int(today.ts[fill[0]]), "Gap filled"))
    for e in tev:
        evs.append((e["time"], f"{e['type']} {'up' if e['dir'] == 'bull' else 'down'}"))
    for z in fv_all:
        if z["time"] >= t0:
            evs.append((z["time"], f"FVG formed ({'bull' if z['dir'] > 0 else 'bear'})"))
    for z in obs_all:
        if z["formed"] >= t0:
            evs.append((z["formed"], f"Order block formed ({'bull' if z['dir'] > 0 else 'bear'})"))
    for s in swept:
        k, hhmm = s.split(" ")
        h_, m_ = (int(x) for x in hhmm.split(":"))
        evs.append((int(day[-1]) * 86400 + h_ * 3600 + m_ * 60 - IST_OFFSET_S, f"{k} swept"))
    evs.sort(key=lambda x: x[0])
    gaps = [(t, x) for t, x in evs if x.startswith("Gap")]
    rest = [(t, x) for t, x in evs if not x.startswith("Gap")]
    out["events"] = [{"time": hm(t), "text": x} for t, x in gaps + rest[-(16 - len(gaps)) :]]
    chart_events = [
        {"time": e["time"], "type": e["type"], "dir": e["dir"], "level": float(e["level"])} for e in ev
    ][-40:]
    vwap_pts = []
    if vw is not None:
        vwap_pts = [
            {"time": int(t), "value": round(float(v), 2)}
            for t, v in zip(b.ts, vw, strict=True)
            if t >= t0 and not np.isnan(v)
        ]
    out["chart"] = {
        "events": chart_events,
        "obs": [{**_fmt(z, last), "end": None} for z in obs[-6:]],
        "fvgs": [_fmt(z, last) for z in fv[-8:]],
        "vwap": vwap_pts,
    }
    return out
