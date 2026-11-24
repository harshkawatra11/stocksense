"""Ported from market-lab lab/swing.py features() and lab/features.py (MIT, ba3f771); changes: numpy only
(pandas helpers from marketlab.npx), Bars and DailyFrame instead of DataFrames, dates as datetime64[D].

Every indicator the strategies need, computed bar by bar with no look-ahead: a value at bar i only uses bars
to and including i (signals are acted on at the NEXT bar's open).
"""

# ruff: noqa: E741 (Market Lab's h, l, c names)
from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars
from stocksense.strategies.data import DailyFrame, NiftyTable

Feat = dict[str, Any]


def supertrend(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    mult: float = 3.0,  # noqa: E741
) -> tuple[np.ndarray, np.ndarray]:
    """(direction +1/-1, line) of the Supertrend with ATR multiplier `mult`."""
    hl2 = (h + l) / 2
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    n = len(c)
    fu, fd, d = up.copy(), dn.copy(), np.ones(n)
    for i in range(1, n):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else d[i - 1]
    return d, np.where(d > 0, fu, fd)


def swings(h: np.ndarray, l: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:  # noqa: E741
    """Fractal swing highs/lows, CONFIRMED k bars later: arrays hold the last confirmed level at each bar."""
    n = len(h)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(2 * k, n):
        j = i - k
        if h[j] == h[j - k : i + 1].max():
            last_h = h[j]
        if l[j] == l[j - k : i + 1].min():
            last_l = l[j]
        sh[i], sl[i] = last_h, last_l
    return sh, sl


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:  # noqa: E741
    """True range with the first bar's previous close taken as its own close."""
    pc = npx.shift(c, 1, c[0])
    return np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    return 100 - 100 / (1 + npx.rma(np.maximum(d, 0), n) / np.maximum(npx.rma(np.maximum(-d, 0), n), 1e-12))


def _adx(h: np.ndarray, l: np.ndarray, atr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: E741
    up, dn = h - npx.shift(h, 1, h[0]), npx.shift(l, 1, l[0]) - l
    pdm = np.where((up > dn) & (up > 0), up, 0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0)
    a = np.maximum(atr, 1e-12)
    pdi, mdi = 100 * npx.rma(pdm, 14) / a, 100 * npx.rma(mdm, 14) / a
    return npx.rma(100 * abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12), 14), pdi, mdi


def daily_features(d: DailyFrame, nifty: NiftyTable) -> Feat:
    """Daily indicator dict of one stock (port of lab/swing.features); `nifty` supplies the market regime."""
    o, h, l, c, v = d.open, d.high, d.low, d.close, d.volume  # noqa: E741
    n = len(c)
    f: Feat = {"o": o, "h": h, "l": l, "c": c, "v": v, "n": n, "date": d.date}
    tr = true_range(h, l, c)
    f["atr"] = npx.rma(tr, 14)
    for k in (5, 10, 20, 50, 100, 200):
        f[f"sma{k}"] = npx.roll(c, k, "mean")
    for k in (10, 20, 50):
        f[f"ema{k}"] = npx.ema(c, k)
    for k in (2, 14):
        f[f"rsi{k}"] = _rsi(c, k)
    sd = npx.roll(c, 20, "std")
    f["bbu"], f["bbl"] = f["sma20"] + 2 * sd, f["sma20"] - 2 * sd
    f["bbw"] = 4 * sd / f["sma20"]
    f["bbw_min"] = npx.roll(f["bbw"], 120, "min")
    f["hh20"] = npx.shift(npx.roll(h, 20, "max"))
    f["hh55"] = npx.shift(npx.roll(h, 55, "max"))
    f["hh252"] = npx.shift(npx.roll(h, 252, "max"))
    f["ll10"] = npx.roll(l, 10, "min")
    f["ll20"] = npx.roll(l, 20, "min")
    f["vavg"] = npx.shift(npx.roll(v, 50, "mean"))
    m = npx.ema(c, 12) - npx.ema(c, 26)
    f["macd"], f["macds"] = m, npx.ema(m, 9)
    f["st"], f["stl"] = supertrend(h, l, c, npx.rma(tr, 10), 3.0)
    f["adx"], f["pdi"], f["mdi"] = _adx(h, l, f["atr"])
    rng = h - l
    f["nr7"] = rng <= npx.roll(rng, 7, "min")
    f["inside"] = (h < npx.shift(h)) & (l > npx.shift(l))
    f["roc63"] = npx.pct_change(c, 63) * 100
    f["roc126"] = npx.pct_change(c, 126) * 100
    f["gap"] = (o / npx.shift(c) - 1) * 100
    f["chg"] = (c / npx.shift(c) - 1) * 100
    body = abs(c - o)
    f["bull_engulf"] = (c > o) & (npx.shift(c) < npx.shift(o)) & (c >= npx.shift(o)) & (o <= npx.shift(c))
    f["hammer"] = ((np.minimum(o, c) - l) >= 2 * body) & (
        (h - np.maximum(o, c)) <= 0.3 * np.maximum(rng, 1e-9)
    )
    nx = nifty.reindex(d.date)  # market regime and relative strength vs Nifty, aligned by date
    f["mkt_up"] = nx["c"] > nx["sma200"]
    f["mkt_roc63"] = nx["roc63"]
    f["rs63"] = f["roc63"] - f["mkt_roc63"]
    f["uptrend"] = (c > f["sma200"]) & (f["sma50"] > f["sma200"]) & (f["ema20"] > f["sma50"])
    return f


def mirror(b: Bars) -> Bars:
    """Upside-down copy of a bar series: a short setup on `b` is exactly a long setup on its mirror."""
    return Bars(b.ts, -b.open, -b.low, -b.high, -b.close, b.volume)


def _daily_context(daily: DailyFrame, day_dates: np.ndarray, sign: int) -> dict[str, np.ndarray]:
    """Yesterday's daily trend flags, ATR %, 5-day return and NR7, looked up by the intraday bars' dates."""
    dc, dh, dl = daily.close, daily.high, daily.low
    e20, e50 = npx.ema(dc, 20), npx.ema(dc, 50)
    pc = npx.shift(dc)
    dtr = np.fmax(np.fmax(dh - dl, np.abs(dh - pc)), np.abs(dl - pc))
    datr = npx.rma(dtr, 14)
    drng = dh - dl
    cols = {
        "dup": ((dc > e20) & (e20 > e50)).astype(float),
        "ddn": ((dc < e20) & (e20 < e50)).astype(float),
        "datr_pct": np.abs(datr / dc * 100),
        "d_ret5": sign * npx.pct_change(dc, 5) * 100,
        "nr7": (drng <= npx.roll(drng, 7, "min")).astype(float),
    }
    keep = np.r_[daily.date[1:] != daily.date[:-1], True]  # duplicate dates keep the last
    dates = daily.date[keep]
    pos = np.minimum(np.searchsorted(dates, day_dates), len(dates) - 1)
    ok = dates[pos] == day_dates
    out = {}
    for k, x in cols.items():
        shifted = npx.shift(x, 1)[keep]  # yesterday's value, seen at today's date
        out[k] = np.where(ok, shifted[pos], np.nan)
    return out


def intraday_features(
    b: Bars,
    daily: DailyFrame | None,
    tf_min: int = 5,
    index_feat: dict[str, np.ndarray] | None = None,
    sign: int = 1,
) -> Feat:
    """Intraday indicator dict (port of lab/features.build). `b` and `daily` are already mirrored for sign=-1;
    percent features carry `sign` so they read 'in the trade's direction'."""
    o, h, l, c, v = b.open, b.high, b.low, b.close, b.volume  # noqa: E741
    n = len(c)
    day_no = b.day_index()
    day = npx.factorize(day_no)
    first = np.r_[True, day[1:] != day[:-1]]
    last = np.r_[day[1:] != day[:-1], True]
    k = (npx.group_transform(np.ones(n), day, "cumsum") - 1).astype(int)
    mins = b.minute_of_day()
    f: Feat = {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "v": v,
        "day": day,
        "k": k,
        "first": first,
        "last": last,
        "mins": mins,
        "n": n,
        "sign": sign,
        "ts": b.ts,
        "tf": tf_min,
        "date": day_no.astype("datetime64[D]"),
    }
    tr = true_range(h, l, c)
    f["atr"] = npx.rma(tr, 14)
    for p in (9, 20, 21, 50, 200):
        f[f"ema{p}"] = npx.ema(c, p)
    w = np.where(v > 0, v, 1.0)  # VWAP per day (indices have no volume: equal weights)
    tp = (h + l + c) / 3
    f["vwap"] = npx.group_transform(tp * w, day, "cumsum") / npx.group_transform(w, day, "cumsum")
    f["rsi"], f["rsi2"] = _rsi(c, 14), _rsi(c, 2)
    m = npx.ema(c, 12) - npx.ema(c, 26)
    f["macd"], f["macds"] = m, npx.ema(m, 9)
    sma, sd = npx.roll(c, 20, "mean"), npx.roll(c, 20, "std")
    f["bbu"], f["bbl"], f["bbm"] = sma + 2 * sd, sma - 2 * sd, sma
    f["bbw"] = abs(4 * sd / sma)
    f["kcu"], f["kcl"] = f["ema20"] + 1.5 * f["atr"], f["ema20"] - 1.5 * f["atr"]
    f["squeeze"] = (f["bbu"] < f["kcu"]) & (f["bbl"] > f["kcl"])
    f["st_dir"], f["st_line"] = supertrend(h, l, c, npx.rma(tr, 10), 3.0)
    f["adx"], f["pdi"], f["mdi"] = _adx(h, l, f["atr"])
    f["dch"] = npx.shift(npx.roll(h, 20, "max"))  # Donchian of the previous 20 bars
    f["dcl"] = npx.shift(npx.roll(l, 20, "min"))
    hac = (o + h + l + c) / 4  # Heikin-Ashi
    hao = npx.ewm_alpha(npx.shift(hac, 1, o[0]), 0.5)
    f["ha_bull"], f["ha_bear"] = hac > hao, hac < hao
    f["ha_nowick_bull"] = (hac > hao) & (hao <= l + 1e-9)
    f["ha_bull_prev"] = npx.shift(f["ha_bull"].astype(float), 1, 0) > 0

    # day structure
    f["dopen"] = npx.group_transform(o, day, "first")
    f["hod"] = npx.group_transform(h, day, "cummax")  # includes current bar
    f["lod"] = npx.group_transform(l, day, "cummin")
    f["hod_prev"] = np.where(first, np.nan, npx.shift(f["hod"]))
    f["lod_prev"] = np.where(first, np.nan, npx.shift(f["lod"]))
    for name, nb in (("or15", max(1, 15 // tf_min)), ("or30", max(1, 30 // tf_min))):
        hh = npx.group_transform(np.where(k < nb, h, np.nan), day, "max")
        ll = npx.group_transform(np.where(k < nb, l, np.nan), day, "min")
        f[name + "h"], f[name + "l"] = (
            np.where(k >= nb, hh, np.nan),
            np.where(k >= nb, ll, np.nan),
        )  # known after
    starts = np.flatnonzero(first)  # previous day, from the intraday bars
    day_h, day_l = np.maximum.reduceat(h, starts), np.minimum.reduceat(l, starts)
    day_c = c[np.flatnonzero(last)]
    for col, arr in (("h", day_h), ("l", day_l), ("c", day_c)):
        f["pd" + col] = np.r_[np.nan, arr[:-1]][day]
    p_ = (f["pdh"] + f["pdl"] + f["pdc"]) / 3
    bc = (f["pdh"] + f["pdl"]) / 2
    f["cpr_top"], f["cpr_bot"] = np.maximum(bc, 2 * p_ - bc), np.minimum(bc, 2 * p_ - bc)
    f["r1"], f["s1"] = 2 * p_ - f["pdl"], 2 * p_ - f["pdh"]
    f["gap"] = sign * (f["dopen"] / f["pdc"] - 1) * 100
    f["chg"] = sign * (c / f["pdc"] - 1) * 100

    # daily context (yesterday's values only)
    if daily is not None and len(daily) > 60:
        ctx = _daily_context(daily, f["date"], sign)
        f["dup"], f["ddn"] = np.nan_to_num(ctx["dup"]) > 0, np.nan_to_num(ctx["ddn"]) > 0
        f["datr_pct"], f["d_ret5"] = ctx["datr_pct"], ctx["d_ret5"]
        f["nr7"] = np.nan_to_num(ctx["nr7"]) > 0
    else:
        f["dup"] = f["ddn"] = f["nr7"] = np.zeros(n, bool)
        f["datr_pct"] = f["d_ret5"] = np.full(n, np.nan)

    f["rvol"] = npx.rvol_same_bar(v, day, k)  # today's cumulative volume vs the same bar over 10 days
    f["vavg"] = npx.roll(v, 20, "mean")

    # candles
    body, rng = abs(c - o), np.maximum(h - l, 1e-12)
    uw, lw = h - np.maximum(o, c), np.minimum(o, c) - l
    po, pc, ph, pl_ = npx.shift(o), npx.shift(c), npx.shift(h), npx.shift(l)
    f["bull_engulf"] = (c > o) & (pc < po) & (c >= po) & (o <= pc) & (body > 0.5 * rng)
    f["bear_engulf"] = (c < o) & (pc > po) & (c <= po) & (o >= pc) & (body > 0.5 * rng)
    f["hammer"] = (lw >= 2 * body) & (uw <= 0.25 * rng) & (rng > 0.5 * f["atr"])
    f["star"] = (uw >= 2 * body) & (lw <= 0.25 * rng) & (rng > 0.5 * f["atr"])
    f["inside"] = (h < ph) & (l > pl_)
    f["inside_prev"] = npx.shift(f["inside"].astype(float), 1, 0) > 0
    f["marubozu_bull"] = (c > o) & (body >= 0.8 * rng) & (rng > 1.2 * f["atr"])
    f["marubozu_bear"] = (c < o) & (body >= 0.8 * rng) & (rng > 1.2 * f["atr"])
    c1, c2, o1, o2 = npx.shift(c), npx.shift(c, 2), npx.shift(o), npx.shift(o, 2)
    f["soldiers"] = (c > o) & (c1 > o1) & (c2 > o2) & (c > c1) & (c1 > c2)
    f["crows"] = (c < o) & (c1 < o1) & (c2 < o2) & (c < c1) & (c1 < c2)
    b2, b1 = abs(c2 - o2), abs(pc - po)  # morning / evening star: big move, small body, big reversal
    atr2 = npx.shift(f["atr"], 2)
    f["morning_star"] = (c2 < o2) & (b2 > 0.6 * atr2) & (b1 < 0.4 * b2) & (c > o) & (c > (o2 + c2) / 2)
    f["evening_star"] = (c2 > o2) & (b2 > 0.6 * atr2) & (b1 < 0.4 * b2) & (c < o) & (c < (o2 + c2) / 2)
    f["body"], f["rng"] = body, rng

    f["swh"], f["swl"] = swings(h, l, 2)  # structure
    f["low3"], f["high3"] = npx.roll(l, 3, "min"), npx.roll(h, 3, "max")
    f["low6"], f["high6"] = npx.roll(l, 6, "min"), npx.roll(h, 6, "max")

    if index_feat is not None:  # index context aligned by timestamp (duplicate stamps keep the last)
        its = np.asarray(index_feat["ts"])
        lookup = {int(t): i for i, t in enumerate(its)}
        idx = np.array([lookup.get(int(t), -1) for t in b.ts])
        ok = idx >= 0
        f["ix_up"] = np.where(ok, np.asarray(index_feat["up"])[np.maximum(idx, 0)], False).astype(bool)
        f["ix_dn"] = np.where(ok, np.asarray(index_feat["dn"])[np.maximum(idx, 0)], False).astype(bool)
        f["ix_chg"] = np.where(ok, np.asarray(index_feat["chg"], dtype=float)[np.maximum(idx, 0)], np.nan)
        f["rs"] = f["chg"] - f["ix_chg"]
    else:
        f["ix_up"] = f["ix_dn"] = np.ones(n, bool)
        f["ix_chg"] = np.zeros(n)
        f["rs"] = np.zeros(n)
    for key in (
        "bull_engulf",
        "bear_engulf",
        "hammer",
        "star",
        "inside",
        "marubozu_bull",
        "marubozu_bear",
        "soldiers",
        "crows",
        "morning_star",
        "evening_star",
        "squeeze",
    ):
        f[key] = np.nan_to_num(f[key].astype(float)) > 0
    return f


def index_context(f: Feat) -> dict[str, np.ndarray]:
    """What a stock strategy may know about the index at each bar."""
    return {
        "ts": f["ts"],
        "up": (f["c"] > f["vwap"]) & (f["ema21"] > f["ema50"]),
        "dn": (f["c"] < f["vwap"]) & (f["ema21"] < f["ema50"]),
        "chg": f["chg"],
    }
