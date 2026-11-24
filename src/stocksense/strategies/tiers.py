"""Tier-list strategies (PLAN-M M5.3): F-tier negative controls, C-tier and B-tier rules, exactly as the card.

Every per-stock rule is written once for a LONG on the feature dict `f` and returns `(signal bool[n], stop
float[n])`; a signal at bar i is acted on at bar i+1's open (BTST: at bar i's close). Shorts run the same
function on the mirrored chart (`f["sign"] == -1`), so no rule has a separate short branch. Everything a rule
reads at bar i comes from bars up to and including i (the truncation tests prove it).

Rules that need more than the indicator dict build it from the bars inside `f` (session VWAP and sigma,
volume profile, TPO, absorption, CVD): they never need a second input, so the runner only has to supply `f`.
Daily rules that need the Nifty (BTST) read `f["nifty_gt_sma50"]`, added by `extend_daily`.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.marketlab import npx
from stocksense.marketlab import smc as SMC
from stocksense.marketlab import structure as ST
from stocksense.marketlab.bars import Bars
from stocksense.orderflow import absorption as AB
from stocksense.orderflow import delta as DL
from stocksense.orderflow import profile as PR
from stocksense.orderflow import vwap as VW
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.data import NiftyTable
from stocksense.strategies.exits import EXITS_1D, EXITS_INTRA, levels, real_bars
from stocksense.strategies.indicators import true_range
from stocksense.strategies.intraday import nz

Feat = dict[str, Any]
Out = tuple[np.ndarray, np.ndarray]
sh = npx.shift
_NO_MR5 = tuple(e for e in EXITS_1D if e != "mr5")
_OPEN, _IB_END = 555, 615  # 09:15 and 10:15 in minutes of the day


def _none(n: int) -> Out:
    return np.zeros(n, bool), np.full(n, np.nan)


def _bars(f: Feat) -> Bars:
    """The feature dict's bars in the trade's frame (mirrored for a short)."""
    return Bars(np.asarray(f["ts"]), f["o"], f["h"], f["l"], f["c"], f["v"])


# =============================================================================================== F-tier
def tier_f_trendline(f: Feat) -> Out:
    """Low touches a rising line through the last two swing lows (fractal 3, 10+ bars apart); green close
    above it."""
    o, low, c, atr, n = f["o"], f["l"], f["c"], f["atr"], f["n"]
    k = 3
    sig, stop = _none(n)
    lows: list[tuple[int, float]] = []
    for t in range(n):
        j = t - k
        if j >= k and low[j] < low[j - k : j].min() and low[j] <= low[j + 1 : t + 1].min():
            lows.append((j, float(low[j])))  # the fractal is only known 3 bars later, at t
        if len(lows) < 2:
            continue
        (i1, l1), (i2, l2) = lows[-2], lows[-1]
        if l2 <= l1 or i2 - i1 < 10:
            continue
        line = l1 + (l2 - l1) * (t - i1) / (i2 - i1)
        if low[t] <= line + 0.3 * atr[t] and c[t] > line and c[t] > o[t]:
            sig[t], stop[t] = True, line - 0.5 * atr[t]
    return sig, stop


def tier_f_rsi_oversold(f: Feat) -> Out:
    """RSI(14) crosses up through 30; stop 2.5 ATR under the close; no trend filter."""
    r = f["rsi14"]
    return nz(sh(r) < 30) & nz(r >= 30), f["c"] - 2.5 * f["atr"]


def _unicorn(b: Bars, atr: np.ndarray) -> Out:
    """Bullish unicorn: a bearish order block closed through upward (a breaker), a bullish FVG formed after
    the break that overlaps it, then a bar that trades back into the overlap and closes above it."""
    n = len(b)
    sig, stop = _none(n)
    ev, _ = ST.structure_events(b, ST.swings(b, 3))
    obs = SMC.order_blocks(b, [e for e in ev if e["dir"] == "bear"], lookback=max(n, 1))
    at = {int(t): i for i, t in enumerate(b.ts)}
    c, low, high = b.close, b.low, b.high
    gaps = [(i, float(high[i - 2]), float(low[i])) for i in range(2, n) if low[i] > high[i - 2]]
    for ob in obs:
        k = at[ob["formed"]]
        above = np.flatnonzero(c[k + 1 :] > ob["hi"])
        if not len(above):
            continue
        m = k + 1 + int(above[0])  # the bar that closed through the block: it is now a breaker
        for gi, glo, ghi in gaps:
            lo_ov, hi_ov = max(glo, ob["lo"]), min(ghi, ob["hi"])
            if gi <= m or lo_ov >= hi_ov:
                continue
            for t in range(gi + 1, min(n, gi + 1 + _UNICORN_WINDOW)):
                if c[t] < lo_ov:
                    break  # closed through the zone: dead
                if low[t] <= hi_ov and c[t] > hi_ov:
                    sig[t], stop[t] = True, ob["lo"] - 0.1 * atr[t]
                    break
    return sig, stop


_UNICORN_WINDOW = 75  # a zone is live for one session of 5-minute bars


def tier_f_ict_unicorn(f: Feat) -> Out:
    """ICT unicorn: breaker block plus an overlapping fair value gap, entered on the retest."""
    return _unicorn(_bars(f), f["atr"])


# =============================================================================================== C-tier
def tier_c_vwap_mr(f: Feat) -> Out:
    """From 10:00, close under VWAP - 2 sigma with RSI(2) < 10; stop VWAP - 3 sigma (exit vwap_touch)."""
    vw, sd = VW.session_vwap(_bars(f))
    stop = vw - 3 * sd
    sig = (f["mins"] >= 600) & (sd > 0) & (f["c"] < vw - 2 * sd) & (f["c"] > stop) & nz(f["rsi2"] < 10)
    return sig, stop


def extend_daily(f: Feat, nifty: NiftyTable) -> Feat:
    """Add `nifty_gt_sma50` (Nifty close above its 50-day mean, on the stock's own dates) to a daily dict."""
    c = np.asarray(nifty.c, float)
    out = np.zeros(f["n"], bool)
    if len(c):
        diff = c - npx.roll(c, 50, "mean")
        pos = np.minimum(np.searchsorted(nifty.date, f["date"]), len(c) - 1)
        out = (nifty.date[pos] == f["date"]) & (np.nan_to_num(diff[pos], nan=-1.0) > 0)
    f["nifty_gt_sma50"] = out
    return f


def tier_c_btst(f: Feat) -> Out:
    """Buy the close of a strong day (up over 2%, closing in the top 10% of its range, on 1.5x volume, Nifty
    above its 50-day mean); sell the next open."""
    h, low, c = f["h"], f["l"], f["c"]
    ok = (
        nz(f["chg"] > 2)
        & nz(c >= low + 0.9 * (h - low))
        & nz(f["v"] > 1.5 * f["vavg"])
        & np.asarray(f.get("nifty_gt_sma50", np.zeros(f["n"], bool)), bool)
    )
    return ok, low  # the stop is a risk reference only: the trade leaves at the next open


def _orb(f: Feat, hi: str, lo: str) -> Out:
    sig = nz(f[hi] < f["c"]) & nz(sh(f["c"]) <= f[hi]) & nz(f["rvol"] >= 1.2)
    return sig, f[lo]


def tier_c_orb_15(f: Feat) -> Out:
    """Close crosses above the 15-minute opening-range high on 1.2x relative volume; stop the range low."""
    return _orb(f, "or15h", "or15l")


def tier_c_orb_30(f: Feat) -> Out:
    """Close crosses above the 30-minute opening-range high on 1.2x relative volume; stop the range low."""
    return _orb(f, "or30h", "or30l")


# --- macro timing: pure frame functions first, thin readers second
def _session_cut(dates: pl.DataFrame) -> pl.DataFrame:
    """Sessions with `_cut`, the 15:30 IST close of each."""
    return dates.sort("date").with_columns(
        (pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=15, minutes=30))
        .dt.replace_time_zone("Asia/Kolkata")
        .alias("_cut")
    )


def macro_trend_position(sessions: pl.DataFrame, macro: pl.DataFrame, fii_chg5: pl.DataFrame) -> pl.DataFrame:
    """(date, position): 1 when 3 of 4 hold: USDINR, BRENT, US10Y 20-session change < 0; FII 5-day ratio
    change > 0.

    A macro value counts for session t when its `available_ts` is at or before t's 15:30 IST close.
    """
    sp = _session_cut(sessions.select("date"))
    n_true = pl.lit(0)
    for name in ("USDINR", "BRENT", "US10Y"):
        m = (
            macro.filter(pl.col("series").str.to_uppercase() == name)
            .select(pl.col("available_ts"), pl.col("close").alias(f"_{name}"))
            .drop_nulls("available_ts")
            .sort("available_ts")
        )
        sp = sp.join_asof(m, left_on="_cut", right_on="available_ts", strategy="backward").drop(
            "available_ts"
        )
        down = (pl.col(f"_{name}") < pl.col(f"_{name}").shift(20)).fill_null(False)
        n_true = n_true + down.cast(pl.Int32)
    fii = fii_chg5.select("date", pl.col("fii_ratio_chg5"))
    sp = sp.join(fii, on="date", how="left")
    n_true = n_true + (pl.col("fii_ratio_chg5") > 0).fill_null(False).cast(pl.Int32)
    return sp.select("date", (n_true >= 3).cast(pl.Int8).alias("position")).sort("date")


def fii_ratio_position(participants: pl.DataFrame) -> pl.DataFrame:
    """(date, position): 1 when the FII index-futures long ratio is above 0.5 and rose over 5 sessions."""
    cond = ((pl.col("fii_ratio") > 0.5) & (pl.col("fii_ratio_chg5") > 0)).fill_null(False)
    return participants.select("date", cond.cast(pl.Int8).alias("position")).sort("date")


def _sessions(settings: Settings, start: date, end: date) -> pl.DataFrame:
    return (
        io.read(settings, "index_daily", start, end, columns=["index_name", "date"])
        .filter(pl.col("index_name") == "NIFTY 50")
        .select("date")
        .unique()
        .collect()
    )


def timing_macro_trend(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """(date, position) of tier_c_macro_trend over start..end; decided at close t, held from t+1's open."""
    wu = start - timedelta(days=60)
    macro = io.read(settings, "macro_daily", None, end, columns=["series", "close", "available_ts"]).collect()
    fii = io.read(settings, "flow_participants_daily", wu, end, columns=["date", "fii_ratio_chg5"]).collect()
    out = macro_trend_position(_sessions(settings, wu, end), macro, fii)
    return out.filter(pl.col("date").is_between(start, end))


def timing_macro_fii(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """(date, position) of tier_c_macro_fii over start..end; decided at date t, held from t+1's open."""
    p = io.read(
        settings, "flow_participants_daily", start, end, columns=["date", "fii_ratio", "fii_ratio_chg5"]
    )
    return fii_ratio_position(p.collect())


# =============================================================================================== B-tier
def _absorb_flags(f: Feat) -> np.ndarray:
    """True at bars flagged as bid absorption in the trade's frame (real offer absorption for a short)."""
    b = real_bars(f)
    vw, sd = VW.session_vwap(b)
    want = "bid" if f.get("sign", 1) == 1 else "offer"
    flag = np.zeros(f["n"], bool)
    prev = None
    for d in levels(f)["days"]:
        a, z = d["a"], d["z"]
        for r in AB.absorption(
            b.take(slice(a, z)),
            vw[a:z],
            sd[a:z],
            None if prev is None else prev["vah"],
            None if prev is None else prev["val"],
        ):
            if r["side"] == want:
                flag[a + r["i"]] = True
        prev = d
    return flag


def tier_b_absorption(f: Feat) -> Out:
    """The bar after a bid-absorption bar closes above its high; stop 0.1 ATR under its low."""
    flag = _absorb_flags(f)
    c, h, low, atr = f["c"], f["h"], f["l"], f["atr"]
    same = np.r_[False, f["day"][1:] == f["day"][:-1]]
    sig = same & sh(flag.astype(float), 1, 0).astype(bool) & (c > sh(h))
    return sig, sh(low) - 0.1 * sh(atr)


def tier_b_cvd_divergence(f: Feat) -> Out:
    """The bar after a bullish CVD divergence (new 20-bar low, CVD above its value at the prior low) closes
    above that bar's high; stop its low."""
    b = _bars(f)
    div = DL.divergence(b, DL.cvd(DL.delta_bvc(b), b.day_index()))
    same = np.r_[False, f["day"][1:] == f["day"][:-1]]
    sig = same & (sh(div.astype(float), 1, 0) == 1) & (f["c"] > sh(f["h"]))
    return sig, sh(f["l"])


def str_1d_frame(panel: pl.DataFrame) -> pl.DataFrame:
    """(date, sid, score) = minus the one-day return: yesterday's losers rank first."""
    return (
        panel.select("date", "sid", (-pl.col("ret")).alias("score")).drop_nulls("score").sort("date", "sid")
    )


def str_5d_frame(panel: pl.DataFrame) -> pl.DataFrame:
    """(date, sid, score) = minus the 5-session sum of (ret - the date's industry mean ret)."""
    excess = pl.col("ret") - pl.col("ret").mean().over("date", "industry")
    out = panel.sort("sid", "date").with_columns(excess.alias("_ex"))
    out = out.with_columns((-pl.col("_ex").rolling_sum(5)).over("sid").alias("score"))
    return out.select("date", "sid", "score").drop_nulls("score").sort("date", "sid")


def _panel(settings: Settings, start: date, end: date) -> pl.DataFrame:
    cols = ["date", "sid", "ret", "industry", "series"]
    lf = io.read(settings, "panel_daily", start - timedelta(days=15), end, columns=cols)
    return lf.filter(pl.col("series") == "EQ").collect()


def score_str_1d(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """tier_b_str_1d scores for start..end."""
    return str_1d_frame(_panel(settings, start, end)).filter(pl.col("date").is_between(start, end))


def score_str_5d(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """tier_b_str_5d scores for start..end."""
    return str_5d_frame(_panel(settings, start, end)).filter(pl.col("date").is_between(start, end))


# --- SMT divergence (needs NIFTY 50 and NIFTY BANK 5-minute bars: not stored yet, status await_data)
def smt_signals(nifty: Bars, bank: Bars, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """(long, short) per aligned bar: Nifty sets a new `lookback`-bar low in its session while Nifty Bank's
    low stays above its own (mirror for short)."""
    both = np.intersect1d(nifty.ts, bank.ts)
    a, b = nifty.take(np.isin(nifty.ts, both)), bank.take(np.isin(bank.ts, both))
    n = len(a)
    long_, short_ = np.zeros(n, bool), np.zeros(n, bool)
    day = a.day_index()
    start = 0
    for t in range(n):
        if t == 0 or day[t] != day[t - 1]:
            start = t
        if t - start < lookback:
            continue
        w = slice(t - lookback, t)
        long_[t] = a.low[t] < a.low[w].min() and b.low[t] > b.low[w].min()
        short_[t] = a.high[t] > a.high[w].max() and b.high[t] < b.high[w].max()
    return long_, short_


def timing_smt(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """(date, position): +1 / -1 when the session's first SMT signal is long / short, else 0; empty if no
    bars."""
    from stocksense.strategies.data import intraday_frame

    nf, bk = (
        intraday_frame(settings, "NIFTY 50", start, end),
        intraday_frame(settings, "NIFTY BANK", start, end),
    )
    schema = {"date": pl.Date, "position": pl.Int8}
    if nf is None or bk is None:
        return pl.DataFrame(schema=schema)
    lg, st = smt_signals(nf, bk)
    keep = np.isin(nf.ts, np.intersect1d(nf.ts, bk.ts))
    day = nf.day_index()[keep]
    rows = []
    for d in np.unique(day):
        m = day == d
        first_l = int(np.argmax(lg[m])) if lg[m].any() else 10**9
        first_s = int(np.argmax(st[m])) if st[m].any() else 10**9
        pos = 0 if first_l == first_s else (1 if first_l < first_s else -1)
        rows.append({"date": date(1970, 1, 1) + timedelta(days=int(d)), "position": pos})
    return pl.DataFrame(rows, schema=schema)


def tier_b_amt_80(f: Feat) -> Out:
    """Open below the previous value area, then two consecutive 30-minute periods close inside it: long next
    bar; stop 0.2% under VAL."""
    n, c = f["n"], f["c"]
    lv = levels(f)
    sig, stop = _none(n)
    per = (f["mins"] - _OPEN) // 30
    bounds = _day_bounds(f)
    for a, z in zip(bounds[:-1], bounds[1:], strict=True):
        val, vah = lv["pval"][a], lv["pvah"][a]
        if not (np.isfinite(val) and np.isfinite(vah)) or not f["o"][a] < val:
            continue
        inside = {}
        for i in range(a, z):
            if (f["mins"][i] - _OPEN) % 30 != 25:
                continue  # only the last 5-minute bar of a 30-minute period closes it
            p = int(per[i])
            inside[p] = bool(val <= c[i] <= vah)
            if inside[p] and inside.get(p - 1, False):
                sig[i], stop[i] = True, val - 0.002 * abs(val)
                break
    return sig, stop


def _day_bounds(f: Feat) -> np.ndarray:
    return np.r_[np.flatnonzero(f["first"]), f["n"]]


def tier_b_ib_ext(f: Feat) -> Out:
    """After 10:15 the close crosses above the initial-balance high on relative volume >= 1; stop the IB
    midpoint (target IB high + IB range)."""
    lv = levels(f)
    ibh, ibl = lv["ibh"], lv["ibl"]
    sig = (
        nz(f["c"] > ibh) & nz(sh(f["c"]) <= ibh) & (f["mins"] >= _IB_END) & nz(f["rvol"] >= 1) & nz(ibh > ibl)
    )
    return sig, (ibh + ibl) / 2


def tier_b_open_drive(f: Feat) -> Out:
    """A clean opening drive away from the previous POC: long at 09:45; stop the first-30-minute low."""
    n = f["n"]
    sig, stop = _none(n)
    lv = levels(f)
    rb = real_bars(f)
    bounds = _day_bounds(f)
    for a, z in zip(bounds[:-1], bounds[1:], strict=True):
        w = slice(a, a + 6)  # 09:15 to 09:40 bars
        if z - a < 6 or not np.isfinite(lv["ppoc"][a]) or f["mins"][a + 5] != _OPEN + 25:
            continue
        pvah, pval = lv["pvah"][a], lv["pval"][a]
        real_vah, real_val = (pvah, pval) if f.get("sign", 1) == 1 else (-pval, -pvah)
        if PR.open_type(real_vah, real_val, rb.take(w)) != "open_drive":
            continue
        if f["l"][w].min() >= f["o"][a] and f["o"][a] > lv["ppoc"][a]:
            sig[a + 5], stop[a + 5] = True, f["l"][w].min()
    return sig, stop


def tier_b_poor_extreme(f: Feat) -> Out:
    """Previous session had a poor low; today trades under it, then closes back above it: long; stop 0.1 ATR
    under today's low (target: previous POC)."""
    lv = levels(f)
    pdl = f["pdl"]
    sig = (
        lv["ppoor_lo"]
        & nz(f["lod_prev"] < pdl)
        & nz(f["c"] > pdl)
        & nz(sh(f["c"]) <= pdl)
        & nz(lv["ppoc"] > f["c"])
    )
    return sig, f["lod"] - 0.1 * f["atr"]


def tier_b_turtle(f: Feat) -> Out:
    """Close above the prior 20-day high (Turtle system 1); stop 2 N under the close, N = 20-day Wilder ATR;
    leave on a close under the 10-day low."""
    n20 = npx.rma(true_range(f["h"], f["l"], f["c"]), 20)
    return nz(f["c"] > f["hh20"]), f["c"] - 2 * n20


# =============================================================================================== registry
_TIERS: tuple[dict[str, Any], ...] = (
    dict(
        key="tier_f_trendline",
        name="Trendline bounce",
        tier="F",
        tf="1d",
        kind="trend",
        dir="long",
        fn=tier_f_trendline,
        exits=_NO_MR5,
        default="rr2_20",
    ),
    dict(
        key="tier_f_rsi_oversold",
        name="RSI(14) oversold cross",
        tier="F",
        tf="1d",
        kind="mr",
        dir="long",
        fn=tier_f_rsi_oversold,
        exits=_NO_MR5,
        default="time10",
    ),
    dict(
        key="tier_f_ict_unicorn",
        name="ICT unicorn",
        tier="F",
        tf="5m",
        kind="trend",
        dir="both",
        fn=tier_f_ict_unicorn,
        exits=EXITS_INTRA,
        default="rr2",
    ),
    dict(
        key="tier_c_vwap_mr",
        name="VWAP mean reversion",
        tier="C",
        tf="5m",
        kind="mr",
        dir="both",
        fn=tier_c_vwap_mr,
        exits=("vwap_touch", "rr1", "eod"),
        default="vwap_touch",
    ),
    dict(
        key="tier_c_btst",
        name="BTST strong close",
        tier="C",
        tf="1d",
        kind="event",
        dir="long",
        fn=tier_c_btst,
        exits=("next_open",),
        default="next_open",
    ),
    dict(
        key="tier_c_orb_15",
        name="ORB 15m (rvol)",
        tier="C",
        tf="5m",
        kind="breakout",
        dir="both",
        fn=tier_c_orb_15,
        exits=EXITS_INTRA,
        default="rr2",
    ),
    dict(
        key="tier_c_orb_30",
        name="ORB 30m (rvol)",
        tier="C",
        tf="5m",
        kind="breakout",
        dir="both",
        fn=tier_c_orb_30,
        exits=EXITS_INTRA,
        default="rr2",
    ),
    dict(
        key="tier_b_absorption",
        name="Absorption reversal",
        tier="B",
        tf="5m",
        kind="flow",
        dir="both",
        fn=tier_b_absorption,
        exits=("rr2", "vwap_touch", "eod"),
        default="rr2",
    ),
    dict(
        key="tier_b_cvd_divergence",
        name="CVD divergence",
        tier="B",
        tf="5m",
        kind="flow",
        dir="both",
        fn=tier_b_cvd_divergence,
        exits=EXITS_INTRA,
        default="rr2",
    ),
    dict(
        key="tier_b_amt_80",
        name="80% rule (value area)",
        tier="B",
        tf="5m",
        kind="mr",
        dir="both",
        fn=tier_b_amt_80,
        exits=("va_edge", "eod"),
        default="va_edge",
    ),
    dict(
        key="tier_b_ib_ext",
        name="Initial balance extension",
        tier="B",
        tf="5m",
        kind="breakout",
        dir="both",
        fn=tier_b_ib_ext,
        exits=("ib_ext", "eod"),
        default="ib_ext",
    ),
    dict(
        key="tier_b_open_drive",
        name="Open drive away from value",
        tier="B",
        tf="5m",
        kind="trend",
        dir="both",
        fn=tier_b_open_drive,
        exits=("eod",),
        default="eod",
    ),
    dict(
        key="tier_b_poor_extreme",
        name="Poor extreme repair",
        tier="B",
        tf="5m",
        kind="mr",
        dir="both",
        fn=tier_b_poor_extreme,
        exits=("poc_touch", "eod"),
        default="poc_touch",
    ),
    dict(
        key="tier_b_turtle",
        name="Turtle system 1",
        tier="B",
        tf="1d",
        kind="breakout",
        dir="long",
        fn=tier_b_turtle,
        exits=("ll10_exit",),
        default="ll10_exit",
    ),
)


def _register_all() -> None:
    for t in _TIERS:
        fn = t["fn"]
        register(
            StrategySpec(
                key=t["key"],
                name=t["name"],
                source="tier",
                tier=t["tier"],
                family="m-swing" if t["tf"] == "1d" else "m-intra",
                timeframe=t["tf"],
                kind=t["kind"],
                direction=t["dir"],
                exits=t["exits"],
                default_exit=t["default"],
                signal=fn,
                description=(fn.__doc__ or "").strip(),
                reference="PLAN-M M5.3",
            )
        )
    for key, name, tier, fn, exits in (
        ("tier_b_str_1d", "Short-term reversal 1d", "B", score_str_1d, "next_open"),
        ("tier_b_str_5d", "Short-term reversal 5d industry-adjusted", "B", score_str_5d, "time5"),
    ):
        register(
            StrategySpec(
                key=key,
                name=name,
                source="tier",
                tier=tier,
                family="m-xs",
                timeframe="1d",
                kind="xs_factor",
                direction="long",
                exits=(exits,),
                default_exit=exits,
                signal=None,
                score=fn,
                description=(fn.__doc__ or "").strip(),
                reference="PLAN-M M5.3",
            )
        )
    for key, name, tier, tf, fn, status in (
        (
            "tier_c_macro_trend",
            "Macro trend (USDINR, Brent, US10Y, FII)",
            "C",
            "1d",
            timing_macro_trend,
            "ready",
        ),
        ("tier_c_macro_fii", "FII index-futures long ratio", "C", "1d", timing_macro_fii, "ready"),
        ("tier_b_smt", "SMT divergence Nifty vs Bank Nifty", "B", "5m", timing_smt, "await_data"),
    ):
        register(
            StrategySpec(
                key=key,
                name=name,
                source="tier",
                tier=tier,
                family="m-index",
                timeframe=tf,
                kind="index_timing",
                direction="long" if tf == "1d" else "both",
                exits=("state_exit",) if tf == "1d" else ("vwap_touch", "eod"),
                default_exit="state_exit" if tf == "1d" else "vwap_touch",
                signal=None,
                timing=fn,
                description=(fn.__doc__ or "").strip(),
                reference="PLAN-M M5.3",
                status=status,
            )
        )


_register_all()
