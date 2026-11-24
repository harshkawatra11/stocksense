"""Ported from market-lab lab/strategies.py (MIT, ba3f771); changes: numpy via npx, StrategySpec registry.

The 39 Market Lab intraday strategies, each written once for longs: `f -> (signal bool[n], stop float[n])`;
a signal at bar i is acted on at bar i+1's open. Shorts run the same function on the mirrored chart.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab import npx
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.exits import EXITS_INTRA

Feat = dict[str, Any]
Out = tuple[np.ndarray, np.ndarray]
sh = npx.shift
TRENDS = ("D", "I", "DI")


def cross_up(a: Any, b: Any) -> np.ndarray:
    """True where a crosses above b (a > b now, a <= b on the previous bar)."""
    a = np.asarray(a, float)
    b = np.broadcast_to(np.asarray(b, float), np.shape(a))
    return (a > b) & (sh(a) <= sh(b))


def nz(x: Any) -> np.ndarray:
    """Boolean array: True where x is positive (NaN counts as False)."""
    return np.nan_to_num(np.asarray(x, float), nan=0.0) > 0


def mn(*xs: Any) -> np.ndarray:
    """Element-wise minimum ignoring NaN where the other side has a value."""
    return np.fmin.reduce(xs)


def _first(f: Feat, key: str) -> np.ndarray:
    return npx.ffill(np.where(f["first"], f[key], np.nan))


# ------------------------------------------------------------------ breakouts
def orb15(f: Feat) -> Out:
    """Close crosses above the 15-minute opening range high; stop at the range midpoint."""
    return nz(cross_up(f["c"], f["or15h"])), (f["or15h"] + f["or15l"]) / 2


def orb30(f: Feat) -> Out:
    """Close crosses above the 30-minute opening range high; stop at the range midpoint."""
    return nz(cross_up(f["c"], f["or30h"])), (f["or30h"] + f["or30l"]) / 2


def pdh_break(f: Feat) -> Out:
    """Close crosses above the previous day's high after the first bar."""
    return nz(cross_up(f["c"], f["pdh"])) & (f["k"] > 0), f["low3"] - 0.2 * f["atr"]


def cpr_break(f: Feat) -> Out:
    """Close crosses above a narrow CPR's top."""
    narrow = (f["cpr_top"] - f["cpr_bot"]) / f["c"] < 0.004
    sig = nz(cross_up(f["c"], f["cpr_top"])) & narrow & (f["k"] > 0)
    return sig, mn(f["cpr_bot"], f["low3"]) - 0.1 * f["atr"]


def pivot_r1(f: Feat) -> Out:
    """Close crosses above the classic pivot R1."""
    return nz(cross_up(f["c"], f["r1"])) & (f["k"] > 0), f["low3"] - 0.2 * f["atr"]


def donchian(f: Feat) -> Out:
    """Close crosses above the prior 20-bar high."""
    return nz(cross_up(f["c"], f["dch"])), f["low6"] - 0.1 * f["atr"]


def keltner(f: Feat) -> Out:
    """Close crosses above the upper Keltner band; stop at the 20 EMA."""
    return nz(cross_up(f["c"], f["kcu"])), f["ema20"]


def bb_squeeze(f: Feat) -> Out:
    """Bollinger bands inside Keltner on the previous bar, close above the upper band now."""
    was = sh(f["squeeze"].astype(float), 1, 0) > 0
    return was & (f["c"] > f["bbu"]), f["bbm"]


def gap_and_go(f: Feat) -> Out:
    """A gap-up day: close crosses above the first bar's high after bar 1, within the first 6 bars."""
    first_h = np.where(f["k"] >= 1, _first(f, "h"), np.nan)
    first_l = _first(f, "l")
    green = _first(f, "c") > _first(f, "o")
    return nz(cross_up(f["c"], first_h)) & (f["gap"] > 0.4) & green & (f["k"] <= 6), first_l


def open_drive(f: Feat) -> Out:
    """First candle of the day is a strong one-way bar in the gap's direction: go with it at its close."""
    sig = (
        (f["k"] == 0)
        & (f["c"] > f["o"])
        & (f["body"] > 0.7 * f["rng"])
        & (f["gap"] > -0.2)
        & (f["rng"] > f["atr"])
    )
    return sig, f["l"]


def afternoon_hod(f: Feat) -> Out:
    """After 13:30, close crosses above the high of day so far."""
    return (f["mins"] >= 810) & nz(cross_up(f["c"], f["hod_prev"])), f["low6"] - 0.1 * f["atr"]


def nr7_orb(f: Feat) -> Out:
    """The ORB 15m signal on an NR7 day."""
    s, st = orb15(f)
    return s & f["nr7"], st


def flag(f: Feat) -> Out:
    """Pole (>2 ATR in 6 bars) then a tight 4-bar flag holding the top half of the pole, then a flag break."""
    c, h, l, atr = f["c"], f["h"], f["l"], f["atr"]  # noqa: E741
    fh = sh(npx.roll(h, 4, "max"))
    fl = sh(npx.roll(l, 4, "min"))
    pole_top, pole_bot = sh(c, 5), sh(c, 11)
    pole = pole_top - pole_bot
    ok = (pole > 2 * atr) & (fh - fl < 0.8 * atr) & (fl > pole_bot + 0.5 * pole)
    return nz(ok) & (c > fh), fl - 0.1 * atr


def inside_break(f: Feat) -> Out:
    """Close above the mother bar's high the bar after an inside bar."""
    ph, pl = sh(f["h"]), sh(f["l"])
    sig = f["inside_prev"] & (f["c"] > sh(f["h"], 2)) & (f["c"] > ph)
    return sig, mn(pl, f["l"]) - 0.1 * f["atr"]


# ------------------------------------------------------------------ VWAP / averages / momentum
def vwap_pullback(f: Feat) -> Out:
    """Dip to VWAP then a green close back above it (after the third bar)."""
    touch = f["l"] <= f["vwap"] * 1.0008
    sig = touch & (f["c"] > f["vwap"]) & (f["c"] > f["o"]) & (f["k"] >= 3)
    return sig, mn(f["l"], f["low3"]) - 0.2 * f["atr"]


def vwap_reclaim(f: Feat) -> Out:
    """Close crosses back above VWAP (after the third bar)."""
    return nz(cross_up(f["c"], f["vwap"])) & (f["k"] >= 3), f["low3"] - 0.2 * f["atr"]


def ema_cross(f: Feat) -> Out:
    """9 EMA crosses above the 21 EMA."""
    return nz(cross_up(f["ema9"], f["ema21"])), f["low6"] - 0.1 * f["atr"]


def ema20_pullback(f: Feat) -> Out:
    """Low touches the 20 EMA, green close back above it, with 20 EMA above 50 EMA."""
    sig = (f["l"] <= f["ema20"]) & (f["c"] > f["ema20"]) & (f["c"] > f["o"]) & (f["ema20"] > f["ema50"])
    return sig, f["low3"] - 0.2 * f["atr"]


def supertrend_flip(f: Feat) -> Out:
    """Supertrend flips from down to up; stop is the supertrend line."""
    return (f["st_dir"] > 0) & (sh(f["st_dir"]) < 0), f["st_line"]


def macd_cross(f: Feat) -> Out:
    """MACD crosses above its signal line above the 50 EMA."""
    return nz(cross_up(f["macd"], f["macds"])) & (f["c"] > f["ema50"]), f["low6"] - 0.1 * f["atr"]


def rsi_pullback(f: Feat) -> Out:
    """RSI crosses back above 50 after dipping under 40 within the last six bars."""
    low_rsi = npx.roll(f["rsi"], 6, "min")
    return nz(cross_up(f["rsi"], 50)) & (sh(low_rsi) < 40), f["low6"] - 0.1 * f["atr"]


def rsi2_dip(f: Feat) -> Out:
    """Connors-style: a 2-period RSI washout inside an uptrend, bought as it turns."""
    return (sh(f["rsi2"]) < 10) & (f["c"] > f["o"]) & (f["c"] > f["ema50"]), f["c"] - 1.5 * f["atr"]


def adx_trend(f: Feat) -> Out:
    """Strong ADX with +DI over -DI, close crosses above the 9 EMA."""
    strong = (f["adx"] > 25) & (f["pdi"] > f["mdi"])
    return strong & nz(cross_up(f["c"], f["ema9"])), f["low3"] - 0.2 * f["atr"]


def heikin_ashi(f: Feat) -> Out:
    """First Heikin-Ashi bull candle without a lower wick after a non-bull one."""
    return f["ha_nowick_bull"] & ~f["ha_bull_prev"], f["low3"] - 0.2 * f["atr"]


def momentum_combo(f: Feat) -> Out:
    """VWAP + EMA stack + RSI turning up through 50."""
    sig = (f["c"] > f["vwap"]) & (f["ema9"] > f["ema21"]) & nz(cross_up(f["rsi"], 50))
    return sig, f["low6"] - 0.1 * f["atr"]


def orb_vwap_rvol(f: Feat) -> Out:
    """ORB 15m signal above VWAP on relative volume of at least 1.5."""
    s, st = orb15(f)
    return s & (f["c"] > f["vwap"]) & (np.nan_to_num(f["rvol"], nan=1.5) >= 1.5), st


def rs_pullback(f: Feat) -> Out:
    """Stock leading its index (relative strength) bought on a pullback to the 20 EMA."""
    s, st = ema20_pullback(f)
    return s & (f["rs"] > 0.5), st


# ------------------------------------------------------------------ candles
def engulf_ema(f: Feat) -> Out:
    """Bullish engulfing near the 20 EMA or VWAP."""
    near = f["l"] <= np.maximum(f["ema20"], f["vwap"]) * 1.002
    return f["bull_engulf"] & near, mn(f["l"], sh(f["l"])) - 0.1 * f["atr"]


def hammer_vwap(f: Feat) -> Out:
    """Hammer at VWAP that closes above it."""
    return f["hammer"] & (f["l"] <= f["vwap"] * 1.001) & (f["c"] > f["vwap"]), f["l"] - 0.1 * f["atr"]


def soldiers(f: Feat) -> Out:
    """Three advancing green candles above VWAP."""
    return f["soldiers"] & (f["c"] > f["vwap"]), f["low3"] - 0.1 * f["atr"]


def morning_star(f: Feat) -> Out:
    """Three-candle morning star."""
    return f["morning_star"], f["low3"] - 0.1 * f["atr"]


def marubozu(f: Feat) -> Out:
    """Bullish marubozu closing above the previous high of day."""
    return f["marubozu_bull"] & (f["c"] > f["hod_prev"]), f["l"]


# ------------------------------------------------------------------ smart money concepts
def sweep_pdl(f: Feat) -> Out:
    """Wick under the previous day's low that closes back above it."""
    return (f["l"] < f["pdl"]) & (f["c"] > f["pdl"]) & (f["k"] > 0), f["l"] - 0.1 * f["atr"]


def sweep_orl(f: Feat) -> Out:
    """Wick under the 15-minute opening-range low that closes back above it (after the third bar)."""
    return (f["k"] >= 3) & (f["l"] < f["or15l"]) & (f["c"] > f["or15l"]), f["l"] - 0.1 * f["atr"]


def bos(f: Feat) -> Out:
    """Break of structure: a decisive close through the last confirmed swing high."""
    return nz(cross_up(f["c"], f["swh"])) & (f["body"] > 0.6 * f["rng"]), f["swl"] - 0.1 * f["atr"]


def order_block(f: Feat) -> Out:
    """After a BOS, buy the return into the last down-candle before the impulse (same day, 30 bars)."""
    o, h, l, c, atr, day = f["o"], f["h"], f["l"], f["c"], f["atr"], f["day"]  # noqa: E741
    b, _ = bos(f)
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    zone, born, used = None, -1, True
    for i in range(1, f["n"]):
        if zone and (day[i] != day[born] or i - born > 30 or c[i] < zone[1]):
            zone = None
        if b[i]:
            for j in range(i - 1, max(i - 12, 0), -1):
                if c[j] < o[j] and day[j] == day[i]:
                    zone, born, used = (h[j], l[j]), i, False
                    break
            continue
        if zone and not used and l[i] <= zone[0] and c[i] > zone[0]:
            sig[i], stop[i], used = True, zone[1] - 0.1 * atr[i], True
    return sig, stop


def fvg_retest(f: Feat) -> Out:
    """A bullish fair-value gap left by a strong candle; buy the first dip into it that closes back above."""
    h, l, c, o, atr, day = f["h"], f["l"], f["c"], f["o"], f["atr"], f["day"]  # noqa: E741
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    gap, born = None, -1
    for i in range(2, f["n"]):
        if gap and (day[i] != day[born] or i - born > 24 or c[i] < gap[0]):
            gap = None
        if l[i] > h[i - 2] and (c[i - 1] - o[i - 1]) > 0.8 * atr[i] and day[i - 2] == day[i]:
            gap, born = (h[i - 2], l[i]), i
            continue
        if gap and i > born + 1 and l[i] <= gap[1] and c[i] > gap[1] and c[i] > o[i]:
            sig[i], stop[i] = True, gap[0] - 0.1 * atr[i]
            gap = None
    return sig, stop


def bos_fib(f: Feat) -> Out:
    """BOS, then the pullback into the 50-62% zone of the impulse leg, entered on a bullish close."""
    h, l, c, o, atr, day = f["h"], f["l"], f["c"], f["o"], f["atr"], f["day"]  # noqa: E741
    b, _ = bos(f)
    swl = f["swl"]
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    leg, born = None, -1
    for i in range(1, f["n"]):
        if leg:
            top = max(leg[1], h[i])
            leg = (leg[0], top)
            if day[i] != day[born] or i - born > 30 or l[i] < leg[0]:
                leg = None
        if b[i] and np.isfinite(swl[i]):
            leg, born = (swl[i], h[i]), i
            continue
        if leg:
            lo, hi = leg
            mid = hi - 0.5 * (hi - lo)
            if l[i] <= mid and c[i] > o[i] and c[i] > hi - 0.62 * (hi - lo):
                sig[i], stop[i] = True, lo - 0.1 * atr[i]
                leg = None
    return sig, stop


def double_bottom(f: Feat) -> Out:
    """Two swing lows within 0.3 ATR, then a close through the high between them."""
    c, atr, day = f["c"], f["atr"], f["day"]
    swl, swh = f["swl"], f["swh"]
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    lows: list[tuple[float, int]] = []
    for i in range(1, f["n"]):
        if np.isfinite(swl[i]) and swl[i] != swl[i - 1]:
            lows = [x for x in lows if day[x[1]] == day[i]][-1:] + [(swl[i], i)]
        if (
            len(lows) == 2
            and abs(lows[0][0] - lows[1][0]) < 0.3 * atr[i]
            and np.isfinite(swh[i])
            and c[i] > swh[i]
            and c[i - 1] <= swh[i]
            and swh[i] > max(lows[0][0], lows[1][0]) + 0.5 * atr[i]
        ):
            sig[i], stop[i] = True, min(lows[0][0], lows[1][0]) - 0.1 * atr[i]
            lows = []
    return sig, stop


# (display name, function). Keys are `ml_intra_<snake>`.
INTRADAY: tuple[tuple[str, str, Any], ...] = (
    ("orb_15m", "ORB 15m", orb15),
    ("orb_30m", "ORB 30m", orb30),
    ("gap_and_go", "Gap & go", gap_and_go),
    ("opening_drive", "Opening drive", open_drive),
    ("pdh_breakout", "PDH breakout", pdh_break),
    ("cpr_breakout", "CPR breakout", cpr_break),
    ("pivot_r1_break", "Pivot R1 break", pivot_r1),
    ("donchian_20", "Donchian 20", donchian),
    ("keltner_break", "Keltner break", keltner),
    ("bb_squeeze", "BB squeeze", bb_squeeze),
    ("nr7_orb", "NR7 + ORB", nr7_orb),
    ("bull_flag", "Bull flag", flag),
    ("inside_bar_break", "Inside bar break", inside_break),
    ("afternoon_hod", "Afternoon HOD", afternoon_hod),
    ("vwap_pullback", "VWAP pullback", vwap_pullback),
    ("vwap_reclaim", "VWAP reclaim", vwap_reclaim),
    ("ema_9_21_cross", "EMA 9/21 cross", ema_cross),
    ("ema20_pullback", "EMA20 pullback", ema20_pullback),
    ("supertrend_flip", "Supertrend flip", supertrend_flip),
    ("macd_cross", "MACD cross", macd_cross),
    ("rsi_pullback", "RSI pullback", rsi_pullback),
    ("rsi2_dip", "RSI2 dip", rsi2_dip),
    ("adx_trend", "ADX trend", adx_trend),
    ("heikin_ashi", "Heikin-Ashi", heikin_ashi),
    ("vwap_ema_rsi", "VWAP+EMA+RSI", momentum_combo),
    ("orb_vwap_rvol", "ORB+VWAP+RVOL", orb_vwap_rvol),
    ("rs_pullback", "RS pullback", rs_pullback),
    ("engulfing_ema", "Engulfing @EMA", engulf_ema),
    ("hammer_vwap", "Hammer @VWAP", hammer_vwap),
    ("three_soldiers", "3 soldiers", soldiers),
    ("morning_star", "Morning star", morning_star),
    ("marubozu_hod", "Marubozu HOD", marubozu),
    ("pdl_sweep", "PDL sweep", sweep_pdl),
    ("or_low_sweep", "OR-low sweep", sweep_orl),
    ("bos", "BOS", bos),
    ("order_block", "Order block", order_block),
    ("fvg_retest", "FVG retest", fvg_retest),
    ("bos_50_pullback", "BOS 50% pullback", bos_fib),
    ("double_bottom", "Double bottom", double_bottom),
)
STRATEGIES: dict[str, Any] = {name: fn for _, name, fn in INTRADAY}
DIP_KEYS = ("vwap_pullback", "ema20_pullback", "rsi2_dip", "rsi_pullback", "rs_pullback", "hammer_vwap")


def _register_all() -> None:
    from stocksense.strategies.swing import STRATS as SWING_NAMES

    for key, name, fn in INTRADAY:
        register(
            StrategySpec(
                key=f"ml_intra_{key}",
                name=f"{name} (5m)" if name in SWING_NAMES else name,  # registry names must be unique
                source="marketlab_intraday",
                tier="none",
                family="m-intra",
                timeframe="5m",
                kind="mr" if key in DIP_KEYS else "breakout",
                direction="both",
                exits=EXITS_INTRA,
                default_exit="rr2",
                signal=fn,
                strength="dev_sigma" if key in ("vwap_pullback", "vwap_reclaim", "hammer_vwap") else "signal",
                description=(fn.__doc__ or "").strip(),
                reference="market-lab lab/strategies.py",
            )
        )


_register_all()
