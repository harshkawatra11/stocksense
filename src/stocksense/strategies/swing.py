"""Ported from market-lab lab/swing.py and swing_live.py (MIT, ba3f771); changes: numpy via npx, registry.

The 23 Market Lab swing strategies. Each is `f -> (signal bool[n], stop float[n])` on the daily feature dict;
a signal at the close of day i is acted on at day i+1's open.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab import npx
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.exits import EXITS_1D

Feat = dict[str, Any]
Out = tuple[np.ndarray, np.ndarray]
sh = npx.shift


def X(cond: Any) -> np.ndarray:  # noqa: N802
    """Boolean array: True where cond is positive (NaN counts as False)."""
    return np.nan_to_num(np.asarray(cond, float)) > 0


def s_breakout20(f: Feat) -> Out:
    """Close above the prior 20-day high in an uptrend."""
    return X(f["c"] > f["hh20"]) & f["uptrend"], f["c"] - 2 * f["atr"]


def s_breakout55(f: Feat) -> Out:
    """Close above the prior 55-day high in an uptrend (Turtle system 2)."""
    return X(f["c"] > f["hh55"]) & f["uptrend"], f["c"] - 2.5 * f["atr"]


def s_52wk(f: Feat) -> Out:
    """Close above the prior 252-day high on 1.5x average volume."""
    return X(f["c"] > f["hh252"]) & X(f["v"] > 1.5 * f["vavg"]), f["c"] - 2.5 * f["atr"]


def s_pullback20(f: Feat) -> Out:
    """Low touches the 20 EMA, close back above it on a green candle, in an uptrend."""
    touch = X(f["l"] <= f["ema20"]) & X(f["c"] > f["ema20"]) & X(f["c"] > f["o"])
    return touch & f["uptrend"], np.fmin(f["l"], sh(f["l"])) - 0.5 * f["atr"]


def s_pullback50(f: Feat) -> Out:
    """Low within 1% of the 50 SMA, close above it, with 50 SMA above 200 SMA."""
    touch = X(f["l"] <= f["sma50"] * 1.01) & X(f["c"] > f["sma50"]) & X(f["c"] > f["o"])
    return touch & X(f["c"] > f["sma200"]) & X(f["sma50"] > f["sma200"]), f["l"] - 0.5 * f["atr"]


def s_rsi2(f: Feat) -> Out:
    """RSI(2) under 10 above the 200 SMA."""
    return X(f["rsi2"] < 10) & X(f["c"] > f["sma200"]), f["c"] - 2.5 * f["atr"]


def s_3down(f: Feat) -> Out:
    """Three lower closes in a row inside an uptrend."""
    c = f["c"]
    d3 = X(c < sh(c)) & X(sh(c) < sh(c, 2)) & X(sh(c, 2) < sh(c, 3))
    return d3 & f["uptrend"], c - 2.5 * f["atr"]


def s_bb_low(f: Feat) -> Out:
    """Close under the lower Bollinger band above the 200 SMA."""
    return X(f["c"] < f["bbl"]) & X(f["c"] > f["sma200"]), f["c"] - 2.5 * f["atr"]


def s_squeeze(f: Feat) -> Out:
    """Band width near its 120-day low yesterday, close above the upper band today."""
    tight = X(sh(f["bbw"]) <= sh(f["bbw_min"]) * 1.1)
    return tight & X(f["c"] > f["bbu"]) & X(f["c"] > f["sma200"]), f["sma20"] - 0.5 * f["atr"]


def s_nr7(f: Feat) -> Out:
    """Close above yesterday's high the day after an NR7 bar."""
    was = sh(f["nr7"].astype(float), 1, 0) > 0
    return was & X(f["c"] > sh(f["h"])) & f["uptrend"], sh(f["l"]) - 0.3 * f["atr"]


def s_inside(f: Feat) -> Out:
    """Close above the mother bar's high the day after an inside day."""
    was = sh(f["inside"].astype(float), 1, 0) > 0
    return was & X(f["c"] > sh(f["h"], 2)) & f["uptrend"], sh(f["l"]) - 0.3 * f["atr"]


def s_flag(f: Feat) -> Out:
    """12% pole in ten days, a tight five-day flag, then a break of the flag."""
    pole = (sh(f["c"], 5) / sh(f["c"], 15) - 1) * 100
    fh = sh(npx.roll(f["h"], 5, "max"))
    fl = sh(npx.roll(f["l"], 5, "min"))
    ok = X(pole > 12) & X((fh - fl) < 2.0 * f["atr"]) & X(f["c"] > fh)
    return ok & X(f["c"] > f["sma50"]), fl - 0.3 * f["atr"]


def s_ema_cross(f: Feat) -> Out:
    """20 EMA crosses above the 50 SMA above the 200 SMA."""
    x = X(f["ema20"] > f["sma50"]) & X(sh(f["ema20"]) <= sh(f["sma50"]))
    return x & X(f["c"] > f["sma200"]), f["c"] - 2.5 * f["atr"]


def s_golden(f: Feat) -> Out:
    """50 SMA crosses above the 200 SMA."""
    x = X(f["sma50"] > f["sma200"]) & X(sh(f["sma50"]) <= sh(f["sma200"]))
    return x, f["c"] - 3 * f["atr"]


def s_macd(f: Feat) -> Out:
    """MACD crosses above its signal line while above zero, in an uptrend."""
    x = X(f["macd"] > f["macds"]) & X(sh(f["macd"]) <= sh(f["macds"])) & X(f["macd"] > 0)
    return x & f["uptrend"], f["c"] - 2.5 * f["atr"]


def s_supertrend(f: Feat) -> Out:
    """Supertrend flips up above the 200 SMA; stop is the supertrend line."""
    return (f["st"] > 0) & (sh(f["st"]) < 0) & X(f["c"] > f["sma200"]), f["stl"]


def s_adx(f: Feat) -> Out:
    """Strong ADX with +DI over -DI, close back above the 10 EMA after a dip, in an uptrend."""
    strong = X(f["adx"] > 25) & X(f["pdi"] > f["mdi"])
    dip = X(f["c"] > f["ema10"]) & X(sh(f["c"]) <= sh(f["ema10"]))
    return strong & dip & f["uptrend"], f["ll10"] - 0.3 * f["atr"]


def s_engulf(f: Feat) -> Out:
    """Bullish engulfing near the 20 EMA in an uptrend."""
    near = X(f["l"] <= f["ema20"] * 1.01)
    return f["bull_engulf"] & near & f["uptrend"], np.fmin(f["l"], sh(f["l"])) - 0.3 * f["atr"]


def s_hammer(f: Feat) -> Out:
    """Hammer at the 50 SMA or 20 EMA in an uptrend."""
    near = X(f["l"] <= f["sma50"] * 1.01) | X(f["l"] <= f["ema20"])
    return f["hammer"] & near & f["uptrend"], f["l"] - 0.3 * f["atr"]


def s_pocket(f: Feat) -> Out:
    """Up day on volume above the largest down-day volume of the last ten days (pocket pivot)."""
    v, c = f["v"], f["c"]
    down_v = np.where(c < sh(c), v, 0)
    mx = sh(npx.roll(down_v, 10, "max"))
    sig = X(c > sh(c)) & X(v > mx) & X(c > f["ema10"]) & f["uptrend"] & X(f["l"] <= f["sma50"] * 1.05)
    return sig, c - 2 * f["atr"]


def s_gap_go(f: Feat) -> Out:
    """Gap above 3% on 2.5x volume that closes green above the 50 SMA."""
    sig = X(f["gap"] > 3) & X(f["v"] > 2.5 * f["vavg"]) & X(f["c"] > f["o"]) & X(f["c"] > f["sma50"])
    return sig, f["l"] - 0.3 * f["atr"]


def s_vcp(f: Feat) -> Out:
    """Tightening near highs: 10-day range under 3.6 ATR within 8% of the 52-week high, then a break.

    Market Lab writes the width as `1.8 * atr * 2`, which is 3.6 ATR; kept as is.
    """
    hh10 = sh(npx.roll(f["h"], 10, "max"))
    ll10 = sh(npx.roll(f["l"], 10, "min"))
    near = X(f["c"] >= 0.92 * f["hh252"])
    sig = (
        X((hh10 - ll10) < 1.8 * f["atr"] * 2)
        & near
        & X(f["c"] > hh10)
        & f["uptrend"]
        & X(f["v"] > 1.3 * f["vavg"])
    )
    return sig, ll10 - 0.3 * f["atr"]


def s_rs_leader(f: Feat) -> Out:
    """Relative strength over 15 points vs Nifty, bought on a close back above the 10 EMA after a dip."""
    dip = X(f["c"] > f["ema10"]) & X(sh(f["c"]) <= sh(f["ema10"]))
    return dip & X(f["rs63"] > 15) & f["uptrend"], f["ll10"] - 0.3 * f["atr"]


def setup_state(f: Feat) -> dict[str, Any]:
    """VCP radar state at the last bar: FORMING when tight near highs in an uptrend."""
    i = f["n"] - 1
    h, l, c, atr = f["h"], f["l"], f["c"], f["atr"]  # noqa: E741
    hh10, ll10 = float(h[i - 9 : i + 1].max()), float(l[i - 9 : i + 1].min())
    tight = (hh10 - ll10) < 3.6 * atr[i]
    ok52 = bool(np.isfinite(f["hh252"][i]))
    near = bool(c[i] >= 0.92 * f["hh252"][i]) if ok52 else False
    trend = bool(f["uptrend"][i])
    rs = f["rs63"][i]
    base = {
        "close": round(float(c[i]), 2),
        "atr_pct": round(float(atr[i] / c[i] * 100), 2),
        "rs63": round(float(rs), 1) if np.isfinite(rs) else None,
        "from_high": round(float((c[i] / f["hh252"][i] - 1) * 100), 1) if ok52 else None,
        "trend": trend,
        "tight": bool(tight),
        "near_high": near,
        "mkt_up": bool(f["mkt_up"][i]),
    }
    if trend and near and tight:
        trig = hh10 + 0.05
        stop = ll10 - 0.3 * atr[i]
        r = trig - stop
        return {
            **base,
            "state": "FORMING",
            "trigger": round(trig, 2),
            "stop": round(float(stop), 2),
            "target": round(float(trig + 3 * r), 2),
            "risk_pct": round(float(r / trig * 100), 2),
            "dist_pct": round(float((trig / c[i] - 1) * 100), 2),
            "vol_need": round(float(1.3 * f["vavg"][i]), 0) if np.isfinite(f["vavg"][i]) else None,
        }
    return {**base, "state": "NONE"}


# (key suffix, Market Lab display name, function, kind)
SWING: tuple[tuple[str, str, Any, str], ...] = (
    ("breakout20", "20-day breakout", s_breakout20, "trend"),
    ("breakout55", "55-day breakout (Turtle)", s_breakout55, "trend"),
    ("52wk", "52-week high + volume", s_52wk, "trend"),
    ("pullback20", "Pullback to 20 EMA", s_pullback20, "trend"),
    ("pullback50", "Pullback to 50 SMA", s_pullback50, "trend"),
    ("rsi2", "RSI(2) dip", s_rsi2, "mr"),
    ("3down", "3 down days in uptrend", s_3down, "mr"),
    ("bb_low", "Close under lower Bollinger", s_bb_low, "mr"),
    ("squeeze", "Volatility squeeze break", s_squeeze, "trend"),
    ("nr7", "NR7 breakout", s_nr7, "trend"),
    ("inside", "Inside-day breakout", s_inside, "trend"),
    ("flag", "Bull flag", s_flag, "trend"),
    ("ema_cross", "EMA20/SMA50 cross", s_ema_cross, "trend"),
    ("golden", "Golden cross", s_golden, "trend"),
    ("macd", "MACD cross > 0", s_macd, "trend"),
    ("supertrend", "Supertrend flip", s_supertrend, "trend"),
    ("adx", "ADX trend dip-buy", s_adx, "trend"),
    ("engulf", "Bullish engulfing @EMA", s_engulf, "trend"),
    ("hammer", "Hammer @MA", s_hammer, "trend"),
    ("pocket", "Pocket pivot", s_pocket, "trend"),
    ("gap_go", "Earnings-style gap & go", s_gap_go, "trend"),
    ("vcp", "VCP tightening break", s_vcp, "trend"),
    ("rs_leader", "RS leader dip-buy", s_rs_leader, "trend"),
)
STRATS: dict[str, tuple[Any, str]] = {name: (fn, kind) for _, name, fn, kind in SWING}
BREAKOUT_KEYS = ("breakout20", "breakout55", "52wk", "vcp")


def _register_all() -> None:
    for key, name, fn, kind in SWING:
        register(
            StrategySpec(
                key=f"ml_swing_{key}",
                name=name,
                source="marketlab_swing",
                tier="none",
                family="m-swing",
                timeframe="1d",
                kind="mr" if kind == "mr" else "trend",
                direction="long",
                exits=EXITS_1D if kind == "mr" else tuple(e for e in EXITS_1D if e != "mr5"),
                default_exit="rr3_40" if key == "vcp" else ("mr5" if kind == "mr" else "trail3"),
                signal=fn,
                strength="breakout_atr" if key in BREAKOUT_KEYS else "signal",
                forming=setup_state if key == "vcp" else None,
                description=(fn.__doc__ or "").strip(),
                reference="market-lab lab/swing.py",
            )
        )


_register_all()
