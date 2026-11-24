"""Flow strategies (PLAN-M M5.5): does following the money pay? Open interest, delivery, participants, gamma,
pre-open and cumulative-delta rules, exactly as the card.

Same conventions as `tiers.py`: a per-stock rule is written for a long on the feature dict `f` and returns
`(signal, stop)`; a signal at bar i is acted on at bar i+1's open; a short runs on the mirrored chart. The
daily flow facts (OI buildup label, delivery spike) and the pre-open numbers are attached to `f` by
`attach_daily_flow` and `attach_preopen`; without them a rule stays silent. Rules whose data is not stored yet
carry `status="await_data"`.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars
from stocksense.orderflow import delta as DL
from stocksense.orderflow import vwap as VW
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.intraday import nz

Feat = dict[str, Any]
Out = tuple[np.ndarray, np.ndarray]
sh = npx.shift
_OPEN, _IB_END, _LAST_ENTRY = 555, 615, 870  # 09:15, 10:15 and 14:30 in minutes of the day


def _none(n: int) -> Out:
    return np.zeros(n, bool), np.full(n, np.nan)


def _day_bounds(f: Feat) -> np.ndarray:
    return np.r_[np.flatnonzero(f["first"]), f["n"]]


# ------------------------------------------------------------------------------------------ attach data
def attach_daily_flow(f: Feat, oi: pl.DataFrame | None, deliv: pl.DataFrame | None) -> Feat:
    """Add the stock's `flow_oi_daily` and `flow_delivery_daily` rows to a daily dict, aligned by date.

    `oi`: date, quad_label, oi_chg_pct, price_chg_pct; `deliv`: date, deliv_spike. A date with no row gets
    an empty label, NaN changes and no spike.
    """
    n, dates = f["n"], f["date"]
    label = np.full(n, "", dtype=object)
    oi_chg, px_chg = np.full(n, np.nan), np.full(n, np.nan)
    spike = np.zeros(n, bool)
    if oi is not None and oi.height:
        at = {np.datetime64(d, "D"): r for d, r in zip(oi["date"], oi.iter_rows(named=True), strict=True)}
        for i, d in enumerate(dates):
            r = at.get(d)
            if r is not None:
                label[i] = r["quad_label"] or ""
                oi_chg[i] = np.nan if r["oi_chg_pct"] is None else r["oi_chg_pct"]
                px_chg[i] = np.nan if r["price_chg_pct"] is None else r["price_chg_pct"]
    if deliv is not None and deliv.height:
        sp = {
            np.datetime64(d, "D"): bool(s) for d, s in zip(deliv["date"], deliv["deliv_spike"], strict=True)
        }
        spike = np.array([sp.get(d, False) for d in dates], bool)
    f["oi_label"], f["oi_chg_pct"], f["price_chg_pct"], f["deliv_spike"] = label, oi_chg, px_chg, spike
    return f


def attach_preopen(f: Feat, pre: pl.DataFrame) -> Feat:
    """Add `pre_imb` and `pre_gap` (per bar, the session's pre-open imbalance and indicative-price gap) by
    date.

    `pre` has date, imb (-1..1, buy minus sell over total) and gap (fraction of the previous close), both in
    the real chart's frame (the rules apply `f["sign"]`).
    """
    at = {np.datetime64(d, "D"): (i, g) for d, i, g in pre.select("date", "imb", "gap").iter_rows()}
    imb, gap = np.full(f["n"], np.nan), np.full(f["n"], np.nan)
    for k, d in enumerate(f["date"]):
        if d in at:
            imb[k], gap[k] = at[d]
    f["pre_imb"], f["pre_gap"] = imb, gap
    return f


# ------------------------------------------------------------------------------------------ daily rules
def flow_oi_long_buildup(f: Feat) -> Out:
    """Long buildup (price and open interest both up) with OI up over 5% and price over 1%: long next open."""
    sig = (
        (np.asarray(f.get("oi_label", np.full(f["n"], "")), object) == "long buildup")
        & nz(np.asarray(f.get("oi_chg_pct", np.full(f["n"], np.nan)), float) > 5)
        & nz(np.asarray(f.get("price_chg_pct", np.full(f["n"], np.nan)), float) > 1)
    )
    return sig, f["c"] - 2 * f["atr"]


def flow_short_covering_rev(f: Feat) -> Out:
    """Short covering after a 10-day fall of more than 8%: long; stop 0.3 ATR under the 10-day low."""
    covering = np.asarray(f.get("oi_label", np.full(f["n"], "")), object) == "short covering"
    sig = covering & nz(npx.pct_change(f["c"], 10) < -0.08)
    return sig, npx.roll(f["l"], 10, "min") - 0.3 * f["atr"]


def flow_deliv_spike(f: Feat) -> Out:
    """Delivery spike (z over 2 on an up day) with the close above its 50-day mean: long; stop 2.5 ATR
    under."""
    spike = np.asarray(f.get("deliv_spike", np.zeros(f["n"], bool)), bool)
    return spike & nz(f["c"] > f["sma50"]), f["c"] - 2.5 * f["atr"]


# ------------------------------------------------------------------------------------------ index timing
def fii_index_position(participants: pl.DataFrame, hold: int = 5) -> pl.DataFrame:
    """(date, position): 1 for `hold` sessions after a day on which the FII long ratio rose by over 0.05 in
    5 sessions.

    A signal at close t is held through the next `hold` sessions, so positions decided at t..t+hold-1 are 1.
    """
    sig = (pl.col("fii_ratio_chg5") > 0.05).fill_null(False).cast(pl.Int8)
    out = participants.sort("date").select("date", sig.alias("_s"))
    return out.select(
        "date", pl.col("_s").rolling_max(window_size=hold, min_samples=1).cast(pl.Int8).alias("position")
    )


def timing_fii_index(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """(date, position) of flow_fii_index over start..end (decided at close of `date`, held from the next
    open)."""
    p = io.read(
        settings,
        "flow_participants_daily",
        start - timedelta(days=14),
        end,
        columns=["date", "fii_ratio_chg5"],
    )
    return fii_index_position(p.collect()).filter(pl.col("date").is_between(start, end))


def gex_day_trade(day: Bars, gex_prev: float) -> dict[str, Any] | None:
    """The one NIFTY trade of a session given the previous session's total GEX (None when no trade).

    Negative GEX (dealers short gamma): follow the 09:15-10:15 move from 10:15 to 15:15.
    Positive GEX (moves are damped): fade the first close beyond VWAP +/- 1 sigma, back to VWAP, out by 15:15.
    Returns side (+1/-1), entry_i (entered at that bar open), exit_i (the last bar closing by 15:15), regime.
    """
    if not np.isfinite(gex_prev) or len(day) < 12:
        return None
    mins = day.minute_of_day()
    last = int(np.flatnonzero(mins + 5 <= 915)[-1])
    if gex_prev < 0:
        window = mins < _IB_END
        late = np.flatnonzero(mins >= _IB_END)
        if window.sum() < 12 or not len(late):
            return None
        move = float(day.close[window][-1] - day.open[0])
        if move == 0:
            return None
        return {"regime": "negative", "side": 1 if move > 0 else -1, "entry_i": int(late[0]), "exit_i": last}
    vw, sd = VW.session_vwap(day)
    k = np.arange(len(day))
    for t in range(len(day) - 1):
        if k[t] < 6 or mins[t + 1] > _LAST_ENTRY or sd[t] <= 0:
            continue
        if day.close[t] > vw[t] + sd[t]:
            return {"regime": "positive", "side": -1, "entry_i": t + 1, "exit_i": last, "target": "vwap"}
        if day.close[t] < vw[t] - sd[t]:
            return {"regime": "positive", "side": 1, "entry_i": t + 1, "exit_i": last, "target": "vwap"}
    return None


def timing_gex(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """(date, position in -1/0/1): side of each session GEX-regime trade; empty without NIFTY 5m bars."""
    from stocksense.strategies.data import intraday_frame

    schema = {"date": pl.Date, "position": pl.Int8}
    nf = intraday_frame(settings, "NIFTY 50", start, end)
    if nf is None:
        return pl.DataFrame(schema=schema)
    opt = (
        io.read(
            settings,
            "flow_options_daily",
            start - timedelta(days=14),
            end,
            columns=["date", "underlying", "gex_total_cr"],
        )
        .filter(pl.col("underlying") == "NIFTY")
        .sort("date")
        .collect()
    )
    gex = dict(zip(opt["date"].to_list(), opt["gex_total_cr"].to_list(), strict=True))
    days = sorted(gex)
    rows = []
    day_no = nf.day_index()
    for d in np.unique(day_no):
        dt = date(1970, 1, 1) + timedelta(days=int(d))
        prev = [x for x in days if x < dt]
        g = gex[prev[-1]] if prev else float("nan")
        tr = gex_day_trade(nf.take(day_no == d), float("nan") if g is None else g)
        rows.append({"date": dt, "position": 0 if tr is None else tr["side"]})
    return pl.DataFrame(rows, schema=schema)


# ------------------------------------------------------------------------------------------ intraday rules
def flow_preopen_imb(f: Feat) -> Out:
    """Pre-open imbalance over 0.3 with the indicative-price gap the same way: follow at 09:20 (out 15:15)."""
    s = f.get("sign", 1)
    imb = s * np.asarray(f.get("pre_imb", np.full(f["n"], np.nan)), float)
    gap = s * np.asarray(f.get("pre_gap", np.full(f["n"], np.nan)), float)
    return f["first"] & nz(imb > 0.3) & nz(gap > 0), f["c"] - 2 * f["atr"]


def cvd_slope_open(f: Feat) -> np.ndarray:
    """Per bar from 10:10 on: the slope of the session CVD (BVC) over 09:15-10:10, in shares of average bar
    volume per bar."""
    b = Bars(np.asarray(f["ts"]), f["o"], f["h"], f["l"], f["c"], f["v"])
    cv = DL.cvd(DL.delta_bvc(b), b.day_index())
    out = np.full(f["n"], np.nan)
    bounds = _day_bounds(f)
    x = np.arange(12, dtype=float)
    for a, z in zip(bounds[:-1], bounds[1:], strict=True):
        m = f["mins"][a:z]
        w = np.flatnonzero((m >= _OPEN) & (m <= _IB_END - 5)) + a
        if len(w) != 12:
            continue
        y = cv[w]
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        vol = float(np.mean(f["v"][w]))
        out[w[-1] : z] = slope / vol if vol > 0 else np.nan
    return out


def cross_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank 0..1 of each value among the finite ones (NaN stays NaN): the universe rank for a
    decile cut."""
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() == 1:
        out[ok] = 0.5
    elif ok.sum() > 1:
        order = np.argsort(np.argsort(v[ok], kind="stable"), kind="stable")
        out[ok] = order / (ok.sum() - 1)
    return out


def flow_cvd_trend(f: Feat) -> Out:
    """CVD slope over the first hour in the top decile of the universe (`f["cvd_rank"] >= 0.9`): long at
    10:15."""
    rank = np.asarray(f.get("cvd_rank", np.full(f["n"], np.nan)), float)
    sig = (f["mins"] == _IB_END - 5) & nz(rank >= 0.9)
    return sig, f["lod"]


# ------------------------------------------------------------------------------------------ registry
_DAILY = (
    (
        "flow_oi_long_buildup",
        "OI long buildup",
        flow_oi_long_buildup,
        ("rr2_20", "time5", "time10", "trail3"),
        "rr2_20",
    ),
    (
        "flow_short_covering_rev",
        "Short covering reversal",
        flow_short_covering_rev,
        ("time5", "rr2_20"),
        "time5",
    ),
    (
        "flow_deliv_spike",
        "Delivery spike above 50-day mean",
        flow_deliv_spike,
        ("time10", "rr2_20", "trail3"),
        "time10",
    ),
)
_INTRA = (
    ("flow_preopen_imb", "Pre-open imbalance follow", flow_preopen_imb, ("eod",), "await_data"),
    ("flow_cvd_trend", "First-hour CVD trend", flow_cvd_trend, ("eod",), "ready"),
)


def _register_all() -> None:
    for key, name, fn, exits, default in _DAILY:
        register(
            StrategySpec(
                key=key,
                name=name,
                source="flow",
                tier="none",
                family="m-swing",
                timeframe="1d",
                kind="flow",
                direction="long",
                exits=exits,
                default_exit=default,
                signal=fn,
                description=(fn.__doc__ or "").strip(),
                reference="PLAN-M M5.5",
            )
        )
    for key, name, fn, exits, status in _INTRA:
        register(
            StrategySpec(
                key=key,
                name=name,
                source="flow",
                tier="none",
                family="m-intra",
                timeframe="5m",
                kind="flow",
                direction="both",
                exits=exits,
                default_exit="eod",
                signal=fn,
                description=(fn.__doc__ or "").strip(),
                reference="PLAN-M M5.5",
                status=status,  # type: ignore[arg-type]
            )
        )
    for key, name, tf, _fam, fn, status in (
        ("flow_fii_index", "FII index-futures long ratio rising", "1d", "m-index", timing_fii_index, "ready"),
        ("flow_gex_regime", "Dealer gamma regime on Nifty", "5m", "m-index", timing_gex, "await_data"),
    ):
        register(
            StrategySpec(
                key=key,
                name=name,
                source="flow",
                tier="none",
                family="m-index",
                timeframe=tf,
                kind="index_timing",
                direction="long" if tf == "1d" else "both",
                exits=("state_exit",) if tf == "1d" else ("eod",),
                default_exit="state_exit" if tf == "1d" else "eod",
                signal=None,
                timing=fn,
                description=(fn.__doc__ or "").strip(),
                reference="PLAN-M M5.5",
                status=status,  # type: ignore[arg-type]
            )
        )


_register_all()
