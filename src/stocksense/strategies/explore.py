"""Exploratory engine: every strategy on every stock, with our costs, on completed bars only (PLAN-M M5.6).

Ported from market-lab lab/swing_live.py trades_for and lab/engine.py run_symbol / lab/live.py chart_trades
(MIT, ba3f771); changes: numpy only, our registry, exits and costs, split-adjusted completed sessions.

Results here are exploratory: they are never evidence. Only the gated runner (M5.7) produces evidence.
Units: `Trade.gross_pct`, `cost_pct`, `net_pct` are percent (as Market Lab's JSON); `R` is in risk multiples.
Epoch ints (`t_sig`, `t_in`, `t_out`) are seconds; a daily bar's stamp is its date at 00:00 UTC, and the
`t_in` of a SIGNAL is the nominal next calendar day (`t_sig + 86400`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.marketlab import npx
from stocksense.strategies import base
from stocksense.strategies.base import StrategySpec
from stocksense.strategies.costs import When, trade_cost_frac
from stocksense.strategies.data import (
    DailyFrame,
    NiftyTable,
    completed_end,
    intraday_frame,
    nifty_table,
)
from stocksense.strategies.exits import (
    EXITS_INTRA,
    entry_window,
    follow_1d,
    follow_intra,
    trend_mask,
)
from stocksense.strategies.indicators import daily_features, index_context, intraday_features, mirror

log = get_logger(__name__)

OOS_START = date(2024, 7, 1)
WARMUP = 260  # minimum daily bars; signals before index WARMUP - 40 (= Market Lab's 220) are ignored
MAX_RISK = 0.25  # Market Lab skips a trade whose stop is more than 25% under the entry
HOLD_DAYS = {"rr2_20": 20, "rr3_40": 40, "time5": 5, "time10": 10, "mr5": 10, "trail3": 120, "ema20": 120}
EXIT_NAMES = {
    "trail3": "3-ATR trailing stop",
    "ema20": "close under 20 EMA",
    "rr2_20": "2R or 20 days",
    "rr3_40": "3R or 40 days",
    "time5": "5 days",
    "time10": "10 days",
    "mr5": "close above 5 SMA",
}
CANDIDATE = {"strat": "ml_swing_vcp", "exit": "rr3_40"}
COSTS_MODEL = "stocksense.costs ({product}) + slippage model"
TRENDS = ("D", "I", "DI")
DEFAULT_INTRA = {"strat": "ml_intra_orb_15m", "trend": "DI", "exit": "rr2"}
INTRA_DAYS = 60  # calendar days of 5m history a chart request reads
_OUT_WHEN: dict[str, When] = {"gap_stop": "open", "stop": "now", "target": "now"}


@dataclass(frozen=True)
class Trade:
    """One simulated trade. Percent fields are percent; R is a multiple of the initial risk."""

    sym: str
    strat: str
    exit: str
    side: int
    t_sig: int
    t_in: int
    p_in: float
    stop: float
    target: float | None
    t_out: int | None
    p_out: float | None
    reason: str
    R: float | None
    gross_pct: float | None
    cost_pct: float
    net_pct: float | None
    days: int | None
    status: Literal["CLOSED", "OPEN", "SIGNAL"]
    mkt_up: bool
    rs63: float | None


@dataclass(frozen=True)
class SymCtx:
    """One stock's daily bars plus the per-bar inputs of the cost model (aligned to `frame.date`)."""

    sym: str
    frame: DailyFrame
    rank: np.ndarray  # turnover rank at that date (NaN when unknown)
    vol21: np.ndarray  # 21-day standard deviation of daily returns (fraction)
    med_turn: np.ndarray  # 20-day median of close x volume (INR)


@dataclass
class Prep:
    """A stock ready to run strategies on: context, daily features, Nifty table and slippage table."""

    settings: Settings
    ctx: SymCtx
    f: dict[str, Any]
    slip: SlippageTable


# ---- data loading ------------------------------------------------------------------------------


def _roll_median(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = np.median(sliding_window_view(x, n), axis=1)
    return out


def _epoch(d: np.datetime64) -> int:
    return int(d.astype("datetime64[s]").astype(np.int64))


def epoch_date(t: int) -> date:
    """Calendar date (UTC) of an epoch-second stamp."""
    return date(1970, 1, 1) + timedelta(seconds=int(t))


def make_ctx(sym: str, frame: DailyFrame, rank: np.ndarray | None = None) -> SymCtx:
    """A SymCtx from a daily frame: volatility and median turnover from the bars, rank as given."""
    n = len(frame)
    c = frame.close
    return SymCtx(
        sym,
        frame,
        np.full(n, np.nan) if rank is None else np.asarray(rank, dtype=float),
        npx.roll(npx.pct_change(c, 1), 21, "std"),
        _roll_median(c * frame.volume, 20),
    )


def load_ctx(settings: Settings, symbols: list[str], end: date) -> dict[str, SymCtx]:
    """Contexts of many stocks from panel_daily (EQ, adjusted, volume > 0, dates <= end) in one read."""
    cols = ["sid", "symbol", "series", "date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
    df = (
        io.read(settings, "panel_daily", start=None, end=end, columns=cols)
        .filter(pl.col("symbol").is_in(symbols) & (pl.col("series") == "EQ"))
        .sort("symbol", "date")
        .unique(subset=["symbol", "date"], keep="last", maintain_order=True)
        .filter(pl.col("adj_volume") > 0)
        .drop_nulls(["adj_close"])
        .collect()
    )
    if not df.height:
        return {}
    uni = (
        io.read(settings, "panel_universe", start=None, end=end, columns=["date", "sid", "turnover_rank"])
        .filter(pl.col("sid").is_in(df["sid"].unique().to_list()))
        .sort("sid", "date")
        .collect()
    )
    ranks = {k[0]: g for k, g in uni.partition_by("sid", as_dict=True).items()}
    out: dict[str, SymCtx] = {}
    for key, g in df.partition_by("symbol", as_dict=True).items():
        dates = np.array(g["date"].to_list(), dtype="datetime64[D]")

        def col(name: str, g: pl.DataFrame = g) -> np.ndarray:
            return g[name].to_numpy().astype(float)

        frame = DailyFrame(
            dates,
            col("adj_open"),
            col("adj_high"),
            col("adj_low"),
            col("adj_close"),
            col("adj_volume"),
        )
        rank = None
        u = ranks.get(g["sid"][-1])
        if u is not None and u.height:
            ud = np.array(u["date"].to_list(), dtype="datetime64[D]")
            pos = np.searchsorted(ud, dates, side="right") - 1
            r = u["turnover_rank"].to_numpy().astype(float)
            rank = np.where(pos >= 0, r[np.maximum(pos, 0)], np.nan)
        out[key[0]] = make_ctx(key[0], frame, rank)
    return out


def universe_symbols(settings: Settings, asof: date) -> list[str]:
    """Symbols in the u500 or F&O universe on the last stored universe date <= asof, sorted."""
    uni = io.read(
        settings,
        "panel_universe",
        start=asof - timedelta(days=10),
        end=asof,
        columns=["date", "sid", "u500", "in_fno"],
    ).collect()
    if not uni.height:
        return []
    last = uni["date"].max()
    sids = uni.filter((pl.col("date") == last) & (pl.col("u500") | pl.col("in_fno")))["sid"].to_list()
    px = (
        io.read(
            settings,
            "panel_daily",
            start=asof - timedelta(days=10),
            end=asof,
            columns=["sid", "symbol", "series", "date"],
        )
        .filter(pl.col("sid").is_in(sids) & (pl.col("series") == "EQ"))
        .collect()
    )
    return sorted(px["symbol"].unique().to_list())


def make_prep(
    settings: Settings,
    ctx: SymCtx,
    nifty: NiftyTable | None = None,
    slip: SlippageTable | None = None,
) -> Prep:
    """Daily features for one stock (Nifty regime from `nifty` or the stored index)."""
    last = ctx.frame.date[-1].astype("datetime64[D]").item()
    nf = nifty if nifty is not None else nifty_table(settings, end=last)
    sl = slip if slip is not None else SlippageTable.load_latest(settings)
    return Prep(settings, ctx, daily_features(ctx.frame, nf), sl)


def _resolve_ctx(
    sym: str,
    settings: Settings,
    clock: Clock,
    ctx: SymCtx | None,
    end: date | None,
    cal: TradingCalendar | None,
) -> SymCtx | None:
    if ctx is not None:
        return ctx
    if end is None:
        end = completed_end(clock, cal or TradingCalendar.from_store(settings, clock), settings)
    return load_ctx(settings, [sym], end).get(sym)


# ---- costs -------------------------------------------------------------------------------------


def _cost(prep: Prep, i: int, notional: float, product: Literal["delivery", "intraday"], out: When) -> float:
    """Round-trip cost fraction at daily bar i: entry at an open, exit as `out`."""
    c = prep.ctx
    rank = c.rank[i]
    vol, med = c.vol21[i], c.med_turn[i]
    return trade_cost_frac(
        product,
        notional,
        prep.settings,
        prep.slip,
        int(rank) if np.isfinite(rank) else None,
        float(vol) if np.isfinite(vol) else 0.02,
        float(med) if np.isfinite(med) and med > 0 else notional * 100,
        "open",
        out,
        as_of=c.frame.date[i].astype("datetime64[D]").item(),
    )


# ---- daily trades ------------------------------------------------------------------------------


def _target(exit_key: str, e: float, r: float) -> float | None:
    return {"rr2_20": e + 2 * r, "rr3_40": e + 3 * r}.get(exit_key)


def _rs(f: dict[str, Any], i: int) -> float | None:
    v = f["rs63"][i]
    return float(v) if np.isfinite(v) else None


def trades_1d(
    prep: Prep, spec: StrategySpec, sig: Any, stop: Any, exit_key: str, *, warmup: int, notional: float
) -> list[Trade]:
    """Market Lab's trades_for loop: a signal at close i enters at open i+1, one position at a time."""
    f, ctx = prep.f, prep.ctx
    n = f["n"]
    if n < warmup or exit_key not in HOLD_DAYS:
        return []
    sg = np.asarray(sig, bool).copy()
    sg[: max(warmup - 40, 0)] = False
    stop = np.asarray(stop, float)
    ts = ctx.frame.date.astype("datetime64[s]").astype(np.int64)
    o, c = f["o"], f["c"]
    maxd = HOLD_DAYS[exit_key]
    out: list[Trade] = []
    busy = -1
    for i in np.flatnonzero(sg):
        i = int(i)
        s = float(stop[i])
        if i <= busy or not np.isfinite(s):
            continue
        base_args = {"sym": ctx.sym, "strat": spec.key, "exit": exit_key, "side": 1, "t_sig": int(ts[i])}
        if i == n - 1:  # fired on the latest completed close: buy at the next open
            e = float(c[i])
            r = e - s
            if r > 0 and r / e <= MAX_RISK:
                out.append(
                    Trade(
                        **base_args,
                        t_in=int(ts[i]) + 86400,
                        p_in=e,
                        stop=s,
                        target=_target(exit_key, e, r),
                        t_out=None,
                        p_out=None,
                        reason="enter at next open",
                        R=None,
                        gross_pct=None,
                        cost_pct=100 * _cost(prep, i, notional, "delivery", "now"),
                        net_pct=None,
                        days=None,
                        status="SIGNAL",
                        mkt_up=bool(f["mkt_up"][i]),
                        rs63=_rs(f, i),
                    )
                )
            continue
        ie = i + 1
        e = float(o[ie])
        r = e - s
        if r <= 0 or r / e > MAX_RISK:
            continue
        j, px, why = follow_1d(f, ie, s, exit_key)
        busy = j
        live = j == n - 1 and why == "time" and j - ie + 1 < maxd
        cost = _cost(prep, i, notional, "delivery", "close" if live else _OUT_WHEN.get(why, "close"))
        g = (px - e) / e
        net = g - cost
        out.append(
            Trade(
                **base_args,
                t_in=int(ts[ie]),
                p_in=e,
                stop=s,
                target=_target(exit_key, e, r),
                t_out=int(ts[j]),
                p_out=float(px),
                reason="open" if live else why,
                R=float(net * e / r),
                gross_pct=100 * g,
                cost_pct=100 * cost,
                net_pct=100 * net,
                days=int(j - ie + 1),
                status="OPEN" if live else "CLOSED",
                mkt_up=bool(f["mkt_up"][i]),
                rs63=_rs(f, i),
            )
        )
    return out


def run_symbol_1d(
    sym: str,
    spec: StrategySpec,
    exit_key: str,
    settings: Settings,
    clock: Clock,
    *,
    warmup: int = WARMUP,
    oos_start: date = OOS_START,
    notional: float = 100_000.0,
    ctx: SymCtx | None = None,
    nifty: NiftyTable | None = None,
    slip: SlippageTable | None = None,
    end: date | None = None,
    cal: TradingCalendar | None = None,
) -> list[Trade]:
    """Trades of one daily strategy and exit on one stock's completed adjusted bars (costs in percent).

    `oos_start` is accepted for symmetry with `summarize`; the split is applied there. The keyword-only
    `ctx`, `nifty`, `slip`, `end` and `cal` let callers inject data (tests, the scan); otherwise the bars
    end at the last completed session.
    """
    if spec.timeframe != "1d" or spec.signal is None:
        raise ValueError(f"{spec.key} is not a daily per-stock strategy")
    cx = _resolve_ctx(sym, settings, clock, ctx, end, cal)
    if cx is None or len(cx.frame) < max(warmup, 1):
        return []
    prep = make_prep(settings, cx, nifty, slip)
    sig, stop = spec.signal(prep.f)
    return trades_1d(prep, spec, sig, stop, exit_key, warmup=warmup, notional=notional)


# ---- intraday trades ---------------------------------------------------------------------------


@dataclass
class IntraPrep:
    """Feature dicts of a stock's 5m bars: as is (long) and mirrored (short)."""

    sym: str
    long: dict[str, Any]
    short: dict[str, Any] | None
    settings: Settings


def intra_prep(
    sym: str, settings: Settings, start: date, end: date, *, daily: SymCtx | None
) -> IntraPrep | None:
    """5m features of a stock (None under 60 bars); index context from stored NIFTY 5m bars when present."""
    b = intraday_frame(settings, sym, start, end, "5m")
    if b is None or len(b) < 60:
        return None
    dfr = daily.frame if daily is not None else None
    nb = intraday_frame(settings, "NIFTY", start, end, "5m")
    ix = ixm = None
    if nb is not None and len(nb) >= 60:
        ix = index_context(intraday_features(nb, None, 5, None, 1))
        ixm = index_context(intraday_features(mirror(nb), None, 5, None, -1))
    f = intraday_features(b, dfr, 5, ix, 1)
    fm = intraday_features(mirror(b), dfr.mirrored() if dfr is not None else None, 5, ixm, -1)
    return IntraPrep(sym, f, fm, settings)


def _intra_reason(exit_key: str, gross_r: float) -> str:
    if gross_r <= -1 + 1e-9:
        return "stop"
    rr = {"rr1": 1.0, "rr15": 1.5, "rr2": 2.0, "rr3": 3.0}.get(exit_key)
    if rr is not None:
        return "target" if abs(gross_r - rr) < 1e-9 else "15:15 exit"
    return "15:15 exit" if exit_key == "eod" else exit_key


def trades_intra(
    ip: IntraPrep, spec: StrategySpec, exit_key: str, trend: str, *, notional: float
) -> list[Trade]:
    """lab/engine.run_symbol for one strategy, exit and trend: one trade a day, entry at the next open."""
    if exit_key not in EXITS_INTRA:
        raise ValueError(f"unknown intraday exit {exit_key}")
    if spec.signal is None:
        return []
    out: list[Trade] = []
    sides = (
        ((1, ip.long), (-1, ip.short))
        if spec.direction == "both" and ip.short is not None
        else ((1, ip.long),)
    )
    slip = SlippageTable.load_latest(ip.settings)
    rr = {"rr1": 1.0, "rr15": 1.5, "rr2": 2.0, "rr3": 3.0}.get(exit_key)
    when_out: When = "close" if exit_key == "eod" else "now"
    cost = trade_cost_frac(
        "intraday", notional, ip.settings, slip, None, 0.02, notional * 100, "open", when_out
    )
    for side, g in sides:
        n = g["n"]
        entry_ok, exit_idx = entry_window(g)
        sig, stop = spec.signal(g)
        sig = np.asarray(sig, bool) & entry_ok & trend_mask(g, trend)
        taken: set[int] = set()
        for i in np.flatnonzero(sig):
            i = int(i)
            ie, d = i + 1, int(g["day"][i])
            last_bar = ie >= n
            if d in taken or (not last_bar and g["day"][ie] != d):
                continue
            s = float(stop[i])
            if not np.isfinite(s):
                continue
            mk = bool(g["ix_up"][i])
            common = {
                "sym": ip.sym,
                "strat": spec.key,
                "exit": exit_key,
                "side": side,
                "t_sig": int(g["ts"][i]),
            }
            if last_bar:
                e = float(g["c"][i])
                r = e - s
                if r > 0:
                    out.append(
                        _intra_trade(
                            common,
                            side,
                            g["ts"][i] + g["tf"] * 60,
                            e,
                            s,
                            r,
                            rr,
                            None,
                            None,
                            "enter at next open",
                            cost,
                            "SIGNAL",
                            mk,
                        )
                    )
                continue
            e = float(g["o"][ie])
            r = e - s
            xe = int(exit_idx[d])
            if r <= 0 or xe < ie:
                continue
            taken.add(d)
            xe_c = min(xe, n - 1)
            gross_r = follow_intra(g, ie, s, e, r, xe_c)[exit_key]
            live = xe >= n - 1 and g["mins"][n - 1] + g["tf"] < 915
            net_r = gross_r - cost * abs(e) / r
            reason = "open" if live else _intra_reason(exit_key, gross_r)
            out.append(
                _intra_trade(
                    common,
                    side,
                    g["ts"][ie],
                    e,
                    s,
                    r,
                    rr,
                    int(g["ts"][xe_c]),
                    (gross_r, net_r, cost),
                    reason,
                    cost,
                    "OPEN" if live else "CLOSED",
                    mk,
                )
            )
    out.sort(key=lambda t: t.t_in)
    return out


def _intra_trade(
    common: dict[str, Any],
    side: int,
    t_in: Any,
    e: float,
    s: float,
    r: float,
    rr: float | None,
    t_out: int | None,
    res: tuple[float, float, float] | None,
    reason: str,
    cost: float,
    status: Literal["CLOSED", "OPEN", "SIGNAL"],
    mkt_up: bool,
) -> Trade:
    """Build an intraday Trade; prices are converted back from the mirrored orientation (x side)."""
    tgt = side * (e + rr * r) if rr else None
    gross_r, net_r = (res[0], res[1]) if res else (None, None)
    return Trade(
        **common,
        t_in=int(t_in),
        p_in=side * e,
        stop=side * s,
        target=tgt,
        t_out=t_out,
        p_out=None if gross_r is None else side * (e + gross_r * r),
        reason=reason,
        R=net_r,
        gross_pct=None if gross_r is None else 100 * gross_r * r / abs(e),
        cost_pct=100 * cost,
        net_pct=None if net_r is None else 100 * net_r * r / abs(e),
        days=None,
        status=status,
        mkt_up=mkt_up,
        rs63=None,
    )


def run_symbol_intra(
    sym: str,
    spec: StrategySpec,
    exit_key: str,
    trend: str,
    settings: Settings,
    clock: Clock,
    *,
    start: date,
    end: date,
    notional: float = 250_000.0,
    prep: IntraPrep | None = None,
    daily: SymCtx | None = None,
) -> list[Trade]:
    """lab/engine.run_symbol on stored 5m bars for one strategy, exit and trend (R converted to prices)."""
    if spec.timeframe not in ("5m", "60m") or spec.signal is None:
        raise ValueError(f"{spec.key} is not an intraday per-stock strategy")
    ip = prep or intra_prep(sym, settings, start, end, daily=daily or load_ctx(settings, [sym], end).get(sym))
    return [] if ip is None else trades_intra(ip, spec, exit_key, trend, notional=notional)


# ---- statistics --------------------------------------------------------------------------------


def summarize(trades: list[Trade], oos_start: date = OOS_START) -> dict[str, Any]:
    """Closed-trade stats: n, win (%), avgR, totR, pct (net %), avg_hold, pf, max_consec_losses, is, oos."""
    cl = [t for t in trades if t.status == "CLOSED" and t.R is not None]
    rs = np.array([t.R for t in cl], dtype=float)
    pct = np.array([t.net_pct for t in cl], dtype=float)
    days = [t.days for t in cl if t.days is not None]
    gain, loss = float(rs[rs > 0].sum()), float(-rs[rs < 0].sum())
    run = best = 0
    for p in pct:
        run = run + 1 if p <= 0 else 0
        best = max(best, run)
    in_s = np.array([epoch_date(t.t_in) < oos_start for t in cl], dtype=bool)

    def part(m: np.ndarray) -> dict[str, Any]:
        return {"n": int(m.sum()), "avgR": float(rs[m].mean()) if m.any() else None}

    return {
        "n": len(cl),
        "win": float(100 * (rs > 0).mean()) if len(cl) else None,
        "avgR": float(rs.mean()) if len(cl) else None,
        "totR": float(rs.sum()) if len(cl) else 0.0,
        "pct": float(pct.sum()) if len(cl) else 0.0,
        "avg_hold": float(np.mean(days)) if days else None,
        "pf": gain / loss if loss > 0 else None,
        "max_consec_losses": best,
        "is": part(in_s),
        "oos": part(~in_s),
    }


def expected_max_sharpe_null(n_tests: int, n_obs: int) -> float:
    """Expected maximum annualised Sharpe of n_tests independent null strategies seen over n_obs daily
    observations (Bailey and Lopez de Prado 2014); 0.0 for a single test."""
    if n_tests <= 1 or n_obs <= 0:
        return 0.0
    g = 0.5772156649015329
    inv = NormalDist().inv_cdf
    z = (1 - g) * inv(1 - 1 / n_tests) + g * inv(1 - 1 / (n_tests * math.e))
    return float(math.sqrt(252 / n_obs) * z)


# ---- gated verdicts ----------------------------------------------------------------------------


def gated_verdicts(settings: Settings) -> dict[str, str]:
    """Strategy key -> latest gate verdict from reports/m/strategies/*/eval.json (empty until M5.7 runs)."""
    root = Paths(settings).reports / "m" / "strategies"
    found: dict[str, tuple[float, str]] = {}
    for p in root.glob("*/eval.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("eval_unreadable", path=str(p), error=type(e).__name__)
            continue
        key, verdict = j.get("strategy") or j.get("strat"), j.get("verdict")
        if key and verdict:
            mt = p.stat().st_mtime
            if key not in found or mt >= found[key][0]:
                found[key] = (mt, str(verdict))
    return {k: v for k, (_, v) in found.items()}


# ---- cache -------------------------------------------------------------------------------------


def _cache_file(settings: Settings, sym: str, parts: tuple[Any, ...]) -> Path:
    h = hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]
    return Paths(settings).lab / "explore_cache" / sym / f"{h}.json"


def _cache_get(settings: Settings, sym: str, parts: tuple[Any, ...]) -> list[Trade] | None:
    p = _cache_file(settings, sym, parts)
    if not p.is_file():
        return None
    try:
        return [Trade(**d) for d in json.loads(p.read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError) as e:
        log.warning("explore_cache_bad", path=str(p), error=type(e).__name__)
        return None


def _cache_put(settings: Settings, sym: str, parts: tuple[Any, ...], trades: list[Trade]) -> None:
    p = _cache_file(settings, sym, parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps([asdict(t) for t in trades]), encoding="utf-8")
    tmp.replace(p)


def _parts(spec: StrategySpec, exit_key: str, trend: str, last: str, notional: float) -> tuple[Any, ...]:
    return (spec.key, exit_key, trend, last, notional, base.code_hash())


def _last_date(ctx: SymCtx) -> str:
    return str(ctx.frame.date[-1])


# ---- Market Lab shapes -------------------------------------------------------------------------


def _r(x: float | None, nd: int) -> float | None:
    return None if x is None else round(float(x), nd)


def _ml_trade_1d(t: Trade) -> dict[str, Any]:
    return {
        "sym": t.sym,
        "side": t.side,
        "t_in": t.t_in,
        "p_in": _r(t.p_in, 2),
        "stop": _r(t.stop, 2),
        "target": _r(t.target, 2),
        "t_out": t.t_out,
        "p_out": _r(t.p_out, 2),
        "R": _r(t.R, 2),
        "pct": _r(t.net_pct, 2),
        "gross_pct": _r(t.gross_pct, 2),
        "cost_pct": _r(t.cost_pct, 3),
        "days": t.days,
        "status": t.status,
        "reason": t.reason,
        "mkt_up": t.mkt_up,
        "rs63": _r(t.rs63, 1),
    }


def _ml_trade_intra(t: Trade) -> dict[str, Any]:
    risk = abs(t.p_in - t.stop)
    gross = None
    if t.R is not None and risk > 0:
        gross = t.R + t.cost_pct / 100 * abs(t.p_in) / risk
    return {
        "side": t.side,
        "t_in": t.t_in,
        "p_in": _r(t.p_in, 2),
        "stop": _r(t.stop, 2),
        "target": _r(t.target, 2),
        "t_out": t.t_out,
        "p_out": _r(t.p_out, 2),
        "R": _r(t.R, 2),
        "gross": _r(gross, 2),
        "reason": t.reason,
        "status": t.status,
        "risk_pct": _r(risk / abs(t.p_in) * 100, 2) if t.p_in else None,
    }


def _ml_summary(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": s["n"],
        "win": _r(s["win"], 1),
        "avgR": _r(s["avgR"], 3),
        "totR": _r(s["totR"], 1),
        "pct": _r(s["pct"], 1),
    }


def _spec(strat: str | None, default: str) -> StrategySpec:
    return base.by_name_or_key(strat or default)


def _check_exit(spec: StrategySpec, exit_key: str) -> None:
    if exit_key not in spec.exits:
        raise ValueError(f"unknown exit {exit_key!r} for {spec.key}")


def trades_for(
    sym: str,
    strat: str | None,
    exit_key: str | None,
    settings: Settings,
    clock: Clock,
    tf: str = "1d",
    *,
    ctx: SymCtx | None = None,
    nifty: NiftyTable | None = None,
    end: date | None = None,
    cal: TradingCalendar | None = None,
    trend: str | None = None,
) -> dict[str, Any]:
    """Market Lab /api/lab/trades shape (percent units) plus exploratory, costs_model, gated and cost_pct.

    A strategy may be given by registry key or display name; an unknown one raises KeyError and an
    exit not allowed for it raises ValueError.
    """
    base.load_library()
    if tf == "1d":
        return _trades_for_1d(sym, strat, exit_key, settings, clock, ctx, nifty, end, cal)
    if tf != "5m":
        return {"trades": [], "note": f"{tf} bars are not stored; use 1d or 5m.", "exploratory": True}
    spec = _spec(strat, DEFAULT_INTRA["strat"])
    if spec.timeframe != "5m":
        raise KeyError(strat)
    xk = exit_key or DEFAULT_INTRA["exit"]
    _check_exit(spec, xk)
    tr = trend or DEFAULT_INTRA["trend"]
    if tr not in TRENDS:
        raise ValueError(f"unknown trend {tr!r}")
    last = end or completed_end(clock, cal or TradingCalendar.from_store(settings, clock), settings)
    dctx = ctx or load_ctx(settings, [sym], last).get(sym)
    trades = run_symbol_intra(
        sym, spec, xk, tr, settings, clock, start=last - timedelta(days=INTRA_DAYS), end=last, daily=dctx
    )
    s = summarize(trades)
    return {
        "sym": sym,
        "tf": tf,
        "strat": spec.key,
        "trend": tr,
        "exit": xk,
        "candidate": False,
        "cost_pct": _r(trades[0].cost_pct, 3) if trades else None,
        "trades": [_ml_trade_intra(t) for t in trades],
        "summary": {k: _ml_summary(s)[k] for k in ("n", "win", "avgR", "totR")},
        "exploratory": True,
        "costs_model": COSTS_MODEL.format(product="intraday"),
        "gated": gated_verdicts(settings).get(spec.key),
    }


def _trades_for_1d(
    sym: str,
    strat: str | None,
    exit_key: str | None,
    settings: Settings,
    clock: Clock,
    ctx: SymCtx | None,
    nifty: NiftyTable | None,
    end: date | None,
    cal: TradingCalendar | None,
) -> dict[str, Any]:
    spec = _spec(strat, CANDIDATE["strat"])
    if spec.timeframe != "1d" or spec.signal is None:
        raise KeyError(strat)
    xk = exit_key or spec.default_exit
    _check_exit(spec, xk)
    cx = _resolve_ctx(sym, settings, clock, ctx, end, cal)
    if cx is None or len(cx.frame) < WARMUP:
        return {"trades": [], "note": "Not enough daily history for a swing read.", "exploratory": True}
    notional = 100_000.0
    parts = _parts(spec, xk, "-", _last_date(cx), notional)
    prep: Prep | None = None
    trades = _cache_get(settings, sym, parts)
    if trades is None:
        prep = make_prep(settings, cx, nifty)
        trades = trades_1d(prep, spec, *spec.signal(prep.f), xk, warmup=WARMUP, notional=notional)
        _cache_put(settings, sym, parts, trades)
    state = None
    if spec.forming is not None:
        prep = prep or make_prep(settings, cx, nifty)
        state = spec.forming(prep.f)
        if state is not None and trades and trades[-1].status in ("SIGNAL", "OPEN"):
            lt = trades[-1]
            state = {
                **state,
                "state": lt.status,
                "p_in": _r(lt.p_in, 2),
                "stop": _r(lt.stop, 2),
                "target": _r(lt.target, 2),
            }
    closed = [t for t in trades if t.status == "CLOSED"]
    if trades:
        cost_pct = trades[0].cost_pct
    else:
        prep = prep or make_prep(settings, cx, nifty)
        cost_pct = 100 * _cost(prep, len(cx.frame) - 1, notional, "delivery", "now")
    return {
        "sym": sym,
        "tf": "1d",
        "state": state,
        "days_held": [t.days for t in closed],
        "strat": spec.key,
        "exit": xk,
        "exit_name": EXIT_NAMES[xk],
        "swing": True,
        "candidate": spec.key == CANDIDATE["strat"] and xk == CANDIDATE["exit"],
        "cost_pct": _r(cost_pct, 3),
        "trades": [_ml_trade_1d(t) for t in trades],
        "summary": _ml_summary(summarize(trades)),
        "exploratory": True,
        "costs_model": COSTS_MODEL.format(product="delivery"),
        "gated": gated_verdicts(settings).get(spec.key),
    }


def daily_specs() -> list[StrategySpec]:
    """Registered per-stock daily strategies (those with a signal function), in registry order."""
    return [s for s in base.load_library().values() if s.timeframe == "1d" and s.signal is not None]


def matrix(
    sym: str,
    settings: Settings,
    clock: Clock,
    tf: str = "1d",
    *,
    ctx: SymCtx | None = None,
    nifty: NiftyTable | None = None,
    end: date | None = None,
    cal: TradingCalendar | None = None,
) -> dict[str, Any]:
    """Every registered strategy x allowed exit on one stock (daily only; other timeframes give no rows).

    Returns {sym, tf, n_tests, expected_max_sharpe_null, rows[{strat, name, tier, source, exit, n, win, avgR,
    totR, pct, is_avgR, oos_avgR, gated}]}.
    """
    base.load_library()
    verdicts = gated_verdicts(settings)
    rows: list[dict[str, Any]] = []
    n_obs = 0
    if tf == "1d":
        cx = _resolve_ctx(sym, settings, clock, ctx, end, cal)
        if cx is not None and len(cx.frame) >= WARMUP:
            n_obs = len(cx.frame) - WARMUP
            prep: Prep | None = None
            notional = 100_000.0
            for spec in daily_specs():
                sig_stop: tuple[Any, Any] | None = None
                for xk in spec.exits:
                    parts = _parts(spec, xk, "-", _last_date(cx), notional)
                    trades = _cache_get(settings, sym, parts)
                    if trades is None:
                        prep = prep or make_prep(settings, cx, nifty)
                        try:
                            sig_stop = sig_stop or spec.signal(prep.f)  # type: ignore[misc]
                        except (KeyError, ValueError, IndexError) as e:
                            log.warning("strategy_error", strat=spec.key, sym=sym, error=type(e).__name__)
                            break
                        trades = trades_1d(prep, spec, *sig_stop, xk, warmup=WARMUP, notional=notional)
                        _cache_put(settings, sym, parts, trades)
                    s = summarize(trades)
                    rows.append(
                        {
                            "strat": spec.key,
                            "name": spec.name,
                            "tier": spec.tier,
                            "source": spec.source,
                            "exit": xk,
                            "n": s["n"],
                            "win": _r(s["win"], 1),
                            "avgR": _r(s["avgR"], 3),
                            "totR": _r(s["totR"], 1),
                            "pct": _r(s["pct"], 1),
                            "is_avgR": _r(s["is"]["avgR"], 3),
                            "oos_avgR": _r(s["oos"]["avgR"], 3),
                            "gated": verdicts.get(spec.key),
                        }
                    )
    return {
        "sym": sym,
        "tf": tf,
        "n_tests": len(rows),
        "expected_max_sharpe_null": expected_max_sharpe_null(len(rows), n_obs),
        "rows": rows,
    }


__all__ = [
    "Trade",
    "SymCtx",
    "load_ctx",
    "make_ctx",
    "run_symbol_1d",
    "run_symbol_intra",
    "summarize",
    "expected_max_sharpe_null",
    "trades_for",
    "matrix",
]
