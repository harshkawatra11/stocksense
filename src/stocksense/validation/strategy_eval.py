"""Gated walk-forward runner of the strategy library (PLAN-M M5.7).

One call evaluates one pre-registered strategy (`config/strategy_library.toml`) the way the gate wants it:

1. the trial is registered first (`register_trial`, one per intraday trend filter), then evaluated;
2. each test fold (calendar years 2014-2025 for daily strategies, calendar months 2024-10..2025-09 for intraday
   ones) takes the exit rule that had the best net Sharpe on all earlier data minus the embargo (at least 30
   inner trades, else the default exit); the inner simulation sees bars up to the cut only;
3. the chosen test-fold trades are simulated once at every capital of the grid (`strategies/portfolio.py`), and once
   with doubled slippage at the reference capital;
4. the daily series, the IC and the registry link go to `validate_candidate(..., gate_version=2)`; the verdict and
   the numbers are written to `reports/m/strategies/<trial_id>/eval.json`.

Everything is a fraction in code; `eval.json` percentages are fractions too. A strategy whose `eval.json` exists is
skipped (resume). Strategies marked `await_data` or informational in the library are listed but never run.
This module never touches the lockbox: every read stops at `research_end`.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from stocksense.backtest.metrics import ic_stats, perf_stats
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.strategies import base, explore, library, portfolio
from stocksense.strategies.base import STRENGTHS, StrategySpec
from stocksense.strategies.data import NiftyTable, nifty_table
from stocksense.strategies.explore import WARMUP, SymCtx
from stocksense.strategies.portfolio import Candidate, PanelView, SimConfig, SymView
from stocksense.validation.gate import CandidateResult, save_candidate_result, validate_candidate
from stocksense.validation.splits import Fold, walk_forward_folds
from stocksense.validation.trials import finish_trial, register_trial

log = get_logger(__name__)

MIN_INNER_TRADES = 30
EMBARGO_DAILY = 5
EMBARGO_INTRA = 1
INTRA_START = date(2024, 9, 13)
IC_BLOCK = 21
XS_BASE = "s1-gbm-y_c2o_10-e0f7276025"  # the baseline GBM trial evaluate_model compares against
NO_IC = {
    "mean_ic": 0.0,
    "ic_tstat": 0.0,
}  # series-only books: no ranked signal, the IC lines fail (DECISIONS M5.7)


# ---- data ------------------------------------------------------------------------------------------------


@dataclass
class MarketData:
    """Everything the runner reads from the stored market, so tests can pass a synthetic one.

    `member[sym]` is the point-in-time u500 flag aligned to `ctxs[sym].frame.date`; `intra_members` maps a
    session to the u200 symbols of that session; `labels` has date, sid and the IC labels (y_c2o_10, y_oc).
    """

    ctxs: dict[str, SymCtx]
    sids: dict[str, int]
    member: dict[str, np.ndarray]
    nifty: NiftyTable
    nifty_bars: pl.DataFrame
    benchmark: pl.DataFrame
    labels: pl.DataFrame
    slip: SlippageTable
    intra_members: dict[date, frozenset[str]] = field(default_factory=dict)


def load_market(settings: Settings, end: date, *, intraday: bool = False) -> MarketData:
    """The research-period market: daily bars of every stock ever in the u500 (or u200 for intraday), flags,
    NIFTY 50, the equal-weight benchmark and the labels."""
    flag = "u200" if intraday else "u500"
    uni = (
        io.read(settings, "panel_universe", start=None, end=end, columns=["date", "sid", "u500", "u200"])
        .collect()
        .sort("sid", "date")
    )
    ever = uni.filter(pl.col(flag))["sid"].unique().to_list()
    px = (
        io.read(settings, "panel_daily", start=None, end=end, columns=["sid", "symbol", "series", "date"])
        .filter(pl.col("sid").is_in(ever) & (pl.col("series") == "EQ"))
        .sort("date")
        .group_by("sid")
        .agg(pl.col("symbol").last())
        .collect()
    )
    sids = {r["symbol"]: int(r["sid"]) for r in px.iter_rows(named=True)}
    ctxs = explore.load_ctx(settings, sorted(sids), end)
    by_sid = {
        k[0]: g for k, g in uni.filter(pl.col("sid").is_in(ever)).partition_by("sid", as_dict=True).items()
    }
    member: dict[str, np.ndarray] = {}
    for sym, ctx in ctxs.items():
        g = by_sid.get(sids[sym])
        if g is None:
            continue
        ud = np.array(g["date"].to_list(), dtype="datetime64[D]")
        pos = np.searchsorted(ud, ctx.frame.date, side="right") - 1
        flags = g[flag].to_numpy().astype(bool)
        ok = (pos >= 0) & (ud[np.maximum(pos, 0)] == ctx.frame.date)  # the flag of that very session
        member[sym] = ok & flags[np.maximum(pos, 0)]
    intra: dict[date, frozenset[str]] = {}
    if intraday:
        sym_of = {v: k for k, v in sids.items()}
        u = uni.filter(pl.col("u200") & (pl.col("date") >= INTRA_START))
        for d, g in u.group_by("date"):
            intra[d[0]] = frozenset(sym_of[s] for s in g["sid"].to_list() if s in sym_of)
    ixb = (
        io.read(settings, "index_daily", start=None, end=end, columns=["index_name", "date", "open", "close"])
        .filter(pl.col("index_name") == "NIFTY 50")
        .select("date", "open", "close")
        .sort("date")
        .unique("date", keep="last", maintain_order=True)
        .collect()
    )
    bench = io.read(settings, "panel_market", start=None, end=end, columns=["date", "ew500_ret"]).collect()
    from stocksense.labels.forward import load_labels

    labels = load_labels(settings, "research", columns=["y_c2o_10", "y_oc"]).collect()
    return MarketData(
        ctxs, sids, member, nifty_table(settings, end=end), ixb, bench, labels,
        SlippageTable.load_latest(settings), intra,
    )  # fmt: skip


# ---- helpers ---------------------------------------------------------------------------------------------


def eval_dir(settings: Settings, trial_id: str) -> Path:
    """`reports/m/strategies/<trial_id>`."""
    return Paths(settings).reports / "m" / "strategies" / trial_id


def month_folds(
    calendar: TradingCalendar, first: date = date(2024, 10, 1), last: date = date(2025, 9, 30)
) -> list[Fold]:
    """Calendar-month test folds 2024-10..2025-09 (the intraday protocol); train_end is the session before."""
    out: list[Fold] = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        a = date(y, m, 1)
        b = (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)
        sess = calendar.sessions(a, min(b, last))
        if sess:
            out.append(Fold(f"{y}-{m:02d}", calendar.prev_session(sess[0]), sess[0], sess[-1]))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _sharpe(net: np.ndarray) -> float:
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    return float(net.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0


def block_ic(position: np.ndarray, next_ret: np.ndarray, block: int = IC_BLOCK) -> dict[str, float]:
    """Mean Spearman correlation of `position` with the next day's return over blocks of `block` days, and its
    t statistic over the blocks (blocks where either series is constant are skipped)."""
    ics = []
    for a in range(0, len(position) - block + 1, block):
        p, r = position[a : a + block], next_ret[a : a + block]
        ok = np.isfinite(p) & np.isfinite(r)
        if ok.sum() < 5 or np.ptp(p[ok]) == 0 or np.ptp(r[ok]) == 0:
            continue
        ics.append(float(np.corrcoef(rankdata(p[ok]), rankdata(r[ok]))[0, 1]))
    if len(ics) < 2:
        return {"mean_ic": 0.0, "ic_tstat": 0.0, "n_blocks": len(ics)}
    v = np.array(ics)
    sd = float(v.std(ddof=1))
    return {
        "mean_ic": float(v.mean()),
        "ic_tstat": float(v.mean() / sd * math.sqrt(len(v))) if sd > 0 else 0.0,
        "n_blocks": len(v),
    }


def _years(daily: pl.DataFrame) -> dict[str, float]:
    g = (
        daily.with_columns(pl.col("date").dt.year().alias("y"))
        .group_by("y")
        .agg(((1 + pl.col("net")).product() - 1).alias("net"))
    )
    return {str(y): float(v) for y, v in sorted(zip(g["y"].to_list(), g["net"].to_list(), strict=True))}


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return _clean(x.item())
    if isinstance(x, date):
        return x.isoformat()
    return x


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8", newline="\n")
    tmp.replace(path)


@dataclass
class RunOut:
    """One trial's simulated result: the reference-capital daily series (date, net, net_2x, gross), the net series
    at every capital, the IC statistics, the exit chosen per fold and one record per trade."""

    daily: pl.DataFrame
    by_capital: dict[int, pl.DataFrame]
    ic: dict[str, float]
    chosen: dict[str, str] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


# ---- exit selection and the per-stock books --------------------------------------------------------------


def _by_date(cands: list[Candidate]) -> dict[date, list[Candidate]]:
    out: dict[date, list[Candidate]] = {}
    for c in cands:
        out.setdefault(c.t_sig, []).append(c)
    return out


def select_exit(
    exits: tuple[str, ...],
    default: str,
    fold: Fold,
    calendar: TradingCalendar,
    embargo: int,
    sim: Callable[[str, date], tuple[pl.DataFrame, int]],
) -> str:
    """The exit with the best net Sharpe on everything before the fold (minus the embargo).

    `sim(exit_key, inner_end)` runs the book on [start of data, inner_end] and returns (daily, n_trades);
    `inner_end` is `embargo` sessions before the session that precedes the fold, so no test bar is read.
    """
    inner_end = calendar.shift(fold.train_end, -embargo)
    best: tuple[float, str] | None = None
    for ex in exits:
        daily, n = sim(ex, inner_end)
        if n < MIN_INNER_TRADES:
            continue
        sh = _sharpe(daily["net"].to_numpy())
        if best is None or sh > best[0]:
            best = (sh, ex)
    return best[1] if best else default


def _gen_swing(
    spec: StrategySpec, market: MarketData, settings: Settings
) -> tuple[list[Candidate], PanelView, pl.DataFrame]:
    """Every candidate signal (exit-agnostic), the views the simulator needs and the strength frame for the IC."""
    cands: list[Candidate] = []
    syms: dict[str, SymView] = {}
    parts: list[pl.DataFrame] = []
    strength_fn = STRENGTHS[spec.strength]
    for sym, ctx in market.ctxs.items():
        if len(ctx.frame) < WARMUP or sym not in market.member or spec.signal is None:
            continue
        prep = explore.make_prep(settings, ctx, market.nifty, market.slip)
        f = prep.f
        try:
            sig, stop = spec.signal(f)
        except (KeyError, ValueError, IndexError) as e:
            log.warning("strategy_error", strat=spec.key, sym=sym, error=type(e).__name__)
            continue
        sig = np.asarray(sig, bool).copy()
        sig[: max(WARMUP - 40, 0)] = False
        sig &= market.member[sym]
        stop = np.asarray(stop, dtype=float)
        mem = market.member[sym]
        st = np.where(mem, np.nan_to_num(strength_fn(f, sig)), np.nan).astype(np.float32)
        parts.append(
            pl.DataFrame({"date": ctx.frame.date, "sid": market.sids[sym], "score": st}).filter(
                pl.col("score").is_not_nan()
            )
        )
        take = sig & np.isfinite(stop)
        take[-1] = False  # a signal on the last bar has no entry bar
        idx = np.flatnonzero(take)
        if not len(idx):
            continue
        rs = np.nan_to_num(np.asarray(f["rs63"], dtype=float), nan=0.0)
        days = ctx.frame.date[idx].astype("datetime64[D]").tolist()
        cands += [
            Candidate(sym, market.sids[sym], d, spec.key, "", float(rs[i]), float(stop[i]))
            for i, d in zip(idx.tolist(), days, strict=True)
        ]
        syms[sym] = SymView(ctx, portfolio.slim_features(f, spec.key))
    strength = (
        pl.concat(parts)
        if parts
        else pl.DataFrame(schema={"date": pl.Date, "sid": pl.Int64, "score": pl.Float32})
    )
    return cands, PanelView(syms, market.slip), strength


def _frame_ic(frame: pl.DataFrame, label: str) -> dict[str, float]:
    """Mean daily Spearman IC of `score` with `label` over the dates on which the score is not constant.

    A strategy that fires on a few names a day has no ranking on the other days (the correlation is undefined),
    so those dates are left out rather than counted as zero or as NaN (DECISIONS M5.7). No dates: IC 0.
    """
    f = frame.filter(pl.col("score").n_unique().over("date") > 1)
    if f.is_empty():
        return {"mean_ic": 0.0, "ic_tstat": 0.0, "n_dates": 0}
    s = ic_stats(f, label)
    mean, t = float(s["mean_ic"]), float(s["ic_tstat"])
    if not math.isfinite(mean):
        return {"mean_ic": 0.0, "ic_tstat": 0.0, "n_dates": 0}
    return {"mean_ic": mean, "ic_tstat": t if math.isfinite(t) else 0.0, "n_dates": int(s["n_dates"])}


def _ic_from_strength(
    strength: pl.DataFrame, labels: pl.DataFrame, label: str, folds: list[Fold]
) -> dict[str, float]:
    keep = pl.lit(False)
    for f in folds:
        keep = keep | pl.col("date").is_between(f.test_start, f.test_end)
    frame = strength.filter(keep).join(labels.select("date", "sid", label), on=["date", "sid"], how="inner")
    return _frame_ic(frame, label)


def _compose(
    daily_by_cap: dict[int, pl.DataFrame], daily_2x: pl.DataFrame, ref: int
) -> tuple[pl.DataFrame, dict[int, pl.DataFrame]]:
    d = daily_by_cap[ref].select("date", "net", "gross")
    d = d.join(daily_2x.select("date", pl.col("net").alias("net_2x")), on="date", how="left")
    return d.with_columns(pl.col("net_2x").fill_null(pl.col("net"))), {
        c: v.select("date", "net") for c, v in daily_by_cap.items()
    }


def _run_swing(
    spec: StrategySpec,
    section: dict[str, Any],
    market: MarketData,
    settings: Settings,
    calendar: TradingCalendar,
    folds: list[Fold],
    proto: dict[str, Any],
) -> RunOut:
    cands, panel, strength = _gen_swing(spec, market, settings)
    by_sig = _by_date(cands)
    exits = tuple(section["exits"])
    ref, grid, slots = (
        int(proto["reference_capital"]),
        [int(c) for c in proto["capital_grid"]],
        int(proto["slots"]),
    )
    first = calendar.next_session(min(by_sig)) if by_sig else folds[0].test_start
    cfg_ref = SimConfig(float(ref), slots)

    chosen: dict[str, str] = {}
    final: dict[date, list[Candidate]] = {}
    for fold in folds:

        def inner(ex: str, inner_end: date, _first: date = first) -> tuple[pl.DataFrame, int]:
            if inner_end <= _first:
                return pl.DataFrame(schema=portfolio.SIM_COLUMNS_SCHEMA), 0
            r = portfolio.run_daily(
                by_sig, panel, cfg_ref, settings, calendar, _first, inner_end, exit_key=ex
            )
            return r.daily, len(r.trades)

        ex = select_exit(exits, section["default_exit"], fold, calendar, EMBARGO_DAILY, inner)
        chosen[fold.name] = ex
        for d in calendar.sessions(
            calendar.prev_session(fold.test_start), calendar.prev_session(fold.test_end)
        ):
            final[d] = [replace(c, exit_key=ex) for c in by_sig.get(d, ())]
    start, end = folds[0].test_start, folds[-1].test_end
    res: dict[int, portfolio.SimResult] = {}
    for cap in grid:
        res[cap] = portfolio.run_daily(
            final, panel, SimConfig(float(cap), slots), settings, calendar, start, end
        )
    r2 = portfolio.run_daily(
        final, panel, SimConfig(float(ref), slots, slip_scale=2.0), settings, calendar, start, end
    )
    daily, by_cap = _compose({c: r.daily for c, r in res.items()}, r2.daily, ref)
    ic = _ic_from_strength(strength, market.labels, "y_c2o_10", folds)
    return RunOut(daily, by_cap, ic, chosen, res[ref].trades)


# ---- intraday --------------------------------------------------------------------------------------------


def _run_intra(
    spec: StrategySpec,
    section: dict[str, Any],
    trends: list[str],
    market: MarketData,
    settings: Settings,
    calendar: TradingCalendar,
    folds: list[Fold],
    proto: dict[str, Any],
    research_end: date,
) -> dict[str, RunOut]:
    exits = tuple(section["exits"])
    ref, grid, slots = (
        int(proto["reference_capital"]),
        [int(c) for c in proto["capital_grid"]],
        int(proto["slots"]),
    )
    trades: dict[tuple[str, str], list[explore.Trade]] = {(t, e): [] for t in trends for e in exits}
    syms = sorted({s for v in market.intra_members.values() for s in v})
    for sym in syms:
        ip = explore.intra_prep(sym, settings, INTRA_START, research_end, daily=market.ctxs.get(sym))
        if ip is None:
            continue
        for t in trends:
            for e in exits:
                for tr in portfolio.intra_trades(ip, spec, e, t, notional=float(ref) / slots):
                    d = explore.epoch_date(tr.t_in)
                    if sym in market.intra_members.get(d, ()):
                        trades[(t, e)].append(tr)
    out: dict[str, RunOut] = {}
    start, end = folds[0].test_start, folds[-1].test_end
    for t in trends:
        chosen: dict[str, str] = {}
        picked: list[explore.Trade] = []
        for fold in folds:

            def inner(ex: str, inner_end: date, _t: str = t) -> tuple[pl.DataFrame, int]:
                if inner_end <= INTRA_START:
                    return pl.DataFrame(schema=portfolio.SIM_COLUMNS_SCHEMA), 0
                r = portfolio.run_intraday(
                    trades[(_t, ex)],
                    SimConfig(float(ref), slots, "intraday"),
                    settings,
                    calendar,
                    INTRA_START,
                    inner_end,
                )
                return r.daily, len(r.trades)

            ex = select_exit(exits, section["default_exit"], fold, calendar, EMBARGO_INTRA, inner)
            chosen[fold.name] = ex
            picked += [
                x for x in trades[(t, ex)] if fold.test_start <= explore.epoch_date(x.t_in) <= fold.test_end
            ]
        res = {
            cap: portfolio.run_intraday(
                picked, SimConfig(float(cap), slots, "intraday"), settings, calendar, start, end
            )
            for cap in grid
        }
        r2 = portfolio.run_intraday(
            picked, SimConfig(float(ref), slots, "intraday", slip_scale=2.0), settings, calendar, start, end
        )
        daily, by_cap = _compose({c: r.daily for c, r in res.items()}, r2.daily, ref)
        flags = pl.DataFrame(
            {
                "date": [explore.epoch_date(x.t_in) for x in picked if x.sym in market.sids],
                "sid": [market.sids[x.sym] for x in picked if x.sym in market.sids],
                "score": 1.0,
            },
            schema={"date": pl.Date, "sid": pl.Int64, "score": pl.Float32},
        ).unique(["date", "sid"])
        out[t] = RunOut(daily, by_cap, _ic_from_flags(flags, market, folds), chosen, res[ref].trades)
    return out


def _ic_from_flags(flags: pl.DataFrame, market: MarketData, folds: list[Fold]) -> dict[str, float]:
    """IC of "the stock traded that day" (1) against y_oc over the u200 member-days of the test folds."""
    rows = [
        (d, market.sids[s])
        for d, syms in market.intra_members.items()
        if any(f.test_start <= d <= f.test_end for f in folds)
        for s in syms
        if s in market.sids
    ]
    base_frame = pl.DataFrame(rows, schema={"date": pl.Date, "sid": pl.Int64}, orient="row")
    frame = base_frame.join(flags, on=["date", "sid"], how="left").with_columns(
        pl.col("score").fill_null(0.0)
    )
    frame = frame.join(market.labels.select("date", "sid", "y_oc"), on=["date", "sid"], how="inner")
    return _frame_ic(frame, "y_oc")


# ---- series books ----------------------------------------------------------------------------------------


def _nifty_next_return(market: MarketData) -> pl.DataFrame:
    b = market.nifty_bars.sort("date")
    return b.select("date", (pl.col("close").shift(-1) / pl.col("close") - 1.0).alias("next_ret"))


def _position_ic(position: pl.DataFrame, market: MarketData) -> dict[str, float]:
    j = position.join(_nifty_next_return(market), on="date", how="inner").drop_nulls().sort("date")
    if j.height < 2 * IC_BLOCK:
        return dict(NO_IC)
    return block_ic(j["position"].to_numpy().astype(float), j["next_ret"].to_numpy().astype(float))


def _window(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end)).sort("date")


def _book_out(
    by_cap: dict[int, pl.DataFrame], ref: int, ic: dict[str, float], two_x: pl.DataFrame | None = None
) -> RunOut:
    """Series books carry `net`, `net_2x` and `gross` already; `two_x` overrides the doubled-cost series."""
    d = by_cap[ref]
    n2 = d.select("date", pl.col("net").alias("net_2x")) if two_x is None else two_x
    daily = d.select("date", "net", "gross").join(n2.select("date", "net_2x"), on="date", how="left")
    daily = daily.with_columns(pl.col("net_2x").fill_null(pl.col("net")))
    return RunOut(daily, {c: v.select("date", "net") for c, v in by_cap.items()}, ic)


def _run_book_index(spec, section, market, settings, calendar, folds, proto) -> RunOut:
    from stocksense.strategies.k151_idx import run_book

    start, end = folds[0].test_start, folds[-1].test_end
    book = _window(run_book(settings, spec.key, start, end), start, end)
    ref, grid = int(proto["reference_capital"]), [int(c) for c in proto["capital_grid"]]
    w = spec.timing(settings, start, end)  # type: ignore[misc]
    pos = w.group_by("date").agg(pl.col("weight").sum().alias("position")).sort("date") if w.height else None
    ic = _position_ic(pos, market) if pos is not None else dict(NO_IC)
    by_cap = {
        c: book for c in grid
    }  # the cost of an index rule is set at a 100,000 INR notional, not by capital
    out = _book_out(by_cap, ref, ic, book.select("date", pl.col("net_2x").alias("net_2x")))
    out.notes["capital_note"] = (
        "index-rule costs do not depend on capital (k151_idx.one_way_cost, 100,000 INR)"
    )
    return out


def _run_book_capital(spec, section, market, settings, calendar, folds, proto) -> RunOut:
    start, end = folds[0].test_start, folds[-1].test_end
    ref, grid = int(proto["reference_capital"]), [int(c) for c in proto["capital_grid"]]
    by_cap = {
        c: _window(spec.timing(settings, start, end, capital_inr=float(c)), start, end)  # type: ignore[misc,call-arg]
        for c in grid
    }
    return _book_out(by_cap, ref, dict(NO_IC), by_cap[ref].select("date", "net_2x"))


def _run_position(spec, section, market, settings, calendar, folds, proto) -> RunOut:
    start, end = folds[0].test_start, folds[-1].test_end
    ref, grid = int(proto["reference_capital"]), [int(c) for c in proto["capital_grid"]]
    pos = spec.timing(settings, start, end)  # type: ignore[misc]
    bars = market.nifty_bars.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    sims = {c: portfolio.simulate_timing(pos, bars, SimConfig(float(c), 1), settings) for c in grid}
    two = portfolio.simulate_timing(pos, bars, SimConfig(float(ref), 1, slip_scale=2.0), settings)
    daily, by_cap = _compose({c: s for c, s in sims.items()}, two, ref)
    return RunOut(daily, by_cap, _position_ic(pos, market))


def _run_options(spec, section, market, settings, calendar, folds, proto) -> RunOut:
    start, end = folds[0].test_start, folds[-1].test_end
    ref, grid = int(proto["reference_capital"]), [int(c) for c in proto["capital_grid"]]
    fn = spec.timing
    with_cost = getattr(fn, "with_cost", None)
    raw = (with_cost or fn)(settings, start, end)  # type: ignore[misc]
    raw = _window(raw, start, end)
    cost = raw["cost"] if "cost" in raw.columns else pl.Series([0.0] * raw.height)
    # m-opt: a return on the underlying notional (capital-free); a day with P&L counts as invested (gross 1)
    d = pl.DataFrame(
        {
            "date": raw["date"],
            "net": raw["ret"],
            "net_2x": raw["ret"] - cost,
            "gross": (raw["ret"] != 0).cast(pl.Float64),
        }
    )
    out = _book_out({c: d for c in grid}, ref, dict(NO_IC), d.select("date", "net_2x"))
    out.notes["capital_note"] = "option P&L is a return on underlying notional; it does not depend on capital"
    return out


# ---- cross-sectional factors -----------------------------------------------------------------------------


def _xs_backend(
    spec: StrategySpec,
    trial_id: str,
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    start: date,
    end: date,
    clock: Clock,
) -> tuple[pl.DataFrame, dict[str, float], dict[str, Any]]:
    """Scores -> preds/<trial>/ -> `evaluate_model` -> the S1 backtest daily series of that trial.

    Returns (daily with date, net, net_2x, gross, benchmark, net_<capital> columns, the IC statistics,
    extra info). The S1 backtest is registered by `evaluate_model` as a `g2-s1` trial of its own, which also
    counts in the DSR's number of trials (recorded in DECISIONS)."""
    from stocksense.validation import g1, g2_eval

    sc = spec.score(settings, start, end)  # type: ignore[misc]
    if not sc.height:
        raise ValueError(f"{spec.key} produced no scores")
    df = sc.select(
        "date",
        pl.col("sid").cast(pl.Int32),
        pl.col("score").cast(pl.Float32),
        pl.lit(clock.now()).alias("made_at"),
        pl.lit("research").alias("mode"),
    ).drop_nulls("score")
    io.upsert(settings, "preds", df, sub=trial_id)
    ev = g2_eval.evaluate_model(trial_id, settings, state, calendar, base_id=XS_BASE)
    s1 = (ev.get("net_sharpe_s1") or {}).get("s1_trial")
    if s1 is None:
        raise ValueError(f"no S1 backtest for {trial_id}")
    res = g1.load_result(settings, s1)
    if res is None:
        raise FileNotFoundError(f"S1 trial {s1} has no result")
    ic = {"mean_ic": float(ev["mean_ic"]), "ic_tstat": float(ev["ic_tstat"])}
    return res.daily(), ic, {"s1_trial": s1, "admitted": ev.get("admitted")}


def _run_xs(spec, section, trial_id, market, settings, state, calendar, folds, proto, clock) -> RunOut:
    from stocksense.strategies.k151 import VOL_TARGET_BOOKS, voltgt_scale

    start, end = folds[0].test_start, folds[-1].test_end
    daily, ic, extra = _xs_backend(spec, trial_id, settings, state, calendar, folds[0].test_start, end, clock)
    ref = int(proto["reference_capital"])
    daily = _window(daily, start, end)
    if spec.key in VOL_TARGET_BOOKS:  # eq. 519: scale the long book's return by min(target / sigma_hat, 1)
        k = np.nan_to_num(voltgt_scale(daily["net"].to_numpy(), VOL_TARGET_BOOKS[spec.key]), nan=1.0)
        daily = daily.with_columns(
            (pl.col("net") * k).alias("net"),
            (pl.col("net_2x") * k).alias("net_2x"),
            (pl.col("gross") * k).alias("gross"),
        )
        extra["vol_target"] = VOL_TARGET_BOOKS[spec.key]
    caps = [c for c in daily.columns if c.startswith("net_") and c[4:].isdigit()]
    by_cap = {int(c[4:]): daily.select("date", pl.col(c).alias("net")) for c in caps} or {
        ref: daily.select("date", "net")
    }
    d = daily.select("date", "net", "net_2x", "gross")
    return RunOut(
        d, by_cap, ic, notes={"capital_grid": "the G1 grid of settings.costs.grid (S1 backtest)", **extra}
    )


# ---- finishing -------------------------------------------------------------------------------------------


def _benchmark(kind: str, market: MarketData, dates: pl.Series) -> pl.DataFrame:
    if kind == "cash":
        return pl.DataFrame(
            {"date": dates, "benchmark": [0.0] * len(dates)},
            schema={"date": pl.Date, "benchmark": pl.Float64},
        )
    if kind == "nifty":
        b = market.nifty_bars.sort("date").select(
            "date", (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("benchmark")
        )
    else:
        b = market.benchmark.select("date", pl.col("ew500_ret").alias("benchmark"))
    return b


def _finalize(
    key: str,
    spec: StrategySpec,
    section: dict[str, Any],
    trend: str | None,
    trial_id: str,
    out: RunOut,
    market: MarketData,
    settings: Settings,
    state: State,
    clock: Clock,
    proto: dict[str, Any],
) -> dict[str, Any]:
    family, target = section["family"], section["target"]
    daily = out.daily.sort("date")
    daily = daily.join(_benchmark(section["benchmark"], market, daily["date"]), on="date", how="left")
    daily = daily.with_columns(pl.col("benchmark").fill_null(0.0))
    net = daily["net"].to_numpy().astype(float)
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    rpath = eval_dir(settings, trial_id) / "daily.parquet"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    daily.select("date", pl.col("net").alias("ret"), "net", "net_2x", "benchmark", "gross").write_parquet(
        rpath
    )
    con = state.connect()
    try:
        st = con.execute("SELECT status FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
    finally:
        con.close()
    if st is None or st[0] != "done":
        finish_trial(
            state, trial_id,
            {"sharpe_daily": float(net.mean() / sd) if sd > 0 else 0.0, "n_days": len(net), "strategy": key},
            rpath,
        )  # fmt: skip
    cand = CandidateResult(
        trial_id, family, target, daily.select("date", "net", "net_2x", "benchmark", "gross"), ic=out.ic,
        trial_ids=(trial_id,),
    )  # fmt: skip
    save_candidate_result(cand, settings)
    report = validate_candidate(trial_id, settings, state, gate_version=2, clock=clock)
    ref = int(proto["reference_capital"])
    tr = out.trades
    pnl = np.array([t["pnl"] for t in tr], dtype=float)
    rs = np.array([t["r_mult"] for t in tr], dtype=float)
    stats = perf_stats(net, daily["benchmark"].to_numpy(), dates=daily["date"].to_list())
    payload: dict[str, Any] = {
        "strategy": key, "strat": key, "trial_id": trial_id, "trend": trend, "family": family, "target": target,
        "eval": section["eval"], "reference_capital": ref, "n_days": len(net),
        "first_date": daily["date"].min(), "last_date": daily["date"].max(),
        "trades": len(tr), "win": float((pnl > 0).mean()) if len(pnl) else None,
        "avg_R": float(rs.mean()) if len(rs) else None,
        "net_sharpe": stats["sharpe"], "net_sharpe_2x": perf_stats(daily["net_2x"].to_numpy().astype(float), None)["sharpe"],
        "net_sharpe_by_capital": {
            str(c): _sharpe(v["net"].to_numpy().astype(float)) for c, v in sorted(out.by_capital.items())
        },
        "max_drawdown": stats["max_drawdown"], "cagr": stats["cagr"],
        "per_year": _years(daily), "chosen_exits": out.chosen, "ic": out.ic, "notes": out.notes,
        "gate": report.to_dict(), "lines_passed": sum(1 for ln in report.lines if ln.passed),
        "verdict": "PASS" if report.passed else "FAIL",
        "written_at": clock.now().isoformat(timespec="seconds"),
    }  # fmt: skip
    _write_json(eval_dir(settings, trial_id) / "eval.json", payload)
    return payload


# ---- the entry points ------------------------------------------------------------------------------------


def trial_spec(key: str, section: dict[str, Any], trend: str | None) -> dict[str, Any]:
    """The spec registered for a trial: the strategy key, its library section, the code hash and the trend."""
    s = {"key": key, **section, "code_hash": base.code_hash()}
    if trend is not None:
        s["trend"] = trend
    return s


def run_strategy(
    key: str,
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    clock: Clock,
    *,
    max_hours: float = 6.0,
    market: MarketData | None = None,
    folds: list[Fold] | None = None,
    spec: StrategySpec | None = None,
    section: dict[str, Any] | None = None,
    proto: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one pre-registered strategy; returns {"key", "trials": [{trial_id, trend, verdict | skipped}]}.

    `max_hours` is checked by `run_family` between strategies. `market`, `folds`, `spec`, `section` and `proto`
    replace the stored market, the protocol folds and the library lookups (tests only).
    """
    del max_hours
    spec = spec or base.by_name_or_key(key)
    sec = section if section is not None else library.library_section(key)
    pr = proto if proto is not None else library.protocol()
    kind = sec["eval"]
    if sec["status"] != "ready" or kind == library.KIND_INFO:
        reason = "informational" if kind == library.KIND_INFO else sec["status"]
        log.info("strategy_skipped", key=key, reason=reason)
        return {"key": key, "trials": [], "skipped": reason}
    trends: list[str | None] = list(sec.get("trends") or [None])
    todo: list[tuple[str | None, str]] = []
    results: list[dict[str, Any]] = []
    for tr in trends:  # register every trial before anything is evaluated
        tid = register_trial(state, sec["family"], sec["target"], trial_spec(key, sec, tr))
        if (eval_dir(settings, tid) / "eval.json").is_file():
            results.append({"trial_id": tid, "trend": tr, "skipped": "done"})
        else:
            todo.append((tr, tid))
    if not todo:
        return {"key": key, "trials": results}
    cal = calendar
    rend = library.research_end() if proto is None else pr["research_end"]
    fl = folds
    if fl is None:
        fl = month_folds(cal) if kind == library.KIND_INTRA else walk_forward_folds(settings, cal)
    mk = market or load_market(settings, rend, intraday=kind == library.KIND_INTRA)
    outs: dict[str | None, RunOut] = {}
    if kind == library.KIND_SWING:
        outs[None] = _run_swing(spec, sec, mk, settings, cal, fl, pr)
    elif kind == library.KIND_INTRA:
        got = _run_intra(spec, sec, [t for t, _ in todo if t], mk, settings, cal, fl, pr, rend)
        outs.update(got)
    elif kind == library.KIND_XS:
        outs[None] = _run_xs(spec, sec, todo[0][1], mk, settings, state, cal, fl, pr, clock)
    elif kind == library.KIND_INDEX:
        outs[None] = _run_book_index(spec, sec, mk, settings, cal, fl, pr)
    elif kind == library.KIND_CAPITAL:
        outs[None] = _run_book_capital(spec, sec, mk, settings, cal, fl, pr)
    elif kind == library.KIND_POSITION:
        outs[None] = _run_position(spec, sec, mk, settings, cal, fl, pr)
    elif kind == library.KIND_OPTIONS:
        outs[None] = _run_options(spec, sec, mk, settings, cal, fl, pr)
    else:
        raise ValueError(f"unknown evaluation kind {kind!r} for {key}")
    for tr, tid in todo:
        ev = _finalize(key, spec, sec, tr, tid, outs[tr], mk, settings, state, clock, pr)
        results.append(
            {"trial_id": tid, "trend": tr, "verdict": ev["verdict"], "lines_passed": ev["lines_passed"]}
        )
    return {"key": key, "trials": results}


def run_family(
    family: str,
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    clock: Clock,
    *,
    keys: list[str] | None = None,
    max_hours: float = 6.0,
) -> list[dict[str, Any]]:
    """Run every ready strategy of a family (or the given keys) one after the other.

    Stops after the current strategy once `max_hours` of the clock have passed (the rest resume next time,
    since a finished strategy has its eval.json). `await_data` and informational entries are listed and skipped.
    """
    doc = library.load_toml()["strategy"]
    names = keys if keys is not None else sorted(k for k, s in doc.items() if s["family"] == family)
    bad = [k for k in names if k not in doc or doc[k]["family"] != family]
    if bad:
        raise KeyError(f"not in family {family}: {bad}")
    t0 = clock.now()
    out: list[dict[str, Any]] = []
    for k in names:
        if (clock.now() - t0).total_seconds() > max_hours * 3600:
            log.warning("strat_run_wall_cap", done=len(out), left=len(names) - len(out))
            break
        out.append(run_strategy(k, settings, state, calendar, clock))
    return out


def load_evals(settings: Settings) -> list[dict[str, Any]]:
    """Every `eval.json` under reports/m/strategies, sorted by trial id."""
    root = Paths(settings).reports / "m" / "strategies"
    return (
        [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/eval.json"))]
        if root.is_dir()
        else []
    )


def table(rows: list[dict[str, Any]]) -> str:
    """One line per evaluated trial: verdict, gate lines passed, net Sharpe, 2x Sharpe, drawdown, trades."""
    head = f"{'trial':<56} {'verdict':<7} {'lines':>5} {'sharpe':>7} {'2x':>6} {'maxdd':>6} {'trades':>7} {'IC':>7}"
    lines = [head, "-" * len(head)]

    def f(x: Any, nd: int = 2) -> str:
        return "-" if x is None else f"{x:.{nd}f}"

    for r in sorted(rows, key=lambda r: (r.get("family", ""), r["trial_id"])):
        lines.append(
            f"{r['trial_id']:<56} {r['verdict']:<7} {r['lines_passed']:>3}/9 {f(r['net_sharpe']):>7} "
            f"{f(r['net_sharpe_2x']):>6} {f(r['max_drawdown']):>6} {r['trades']:>7} {f((r.get('ic') or {}).get('mean_ic'), 3):>7}"
        )
    return "\n".join(lines)
