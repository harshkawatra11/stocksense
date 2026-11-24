"""Generic option-combo backtester for the *151 Trading Strategies* section 2 library (PLAN-M M5.4c).

A combo is a tuple of `Leg`s (CE / PE / FUT, side, strike offset in steps of kappa, quantity, expiry month).
Each monthly cycle is entered at the `settle_price` of the first session after the previous monthly expiry
and held to the expiry, where option legs settle at intrinsic value against the expiry-day underlying.
Daily P&L is marked to each leg's settle; the P&L is a fraction of the entry spot (return on notional).
Expiry-day option rows are never used for an entry price (the entry date is always before the expiry).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.features.derivs import MIN_DAYS, R_FREE, implied_vol
from stocksense.strategies.costs import (
    DerivCosts,
    LegFill,
    futures_cost_rs,
    load_deriv_costs,
    options_cost_frac,
)

log = get_logger(__name__)

KAPPA = 0.02  # one OTM/ITM step: the listed strike nearest spot * (1 +- 0.02)


@dataclass(frozen=True)
class Leg:
    """One combo leg: `m` steps of kappa from ATM (or `x` as an explicit fraction of spot, overriding m)."""

    kind: Literal["CE", "PE", "FUT"]
    side: int
    m: float = 0.0
    qty: float = 1.0
    expiry: Literal["near", "next"] = "near"
    x: float | None = None


@dataclass(frozen=True)
class EntryRule:
    """Cycle entry: monthly expiries; `condition(entry_date)` False means stay flat this cycle."""

    kappa: float = KAPPA
    condition: Callable[[date], bool] | None = None
    delta_hedge: bool = False


@dataclass(frozen=True)
class ComboResult:
    """`cycles`: one row per traded cycle; `daily`: net daily return fraction of each cycle's entry spot."""

    cycles: pl.DataFrame
    daily: pl.DataFrame


def intrinsic(kind: str, strike: float, s_t: float) -> float:
    """Value at expiry of one long unit: (S-K)+ for CE, (K-S)+ for PE, S for FUT (strike unused)."""
    if kind == "CE":
        return max(s_t - strike, 0.0)
    if kind == "PE":
        return max(strike - s_t, 0.0)
    return s_t


def nearest(listed: np.ndarray, target: float) -> float:
    """The listed strike nearest `target` (ties go to the lower strike)."""
    return float(listed[int(np.argmin(np.abs(listed - target)))])


def resolve_strikes(
    legs: Sequence[Leg], listed: np.ndarray, spot: float, kappa: float = KAPPA
) -> list[float]:
    """Strike of each leg: ATM = nearest listed to spot; a step is the listed distance to spot*(1+kappa)."""
    atm = nearest(listed, spot)
    step = nearest(listed, spot * (1 + kappa)) - atm
    if step <= 0:
        step = spot * kappa
    out: list[float] = []
    for lg in legs:
        if lg.kind == "FUT":
            out.append(float("nan"))
        elif lg.x is not None:
            out.append(nearest(listed, spot * (1 + lg.x)))
        else:
            out.append(nearest(listed, atm + lg.m * step))
    return out


def terminal_pnl(
    legs: Sequence[Leg],
    strikes: Sequence[float],
    entry: Sequence[float],
    s_t: float,
    near_exit: Sequence[float] | None = None,
) -> float:
    """Closed-form P&L per unit of underlying at the expiry: sum of side * qty * (exit - entry).

    `near_exit` overrides a leg's exit value (a next-month leg marked at the near expiry); else intrinsic.
    """
    tot = 0.0
    for i, lg in enumerate(legs):
        ex = (
            intrinsic(lg.kind, strikes[i], s_t) if near_exit is None or near_exit[i] is None else near_exit[i]
        )
        tot += lg.side * lg.qty * (ex - entry[i])
    return tot


def monthly_expiries(fo: pl.DataFrame) -> list[date]:
    """Last futures expiry of each calendar month in the frame (weekly and shifted dates collapse)."""
    ex = fo.filter(pl.col("instrument_type") == "IDF")["expiry_date"].unique().sort().to_list()
    last: dict[tuple[int, int], date] = {}
    for e in ex:
        last[(e.year, e.month)] = e
    return sorted(last.values())


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _delta(kind: str, s: float, k: float, t: float, sigma: float) -> float:
    if kind == "FUT":
        return 1.0
    if t <= 0 or sigma <= 0:
        return (1.0 if s > k else 0.0) if kind == "CE" else (-1.0 if s < k else 0.0)
    d1 = (math.log(s / k) + (R_FREE + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return _ncdf(d1) if kind == "CE" else _ncdf(d1) - 1.0


def _one_cycle(  # noqa: PLR0912, PLR0915
    legs: Sequence[Leg],
    rule: EntryRule,
    fo: pl.DataFrame,
    dates: list[date],
    prev_exp: date,
    exp: date,
    nxt: date | None,
    lot: float,
    costs: DerivCosts,
) -> tuple[dict, list[tuple[date, float]]] | None:
    entry = next((d for d in dates if d > prev_exp), None)
    if entry is None or entry >= exp or exp not in set(dates):
        return None
    if rule.condition is not None and not rule.condition(entry):
        return None
    want = {exp} | ({nxt} if nxt else set())
    win = fo.filter(
        (pl.col("date") >= entry) & (pl.col("date") <= exp) & pl.col("expiry_date").is_in(list(want))
    )
    spot_rows = fo.filter(pl.col("date") == entry)["underlying_price"].drop_nulls()
    st_rows = fo.filter(pl.col("date") == exp)["underlying_price"].drop_nulls()
    if spot_rows.len() == 0 or st_rows.len() == 0:
        return None
    s0, s_t = float(spot_rows.mean()), float(st_rows.mean())
    px: dict[tuple, float] = {}
    for r in win.select(
        "date", "instrument_type", "expiry_date", "option_type", "strike_price", "settle_price"
    ).iter_rows():
        d, it, e, ot, k, sp = r
        if sp is None:
            continue
        kind = "FUT" if it == "IDF" else ("CE" if str(ot).upper().startswith("C") else "PE")
        px[(d, kind, e, 0.0 if kind == "FUT" else float(k))] = float(sp)
    eo = win.filter((pl.col("date") == entry) & (pl.col("instrument_type") == "IDO"))
    strikes = (
        resolve_strikes(
            legs,
            np.array(sorted(eo.filter(pl.col("expiry_date") == exp)["strike_price"].unique().to_list())),
            s0,
            rule.kappa,
        )
        if eo.height
        else None
    )
    if strikes is None:
        return None
    exps = [exp if lg.expiry == "near" else nxt for lg in legs]
    if any(e is None for e in exps):
        return None
    keys = [
        (lg.kind, e, 0.0 if lg.kind == "FUT" else k) for lg, e, k in zip(legs, exps, strikes, strict=True)
    ]
    ent = [px.get((entry, key[0], key[1], key[2])) for key in keys]
    if any(v is None or (lg.kind != "FUT" and v <= 0) for v, lg in zip(ent, legs, strict=True)):
        return None
    marks = [list(ent)]
    cyc_dates = [d for d in dates if entry <= d <= exp]
    last = list(ent)
    spot_by_date = {
        d: float(v)
        for d, v in win.group_by("date").agg(pl.col("underlying_price").drop_nulls().mean()).iter_rows()
        if v is not None
    }
    spot_by_date[exp] = s_t
    ivs: list[float | None] = [None] * len(legs)
    hedge_pos, daily = 0.0, []
    fut_key = ("FUT", exp, 0.0)
    fut_px = {d: px.get((d, *fut_key)) for d in cyc_dates}
    fut_px[exp] = s_t
    hedge_cost_rs = 0.0
    prev_val = 0.0
    for j, d in enumerate(cyc_dates):
        cur: list[float] = []
        for i, lg in enumerate(legs):
            if d == exp and exps[i] == exp:
                v = intrinsic(lg.kind, strikes[i], s_t)
            else:
                v = px.get((d, keys[i][0], keys[i][1], keys[i][2]), last[i])
            cur.append(v)
        last = cur
        val = sum(lg.side * lg.qty * (cur[i] - ent[i]) for i, lg in enumerate(legs))
        pnl = val - prev_val
        prev_val = val
        if rule.delta_hedge and j > 0:
            f0, f1 = fut_px.get(cyc_dates[j - 1]), fut_px.get(d)
            if f0 is not None and f1 is not None:
                pnl += hedge_pos * (f1 - f0)
        if rule.delta_hedge and d != exp:
            s_d = spot_by_date.get(d, s0)
            net = 0.0
            for i, lg in enumerate(legs):
                if lg.kind == "FUT":
                    net += lg.side * lg.qty
                    continue
                t_y = (exps[i] - d).days / 365.0
                if (exps[i] - d).days >= MIN_DAYS:
                    iv = implied_vol(cur[i], s_d, strikes[i], t_y, lg.kind == "CE", R_FREE)
                    if iv is not None:
                        ivs[i] = iv
                net += lg.side * lg.qty * _delta(lg.kind, s_d, strikes[i], t_y, ivs[i] or 0.0)
            new_pos = -net
            dh = new_pos - hedge_pos
            f_d = fut_px.get(d) or s_d
            if abs(dh) > 1e-9:
                hedge_cost_rs += futures_cost_rs(1 if dh > 0 else -1, abs(dh) * f_d * lot, costs)
            hedge_pos = new_pos
        elif rule.delta_hedge and d == exp and abs(hedge_pos) > 1e-9:
            hedge_cost_rs += futures_cost_rs(-1 if hedge_pos > 0 else 1, abs(hedge_pos) * s_t * lot, costs)
        daily.append((d, pnl / s0))
        marks.append(cur)
    fills = []
    for i, lg in enumerate(legs):
        closed = exps[i] != exp
        fills.append(LegFill(lg.kind, lg.side, lg.qty, ent[i], marks[-1][i], closed))
    cost = options_cost_frac(fills, s0, None, lot, costs) + hedge_cost_rs / (s0 * lot)
    daily[-1] = (daily[-1][0], daily[-1][1] - cost)
    gross = sum(v for _, v in daily) + cost
    row = {
        "entry_date": entry,
        "exit_date": exp,
        "spot_entry": s0,
        "spot_exit": s_t,
        "gross": gross,
        "cost": cost,
        "net": gross - cost,
    }
    return row, daily


def run_combo(
    legs: Sequence[Leg],
    entry_rule: EntryRule,
    settings: Settings | None,
    und: str = "NIFTY",
    fo: pl.DataFrame | None = None,
    costs: DerivCosts | None = None,
    lot: float | None = None,
) -> ComboResult:
    """Backtest one combo on `und` monthly options from the F&O bhavcopy; per-cycle and daily net returns.

    `fo` (rows of the underlying's IDF/IDO) may be passed in; otherwise it is read from `fo_bhavcopy`.
    The first cycle starts after the first monthly expiry present in the data, and cycles whose expiry
    is after the last stored date are not traded.
    """
    c = costs or load_deriv_costs()
    lots = lot if lot is not None else c.lots.get(und, 65.0)
    if fo is None:
        assert settings is not None
        fo = (
            io.read(settings, "fo_bhavcopy")
            .filter((pl.col("symbol") == und) & pl.col("instrument_type").is_in(["IDF", "IDO"]))
            .collect()
        )
    dates = sorted(fo["date"].unique().to_list())
    exps = [e for e in monthly_expiries(fo) if e <= dates[-1]] if dates else []
    allx = monthly_expiries(fo)
    cycles: list[dict] = []
    daily: list[tuple[date, float]] = []
    for i in range(1, len(exps)):
        nxt = allx[allx.index(exps[i]) + 1] if allx.index(exps[i]) + 1 < len(allx) else None
        out = _one_cycle(legs, entry_rule, fo, dates, exps[i - 1], exps[i], nxt, float(lots), c)
        if out is None:
            log.info("opt_cycle_skipped", und=und, expiry=str(exps[i]))
            continue
        cycles.append(out[0])
        daily.extend(out[1])
    cyc = pl.DataFrame(
        cycles,
        schema={
            "entry_date": pl.Date,
            "exit_date": pl.Date,
            "spot_entry": pl.Float64,
            "spot_exit": pl.Float64,
            "gross": pl.Float64,
            "cost": pl.Float64,
            "net": pl.Float64,
        },
    )
    got = dict(daily)
    span = [d for d in dates if daily and daily[0][0] <= d <= daily[-1][0]]
    day = pl.DataFrame(
        {"date": span, "ret": [got.get(d, 0.0) for d in span]}, schema={"date": pl.Date, "ret": pl.Float64}
    )
    return ComboResult(cyc, day)
