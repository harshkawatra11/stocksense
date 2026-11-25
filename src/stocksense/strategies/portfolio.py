"""Strategy portfolio simulators: the daily series the gate reads (PLAN-M M5.7).

The backtest engine fills only at the open or the close and cannot place a stop inside a bar, so Program M
simulates its own k-slot books (PLAN-M Part 3.4):

* `simulate_daily`: per-stock daily strategies. A signal at the close of session t enters at the open of the next
  session; stop, target, time and moving-average exits come from `follow_1d` (or `follow_state` and the extra
  exits), re-run here with the actual entry; costs are the statutory schedule plus the slippage model.
* `simulate_intraday`: per-stock 5-minute strategies, up to `slots` trades a day, flat by 15:15.
* `simulate_timing`: a daily position in one index (NIFTY), held from the open after the decision.

Every simulator returns `date, net, gross, n_pos, turnover, costs_frac` where `net` is NAV_d / NAV_{d-1} - 1 and
`gross` the share of NAV invested at the close. Fractions everywhere. `PanelView` keeps only the arrays a trade
needs (bars, cost-model inputs, exit state), never a stock's whole feature dict.
"""

# ruff: noqa: E501
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Settings
from stocksense.costs.india import round_trip_cost_frac, trade_cost
from stocksense.costs.slippage import SlippageTable, slip_frac
from stocksense.strategies import explore
from stocksense.strategies.base import StrategySpec
from stocksense.strategies.exits import (
    EXITS_1D,
    EXITS_EXTRA,
    EXITS_INTRA,
    entry_window,
    follow_1d,
    follow_state,
    trend_mask,
)
from stocksense.strategies.explore import MAX_RISK, IntraPrep, SymCtx, Trade
from stocksense.strategies.k151 import STATE_RULES

Product = Literal["delivery", "intraday"]
When = Literal["open", "close", "now"]
_OUT_WHEN: dict[str, When] = {"gap_stop": "open", "stop": "now", "target": "now"}
_SLIM_KEYS = ("o", "h", "l", "c", "atr", "sma5", "ema20", "ll10")
INTRA_SLIP_BPS_TIMING = 2e-4  # index timing: 2 bps slippage per unit of position changed (x slip_scale)
SIM_COLUMNS = ("date", "net", "gross", "n_pos", "turnover", "costs_frac")


@dataclass(frozen=True)
class SimConfig:
    """Book settings: capital (INR, always an argument), slots, the cost product and the slippage multiplier."""

    capital: float
    slots: int = 10
    product: Product = "delivery"
    slip_scale: float = 1.0


@dataclass(frozen=True)
class Candidate:
    """A signal at the close of `t_sig`: its entry is the next session's open. `stop` is the signal-time stop."""

    sym: str
    sid: int
    t_sig: date
    spec_key: str
    exit_key: str
    priority: float
    stop: float


@dataclass
class SymView:
    """One stock: its bars and cost-model inputs (`ctx`) plus the slim feature dict that exits read."""

    ctx: SymCtx
    slim: dict[str, Any]

    @property
    def dates(self) -> np.ndarray:
        """Session dates of the bars (datetime64[D])."""
        return self.ctx.frame.date

    def index_of(self, d: date) -> int | None:
        """Bar index of session `d`, None when the stock has no bar that day."""
        a = self.ctx.frame.date
        i = int(np.searchsorted(a, np.datetime64(d)))
        return i if i < len(a) and a[i] == np.datetime64(d) else None

    def bar_on_or_before(self, d: date) -> int | None:
        """Index of the latest bar dated on or before `d`, None when there is none."""
        i = int(np.searchsorted(self.ctx.frame.date, np.datetime64(d), side="right")) - 1
        return i if i >= 0 else None


@dataclass
class PanelView:
    """Stocks by symbol, with the slippage table the cost model uses."""

    syms: dict[str, SymView]
    slip: SlippageTable


@dataclass
class SimResult:
    """`daily` (SIM_COLUMNS) and one record per finished trade (sym, t_in, t_out, side, pnl, ret, r_mult, why)."""

    daily: pl.DataFrame
    trades: list[dict[str, Any]] = field(default_factory=list)


def slim_features(f: dict[str, Any], spec_key: str) -> dict[str, Any]:
    """The arrays `follow_1d`, `follow_state` and the extra daily exits read from a stock's feature dict, plus
    the exit state and target of a `state_exit` rule (computed once here, not at every entry)."""
    out: dict[str, Any] = {k: np.asarray(f[k], dtype=float) for k in _SLIM_KEYS if k in f}
    out["n"] = int(f["n"])
    rule = STATE_RULES.get(spec_key)
    if rule is not None:
        out["exit_state"] = np.asarray(rule.exit_fn(f), bool)
        out["state_tgt"] = None if rule.target_fn is None else np.asarray(rule.target_fn(f), dtype=float)
        out["use_stop"] = rule.use_stop
    return out


def follow(sv: SymView, spec_key: str, exit_key: str, ie: int, stop: float, m: int) -> tuple[int, float, str]:
    """(exit bar, exit price, reason) of a long entered at bar `ie`, looking at bars `< m` only."""
    s = sv.slim
    f = {k: (v[:m] if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    f["n"] = m
    if exit_key in EXITS_1D:
        return follow_1d(f, ie, stop, exit_key)
    if exit_key == "state_exit" and "exit_state" in s:
        tgt = np.nan
        if f["state_tgt"] is not None and ie >= 1:
            tgt = float(f["state_tgt"][ie - 1])
        return follow_state(f, ie, stop, f["exit_state"], tgt, bool(s["use_stop"]))
    if exit_key in EXITS_EXTRA:
        return EXITS_EXTRA[exit_key](f, ie, stop, "long")
    raise ValueError(f"no daily exit {exit_key!r} for {spec_key}")


def _statutory(side: str, product: Product, value: float, settings: Settings) -> float:
    return trade_cost(side, product, value, settings.costs).total if value > 0 else 0.0  # type: ignore[arg-type]


def _slip(
    sv: SymView, i: int, notional: float, when: When, table: SlippageTable, cfg: SimConfig, d: date
) -> float:
    c = sv.ctx
    j = max(i - 1, 0)  # the cost inputs known at the signal bar
    rank = c.rank[j]
    vol, med = c.vol21[j], c.med_turn[j]
    return slip_frac(
        int(rank) if np.isfinite(rank) else 200,
        float(vol) if np.isfinite(vol) else 0.02,
        notional,
        float(med) if np.isfinite(med) and med > 0 else notional * 100,
        when,
        table.scaled(cfg.slip_scale),
        d,
    )


@dataclass
class _Pos:
    sym: str
    qty: int
    fill_in: float
    cost_in: float
    exit_day: date
    exit_px: float
    why: str
    sv: SymView
    ie: int
    j: int
    risk_inr: float
    t_in: date


def _plan(
    c: Candidate, sv: SymView, d: date, m: int, exit_key: str
) -> tuple[int, float, int, float, str, When] | None:
    """(entry bar, entry price, exit bar, exit price, reason, entry side of the auction) or None."""
    ie = sv.index_of(d)
    if ie is None or ie >= m:
        return None
    fr = sv.ctx.frame
    overnight = exit_key == "next_open"
    e = float(fr.close[ie] if overnight else fr.open[ie])
    if not np.isfinite(c.stop) or not np.isfinite(e) or e <= c.stop or (e - c.stop) / e > MAX_RISK:
        return None
    if overnight:  # in at the close of the entry day, out at the next open
        j, px = (ie + 1, float(fr.open[ie + 1])) if ie + 1 < m else (ie, e)
        return ie, e, j, px, "next_open", "close"
    j, px, why = follow(sv, c.spec_key, exit_key, ie, c.stop, m)
    return ie, e, int(j), float(px), why, "open"


def run_daily(
    trades_by_signal_date: dict[date, list[Candidate]],
    panel: PanelView,
    cfg: SimConfig,
    settings: Settings,
    calendar: TradingCalendar,
    start: date,
    end: date,
    exit_key: str | None = None,
) -> SimResult:
    """The k-slot daily book over sessions `start..end` (see `simulate_daily`); also returns every trade.

    `exit_key` replaces the exit rule of every candidate (the exit-selection loop tries them one by one).

    Exits are resolved on bars up to `end` only: a position still open there leaves at the close of the last bar,
    so nothing after `end` is ever read. NAV starts at `cfg.capital` before the first session.
    """
    sessions = calendar.sessions(start, end)
    try:
        prev_sig = calendar.prev_session(start)
    except ValueError:  # the first stored session: nothing was signalled before it
        prev_sig = date.min
    nav = cfg.capital
    cash = cfg.capital
    held: dict[str, _Pos] = {}
    rows: list[tuple[date, float, float, int, float, float]] = []
    trades: list[dict[str, Any]] = []
    end64 = np.datetime64(end)

    def close_pos(p: _Pos) -> tuple[float, float]:
        nonlocal cash
        notional = p.qty * p.exit_px
        slip = _slip(p.sv, p.j, notional, _OUT_WHEN.get(p.why, "close"), panel.slip, cfg, p.exit_day)
        fill = p.exit_px * (1.0 - slip)
        cost = _statutory("SELL", cfg.product, p.qty * fill, settings)
        cash += p.qty * fill - cost
        pnl = p.qty * (fill - p.fill_in) - cost - p.cost_in
        trades.append(
            {
                "sym": p.sym,
                "t_in": p.t_in,
                "t_out": p.exit_day,
                "side": 1,
                "pnl": pnl,
                "ret": pnl / (p.qty * p.fill_in),
                "r_mult": pnl / p.risk_inr if p.risk_inr > 0 else 0.0,
                "why": p.why,
            }
        )
        return p.qty * fill, cost

    for d in sessions:
        paid = traded = 0.0
        for sym in [s for s, p in held.items() if p.exit_day == d]:
            v, cst = close_pos(held.pop(sym))
            traded += v
            paid += cst
        nav_prev = nav
        for c in sorted(trades_by_signal_date.get(prev_sig, ()), key=lambda x: (-x.priority, x.sym)):
            if len(held) >= cfg.slots:
                break
            sv = panel.syms.get(c.sym)
            if sv is None or c.sym in held:
                continue
            m = int(np.searchsorted(sv.dates, end64, side="right"))
            plan = _plan(c, sv, d, m, exit_key or c.exit_key)
            if plan is None:
                continue
            ie, e, j, px, why, when_in = plan
            slot_inr = nav_prev / cfg.slots
            slip_in = _slip(sv, ie, slot_inr, when_in, panel.slip, cfg, d)
            fill_in = e * (1.0 + slip_in)
            qty = int(slot_inr // fill_in)
            if qty < 1 or qty * fill_in > cash:
                continue
            cost_in = _statutory("BUY", cfg.product, qty * fill_in, settings)
            cash -= qty * fill_in + cost_in
            traded += qty * fill_in
            paid += cost_in
            held[c.sym] = _Pos(
                c.sym, qty, fill_in, cost_in, sv.dates[j].item(), px, why, sv, ie, j,
                qty * (e - c.stop), d,
            )  # fmt: skip
        for sym in [s for s, p in held.items() if p.exit_day == d]:  # entered and left on the same day
            v, cst = close_pos(held.pop(sym))
            traded += v
            paid += cst
        invested = 0.0
        for p in held.values():
            k = p.sv.bar_on_or_before(d)
            invested += p.qty * float(p.sv.ctx.frame.close[k if k is not None else 0])
        nav = cash + invested
        rows.append(
            (d, nav / nav_prev - 1.0 if nav_prev > 0 else 0.0, invested / nav if nav > 0 else 0.0,
             len(held), traded / nav_prev if nav_prev > 0 else 0.0, paid / nav_prev if nav_prev > 0 else 0.0)
        )  # fmt: skip
        prev_sig = d
    return SimResult(_frame(rows), trades)


SIM_COLUMNS_SCHEMA: dict[str, Any] = {
    "date": pl.Date,
    "net": pl.Float64,
    "gross": pl.Float64,
    "n_pos": pl.Int64,
    "turnover": pl.Float64,
    "costs_frac": pl.Float64,
}


def _frame(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SIM_COLUMNS_SCHEMA, orient="row")


def simulate_daily(
    trades_by_signal_date: dict[date, list[Candidate]],
    panel: PanelView,
    cfg: SimConfig,
    settings: Settings,
    calendar: TradingCalendar,
    start: date,
    end: date,
) -> pl.DataFrame:
    """k-slot book: on each session d, first process exits scheduled on d (fills at the `follow_1d` price with
    exit-side cost and slippage), then entries: candidates whose signal date is the previous session, sorted by
    priority desc, while free slots; qty = floor(NAV_{d-1} / slots / (open_d * (1 + slip_in))); buy cost via
    `trade_cost`; then mark every position at close_d. Columns date, net, gross, n_pos, turnover, costs_frac."""
    return run_daily(trades_by_signal_date, panel, cfg, settings, calendar, start, end).daily


# ---- intraday --------------------------------------------------------------------------------------------


def intra_trades(
    ip: IntraPrep, spec: StrategySpec, exit_key: str, trend: str, *, notional: float
) -> list[Trade]:
    """Trades of one intraday strategy, exit and trend filter on one stock's 5-minute bars.

    The eight standard exits (`EXITS_INTRA`) come from the exploratory engine; the extra exits of M5.3 (VWAP
    touch, value-area edge, initial-balance extension, POC touch) run through `EXITS_EXTRA`, long on the real
    chart and short on the mirrored one, with the same one-trade-a-day entry rule.
    """
    if exit_key in EXITS_INTRA:
        return explore.trades_intra(ip, spec, exit_key, trend, notional=notional)
    if exit_key not in EXITS_EXTRA or spec.signal is None:
        raise ValueError(f"no intraday exit {exit_key!r} for {spec.key}")
    fn = EXITS_EXTRA[exit_key]
    sides = (
        ((1, ip.long), (-1, ip.short))
        if spec.direction == "both" and ip.short is not None
        else ((1, ip.long),)
    )
    out: list[Trade] = []
    for side, g in sides:
        n = g["n"]
        entry_ok, exit_idx = entry_window(g)
        sig, stop = spec.signal(g)
        sig = np.asarray(sig, bool) & entry_ok & trend_mask(g, trend)
        taken: set[int] = set()
        for i in np.flatnonzero(sig):
            i = int(i)
            ie, day = i + 1, int(g["day"][i])
            if ie >= n or day in taken or g["day"][ie] != day:
                continue
            s = float(stop[i])
            e = float(g["o"][ie])
            if not np.isfinite(s) or e - s <= 0 or int(exit_idx[day]) < ie:
                continue
            taken.add(day)
            j, px, why = fn(g, ie, s, "long" if side == 1 else "short")
            live = j >= n - 1 and g["mins"][n - 1] + g["tf"] < 915
            gross = (px - e) / abs(e)
            out.append(
                Trade(
                    sym=ip.sym,
                    strat=spec.key,
                    exit=exit_key,
                    side=side,
                    t_sig=int(g["ts"][i]),
                    t_in=int(g["ts"][ie]),
                    p_in=side * e,
                    stop=side * s,
                    target=None,
                    t_out=int(g["ts"][j]),
                    p_out=side * px,
                    reason="open" if live else why,
                    R=(px - e) / (e - s),
                    gross_pct=100 * gross,
                    cost_pct=0.0,
                    net_pct=100 * gross,
                    days=None,
                    status="OPEN" if live else "CLOSED",
                    mkt_up=bool(g["ix_up"][i]),
                    rs63=None,
                )  # fmt: skip
            )
    out.sort(key=lambda t: t.t_in)
    return out


def simulate_intraday(
    trades: list[Trade], cfg: SimConfig, settings: Settings, calendar: TradingCalendar, start: date, end: date
) -> pl.DataFrame:
    """Per session: up to `slots` trades in entry-time order, each sized NAV_{d-1}/slots; P&L from p_in/p_out
    with intraday costs; all flat by 15:15; net = sum pnl / NAV_{d-1}; gross = max concurrent notional / NAV_{d-1}."""
    return run_intraday(trades, cfg, settings, calendar, start, end).daily


def run_intraday(
    trades: list[Trade], cfg: SimConfig, settings: Settings, calendar: TradingCalendar, start: date, end: date
) -> SimResult:
    """`simulate_intraday` plus the trade records."""
    table = SlippageTable.load_latest(settings).scaled(cfg.slip_scale)
    by_day: dict[date, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.status == "CLOSED" and t.p_out is not None and t.t_out is not None:
            by_day[explore.epoch_date(t.t_in)].append(t)
    nav = cfg.capital
    rows: list[tuple[date, float, float, int, float, float]] = []
    recs: list[dict[str, Any]] = []
    for d in calendar.sessions(start, end):
        nav_prev = nav
        todays = sorted(by_day.get(d, ()), key=lambda t: (t.t_in, t.sym))[: cfg.slots]
        pnl = paid = traded = 0.0
        slot = nav_prev / cfg.slots
        for t in todays:
            gross = t.side * (t.p_out / t.p_in - 1.0)  # type: ignore[operator]
            out_when: When = "close" if t.reason == "15:15 exit" else "now"
            s_in = slip_frac(200, 0.02, slot, slot * 100, "open", table, d)
            s_out = slip_frac(200, 0.02, slot, slot * 100, out_when, table, d)
            cost = _statutory("BUY", "intraday", slot, settings) + _statutory(
                "SELL", "intraday", slot, settings
            )
            p = slot * (gross - s_in - s_out) - cost
            pnl += p
            paid += cost
            traded += 2 * slot
            risk = abs(t.p_in - t.stop) / abs(t.p_in) * slot
            recs.append(
                {
                    "sym": t.sym, "t_in": d, "t_out": d, "side": t.side, "pnl": p, "ret": p / slot,
                    "r_mult": p / risk if risk > 0 else 0.0, "why": t.reason,
                }
            )  # fmt: skip
        nav = nav_prev + pnl
        ev = sorted([(t.t_in, 1) for t in todays] + [(t.t_out, -1) for t in todays])  # type: ignore[misc]
        cur = peak = 0
        for _, k in ev:
            cur += k
            peak = max(peak, cur)
        rows.append(
            (d, pnl / nav_prev if nav_prev > 0 else 0.0, peak / cfg.slots, peak,
             traded / nav_prev if nav_prev > 0 else 0.0, paid / nav_prev if nav_prev > 0 else 0.0)
        )  # fmt: skip
    return SimResult(_frame(rows), recs)


# ---- index timing ----------------------------------------------------------------------------------------


def simulate_timing(
    position: pl.DataFrame, index_bars: pl.DataFrame, cfg: SimConfig, settings: Settings | None = None
) -> pl.DataFrame:
    """A position in one index, decided at the close of `date` and held from the next session's open.

    `position` has `date, position` (fractions of NAV, negative for a short); `index_bars` has `date, open,
    close`. Day return = (1 + q (open / prev_close - 1)) (1 + p (close / open - 1)) - 1, with q the position held
    overnight and p the one just entered (so a new entry earns open to close, a held one close to close). Each
    change |p - q| costs one side of the delivery round trip on NAV (statutory, via `trade_cost`) plus
    2 bps x slip_scale. `gross` = |p|.
    """
    bars = index_bars.sort("date")
    d_idx = bars["date"].to_list()
    o, c = bars["open"].to_numpy().astype(float), bars["close"].to_numpy().astype(float)
    pos = dict(zip(position["date"].to_list(), position["position"].to_list(), strict=True))
    rows: list[tuple[date, float, float, int, float, float]] = []
    q = 0.0
    nav = cfg.capital
    for k in range(1, len(d_idx)):
        p = float(pos.get(d_idx[k - 1], 0.0) or 0.0)
        ret = (1.0 + q * (o[k] / c[k - 1] - 1.0)) * (1.0 + p * (c[k] / o[k] - 1.0)) - 1.0
        change = abs(p - q)
        cost = 0.0
        if change > 0:
            side = "BUY" if p > q else "SELL"
            stat = (
                _statutory(side, cfg.product, nav * change, settings) / nav if settings is not None else 0.0
            )
            cost = change * INTRA_SLIP_BPS_TIMING * cfg.slip_scale + stat
        net = ret - cost
        nav *= 1.0 + net
        rows.append((d_idx[k], net, abs(p), int(p != 0), change, cost))
        q = p
    return _frame(rows)


def round_trip(settings: Settings, product: Product, value: float) -> float:
    """Statutory round-trip cost as a fraction of `value` (a convenience for the tests and the reports)."""
    return round_trip_cost_frac(product, value, settings.costs)
