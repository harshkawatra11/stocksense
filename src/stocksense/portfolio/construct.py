"""Portfolio construction (PLAN T9.1, T9.2): the S1 multi-day delivery book and the S2 auction book.

Both policies are thin classes over pure planning functions (`plan_s1` / `build_s1_orders`, `plan_s2` /
`build_s2_orders`) so that the backtest and the live daemon run the same logic. Nothing here places a real
order: an `Order` is the backtest/paper dataclass of `backtest.fills`.

Inputs the policies need are small, explicit value objects:
  * `DayScores`   blended score in [-0.5, 0.5] and expected horizon return per sid (a `ScoreSource` gives one
                  per date; for S1 the date is the decision date t, for S2 the trade date d, known pre-open);
  * `NameRow`     one security's causal facts on a date (close, vol_21, rank, industry, flags, event days);
  * `PanelDay`    the rows of one decision date plus the market-state fields and the next session.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

import numpy as np

from stocksense.backtest.engine import BookState, PanelData
from stocksense.backtest.fills import Order
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Costs, Risk, Settings
from stocksense.costs.india import min_position_for_fixed_cost, round_trip_cost_frac
from stocksense.costs.slippage import SlippageTable, slip_frac

COV_WINDOW = 126
MIN_COV_ROWS = 60
S1_RANK_DECAY_MULT = 3
S1_RANK_DECAY_MIN_HELD = 2
OVERLAY_BREADTH = 0.30
OVERLAY_FACTOR = 0.5
CASH_HAIRCUT = 0.005  # expected sale proceeds are counted at 99.5% when sizing tomorrow's close buys
BUY_CUSHION_INR = 30.0  # the engine leaves about Rs 29.5 of cash per buy for the exit fees
_EPS = 1e-9


# ---- value objects -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DayScores:
    """Per-sid blended score (in [-0.5, 0.5]) and expected horizon return `er`; disagreement is optional."""

    score: Mapping[int, float]
    er: Mapping[int, float]
    disagreement: Mapping[int, float] = field(default_factory=dict)


class ScoreSource(Protocol):
    """Gives the day's scores or None when there are none (a data failure: entries are skipped)."""

    def scores(self, d: date) -> DayScores | None: ...


@dataclass(frozen=True)
class NameRow:
    """One security on one date, everything known at that date's close (S1) or the previous close (S2)."""

    close: float
    vol_21: float
    turnover_rank: int
    med_turn_20: float
    industry: str = "UNKNOWN"
    in_universe: bool = True
    stale: bool = False
    locked_up: bool = False
    in_fno: bool = False
    days_to_split: int | None = (
        None  # sessions until a split/bonus/rights/demerger ex-date (None: none known)
    )
    days_to_results: int | None = None  # sessions until a results board meeting
    days_since_result: int = 90
    rets: np.ndarray | None = None  # the last <= 126 daily returns (S1 covariance)
    upper_circuit: float | None = None  # live S2: the quote's upper circuit price


@dataclass(frozen=True)
class PanelDay:
    """Rows of one decision date; `next_date` is the session on which orders trade."""

    date: date
    next_date: date
    rows: Mapping[int, NameRow]
    breadth_200: float | None = None
    mkt_ret_21: float | None = None


class PanelAccess(Protocol):
    def day(self, t: date, next_date: date) -> PanelDay: ...


class MetaLike(Protocol):
    """The optional meta-labeling layer (portfolio.meta.MetaModel)."""

    def predict_rows(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray: ...

    def size_multiplier(self, p: float) -> float: ...

    def veto(self, p: float) -> bool: ...


# ---- configuration -----------------------------------------------------------------------------------


def horizon_of(target: str) -> int:
    """5 for 'y_c2o_5'."""
    return int(target.rsplit("_", 1)[-1])


@dataclass(frozen=True)
class S1Config:
    h: int
    k_max: int
    risk: Risk
    costs: Costs
    slippage: SlippageTable
    overlay: bool = False
    hysteresis: bool = True  # False = naive daily re-ranking (a test/reference knob, not a strategy)
    min_pos: float = field(init=False)

    def __post_init__(self) -> None:
        mp = max(
            self.risk.min_position_inr_floor,
            min_position_for_fixed_cost("delivery", self.costs, self.risk.max_fixed_cost_bps),
        )
        object.__setattr__(self, "min_pos", float(mp))

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        h: int,
        slippage: SlippageTable,
        *,
        k_max: int | None = None,
        overlay: bool = False,
        hysteresis: bool = True,
    ) -> S1Config:
        return cls(
            h, k_max if k_max is not None else settings.strategies.s1.k_max, settings.risk,
            settings.costs, slippage, overlay, hysteresis,
        )  # fmt: skip


# ---- S1: pure planning -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Holding:
    """A delivery holding as the planner sees it; `held` = sessions since entry (or last extension)."""

    sid: int
    qty: int
    entry_price: float
    held: int
    vol_at_entry: float


@dataclass(frozen=True)
class S1Book:
    holdings: Mapping[int, Holding]
    cash: float  # free cash (cash minus reserved)
    dd: float = 0.0  # drawdown of the book NAV from its peak, as a positive fraction
    stopped: bool = False  # the book-drawdown stop flag is set (until `stocksense resume`)


@dataclass(frozen=True)
class S1Plan:
    orders: list[Order]
    k_raw: int  # breadth from capital alone
    k: int
    gross_target: float
    port_vol: float
    min_pos: float
    cap: float
    exits: dict[int, str]  # sid -> reason (stop, horizon, rank_decay, universe_exit)
    extended: list[int]
    weights: dict[int, float]  # new entries
    entries: list[dict[str, Any]]  # features of each new entry (the meta-label sample log)
    note: str = ""


def _floor(x: float) -> int:
    return int(math.floor(x + _EPS))


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def portfolio_vol(cands: Sequence[NameRow], window: int = COV_WINDOW, min_rows: int = MIN_COV_ROWS) -> float:
    """Annualised vol of the equal-weight portfolio of `cands`: Ledoit-Wolf covariance of their last `window`
    returns (columns with any null dropped; fewer than `min_rows` common rows -> the fallback covariance
    mean(vol_21)^2 * (0.3 + 0.7 I), i.e. correlation 0.3)."""
    n = len(cands)
    if n == 0:
        return 0.0
    w = np.full(n, 1.0 / n)
    cov: np.ndarray | None = None
    if all(c.rets is not None and len(c.rets) for c in cands):
        length = min(min(len(c.rets), window) for c in cands if c.rets is not None)
        m = np.vstack([np.asarray(c.rets, dtype=float)[-length:] for c in cands if c.rets is not None])
        x = m[:, np.all(np.isfinite(m), axis=0)].T
        if x.shape[0] >= min_rows:
            if n == 1:
                cov = np.array([[float(np.var(x[:, 0], ddof=1))]])
            else:
                from sklearn.covariance import LedoitWolf

                cov = LedoitWolf().fit(x).covariance_
    if cov is None:
        v = float(np.mean([c.vol_21 for c in cands])) ** 2
        cov = v * (0.3 * np.ones((n, n)) + 0.7 * np.eye(n))
    return math.sqrt(max(float(w @ cov @ w), 0.0) * 252.0)


def allocate_weights(raw: Mapping[int, float], room: float, cap: float) -> dict[int, float]:
    """Scale `raw` to sum to `room`, cap every name at `cap`, hand the excess pro-rata to the names still
    under the cap once, then clip to the cap again (what cannot be placed stays in cash). Never exceeds the
    cap and the total never exceeds `room`."""
    if not raw or room <= 0 or cap <= 0:
        return {s: 0.0 for s in raw}
    tot = sum(max(v, 0.0) for v in raw.values())
    if tot <= 0:
        return {s: 0.0 for s in raw}
    w = {s: max(v, 0.0) / tot * room for s, v in raw.items()}
    over = {s: x for s, x in w.items() if x > cap}
    excess = sum(x - cap for x in over.values())
    for s in over:
        w[s] = cap
    under = {s: x for s, x in w.items() if x < cap}
    base = sum(under.values())
    if excess > 0 and under:
        for s, x in under.items():
            share = x / base if base > 0 else 1.0 / len(under)
            w[s] = min(x + excess * share, cap)
    return w


def _slip(row: NameRow, value: float, when: str, table: SlippageTable) -> float:
    return slip_frac(row.turnover_rank, max(row.vol_21, 1e-6), value, max(row.med_turn_20, 1.0), when, table)  # type: ignore[arg-type]


def _s1_entry_ok(sid: int, row: NameRow, er: float | None, value: float, cfg: S1Config) -> bool:
    """Entry eligibility other than 'not already held / not sold today' (the caller handles those)."""
    if not row.in_universe or row.stale or row.locked_up or er is None or not math.isfinite(er):
        return False
    if row.close <= 0:
        return False
    net = (
        er
        - round_trip_cost_frac("delivery", value, cfg.costs)
        - 2.0 * _slip(row, value, "close", cfg.slippage)
    )
    if net <= 0:
        return False
    n_split, n_res = cfg.risk.no_entry_sessions_before_split, cfg.risk.no_entry_sessions_before_results
    if row.days_to_split is not None and 0 <= row.days_to_split <= n_split:
        return False
    return not (row.days_to_results is not None and 0 <= row.days_to_results <= n_res)


def plan_s1(
    t: date,
    scores_t: Mapping[int, float],
    er_t: Mapping[int, float],
    book: S1Book,
    equity_inr: float,
    panel_t: PanelDay,
    cfg: S1Config,
    *,
    disagreement: Mapping[int, float] | None = None,
    meta: MetaLike | None = None,
    allow_entries: bool = True,
) -> S1Plan:
    """Everything S1 decides after the close of `t` (PLAN T9.1 steps 2-8)."""
    rows, d1, risk = panel_t.rows, panel_t.next_date, cfg.risk
    min_pos = cfg.min_pos
    disagreement = disagreement or {}
    scored = [
        s for s, r in rows.items() if r.in_universe and not r.stale and s in scores_t
        and math.isfinite(scores_t[s])
    ]  # fmt: skip
    ranked = sorted(scored, key=lambda s: (-scores_t[s], s))
    rank = {s: i + 1 for i, s in enumerate(ranked)}

    # 2. capital-aware breadth, 3. exposure target
    k_raw = _clamp_int(_floor(equity_inr * risk.max_gross / min_pos), 0, cfg.k_max)
    pvol = portfolio_vol([rows[s] for s in ranked[:k_raw]]) if k_raw > 0 else 0.0
    gt = risk.max_gross if pvol <= 0 else min(risk.max_gross, risk.target_vol_annual / pvol)
    if (
        cfg.overlay and panel_t.breadth_200 is not None and panel_t.mkt_ret_21 is not None
        and panel_t.breadth_200 < OVERLAY_BREADTH and panel_t.mkt_ret_21 < 0
    ):  # fmt: skip
        gt *= OVERLAY_FACTOR
    if book.stopped or book.dd >= risk.book_dd_stop:
        gt = 0.0
    elif book.dd >= risk.book_dd_half:
        gt *= 0.5
    k = _clamp_int(_floor(equity_inr * gt / min_pos), 0, cfg.k_max)
    notes: list[str] = []
    if k_raw == 0:
        notes.append("capital below one minimum position")
    elif k == 0:
        notes.append("exposure target is zero (book stopped or too little room)")
    target_value = equity_inr * gt / k if k > 0 else 0.0
    sqrt_h = math.sqrt(cfg.h)

    # 4. exits
    orders: list[Order] = []
    exits: dict[int, str] = {}
    extended: list[int] = []
    scores_known = bool(ranked)
    for sid in sorted(book.holdings):
        hd = book.holdings[sid]
        row = rows.get(sid)
        reason: str | None = None
        if row is None or not row.in_universe or row.stale:
            reason = "universe_exit"
        elif row.close <= hd.entry_price * (
            1.0 - max(risk.stop_min, risk.stop_vol_mult * hd.vol_at_entry * sqrt_h)
        ):
            reason = "stop"
        elif scores_known:
            r = rank.get(sid, math.inf)
            if cfg.hysteresis:
                if hd.held >= S1_RANK_DECAY_MIN_HELD and r > S1_RANK_DECAY_MULT * k:
                    reason = "rank_decay"
                elif hd.held >= cfg.h - 1:
                    if k > 0 and r <= k and _s1_entry_ok(sid, row, er_t.get(sid), target_value, cfg):
                        extended.append(sid)
                    else:
                        reason = "horizon"
            elif r > k:
                reason = "rank_decay"
        elif cfg.hysteresis and hd.held >= cfg.h - 1:
            reason = "horizon"
        if reason is not None:
            exits[sid] = reason
            orders.append(
                Order(sid, "SELL", "delivery", "open", hd.qty, trade_date=d1, priority=0.0, reason=reason)
            )

    remaining = [s for s in book.holdings if s not in exits]
    px_of = {s: (rows[s].close if s in rows else book.holdings[s].entry_price) for s in book.holdings}
    existing_value = sum(book.holdings[s].qty * px_of[s] for s in remaining)
    cap = 0.0
    weights: dict[int, float] = {}
    entries: list[dict[str, Any]] = []
    if k > 0 and equity_inr > 0:
        cap = min(max(risk.max_name_weight, min_pos / equity_inr), 1.5 * gt / k)

    if not allow_entries:
        notes.append("entries blocked by pre-trade checks")
    elif k > 0 and scores_known:
        slots = k - len(remaining)
        max_per_industry = max(1, _floor(risk.max_industry_weight * k))
        ind_count: dict[str, int] = {}
        for s in remaining:
            ind = rows[s].industry if s in rows else "UNKNOWN"
            ind_count[ind] = ind_count.get(ind, 0) + 1
        chosen: list[int] = []
        mult: dict[int, float] = {}
        for sid in ranked:
            if len(chosen) >= slots:
                break
            if sid in book.holdings or sid in exits:
                continue
            row = rows[sid]
            if not _s1_entry_ok(sid, row, er_t.get(sid), target_value, cfg):
                continue
            if ind_count.get(row.industry, 0) >= max_per_industry:
                continue
            feat = {
                "score": float(scores_t[sid]), "score_rank": float(rank[sid]),
                "disagreement": float(disagreement.get(sid, float("nan"))), "vol_21": float(row.vol_21),
                "turnover_rank": float(row.turnover_rank),
                "mkt_ret_21": float("nan") if panel_t.mkt_ret_21 is None else float(panel_t.mkt_ret_21),
                "breadth_200": float("nan") if panel_t.breadth_200 is None else float(panel_t.breadth_200),
                "days_since_result": float(row.days_since_result), "gross_target": float(gt),
            }  # fmt: skip
            m = 1.0
            if meta is not None:
                p = float(meta.predict_rows([feat])[0])
                if meta.veto(p):
                    continue
                m = meta.size_multiplier(p)
            ind_count[row.industry] = ind_count.get(row.industry, 0) + 1
            chosen.append(sid)
            mult[sid] = m
            entries.append({"sid": sid, "date": d1, **feat})
        raw = {s: (0.5 + scores_t[s]) / max(rows[s].vol_21, 1e-4) * mult[s] for s in chosen}
        room = max(gt - existing_value / equity_inr, 0.0)
        alloc = allocate_weights(raw, room, cap)
        weights = {s: w for s, w in alloc.items() if w * equity_inr >= min_pos - 1e-6}
        entries = [e for e in entries if e["sid"] in weights]
        # 8. orders, sized to the cash expected after tomorrow's open sells
        proceeds = sum(book.holdings[s].qty * px_of[s] for s in exits) * (1.0 - CASH_HAIRCUT)
        budget = book.cash + proceeds
        for sid in sorted(weights, key=lambda s: -scores_t[s]):
            px = rows[sid].close
            qty = _floor(weights[sid] * equity_inr / px)
            qty = min(qty, _floor(max(budget - BUY_CUSHION_INR, 0.0) / px))
            if qty <= 0:
                continue
            budget -= qty * px + BUY_CUSHION_INR
            orders.append(
                Order(sid, "BUY", "delivery", "close", qty, trade_date=d1, priority=float(scores_t[sid]),
                      reason="entry")
            )  # fmt: skip
        if not any(o.side == "BUY" for o in orders) and not notes:
            notes.append("no stock clears costs today")
    return S1Plan(
        orders, k_raw, k, gt, pvol, min_pos, cap, exits, extended, weights, entries, "; ".join(notes)
    )  # fmt: skip


def build_s1_orders(
    t: date,
    scores_t: Mapping[int, float],
    er_t: Mapping[int, float],
    book: S1Book,
    equity_inr: float,
    panel_t: PanelDay,
    cfg: S1Config,
    **kw: Any,
) -> list[Order]:
    """The orders of `plan_s1` (SELL at the next open, BUY at the next close)."""
    return plan_s1(t, scores_t, er_t, book, equity_inr, panel_t, cfg, **kw).orders


# ---- S1: the policy ----------------------------------------------------------------------------------


class StopFlags(Protocol):
    def kv_get(self, key: str, default: str | None = None) -> str | None: ...

    def kv_set(self, key: str, value: str) -> None: ...


def book_stop_key(book_name: str) -> str:
    """kv key of the book-drawdown stop; `stocksense resume --book <name>` sets it to '0'."""
    return f"book_stop:{book_name}"


class S1Policy:
    """Implements `Policy.decide` for the multi-day delivery book (same class in backtest and live)."""

    def __init__(
        self,
        spec: Any,
        settings: Settings,
        score_source: ScoreSource,
        panel_access: PanelAccess,
        costs: Costs,
        slippage: SlippageTable,
        *,
        k_max: int | None = None,
        overlay: bool = False,
        hysteresis: bool = True,
        meta: MetaLike | None = None,
        state: StopFlags | None = None,
        book_name: str = "s1",
        entries_allowed: Callable[[date], bool] | None = None,
    ) -> None:
        km = k_max if k_max is not None else settings.strategies.s1.k_max
        h = horizon_of(getattr(spec, "target", spec))
        cfg = S1Config(h, km, settings.risk, costs, slippage, overlay, hysteresis)
        self.cfg, self.source, self.access, self.meta = cfg, score_source, panel_access, meta
        self.state, self.book_name, self._entries_allowed = state, book_name, entries_allowed
        self._peak = 0.0
        self._stopped = False
        self._was_stopped = False
        self._anchor: dict[int, date] = {}
        self._vol_at_entry: dict[int, float] = {}
        self._pending_vol: dict[int, float] = {}
        self.entry_log: list[dict[str, Any]] = []
        self.plans: dict[date, S1Plan] = {}

    # -- the kv stop flag (stays set until resume clears it)
    def _read_stop(self) -> bool:
        if self.state is not None:
            return self.state.kv_get(book_stop_key(self.book_name), "0") == "1"
        return self._stopped

    def _write_stop(self, on: bool) -> None:
        self._stopped = on
        if self.state is not None:
            self.state.kv_set(book_stop_key(self.book_name), "1" if on else "0")

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        cal = book.calendar
        d1 = cal.next_session(t)
        stopped = self._read_stop()
        if self._was_stopped and not stopped:  # `stocksense resume` cleared the flag: re-base the peak
            self._peak = capital_inr
        self._peak = max(self._peak, capital_inr)
        dd = 1.0 - capital_inr / self._peak if self._peak > 0 else 0.0
        if dd >= self.cfg.risk.book_dd_stop and not stopped:
            self._write_stop(True)
            stopped = True
        self._was_stopped = stopped

        holdings: dict[int, Holding] = {}
        for sid, pos in book.positions.items():
            if pos.product != "delivery" or pos.qty <= 0:
                continue
            if sid not in self._anchor:
                self._anchor[sid] = pos.entry_date
                self._vol_at_entry[sid] = self._pending_vol.pop(sid, 0.02)
            held = cal.session_index(t) - cal.session_index(self._anchor[sid])
            holdings[sid] = Holding(sid, pos.qty, pos.entry_price, held, self._vol_at_entry[sid])
        for sid in [s for s in self._anchor if s not in holdings]:
            self._anchor.pop(sid, None)
            self._vol_at_entry.pop(sid, None)

        panel_t = self.access.day(t, d1)
        sc = self.source.scores(t)
        allow = self._entries_allowed(t) if self._entries_allowed is not None else True
        sbook = S1Book(holdings, book.cash - book.reserved, dd, stopped)
        plan = plan_s1(
            t, sc.score if sc else {}, sc.er if sc else {}, sbook, capital_inr, panel_t, self.cfg,
            disagreement=sc.disagreement if sc else None, meta=self.meta,
            allow_entries=allow and sc is not None,
        )  # fmt: skip
        for sid in plan.extended:
            self._anchor[sid] = d1
        for e in plan.entries:
            self._pending_vol[e["sid"]] = e["vol_21"]
        self.entry_log.extend(plan.entries)
        self.plans[t] = plan
        return plan.orders


# ---- a panel access over the engine's PanelData -----------------------------------------------------


class EnginePanelAccess:
    """Builds `PanelDay`s from the engine's `PanelData` (causal: only rows up to the decision date).

    `industry` maps sid -> industry; `split_dates` / `result_dates` map sid -> sorted event dates (days to
    the next event are counted in sessions of `calendar`); `fno` is a set of sids; `market` maps
    date -> (breadth_200, mkt_ret_21); `stale` is a set of (date, sid)."""

    def __init__(
        self,
        panel: PanelData,
        calendar: TradingCalendar,
        *,
        industry: Mapping[int, str] | None = None,
        split_dates: Mapping[int, Sequence[date]] | None = None,
        result_dates: Mapping[int, Sequence[date]] | None = None,
        fno: set[int] | None = None,
        market: Mapping[date, tuple[float | None, float | None]] | None = None,
        stale: set[tuple[date, int]] | None = None,
        universe_rank: int = 500,
    ) -> None:
        self.p, self.cal = panel, calendar
        self.industry = industry or {}
        self.split_dates, self.result_dates = split_dates or {}, result_dates or {}
        self.fno, self.market, self.stale = fno or set(), market or {}, stale or set()
        self.universe_rank = universe_rank
        order = np.argsort(panel.sid, kind="stable")
        sids = panel.sid[order]
        self._hist: dict[int, tuple[list[date], np.ndarray]] = {}
        if len(sids):
            bounds = np.flatnonzero(np.diff(sids)) + 1
            for chunk in np.split(order, bounds):
                sid = int(panel.sid[chunk[0]])
                self._hist[sid] = ([panel.date[j] for j in chunk], panel.ret[chunk])

    def _days_to(self, t: date, events: Sequence[date] | None) -> int | None:
        if not events:
            return None
        i = bisect_right(list(events), t)
        if i >= len(events):
            return None
        ev = events[i]
        if ev - t > timedelta(days=14):
            return None
        return len(self.cal.sessions(t + timedelta(days=1), ev))

    def day(self, t: date, next_date: date) -> PanelDay:
        p = self.p
        sl = p.day(t)
        rows: dict[int, NameRow] = {}
        if sl is not None:
            for j in range(*sl):
                sid = int(p.sid[j])
                dates, rets = self._hist[sid]
                pos = bisect_right(dates, t)
                r = rets[max(0, pos - COV_WINDOW) : pos]
                rows[sid] = NameRow(
                    float(p.close[j]), float(p.vol_21[j]), int(p.rank[j]), float(p.med_turn_20[j]),
                    self.industry.get(sid, "UNKNOWN"), int(p.rank[j]) <= self.universe_rank,
                    (t, sid) in self.stale, bool(p.locked_up[j]), sid in self.fno,
                    self._days_to(t, self.split_dates.get(sid)), self._days_to(t, self.result_dates.get(sid)),
                    90, r,
                )  # fmt: skip
        b, m = self.market.get(t, (None, None))
        return PanelDay(t, next_date, rows, b, m)
