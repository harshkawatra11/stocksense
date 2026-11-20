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

from stocksense.backtest.engine import AuctionView, BookState, PanelData, Position
from stocksense.backtest.fills import Order
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Costs, Risk, Settings
from stocksense.costs.india import min_position_for_fixed_cost, round_trip_cost_frac
from stocksense.costs.slippage import SlippageTable, slip_frac

COV_WINDOW = 126
MIN_COV_ROWS = 60
S1_RANK_DECAY_MULT = 3
S1_RANK_DECAY_MIN_HELD = 2
DAYS_SINCE_RESULT_CAP = 90
VOL_MODELS = ("vol21", "har", "garch")
SIZINGS = ("score_over_vol", "kelly")
OVERLAY_BREADTH = 0.30
OVERLAY_FACTOR = 0.5
REGIME_BEAR_P = 0.6  # G2-E4: the regime overlay applies above this filtered bear probability
REGIME_FACTOR = 0.5
CASH_HAIRCUT = 0.005  # expected sale proceeds are counted at 99.5% when sizing tomorrow's close buys
BUY_CUSHION_INR = 30.0  # the engine leaves about Rs 29.5 of cash per buy for the exit fees
S2_STOP_FRAC = 0.03
S2_UNIVERSE_RANK = 200
S2_GAP_SKIP = 0.18  # backtest stand-in for "within 2% of the upper price band"
S2_BAND_MARGIN = 0.02
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
    vol_har: float | None = None  # G2-E1: HAR-RV daily-vol forecast (daily units, like vol_21)
    vol_garch: float | None = None  # G2-E1: GARCH(1,1) daily-vol forecast (10-day vol / sqrt(10))


@dataclass(frozen=True)
class PanelDay:
    """Rows of one decision date; `next_date` is the session on which orders trade."""

    date: date
    next_date: date
    rows: Mapping[int, NameRow]
    breadth_200: float | None = None
    mkt_ret_21: float | None = None
    p_bear: float | None = None  # G2-E4: filtered HMM bear probability of the date (research worlds only)


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
    # G2-C1 options; the defaults reproduce the pre-G2 behaviour exactly
    exit_rank: int | None = None  # rank_decay fires at rank > exit_rank (default 3*k)
    min_hold: int = S1_RANK_DECAY_MIN_HELD  # sessions held before rank_decay / a switch may fire
    extend_rank: int | None = None  # rank limit for an extension at the horizon (default k)
    free_extension: bool = False  # extend on er > 0 and the non-cost filters only (nothing trades)
    rebuy_block: int = 0  # a sid sold within the last `rebuy_block` sessions is not bought again
    switch_hurdle: bool = False  # swap a weak holding for a better candidate when the gain beats the cost
    # G2-E1 / E3 options; the defaults reproduce the pre-E behaviour exactly
    vol_model: str = (
        "vol21"  # volatility used for raw weights, Kelly sizing and the stop: vol21 | har | garch
    )
    sizing: str = "score_over_vol"  # raw weight: (0.5 + score) / vol, or fractional Kelly c * er / sigma_h^2
    kelly_c: float = 0.5  # the Kelly fraction (sizing == "kelly")
    # G2-E4: gross x 0.5 when the filtered bear probability is above 0.6 (default off)
    regime_overlay: bool = False
    min_pos: float = field(init=False)

    def __post_init__(self) -> None:
        if self.vol_model not in VOL_MODELS:
            raise ValueError(f"vol_model must be one of {VOL_MODELS}, got {self.vol_model!r}")
        if self.sizing not in SIZINGS:
            raise ValueError(f"sizing must be one of {SIZINGS}, got {self.sizing!r}")
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
        **overrides: Any,
    ) -> S1Config:
        risk = settings.risk
        if "max_fixed_cost_bps" in overrides:  # a Risk override (G2-C5), not an S1Config field
            risk = risk.model_copy(update={"max_fixed_cost_bps": float(overrides.pop("max_fixed_cost_bps"))})
        return cls(
            h, k_max if k_max is not None else settings.strategies.s1.k_max, risk,
            settings.costs, slippage, overlay, hysteresis, **overrides,
        )  # fmt: skip


@dataclass(frozen=True)
class S2Config:
    k_max: int
    risk: Risk
    costs: Costs
    slippage: SlippageTable
    shorts_require_fno: bool = True
    universe_rank: int = S2_UNIVERSE_RANK
    min_pos: float = field(init=False)

    def __post_init__(self) -> None:
        mp = min_position_for_fixed_cost("intraday", self.costs, self.risk.max_fixed_cost_bps)
        object.__setattr__(self, "min_pos", float(max(mp, self.risk.min_position_inr_floor)))

    @classmethod
    def from_settings(
        cls, settings: Settings, slippage: SlippageTable, *, k_max: int | None = None
    ) -> S2Config:
        s2 = settings.strategies.s2
        return cls(
            k_max if k_max is not None else s2.k_max, settings.risk, settings.costs, slippage,
            s2.shorts_require_fno,
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
    recent_sells: Mapping[int, int] = field(default_factory=dict)  # sid -> sessions since it was sold


@dataclass(frozen=True)
class S1Plan:
    orders: list[Order]
    k_raw: int  # breadth from capital alone
    k: int
    gross_target: float
    port_vol: float
    min_pos: float
    cap: float
    exits: dict[int, str]  # sid -> reason (stop, horizon, rank_decay, universe_exit, switch)
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


def _sel_vol(row: NameRow, cfg: S1Config) -> float:
    """The daily volatility the S1 config selects for a name (`vol_21`, or the HAR / GARCH forecast; a
    missing or non-positive forecast falls back to `vol_21`)."""
    v = row.vol_har if cfg.vol_model == "har" else row.vol_garch if cfg.vol_model == "garch" else None
    if v is None or not math.isfinite(v) or v <= 0:
        return row.vol_21
    return float(v)


def _raw_weight(cfg: S1Config, score: float, er: float | None, row: NameRow, mult: float) -> float:
    """A new entry's raw weight before caps and scaling: (0.5 + score) / vol (default) or fractional Kelly
    c * er / sigma_h^2 with sigma_h = vol * sqrt(h) (0 when er <= 0), times the meta multiplier."""
    vol = max(_sel_vol(row, cfg), 1e-4)
    if cfg.sizing == "kelly":
        if er is None or not math.isfinite(er) or er <= 0:
            return 0.0
        return cfg.kelly_c * er / (vol * vol * cfg.h) * mult
    return (0.5 + score) / vol * mult


def _s1_entry_ok(
    sid: int, row: NameRow, er: float | None, value: float, cfg: S1Config, *, cost_hurdle: bool = True
) -> bool:
    """Entry eligibility other than 'not already held / not sold today' (the caller handles those).
    `cost_hurdle=False` (free extension) keeps every filter except the round-trip cost test and asks only
    er > 0."""
    if not row.in_universe or row.stale or row.locked_up or er is None or not math.isfinite(er):
        return False
    if row.close <= 0:
        return False
    if cost_hurdle:
        net = (
            er
            - round_trip_cost_frac("delivery", value, cfg.costs)
            - 2.0 * _slip(row, value, "close", cfg.slippage)
        )
        if net <= 0:
            return False
    elif er <= 0:
        return False
    n_split, n_res = cfg.risk.no_entry_sessions_before_split, cfg.risk.no_entry_sessions_before_results
    if row.days_to_split is not None and 0 <= row.days_to_split <= n_split:
        return False
    return not (row.days_to_results is not None and 0 <= row.days_to_results <= n_res)


def _plan_switches(
    rank: Mapping[int, int],
    ranked: Sequence[int],
    rows: Mapping[int, NameRow],
    book: S1Book,
    exits: dict[int, str],
    extended: list[int],
    er_t: Mapping[int, float],
    k: int,
    target_value: float,
    cfg: S1Config,
    orders: list[Order],
    d1: date,
) -> list[int]:
    """With no free slot, swap weak holdings for better candidates while the expected gain beats the round
    trip plus two closing slippages. Adds `switch` exits (SELL at the next open) to `exits`/`orders` and
    returns the candidates to enter (best er first). At most max(1, k // 4) swaps a day."""
    remaining = [s for s in book.holdings if s not in exits]
    if k - len(remaining) > 0:
        return []  # free slots exist: ordinary entries handle them
    ok_hold = [
        s for s in remaining
        if book.holdings[s].held >= cfg.min_hold and s in rows
        and (e := er_t.get(s)) is not None and math.isfinite(e)
    ]  # fmt: skip
    ok_hold.sort(key=lambda s: (er_t[s], s))
    cands = [
        s for s in ranked
        if s not in book.holdings and s not in exits
        and not (cfg.rebuy_block > 0 and book.recent_sells.get(s, cfg.rebuy_block + 1) <= cfg.rebuy_block)
        and _s1_entry_ok(s, rows[s], er_t.get(s), target_value, cfg)
    ]  # fmt: skip
    cands.sort(key=lambda s: (-er_t[s], rank[s]))
    rt = round_trip_cost_frac("delivery", target_value, cfg.costs)
    chosen: list[int] = []
    for _ in range(max(1, k // 4)):
        if not ok_hold or not cands:
            break
        weak, cand = ok_hold[0], cands[0]
        if er_t[cand] - er_t[weak] <= rt + 2.0 * _slip(rows[cand], target_value, "close", cfg.slippage):
            break
        ok_hold.pop(0)
        cands.pop(0)
        exits[weak] = "switch"
        hd = book.holdings[weak]
        orders.append(
            Order(weak, "SELL", "delivery", "open", hd.qty, trade_date=d1, priority=0.0, reason="switch")
        )
        chosen.append(cand)
    if chosen:
        extended[:] = [s for s in extended if s not in exits]
    return chosen


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
    gross_scale: float = 1.0,
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
    if cfg.regime_overlay and panel_t.p_bear is not None and panel_t.p_bear > REGIME_BEAR_P:
        gt *= REGIME_FACTOR
    gt *= gross_scale  # live decay scale (G2-G4); 1.0 in every backtest
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
                exit_thr = 0 if k == 0 else (
                    S1_RANK_DECAY_MULT * k if cfg.exit_rank is None else cfg.exit_rank
                )  # fmt: skip
                ext_thr = k if cfg.extend_rank is None else cfg.extend_rank
                if hd.held >= cfg.min_hold and r > exit_thr:
                    reason = "rank_decay"
                elif hd.held >= cfg.h - 1:
                    if (
                        k > 0 and r <= ext_thr
                        and _s1_entry_ok(sid, row, er_t.get(sid), target_value, cfg,
                                         cost_hurdle=not cfg.free_extension)
                    ):  # fmt: skip
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

    forced: list[int] = []  # switch candidates, entered first
    if cfg.switch_hurdle and allow_entries and k > 0 and scores_known:
        forced = _plan_switches(
            rank, ranked, rows, book, exits, extended, er_t, k, target_value, cfg, orders, d1
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
        for sid in forced + [s for s in ranked if s not in forced]:
            if len(chosen) >= slots:
                break
            if sid in book.holdings or sid in exits:
                continue
            if cfg.rebuy_block > 0 and book.recent_sells.get(sid, cfg.rebuy_block + 1) <= cfg.rebuy_block:
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
            if cfg.vol_model != "vol21":
                feat["vol_sel"] = float(_sel_vol(row, cfg))  # the stop's vol_at_entry follows the model
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
        raw = {s: _raw_weight(cfg, scores_t[s], er_t.get(s), rows[s], mult[s]) for s in chosen}
        room = max(gt - existing_value / equity_inr, 0.0)
        if cfg.sizing == "kelly":
            room = min(room, sum(raw.values()))  # fractional Kelly deploys c * er / sigma^2, not all the room
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
        auto_resume_sessions: int | None = None,
        policy_opts: Mapping[str, Any] | None = None,
    ) -> None:
        if auto_resume_sessions is not None:
            if auto_resume_sessions < 1:
                raise ValueError("auto_resume_sessions must be >= 1")
            if state is not None:
                raise ValueError("auto-resume is backtest-only: a policy with a live state needs `resume`")
        km = k_max if k_max is not None else settings.strategies.s1.k_max
        h = horizon_of(getattr(spec, "target", spec))
        opts = dict(policy_opts or {})
        risk = settings.risk
        if "max_fixed_cost_bps" in opts:  # G2-C5: a risk override, not an S1Config field
            risk = risk.model_copy(update={"max_fixed_cost_bps": float(opts.pop("max_fixed_cost_bps"))})
        cfg = S1Config(h, km, risk, costs, slippage, overlay, hysteresis, **opts)
        self.cfg, self.source, self.access, self.meta = cfg, score_source, panel_access, meta
        self.state, self.book_name, self._entries_allowed = state, book_name, entries_allowed
        self._peak = 0.0
        self._stopped = False
        self._was_stopped = False
        self.auto_resume_sessions = auto_resume_sessions  # backtest only (never with a state)
        self._stop_idx: int | None = None  # session index at which the stop fired (auto-resume only)
        self._anchor: dict[int, date] = {}
        self._vol_at_entry: dict[int, float] = {}
        self._pending_vol: dict[int, float] = {}
        self._sold_idx: dict[int, int] = {}  # sid -> session index at which its disappearance was seen
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
        if stopped and self.auto_resume_sessions is not None and self._stop_idx is not None:
            flat = not any(p.product == "delivery" and p.qty > 0 for p in book.positions.values())
            if flat and cal.session_index(t) - self._stop_idx >= self.auto_resume_sessions:
                self._write_stop(False)  # backtest-only stand-in for `stocksense resume`
                self._stop_idx = None
                stopped = False
        if self._was_stopped and not stopped:  # `stocksense resume` cleared the flag: re-base the peak
            self._peak = capital_inr
        self._peak = max(self._peak, capital_inr)
        dd = 1.0 - capital_inr / self._peak if self._peak > 0 else 0.0
        if dd >= self.cfg.risk.book_dd_stop and not stopped:
            self._write_stop(True)
            self._stop_idx = cal.session_index(t)
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
        t_idx = cal.session_index(t)
        for sid in [s for s in self._anchor if s not in holdings]:
            self._anchor.pop(sid, None)
            self._vol_at_entry.pop(sid, None)
            self._sold_idx[sid] = t_idx  # the sell filled at today's open: 0 sessions ago
        keep = self.cfg.rebuy_block
        self._sold_idx = {s: i for s, i in self._sold_idx.items() if t_idx - i <= keep}
        recent = {s: t_idx - i for s, i in self._sold_idx.items()}

        panel_t = self.access.day(t, d1)
        sc = self.source.scores(t)
        allow = self._entries_allowed(t) if self._entries_allowed is not None else True
        sbook = S1Book(holdings, book.cash - book.reserved, dd, stopped, recent)
        plan = plan_s1(
            t, sc.score if sc else {}, sc.er if sc else {}, sbook, capital_inr, panel_t, self.cfg,
            disagreement=sc.disagreement if sc else None, meta=self.meta,
            allow_entries=allow and sc is not None,
        )  # fmt: skip
        for sid in plan.extended:
            self._anchor[sid] = d1
        for e in plan.entries:
            self._pending_vol[e["sid"]] = e.get("vol_sel", e["vol_21"])
        self.entry_log.extend(plan.entries)
        self.plans[t] = plan
        return plan.orders


# ---- S2: pure planning -------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuctionRow:
    """One name before the open of d: indicative/official open, gap, rank, previous-close vol, F&O flag."""

    open: float
    gap: float
    turnover_rank: int
    med_turn_20: float
    vol_21: float
    in_fno: bool = False
    upper_circuit: float | None = None


@dataclass(frozen=True)
class S2Plan:
    orders: list[Order]
    n: int
    budget: float
    min_pos: float
    n_long_qualify: int
    n_short_qualify: int
    n_long: int
    n_short: int
    note: str = ""


def intraday_stop_hit(position: Position | Any, ltp: float, stop: float = S2_STOP_FRAC) -> bool:
    """True when a position (qty > 0 long, qty < 0 short; `entry_price`) is `stop` or more adverse at `ltp`.
    The hard 3% stop is live-only: the daemon calls this and covers/sells at once; a backtest has no
    intraday path."""
    if position.qty == 0 or position.entry_price <= 0:
        return False
    move = (ltp / position.entry_price - 1.0) * (1.0 if position.qty > 0 else -1.0)
    return move <= -stop + 1e-12


def _split_sides(n: int, n_long: int, n_short: int) -> tuple[int, int]:
    """Split n slots by how many names qualify per side, at most ceil(n/2) per side unless the other side
    is empty; ties favour longs."""
    if n <= 0:
        return 0, 0
    if n_short == 0:
        return min(n, n_long), 0
    if n_long == 0:
        return 0, min(n, n_short)
    q = n_long + n_short
    half = -(-n // 2)
    nl = min(int(math.floor(n * n_long / q + 0.5)), half, n_long)
    ns = min(n - nl, half, n_short)
    nl = min(n - ns, half, n_long)
    return nl, ns


def plan_s2(
    d: date,
    scores_d: Mapping[int, float],
    er_d: Mapping[int, float],
    auction: Mapping[int, AuctionRow],
    free_cash: float,
    equity_inr: float,
    cfg: S2Config,
) -> S2Plan:
    """The auction book for session `d`, decided before the open (PLAN T9.2 steps 1-6)."""
    risk, min_pos = cfg.risk, cfg.min_pos
    budget = max(free_cash, 0.0) * risk.s2_free_cash_frac
    n = min(cfg.k_max, _floor(budget / min_pos))
    if n <= 0:
        return S2Plan([], 0, budget, min_pos, 0, 0, 0, 0, "budget below one minimum position")
    planned = budget / n
    longs: list[tuple[float, int]] = []
    shorts: list[tuple[float, int]] = []
    for sid, a in auction.items():
        if a.turnover_rank > cfg.universe_rank or sid not in scores_d or a.open <= 0:
            continue
        er = er_d.get(sid)
        if er is None or not math.isfinite(er):
            continue
        row = NameRow(a.open, a.vol_21, a.turnover_rank, a.med_turn_20)
        c = round_trip_cost_frac("intraday", planned, cfg.costs) + 2.0 * _slip(
            row, planned, "open", cfg.slippage
        )
        if er - c > 0:
            longs.append((er, sid))
        if -er - c > 0 and (a.in_fno or not cfg.shorts_require_fno):
            near_band = a.upper_circuit is not None and a.open >= (1.0 - S2_BAND_MARGIN) * a.upper_circuit
            if not near_band and a.gap < S2_GAP_SKIP:
                shorts.append((-er, sid))
    longs.sort(key=lambda x: (-x[0], x[1]))
    shorts.sort(key=lambda x: (-x[0], x[1]))
    nl, ns = _split_sides(n, len(longs), len(shorts))
    pick = [(s, "BUY", "s2_long", e) for e, s in longs[:nl]] + [
        (s, "SELL", "s2_short", e) for e, s in shorts[:ns]
    ]
    if not pick:
        return S2Plan([], n, budget, min_pos, len(longs), len(shorts), 0, 0, "no stock clears costs today")
    cap_value = max(risk.max_name_weight * equity_inr, min_pos)
    while pick:
        lockup = sum(1.2 if side == "SELL" else 1.0 for _, side, _, _ in pick)  # a short locks 1.2x its value
        value = min(budget / lockup, cap_value)
        if value >= min_pos - 1e-6 or len(pick) == 1:
            break
        pick.pop()  # drop the weakest pick until every position is at least one minimum position
    orders = []
    for sid, side, why, e in pick:
        qty = _floor(value / auction[sid].open)
        if qty > 0:
            orders.append(Order(sid, side, "intraday", "open", qty, trade_date=d, priority=e, reason=why))  # type: ignore[arg-type]
    return S2Plan(
        orders, n, budget, min_pos, len(longs), len(shorts),
        sum(o.side == "BUY" for o in orders), sum(o.side == "SELL" for o in orders),
    )  # fmt: skip


def build_s2_orders(*args: Any, **kw: Any) -> list[Order]:
    """The orders of `plan_s2` (intraday, at the open of `d`; the engine/daemon squares off at 15:18)."""
    return plan_s2(*args, **kw).orders


class S2Policy:
    """Implements `decide_auction(d, book, equity_inr, auction_view)` for the open-to-close auction book."""

    def __init__(
        self,
        settings: Settings,
        score_source: ScoreSource,
        panel_access: PanelAccess,
        slippage: SlippageTable,
        *,
        k_max: int | None = None,
    ) -> None:
        self.cfg = S2Config.from_settings(settings, slippage, k_max=k_max)
        self.source, self.access = score_source, panel_access
        self.plans: dict[date, S2Plan] = {}

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        return []  # S2 trades only through decide_auction

    def decide_auction(self, d: date, book: BookState, equity_inr: float, view: AuctionView) -> list[Order]:
        sc = self.source.scores(d)
        if sc is None:
            return []
        try:
            prev = book.calendar.prev_session(d)
        except ValueError:  # the first session of the calendar has no previous close
            return []
        rows = self.access.day(prev, d).rows
        auction: dict[int, AuctionRow] = {}
        for i, sid in enumerate(view.sids.tolist()):
            r = rows.get(sid)
            if r is None:
                continue
            auction[sid] = AuctionRow(
                float(view.open[i]), float(view.gap[i]), int(view.turnover_rank[i]),
                float(view.med_turn_20[i]), r.vol_21, r.in_fno, r.upper_circuit,
            )  # fmt: skip
        plan = plan_s2(d, sc.score, sc.er, auction, book.cash - book.reserved, equity_inr, self.cfg)
        self.plans[d] = plan
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
        u500: Mapping[date, set[int]] | None = None,
        universe_flag: bool = False,
        volfc: tuple[np.ndarray, np.ndarray] | None = None,
        regime: Mapping[date, float] | None = None,
    ) -> None:
        if universe_flag and u500 is None:
            raise ValueError("universe_flag=True needs the u500 map (date -> set of sids)")
        self.u500, self.universe_flag = u500 or {}, universe_flag
        if volfc is not None and (len(volfc[0]) != len(panel.sid) or len(volfc[1]) != len(panel.sid)):
            raise ValueError("volfc arrays must be aligned with the panel rows")
        self.regime = regime or {}  # date -> filtered bear probability (G2-E4); empty = no overlay data
        self.volfc = volfc  # (har, garch) daily-vol forecasts per panel row (G2-E1); None = vol_21 only
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

    def _days_since(self, t: date, events: Sequence[date] | None) -> int:
        """Sessions from the latest event on or before t to t (0 on the day itself); 90 when none or older."""
        if not events:
            return DAYS_SINCE_RESULT_CAP
        i = bisect_right(list(events), t)
        if i == 0:
            return DAYS_SINCE_RESULT_CAP
        ev = events[i - 1]
        if (t - ev).days > 2 * DAYS_SINCE_RESULT_CAP:
            return DAYS_SINCE_RESULT_CAP
        return min(DAYS_SINCE_RESULT_CAP, len(self.cal.sessions(ev + timedelta(days=1), t)))

    def _fc(self, which: int, j: int) -> float | None:
        if self.volfc is None:
            return None
        v = float(self.volfc[which][j])
        return v if math.isfinite(v) else None

    def day(self, t: date, next_date: date) -> PanelDay:
        p = self.p
        sl = p.day(t)
        rows: dict[int, NameRow] = {}
        members = self.u500.get(t, set()) if self.universe_flag else None
        if sl is not None:
            for j in range(*sl):
                sid = int(p.sid[j])
                dates, rets = self._hist[sid]
                pos = bisect_right(dates, t)
                r = rets[max(0, pos - COV_WINDOW) : pos]
                rows[sid] = NameRow(
                    float(p.close[j]), float(p.vol_21[j]), int(p.rank[j]), float(p.med_turn_20[j]),
                    self.industry.get(sid, "UNKNOWN"),
                    (sid in members) if members is not None else int(p.rank[j]) <= self.universe_rank,
                    (t, sid) in self.stale, bool(p.locked_up[j]), sid in self.fno,
                    self._days_to(t, self.split_dates.get(sid)), self._days_to(t, self.result_dates.get(sid)),
                    self._days_since(t, self.result_dates.get(sid)), r,
                    vol_har=self._fc(0, j), vol_garch=self._fc(1, j),
                )  # fmt: skip
        b, m = self.market.get(t, (None, None))
        return PanelDay(t, next_date, rows, b, m, self.regime.get(t))
