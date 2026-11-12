"""Paper broker (T10.3): provisional live fills from quotes, evening reconciliation to the official bar.

Every process_* call is one idempotent pass (the scheduler repeats it every minute); an order is filled at
most once and a blocked order stays pending until its cut-off. The fill rules are `backtest.fills.try_fill`,
fed a bar built from the quote, so the paper books and the backtests share one implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, time

import polars as pl

from stocksense.backtest.fills import DayBar, Fill, FillContext, Order, Reject, try_fill
from stocksense.brokers.quotes import QuoteHub
from stocksense.brokers.types import Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Costs
from stocksense.core.log import get_logger
from stocksense.costs.india import trade_cost
from stocksense.costs.slippage import SlippageTable
from stocksense.paper.ledger import FillRow, Ledger, NewOrder, OrderRow

log = get_logger(__name__)

OPEN_AT = time(9, 15, 10)
STRICT_AT = time(9, 16, 0)
OPEN_WAIT_UNTIL = time(9, 25, 0)
CLOSE_AT = time(15, 22, 0)
CUTOFF_OPEN_NOW = time(15, 20, 0)
CUTOFF_CLOSE = time(15, 28, 0)
CIRCUIT_MARGIN = 0.05
STRICT_BOOKS = ("auction_strict", "events")


@dataclass(frozen=True)
class LiveStats:
    """Liquidity inputs of the slippage model for one security (the daemon supplies real ones)."""

    turnover_rank: int = 500
    vol_21: float = 0.03
    med_turn_20: float = 1e9


@dataclass(frozen=True)
class CorpAction:
    sid: int
    factor: float = 1.0  # price multiplier on the ex-date (2-for-1 split: 0.5)
    dividend_per_share: float = 0.0


@dataclass
class PassResult:
    filled: list[str] = field(default_factory=list)
    cancelled: list[tuple[str, str]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)


@dataclass
class ReconcileReport:
    repriced: list[str] = field(default_factory=list)
    reversed: list[tuple[str, str]] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    no_bar: list[str] = field(default_factory=list)


def _is_strict(book: str) -> bool:
    return book in STRICT_BOOKS


def _zero_penalty(t: SlippageTable) -> SlippageTable:
    return replace(t, rows=[replace(r, open_penalty_bps=0.0, close_penalty_bps=0.0) for r in t.rows])


def _tick(price: float, side: str) -> float:
    ticks = round(price / 0.05, 6)
    return (math.ceil(ticks) if side == "BUY" else math.floor(ticks)) * 0.05


class PaperBroker:
    def __init__(
        self,
        ledger: Ledger,
        hub: QuoteHub,
        costs: Costs,
        slippage: SlippageTable,
        clock: Clock,
        calendar: TradingCalendar,
        stats: Callable[[int, date], LiveStats | None] | None = None,
    ) -> None:
        self.ledger, self.hub, self.costs, self.slippage = ledger, hub, costs, slippage
        self.clock, self.calendar, self._stats = clock, calendar, stats
        self._flat = _zero_penalty(slippage)

    # ---- corporate actions ----------------------------------------------------------------------

    def apply_corp_actions(self, d: date, actions: Iterable[CorpAction]) -> int:
        """Adjust every book's positions for today's ex-date actions (idempotent); returns rows written."""
        n = 0
        for book in self.ledger.books():
            cap = self.ledger.decision_capital(book, d)
            held = {p.sid: p for p in self.ledger.positions(book)}
            for a in actions:
                p = held.get(a.sid)
                if p is None or cap is None:
                    continue
                div_frac = p.qty * a.dividend_per_share / cap
                n += int(self.ledger.apply_corp_action(book, a.sid, d, a.factor, div_frac))
        return n

    # ---- live passes ----------------------------------------------------------------------------

    def process_open(self, d: date, corp_actions: Iterable[CorpAction] = ()) -> PassResult:
        acts = list(corp_actions)
        if acts:
            self.apply_corp_actions(d, acts)
        return self._run(d, "open")

    def process_now(self, d: date | None = None) -> PassResult:
        return self._run(d or self.clock.today(), "now")

    def process_close(self, d: date) -> PassResult:
        return self._run(d, "close")

    def squareoff_intraday(self, d: date) -> PassResult:
        """Close every open intraday position at the live price (orders are created once, then worked)."""
        for book in self.ledger.books():
            cap = self.ledger.decision_capital(book, d)
            for p in self.ledger.positions(book):
                if p.product != "intraday" or cap is None:
                    continue
                dec = self.ledger.record_decision(book, "squareoff", d, cap, True, {}, {})
                side = "SELL" if p.qty > 0 else "BUY"
                self.ledger.add_orders(
                    dec,
                    book,
                    [
                        NewOrder(
                            p.sid, p.symbol, side, "intraday", "now", d, int(round(abs(p.qty))), p.last_price,
                            "squareoff",
                        )
                    ],
                )
        return self._run(d, "now")

    def _stat(self, sid: int, d: date) -> LiveStats:
        return (self._stats(sid, d) if self._stats else None) or LiveStats()

    def _run(self, d: date, kind: str) -> PassResult:
        res = PassResult()
        orders = self.ledger.pending_orders(d, kind)
        if not orders:
            return res
        now = self.clock.now()
        if now.date() > d:
            for o in orders:
                self.ledger.cancel_pending(o.order_id, "expired")
                res.cancelled.append((o.order_id, "expired"))
            return res
        earliest = CLOSE_AT if kind == "close" else OPEN_AT
        if now.date() < d or now.time() < earliest:
            res.pending = [o.order_id for o in orders]
            return res
        held = {p.symbol for b in self.ledger.books() for p in self.ledger.positions(b)}
        quotes = self.hub.get({o.symbol for o in orders}, held=held)
        for o in orders:
            self._work(o, quotes.get(o.symbol), kind, d, now.time(), res)
        return res

    def _cancel(self, o: OrderRow, reason: str, res: PassResult) -> None:
        self.ledger.cancel_pending(o.order_id, reason)
        res.cancelled.append((o.order_id, reason))

    def _work(self, o: OrderRow, q: Quote | None, kind: str, d: date, t: time, res: PassResult) -> None:
        if kind == "close":
            cutoff = CUTOFF_CLOSE
        else:
            cutoff = CUTOFF_CLOSE if o.reason == "squareoff" else CUTOFF_OPEN_NOW
        late = t >= cutoff

        def wait(reason: str) -> None:
            if late:
                self._cancel(o, reason, res)
            else:
                res.pending.append(o.order_id)

        if q is None or not q.ltp or q.ltp <= 0:
            return wait("no_quote")
        strict = _is_strict(o.book)
        slip_when, table, from_ltp = "close", self._flat, True
        if kind == "open":
            if strict:
                if t < STRICT_AT or q.ts.timetz().replace(tzinfo=None) < STRICT_AT:
                    return wait("no_quote_after_0916")
                ref, slip_when = q.ltp, "open"
            elif q.open and q.open > 0:
                ref, slip_when, table, from_ltp = q.open, "open", self.slippage, False
            elif t >= OPEN_WAIT_UNTIL:
                ref, slip_when = q.ltp, "open"
            else:
                res.pending.append(o.order_id)
                return None
        elif kind == "close":
            ref, table = q.ltp, self.slippage
        else:
            ref = q.ltp
        buy = o.side == "BUY"
        if buy and ((q.ask_qty is not None and q.ask_qty == 0) or (
            q.upper_circuit and q.ltp >= q.upper_circuit - CIRCUIT_MARGIN
        )):
            return wait("circuit_blocked_buy")
        if not buy and ((q.bid_qty is not None and q.bid_qty == 0) or (
            q.lower_circuit and q.ltp <= q.lower_circuit + CIRCUIT_MARGIN
        )):
            return wait("circuit_blocked_sell")
        qty = o.qty
        if not buy and o.product == "delivery":
            held = sum(p.qty for p in self.ledger.positions(o.book) if p.sid == o.sid)
            qty = int(min(qty, max(held, 0)))
            if qty <= 0:
                return self._cancel(o, "no_position", res)
        hi = max(q.high or ref, ref)
        lo = min(q.low or ref, ref)
        prev = q.prev_close or ref
        bar = DayBar(ref, hi, lo, ref, prev, 0.0, ref / prev - 1.0)
        st = self._stat(o.sid, d)
        ctx = FillContext(st.turnover_rank, st.vol_21, st.med_turn_20, table, self.costs, 0.0)
        out = try_fill(Order(o.sid, o.side, o.product, slip_when, qty, d), bar, ctx)
        if isinstance(out, Reject):
            return self._cancel(o, out.reason, res)
        price, cost = out.price, out.cost
        if from_ltp:  # never better than the touch
            if buy and q.ask and q.ask > 0:
                price = max(price, _tick(q.ask, "BUY"))
            if not buy and q.bid and q.bid > 0:
                price = min(price, _tick(q.bid, "SELL"))
            if price != out.price:
                cost = trade_cost(o.side, o.product, out.qty * price, self.costs)
        self.ledger.fill_order(
            o.order_id,
            self.clock.now().isoformat(timespec="seconds"),
            price,
            out.qty,
            out.slip,
            q.bid,
            q.ask,
            q.ltp,
            q.source,
            cost,
        )
        res.filled.append(o.order_id)

    # ---- reconciliation -------------------------------------------------------------------------

    @staticmethod
    def _official(panel_rows: pl.DataFrame | Mapping[int, dict]) -> dict[int, dict]:
        rows = panel_rows.to_dicts() if isinstance(panel_rows, pl.DataFrame) else list(panel_rows.values())
        out = {}
        for r in rows:
            out[int(r["sid"])] = r
        return out

    @staticmethod
    def _bar(r: dict) -> DayBar:
        prev = float(r["prev_close"]) * float(r.get("gap_factor") or 1.0)
        o, c = float(r["open"]), float(r["close"])
        return DayBar(
            o, float(r["high"]), float(r["low"]), c, prev, o / prev - 1.0, c / prev - 1.0,
            bool(r.get("flag_one_price")), bool(r.get("flag_locked_up")), bool(r.get("flag_locked_dn")),
        )

    def reconcile(self, d: date, panel_rows: pl.DataFrame | Mapping[int, dict]) -> ReconcileReport:
        """Official re-pricing of the day's un-reconciled fills (needs the official bar of the session)."""
        rep = ReconcileReport()
        official = self._official(panel_rows)
        if not official:
            raise ValueError(f"no official rows for {d}: refusing to reconcile")
        for f in self.ledger.fills_for_date(d, reconciled=False):
            row = official.get(f.sid)
            if f.qty <= 0:
                self.ledger.mark_reconciled(f.fill_id, None)
                continue
            if f.book in STRICT_BOOKS or f.when_kind == "now":
                self._reconcile_keep(f, row, rep)
            else:
                self._reconcile_reprice(f, row, d, rep)
        return rep

    def _reconcile_keep(self, f: FillRow, row: dict | None, rep: ReconcileReport) -> None:
        if row is None:
            self.ledger.mark_reconciled(f.fill_id, None)
            rep.no_bar.append(f.fill_id)
            return
        ref = float(row["close"] if f.when_kind == "close" else row["open"])
        ref = ref if f.when_kind in ("open", "close") else None
        lo, hi = float(row["low"]), float(row["high"])
        if f.price < lo - 1e-9 or f.price > hi + 1e-9:
            price = min(max(f.price, lo), hi)
            fees = trade_cost(f.side, f.product, f.qty * price, self.costs)
            self.ledger.add_correction(
                f.fill_id, price, "clamped_to_official_range", official_ref=ref, new_fees=fees
            )
            rep.clamped.append(f.fill_id)
        else:
            self.ledger.mark_reconciled(f.fill_id, ref)
            rep.kept.append(f.fill_id)

    def _reconcile_reprice(self, f: FillRow, row: dict | None, d: date, rep: ReconcileReport) -> None:
        bar = self._bar(row) if row is not None else None
        st = LiveStats(
            int(row.get("turnover_rank") or 500), float(row.get("vol_21") or 0.03),
            float(row.get("med_turn_20") or 1e9),
        ) if row is not None else self._stat(f.sid, d)
        ctx = FillContext(st.turnover_rank, st.vol_21, st.med_turn_20, self.slippage, self.costs, 0.0)
        ref = None if row is None else float(row["open"] if f.when_kind == "open" else row["close"])
        out: Fill | Reject = try_fill(Order(f.sid, f.side, f.product, f.when_kind, f.order_qty, d), bar, ctx)
        if isinstance(out, Reject):
            self.ledger.add_correction(
                f.fill_id, f.price, f"official_reject:{out.reason}", official_ref=ref, new_qty=0
            )
            rep.reversed.append((f.fill_id, out.reason))
            return
        self.ledger.add_correction(
            f.fill_id, out.price, "official_reprice", official_ref=ref,
            new_qty=out.qty if out.qty != f.qty else None, new_fees=out.cost,
        )
        rep.repriced.append(f.fill_id)
