"""Append-only paper ledger (T10.1). Book arithmetic is in weights (fractions of equity); NAV starts at 100.

Nothing here deletes a row. The only updates are `orders.status/cancel_reason` and, through
`add_correction` / `mark_reconciled`, `fills.price/qty/official_ref/reconciled` and the matching `fees` row.
Positions and NAV are derived by replaying fills, corporate actions, dividends and stored mark prices.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import polars as pl

from stocksense.core import ids
from stocksense.core.clock import Clock
from stocksense.core.state import State
from stocksense.costs.india import CostBreakdown

BOOKS = ("main", "auction", "auction_strict", "events")
START_NAV = 100.0
_EPS = 1e-9


def is_valid_book(book: str) -> bool:
    return book in BOOKS or book.startswith("s1-shadow-")


@dataclass(frozen=True)
class NewOrder:
    sid: int
    symbol: str
    side: str
    product: str
    when_kind: str
    trade_date: date
    qty: int
    ref_price: float
    reason: str = ""


@dataclass(frozen=True)
class OrderRow:
    order_id: str
    decision_id: str
    book: str
    sid: int
    symbol: str
    side: str
    product: str
    when_kind: str
    trade_date: date
    qty: int
    ref_price: float
    status: str
    reason: str
    cancel_reason: str | None


@dataclass(frozen=True)
class FillRow:
    fill_id: str
    order_id: str
    ts: str
    price: float
    qty: int
    slip_assumed: float
    quote_ltp: float | None
    source: str
    official_ref: float | None
    reconciled: bool
    book: str
    sid: int
    symbol: str
    side: str
    product: str
    when_kind: str
    trade_date: date
    order_qty: int


@dataclass(frozen=True)
class Position:
    book: str
    sid: int
    symbol: str
    product: str
    qty: float  # signed shares; negative = short
    weight: float  # signed fraction of current book equity
    last_price: float


@dataclass(frozen=True)
class MarkResult:
    book: str
    date: date
    nav: float
    day_ret: float
    cash_frac: float
    gross: float
    n_pos: int


@dataclass
class _Pos:
    symbol: str
    product: str
    qty: float
    w: float
    p: float


_ORDER_COLS = (
    "order_id, decision_id, book, sid, symbol, side, product, when_kind, trade_date, qty, ref_price, "
    "status, reason, cancel_reason"
)


def _order(r: tuple) -> OrderRow:
    return OrderRow(
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        r[5],
        r[6],
        r[7],
        date.fromisoformat(r[8]),
        r[9],
        r[10],
        r[11],
        r[12] or "",
        r[13],
    )


class Ledger:
    def __init__(self, state: State, clock: Clock | None = None) -> None:
        self.state = state
        self._clock: Clock = clock if clock is not None else state._clock

    # ---- helpers --------------------------------------------------------------------------------

    def _now(self) -> str:
        return self._clock.now().isoformat(timespec="seconds")

    def _con(self) -> sqlite3.Connection:
        return self.state.connect()

    # ---- decisions and orders -------------------------------------------------------------------

    def record_decision(
        self,
        book: str,
        strategy: str,
        trade_date: date,
        capital_ref_inr: float,
        capital_stale: bool,
        risk: object,
        payload: object,
    ) -> str:
        if not is_valid_book(book):
            raise ValueError(f"unknown book {book!r}")
        did = ids.decision_id(book, strategy, trade_date)
        con = self._con()
        try:
            con.execute(
                "INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    did,
                    book,
                    strategy,
                    trade_date.isoformat(),
                    self._now(),
                    capital_ref_inr,
                    int(capital_stale),
                    ids.canonical_json(risk),
                    ids.canonical_json(payload),
                ),
            )
        finally:
            con.close()
        return did

    def decision_capital(self, book: str, d: date) -> float | None:
        """Capital reference of the newest decision of `book` on or before d."""
        con = self._con()
        try:
            r = con.execute(
                "SELECT capital_ref_inr FROM decisions WHERE book=? AND trade_date<=? "
                "ORDER BY trade_date DESC, created_ts DESC LIMIT 1",
                (book, d.isoformat()),
            ).fetchone()
        finally:
            con.close()
        return None if r is None else float(r[0])

    def add_orders(self, decision_id: str, book: str, orders: list[NewOrder]) -> int:
        """Insert orders (ignoring ids already present); returns the number newly inserted."""
        n = 0
        con = self._con()
        try:
            for o in orders:
                oid = ids.order_id(decision_id, o.sid, o.side, o.when_kind)
                cur = con.execute(
                    "INSERT OR IGNORE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        oid,
                        decision_id,
                        book,
                        o.sid,
                        o.symbol,
                        o.side,
                        o.product,
                        o.when_kind,
                        o.trade_date.isoformat(),
                        int(o.qty),
                        o.ref_price,
                        self._now(),
                        "pending",
                        o.reason,
                        None,
                    ),
                )
                n += cur.rowcount
        finally:
            con.close()
        return n

    def pending_orders(
        self, trade_date: date, when_kind: str, books: list[str] | None = None
    ) -> list[OrderRow]:
        rows = self._orders(
            "trade_date=? AND when_kind=? AND status='pending'", (trade_date.isoformat(), when_kind)
        )
        return [o for o in rows if books is None or o.book in books]

    def orders_for_date(self, trade_date: date) -> list[OrderRow]:
        return self._orders("trade_date=?", (trade_date.isoformat(),))

    def pending_before(self, d: date) -> list[OrderRow]:
        return self._orders("trade_date<? AND status='pending'", (d.isoformat(),))

    def _orders(self, where: str, args: tuple) -> list[OrderRow]:
        con = self._con()
        try:
            rows = con.execute(
                f"SELECT {_ORDER_COLS} FROM orders WHERE {where} ORDER BY order_id", args
            ).fetchall()
        finally:
            con.close()
        return [_order(r) for r in rows]

    def books(self) -> list[str]:
        con = self._con()
        try:
            return [r[0] for r in con.execute("SELECT DISTINCT book FROM decisions ORDER BY book")]
        finally:
            con.close()

    def cancel_pending(self, order_id: str, reason: str) -> None:
        con = self._con()
        try:
            con.execute(
                "UPDATE orders SET status='cancelled', cancel_reason=? WHERE order_id=? AND status='pending'",
                (reason, order_id),
            )
        finally:
            con.close()

    # ---- fills ----------------------------------------------------------------------------------

    def _order_ctx(self, con: sqlite3.Connection, order_id: str) -> tuple[str, str, float]:
        r = con.execute(
            "SELECT o.side, o.book, d.capital_ref_inr FROM orders o "
            "JOIN decisions d ON d.decision_id=o.decision_id WHERE o.order_id=?",
            (order_id,),
        ).fetchone()
        if r is None:
            raise KeyError(f"order {order_id} or its decision not found")
        return r[0], r[1], float(r[2])

    def fill_order(
        self,
        order_id: str,
        ts: str,
        price: float,
        qty: int,
        slip_assumed: float,
        quote_bid: float | None,
        quote_ask: float | None,
        quote_ltp: float | None,
        source: str,
        fees: CostBreakdown,
    ) -> str:
        """Record the (single) fill of an order; a second call returns the existing fill id."""
        fid = ids._sha1_16(f"{order_id}|fill")
        con = self._con()
        try:
            side, book, cap = self._order_ctx(con, order_id)
            cur = con.execute(
                "INSERT OR IGNORE INTO fills VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",
                (
                    fid,
                    order_id,
                    ts,
                    price,
                    int(qty),
                    slip_assumed,
                    quote_bid,
                    quote_ask,
                    quote_ltp,
                    source,
                    None,
                ),
            )
            if cur.rowcount == 0:
                return fid
            con.execute(
                "INSERT INTO fees VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    fid,
                    fees.brokerage,
                    fees.stt,
                    fees.exchange,
                    fees.sebi,
                    fees.stamp,
                    fees.gst,
                    fees.dp,
                    fees.total,
                ),
            )
            w = qty * price / cap
            amount = (-w if side == "BUY" else w) - fees.total / cap
            con.execute(
                "INSERT INTO cash_events(ts, book, kind, amount_frac, ref) VALUES(?,?,?,?,?)",
                (ts, book, "trade", amount, fid),
            )
            con.execute("UPDATE orders SET status='filled' WHERE order_id=?", (order_id,))
        finally:
            con.close()
        return fid

    def fills_for_date(self, trade_date: date, reconciled: bool | None = None) -> list[FillRow]:
        q = (
            "SELECT f.fill_id, f.order_id, f.ts, f.price, f.qty, f.slip_assumed, f.quote_ltp, f.source, "
            "f.official_ref, f.reconciled, o.book, o.sid, o.symbol, o.side, o.product, o.when_kind, "
            "o.trade_date, o.qty FROM fills f JOIN orders o ON o.order_id=f.order_id WHERE o.trade_date=?"
        )
        args: list = [trade_date.isoformat()]
        if reconciled is not None:
            q += " AND f.reconciled=?"
            args.append(int(reconciled))
        con = self._con()
        try:
            rows = con.execute(q + " ORDER BY f.fill_id", args).fetchall()
        finally:
            con.close()
        return [FillRow(*r[:9], bool(r[9]), *r[10:16], date.fromisoformat(r[16]), r[17]) for r in rows]

    def add_correction(
        self,
        fill_id: str,
        new_price: float,
        reason: str,
        *,
        official_ref: float | None = None,
        new_qty: int | None = None,
        new_fees: CostBreakdown | None = None,
    ) -> None:
        """Change a fill's price (and optionally quantity, fees), log it in `corrections`, mark reconciled.

        `new_qty=0` reverses the fill: its order is marked cancelled with the text after `:` in `reason`.
        """
        con = self._con()
        try:
            r = con.execute(
                "SELECT f.price, f.qty, f.order_id, fe.total FROM fills f "
                "JOIN fees fe ON fe.fill_id=f.fill_id "
                "WHERE f.fill_id=?",
                (fill_id,),
            ).fetchone()
            if r is None:
                raise KeyError(fill_id)
            old_price, old_qty, order_id, old_fee = r
            side, book, cap = self._order_ctx(con, order_id)
            qty = old_qty if new_qty is None else int(new_qty)
            fee = old_fee if new_fees is None else new_fees.total
            if qty == 0 and new_fees is None:
                fee = 0.0
            sign = -1.0 if side == "BUY" else 1.0
            delta = sign * (qty * new_price - old_qty * old_price) / cap - (fee - old_fee) / cap
            now = self._now()
            con.execute(
                "UPDATE fills SET price=?, qty=?, official_ref=COALESCE(?, official_ref), reconciled=1 "
                "WHERE fill_id=?",
                (new_price, qty, official_ref, fill_id),
            )
            if new_fees is not None or qty == 0:
                f = new_fees or CostBreakdown(0, 0, 0, 0, 0, 0, 0, 0)
                con.execute(
                    "UPDATE fees SET brokerage=?, stt=?, exchange=?, sebi=?, stamp=?, gst=?, dp=?, total=? "
                    "WHERE fill_id=?",
                    (f.brokerage, f.stt, f.exchange, f.sebi, f.stamp, f.gst, f.dp, f.total, fill_id),
                )
            con.execute(
                "INSERT INTO corrections(ts, fill_id, old_price, new_price, reason) VALUES(?,?,?,?,?)",
                (now, fill_id, old_price, new_price, reason),
            )
            con.execute(
                "INSERT INTO cash_events(ts, book, kind, amount_frac, ref) VALUES(?,?,?,?,?)",
                (now, book, "correction", delta, fill_id),
            )
            if qty == 0:
                con.execute(
                    "UPDATE orders SET status='cancelled', cancel_reason=? WHERE order_id=?",
                    (reason.split(":", 1)[-1], order_id),
                )
        finally:
            con.close()

    def mark_reconciled(self, fill_id: str, official_ref: float | None) -> None:
        con = self._con()
        try:
            con.execute(
                "UPDATE fills SET reconciled=1, official_ref=COALESCE(?, official_ref) WHERE fill_id=?",
                (official_ref, fill_id),
            )
        finally:
            con.close()

    # ---- corporate actions ----------------------------------------------------------------------

    def apply_corp_action(self, book: str, sid: int, d: date, factor: float, div_frac: float) -> bool:
        """Record an ex-date adjustment (price x factor, quantity / factor) and a dividend fraction of book
        equity; False (nothing written) if already applied."""
        con = self._con()
        try:
            if con.execute(
                "SELECT 1 FROM corp_adjust WHERE book=? AND sid=? AND date=?", (book, sid, d.isoformat())
            ).fetchone():
                return False
            con.execute(
                "INSERT INTO corp_adjust(book, sid, date, factor, note) VALUES(?,?,?,?,?)",
                (book, sid, d.isoformat(), factor, f"div_frac={div_frac}"),
            )
            if div_frac:
                con.execute(
                    "INSERT INTO cash_events(ts, book, kind, amount_frac, ref) VALUES(?,?,?,?,?)",
                    (f"{d.isoformat()}T00:00:00+05:30", book, "dividend", div_frac, f"{sid}|{d.isoformat()}"),
                )
        finally:
            con.close()
        return True

    # ---- replay ---------------------------------------------------------------------------------

    def _replay(self, book: str, extra_mark: date | None = None):
        con = self._con()
        try:
            fills = con.execute(
                "SELECT substr(f.ts,1,10), f.ts, f.fill_id, f.price, f.qty, o.sid, o.symbol, o.side, "
                "o.product, "
                "d.capital_ref_inr, fe.total FROM fills f JOIN orders o ON o.order_id=f.order_id "
                "JOIN decisions d ON d.decision_id=o.decision_id JOIN fees fe ON fe.fill_id=f.fill_id "
                "WHERE o.book=? AND f.qty>0 ORDER BY f.ts, f.fill_id",
                (book,),
            ).fetchall()
            corp = con.execute(
                "SELECT date, sid, factor FROM corp_adjust WHERE book=? ORDER BY id", (book,)
            ).fetchall()
            divs = con.execute(
                "SELECT ref, amount_frac FROM cash_events WHERE book=? AND kind='dividend' ORDER BY id",
                (book,),
            ).fetchall()
            mark_dates = {r[0] for r in con.execute("SELECT date FROM marks WHERE book=?", (book,))}
            prices: dict[str, dict[int, float]] = defaultdict(dict)
            for d_, sid, px in con.execute("SELECT date, sid, price FROM mark_prices WHERE book=?", (book,)):
                prices[d_][sid] = px
            first = con.execute("SELECT MIN(trade_date) FROM decisions WHERE book=?", (book,)).fetchone()[0]
        finally:
            con.close()
        if extra_mark is not None:
            mark_dates.add(extra_mark.isoformat())
        by_date: dict[str, dict[str, list]] = defaultdict(lambda: {"fill": [], "corp": [], "div": []})
        for r in fills:
            by_date[r[0]]["fill"].append(r)
        for r in corp:
            by_date[r[0]]["corp"].append(r)
        for ref, amt in divs:
            sid_s, d_ = ref.split("|")
            by_date[d_]["div"].append((int(sid_s), amt))
        for d_ in mark_dates:
            by_date[d_]
        pos: dict[int, _Pos] = {}
        nav = START_NAV
        realized = cost = div = 0.0
        marks: dict[str, MarkResult] = {}
        for d_ in sorted(by_date):
            ev = by_date[d_]
            for _, sid, factor in ev["corp"]:
                if sid in pos:
                    pos[sid].qty /= factor
                    pos[sid].p *= factor
            for sid, amt in ev["div"]:
                if sid in pos:
                    div += amt
            for _, _ts, _fid, price, qty, sid, symbol, side, product, cap, fee in ev["fill"]:
                cost += fee / cap
                signed = float(qty if side == "BUY" else -qty)
                p = pos.get(sid)
                if p is not None and signed * p.qty > 0:  # adding
                    wf = signed * price / cap
                    wn = p.w + wf
                    p.p = wn / (p.w / p.p + wf / price)
                    p.w, p.qty = wn, p.qty + signed
                    continue
                if p is not None:  # reducing or flipping
                    closed = min(abs(signed), abs(p.qty))
                    f = closed / abs(p.qty)
                    realized += f * p.w * (price / p.p - 1.0)
                    p.w *= 1.0 - f
                    step = closed if signed > 0 else -closed
                    p.qty += step
                    signed -= step
                    if abs(p.qty) < _EPS:
                        del pos[sid]
                if abs(signed) > _EPS:
                    pos[sid] = _Pos(symbol, product, signed, signed * price / cap, price)
            if d_ in mark_dates:
                px = prices.get(d_, {})
                r = realized - cost + div
                for sid, p in pos.items():
                    if sid in px:
                        r += p.w * (px[sid] / p.p - 1.0)
                if 1.0 + r <= 0:
                    raise ValueError(f"book {book} {d_}: return {r} wipes the book out")
                for sid, p in pos.items():
                    if sid in px:
                        p.w = p.w * (px[sid] / p.p) / (1.0 + r)
                        p.p = px[sid]
                    else:
                        p.w = p.w / (1.0 + r)
                nav *= 1.0 + r
                gross = sum(abs(p.w) for p in pos.values())
                marks[d_] = MarkResult(
                    book,
                    date.fromisoformat(d_),
                    nav,
                    r,
                    1.0 - sum(p.w for p in pos.values()),
                    gross,
                    len(pos),
                )
                realized = cost = div = 0.0
        return pos, marks, first, nav

    # ---- public derived state -------------------------------------------------------------------

    def positions(self, book: str) -> list[Position]:
        pos, *_ = self._replay(book)
        return [Position(book, sid, p.symbol, p.product, p.qty, p.w, p.p) for sid, p in sorted(pos.items())]

    def cash_frac(self, book: str) -> float:
        pos, *_ = self._replay(book)
        return 1.0 - sum(p.w for p in pos.values())

    def mark(
        self, book: str, d: date, prices: dict[int, float], capital_ref_inr: float | None = None
    ) -> MarkResult:
        """Mark the book on day d with {sid: price}. The first mark of a day wins; re-marking returns it."""
        con = self._con()
        try:
            row = con.execute(
                "SELECT nav, day_ret, cash_frac, gross, n_pos FROM marks WHERE book=? AND date=?",
                (book, d.isoformat()),
            ).fetchone()
            if row is not None:
                return MarkResult(book, d, row[0], row[1], row[2], row[3], row[4])
            cap = capital_ref_inr if capital_ref_inr is not None else self.decision_capital(book, d)
            for sid, px in prices.items():
                con.execute(
                    "INSERT OR IGNORE INTO mark_prices VALUES(?,?,?,?)",
                    (book, d.isoformat(), int(sid), float(px)),
                )
        finally:
            con.close()
        _, marks, _, _ = self._replay(book, extra_mark=d)
        m = marks[d.isoformat()]
        con = self._con()
        try:
            con.execute(
                "INSERT OR IGNORE INTO marks VALUES(?,?,?,?,?,?,?,?)",
                (book, d.isoformat(), m.nav, m.cash_frac, m.gross, m.n_pos, cap, m.day_ret),
            )
        finally:
            con.close()
        return m

    def nav_series(self, book: str) -> pl.DataFrame:
        """date, nav, day_ret, cash_frac, gross, n_pos (NAV 100 before the first mark)."""
        con = self._con()
        try:
            first = con.execute("SELECT MIN(trade_date) FROM decisions WHERE book=?", (book,)).fetchone()[0]
            rows = con.execute(
                "SELECT date, nav, day_ret, cash_frac, gross, n_pos FROM marks WHERE book=? ORDER BY date",
                (book,),
            ).fetchall()
        finally:
            con.close()
        if first is not None and (not rows or rows[0][0] > first):
            rows = [(first, START_NAV, 0.0, 1.0, 0.0, 0), *rows]
        schema = {
            "date": pl.Utf8,
            "nav": pl.Float64,
            "day_ret": pl.Float64,
            "cash_frac": pl.Float64,
            "gross": pl.Float64,
            "n_pos": pl.Int64,
        }
        df = pl.DataFrame(rows, schema=schema, orient="row")
        return df.with_columns(pl.col("date").str.to_date())
