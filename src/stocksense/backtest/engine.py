from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Protocol

import numpy as np
import polars as pl

from stocksense.backtest.fills import DayBar, Fill, FillContext, Order, Reject, try_fill
from stocksense.backtest.metrics import perf_stats
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Costs, Risk
from stocksense.costs.india import trade_cost
from stocksense.costs.slippage import SlippageTable

DELIST_SESSIONS = 20
DELIST_RECOVERY = 0.8
SHORT_BUFFER = 1.2  # a short locks its proceeds plus 20% (the widest circuit band) so covers cannot overdraw
_DEFAULT_VOL = 0.02
_RETRY_PRIORITY = 1e18

_REQUIRED = (
    "sid", "date", "open", "high", "low", "close", "prev_close", "turnover_inr", "gap_factor", "div_ret",
    "ret", "ret_co", "flag_one_price", "flag_locked_up", "flag_locked_dn",
)  # fmt: skip


# ---- panel access --------------------------------------------------------------------------------------


class PanelData:
    """The panel/daily frame as flat numpy arrays sorted by (date, sid), sliced per date, built once.

    Lookups of a sid inside a date are a binary search. Derived columns: prev_base (previous raw close x
    gap_factor), div_amt (rupees per share), vol_21 (std of the previous 21 returns, causal), med_turn_20
    (median of the previous 20 turnovers, causal) and turnover_rank (ordinal, by med_turn_20, per date)."""

    def __init__(self, df: pl.DataFrame) -> None:
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"panel frame lacks columns: {missing}")
        df = df.sort("date", "sid")
        prev_close = pl.when(pl.col("prev_close") > 0).then(pl.col("prev_close"))
        base = pl.coalesce(prev_close, pl.col("close").shift(1).over("sid"), pl.col("open"))
        df = df.with_columns(base.alias("_base"))
        df = df.with_columns(
            (pl.col("_base") * pl.col("gap_factor")).alias("prev_base"),
            (pl.col("div_ret") * pl.col("_base")).alias("div_amt"),
            pl.col("ret")
            .rolling_std(21, min_samples=5)
            .shift(1)
            .over("sid")
            .fill_null(_DEFAULT_VOL)
            .alias("vol_21"),
            pl.col("turnover_inr")
            .rolling_median(20, min_samples=1)
            .shift(1)
            .over("sid")
            .fill_null(pl.col("turnover_inr"))
            .clip(lower_bound=1.0)
            .alias("med_turn_20"),
        )
        df = df.with_columns(
            pl.col("med_turn_20").rank("ordinal", descending=True).over("date").cast(pl.Int64).alias("rank")
        )
        self.sid = df["sid"].cast(pl.Int64).to_numpy()
        self.date = df["date"].to_list()
        f = {c: df[c].cast(pl.Float64).fill_null(float("nan")).to_numpy() for c in
             ("open", "high", "low", "close", "prev_base", "ret_co", "ret", "gap_factor", "div_amt",
              "vol_21", "med_turn_20")}  # fmt: skip
        self.open, self.high, self.low, self.close = f["open"], f["high"], f["low"], f["close"]
        self.prev_base, self.ret_co, self.ret, self.gap = (
            f["prev_base"],
            f["ret_co"],
            f["ret"],
            f["gap_factor"],
        )
        self.div_amt, self.vol_21, self.med_turn_20 = f["div_amt"], f["vol_21"], f["med_turn_20"]
        self.rank = df["rank"].to_numpy()
        self.one_price = df["flag_one_price"].fill_null(False).to_numpy()
        self.locked_up = df["flag_locked_up"].fill_null(False).to_numpy()
        self.locked_dn = df["flag_locked_dn"].fill_null(False).to_numpy()
        self._slices: dict[date, tuple[int, int]] = {}
        self.last_date: dict[int, date] = {}
        if df.height:
            dates = df["date"]
            bounds = (dates != dates.shift(1)).fill_null(True).arg_true().to_list() + [df.height]
            for a, b in zip(bounds[:-1], bounds[1:], strict=True):
                self._slices[self.date[a]] = (a, b)
            last = df.group_by("sid").agg(pl.col("date").max())
            self.last_date = dict(zip(last["sid"].to_list(), last["date"].to_list(), strict=True))

    @classmethod
    def from_frame(cls, daily: pl.DataFrame, universe: pl.DataFrame | None = None) -> PanelData:
        """Build from a panel/daily-shaped frame; `universe` (date, sid, turnover_rank) overrides ranks."""
        p = cls(daily)
        if universe is not None and universe.height:
            u = pl.DataFrame({"date": p.date, "sid": p.sid}).join(
                universe.select("date", pl.col("sid").cast(pl.Int64), "turnover_rank"),
                on=["date", "sid"],
                how="left",
            )
            r = u["turnover_rank"].to_numpy()
            p.rank = np.where(np.isnan(r.astype(float)), p.rank, r).astype(np.int64)
        return p

    def day(self, d: date) -> tuple[int, int] | None:
        """(lo, hi) row range of session d, or None when no stock has a row."""
        return self._slices.get(d)

    def find(self, d: date, sid: int) -> int:
        """Global row index of (d, sid), or -1."""
        sl = self._slices.get(d)
        if sl is None:
            return -1
        lo, hi = sl
        j = lo + int(np.searchsorted(self.sid[lo:hi], sid))
        return j if j < hi and self.sid[j] == sid else -1

    def bar(self, k: int) -> DayBar:
        return DayBar(
            open=float(self.open[k]),
            high=float(self.high[k]),
            low=float(self.low[k]),
            close=float(self.close[k]),
            prev_base=float(self.prev_base[k]),
            ret_co=float(self.ret_co[k]),
            ret=float(self.ret[k]),
            flag_one_price=bool(self.one_price[k]),
            flag_locked_up=bool(self.locked_up[k]),
            flag_locked_dn=bool(self.locked_dn[k]),
        )

    def close_on(self, sid: int, d: date) -> float | None:
        k = self.find(d, sid)
        return None if k < 0 else float(self.close[k])


@dataclass(frozen=True)
class AuctionView:
    """What a pre-open (S2-style) policy sees on session `date`: the open and its gap per sid."""

    date: date
    sids: np.ndarray
    open: np.ndarray
    prev_base: np.ndarray
    gap: np.ndarray
    med_turn_20: np.ndarray
    turnover_rank: np.ndarray


# ---- state ------------------------------------------------------------------------------------------------


@dataclass
class Position:
    """A held position: long (qty > 0) or an intraday short (qty < 0); prices in current-share terms."""

    sid: int
    qty: int
    product: str
    entry_price: float
    entry_date: date
    entry_idx: int
    entry_cost: float
    last_close: float
    last_seen_idx: int
    reserve: float = 0.0


@dataclass
class BookState:
    """The book a policy sees after a close: cash, positions and price lookups (read only for policies)."""

    date: date
    cash: float
    reserved: float
    positions: dict[int, Position]
    panel: PanelData
    calendar: TradingCalendar
    equity: float

    def next_session(self, d: date | None = None, n: int = 1) -> date:
        """The session n sessions after d (default: the book's current date)."""
        return self.calendar.shift(self.date if d is None else d, n)

    def close(self, sid: int) -> float | None:
        """Today's close of sid, else the last close of a held position, else None."""
        c = self.panel.close_on(sid, self.date)
        if c is not None:
            return c
        p = self.positions.get(sid)
        return None if p is None else p.last_close

    def qty(self, sid: int) -> int:
        p = self.positions.get(sid)
        return 0 if p is None else p.qty


class Policy(Protocol):
    """decide() is called after the close of session t; orders carry trade_date and when. A policy may also
    define decide_auction(d, book, capital_inr, auction_view), called before the open of d."""

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]: ...


@dataclass(frozen=True)
class BacktestResult:
    """daily (one row per session), trades (closed trades), rejects, fills and a summary dict."""

    daily: pl.DataFrame
    trades: pl.DataFrame
    rejects: pl.DataFrame
    summary: dict[str, Any]
    fills: pl.DataFrame = field(default_factory=pl.DataFrame)


# ---- engine ----------------------------------------------------------------------------------------------


class _Engine:
    def __init__(
        self,
        panel: PanelData,
        calendar: TradingCalendar,
        policy: Policy,
        start: date,
        end: date,
        capital: float,
        costs: Costs,
        slippage: SlippageTable,
        risk: Risk,
        debug: bool,
    ) -> None:
        self.p, self.cal, self.policy = panel, calendar, policy
        self.sessions = calendar.sessions(start, end)
        self.sess_idx = {d: i for i, d in enumerate(self.sessions)}
        self.capital, self.costs, self.slip, self.risk, self.debug = capital, costs, slippage, risk, debug
        self.book = BookState(
            self.sessions[0] if self.sessions else start, capital, 0.0, {}, panel, calendar, capital
        )
        self.pending: dict[tuple[date, str], list[Order]] = {}
        self.fills: list[dict] = []
        self.rejects: list[dict] = []
        self.trades: list[dict] = []
        self.sold_today: set[int] = set()
        # cash a new long leaves untouched so that selling a dust position (costs above value) cannot overdraw
        dc = costs.delivery
        self._exit_cushion = (dc.dp_per_scrip_sell_inr + dc.brokerage_min_inr) * (1.0 + dc.gst)
        self.d_costs = self.d_slip = self.d_traded = self.d_div = 0.0

    # ---- order bookkeeping
    def _queue(self, order: Order, today: date | None, strict: bool = False) -> None:
        td = order.trade_date
        if td is None or order.when not in ("open", "close") or order.qty <= 0:
            raise ValueError(f"bad order {order}")
        if td not in self.sess_idx and td <= self.sessions[-1]:
            raise ValueError(f"order trade_date {td} is not a session")
        if today is not None and (td <= today if strict else td < today):
            raise ValueError(f"order trade_date {td} is not after the decision date {today}")
        self.pending.setdefault((td, order.when), []).append(order)

    def _next_opportunity(self, order: Order, d: date, i: int) -> Order | None:
        if order.when == "open":
            return replace(order, trade_date=d, when="close", priority=_RETRY_PRIORITY)
        if i + 1 >= len(self.sessions):
            return None
        return replace(order, trade_date=self.sessions[i + 1], when="open", priority=_RETRY_PRIORITY)

    # ---- execution
    def _reject(self, order: Order, d: date, reason: str) -> None:
        self.rejects.append(
            {
                "date": d, "sid": order.sid, "side": order.side, "product": order.product,
                "when": order.when, "qty": order.qty, "reason": reason,
            }
        )  # fmt: skip

    def _ctx(self, k: int, sid: int) -> FillContext:
        p = self.p
        return FillContext(
            turnover_rank=int(p.rank[k]),
            vol_21=float(p.vol_21[k]),
            med_turn_20=float(p.med_turn_20[k]),
            slippage=self.slip,
            costs=self.costs,
            capital=self.capital,
            max_adv_participation=self.risk.max_adv_participation,
            first_sell_of_scrip_today=sid not in self.sold_today,
        )

    def _execute(self, order: Order, d: date, i: int, squareoff: bool = False) -> None:
        book, p = self.book, self.p
        pos = book.positions.get(order.sid)
        buy = order.side == "BUY"
        if pos is not None and pos.product != order.product:
            self._reject(order, d, "product_conflict")
            return
        closing = pos is not None and ((not buy and pos.qty > 0) or (buy and pos.qty < 0))
        qty = order.qty
        if closing:
            assert pos is not None
            qty = min(qty, abs(pos.qty))
        elif not buy and order.product == "delivery":
            self._reject(order, d, "no_position")
            return
        order = replace(order, qty=qty)
        k = p.find(d, order.sid)
        bar = p.bar(k) if k >= 0 else None
        ctx = self._ctx(k, order.sid) if k >= 0 else None
        if bar is None or ctx is None:
            res: Fill | Reject = Reject(order, "no_trade")
        else:
            res = try_fill(order, bar, ctx)
        if isinstance(res, Reject):
            self._reject(order, d, res.reason)
            if closing and not buy and order.product == "delivery":
                retry = self._next_opportunity(order, d, i)
                if retry is not None:
                    self._queue(retry, None)
            return
        fill = res
        if not closing:
            assert bar is not None and ctx is not None
            fill = self._scale_to_cash(order, bar, ctx, fill, buy)
            if fill is None:
                self._reject(order, d, "no_cash")
                return
        if (
            closing
            and not squareoff
            and book.cash + (-fill.value if buy else fill.value) - fill.cost.total < -0.01
        ):
            self._reject(order, d, "cost_exceeds_cash")
            return
        self._apply(fill, order, pos, closing, d, i)
        if closing and not buy and order.product == "delivery" and fill.qty < qty:
            retry = self._next_opportunity(replace(order, qty=qty - fill.qty), d, i)
            if retry is not None:
                self._queue(retry, None)

    def _scale_to_cash(
        self, order: Order, bar: DayBar, ctx: FillContext, fill: Fill, buy: bool
    ) -> Fill | None:
        book = self.book
        avail = book.cash - book.reserved
        for _ in range(15):
            need = (1.0 if buy else SHORT_BUFFER) * fill.value + fill.cost.total
            if buy:
                need += self._exit_cushion
            if need <= avail + 1e-9:
                return fill
            q = min(fill.qty - 1, math.floor(fill.qty * max(avail, 0.0) / need))
            if q <= 0:
                return None
            r = try_fill(replace(order, qty=q), bar, ctx)
            if isinstance(r, Reject):
                return None
            fill = r
        return None

    def _apply(self, fill: Fill, order: Order, pos: Position | None, closing: bool, d: date, i: int) -> None:
        book = self.book
        buy = order.side == "BUY"
        q, price, cost = fill.qty, fill.price, fill.cost.total
        value = fill.value
        book.cash += (-value if buy else value) - cost
        self.d_costs += cost
        self.d_slip += abs(price - fill.ref_price) * q
        self.d_traded += value
        if order.side == "SELL" and order.product == "delivery":
            self.sold_today.add(order.sid)
        self.fills.append(
            {
                "date": d, "sid": order.sid, "side": order.side, "product": order.product,
                "when": order.when, "qty": q, "price": price, "ref_price": fill.ref_price,
                "slip": fill.slip, "cost_inr": cost, "value_inr": value, "reason": order.reason,
            }
        )  # fmt: skip
        if not closing:
            if pos is None:
                sign = 1 if buy else -1
                reserve = 0.0 if buy else SHORT_BUFFER * value
                book.positions[order.sid] = Position(
                    order.sid, sign * q, order.product, price, d, i, cost, price, i, reserve
                )
                book.reserved += reserve
            else:  # adding to a long (BUY) or to a short (SELL)
                sign = 1 if buy else -1
                held = abs(pos.qty)
                pos.entry_price = (pos.entry_price * held + price * q) / (held + q)
                pos.entry_cost += cost
                pos.qty += sign * q
                if not buy:
                    pos.reserve += SHORT_BUFFER * value
                    book.reserved += SHORT_BUFFER * value
            return
        assert pos is not None
        before = abs(pos.qty)
        share = q / before
        alloc = pos.entry_cost * share
        direction = 1 if pos.qty > 0 else -1
        pnl = direction * (price - pos.entry_price) * q - cost - alloc
        basis = pos.entry_price * q
        self.trades.append(
            {
                "sid": pos.sid, "product": pos.product, "direction": "long" if direction > 0 else "short",
                "entry_date": pos.entry_date, "exit_date": d, "qty": q, "entry_price": pos.entry_price,
                "exit_price": price, "net_pnl_inr": pnl, "ret": pnl / basis,
                "hold_sessions": i - pos.entry_idx,
                "cost_frac": (cost + alloc) / basis, "reason": order.reason or "exit",
            }
        )  # fmt: skip
        pos.entry_cost -= alloc
        if pos.reserve:
            rel = pos.reserve * share
            pos.reserve -= rel
            book.reserved -= rel
        pos.qty -= direction * q
        if pos.qty == 0:
            del book.positions[pos.sid]
            book.reserved = max(book.reserved, 0.0) if book.positions else 0.0

    def _run_phase(self, d: date, i: int, when: str) -> None:
        orders = self.pending.pop((d, when), [])
        if not orders:
            return
        sells = [o for o in orders if o.side == "SELL"]
        buys = sorted((o for o in orders if o.side == "BUY"), key=lambda o: -o.priority)
        for o in sells:
            self._execute(o, d, i)
        for o in buys:
            self._execute(o, d, i)

    def _squareoff(self, d: date, i: int) -> None:
        book = self.book
        for sid in sorted(s for s, ps in book.positions.items() if ps.product == "intraday"):
            pos = book.positions[sid]
            side = "SELL" if pos.qty > 0 else "BUY"
            o = Order(sid, side, "intraday", "close", abs(pos.qty), trade_date=d, reason="squareoff")  # type: ignore[arg-type]
            self._execute(o, d, i, squareoff=True)
            pos = book.positions.get(sid)
            if pos is None:
                continue
            k = self.p.find(d, sid)  # forced close of what could not trade normally
            price = float(self.p.close[k]) if k >= 0 else pos.last_close
            qty = abs(pos.qty)
            f = Fill(
                replace(o, qty=qty), qty, price, price, 0.0,
                trade_cost(side, "intraday", qty * price, self.costs, True),  # type: ignore[arg-type]
            )  # fmt: skip
            self._apply(f, replace(o, qty=qty, reason="squareoff_forced"), pos, True, d, i)

    # ---- the daily loop
    def _corporate_actions(self, d: date, i: int) -> None:
        book, p = self.book, self.p
        for sid, pos in list(book.positions.items()):
            k = p.find(d, sid)
            if k < 0 or pos.qty <= 0:
                continue
            g = float(p.gap[k])
            if g != 1.0:
                new = pos.qty / g
                keep = int(round(new)) if abs(new - round(new)) < 1e-6 else math.floor(new)
                book.cash += (new - keep) * float(p.prev_base[k])
                pos.entry_price *= g
                pos.qty = keep
                if keep <= 0:
                    del book.positions[sid]
                    continue
            div = float(p.div_amt[k])
            if div > 0.0 and pos.qty > 0:
                amt = pos.qty * div
                book.cash += amt
                self.d_div += amt

    def _mark(self, d: date, i: int) -> float:
        book, p = self.book, self.p
        total = 0.0
        gross = 0.0
        for sid, pos in list(book.positions.items()):
            k = p.find(d, sid)
            if k >= 0:
                pos.last_close = float(p.close[k])
                pos.last_seen_idx = i
            elif i - pos.last_seen_idx >= DELIST_SESSIONS and d > p.last_date.get(sid, d):
                px = DELIST_RECOVERY * pos.last_close
                pnl = (px - pos.entry_price) * pos.qty - pos.entry_cost
                basis = pos.entry_price * pos.qty
                self.trades.append(
                    {
                        "sid": sid, "product": pos.product, "direction": "long", "entry_date": pos.entry_date,
                        "exit_date": d, "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": px,
                        "net_pnl_inr": pnl, "ret": pnl / basis, "hold_sessions": i - pos.entry_idx,
                        "cost_frac": pos.entry_cost / basis, "reason": "delisted",
                    }
                )  # fmt: skip
                book.cash += pos.qty * px
                del book.positions[sid]
                continue
            total += pos.qty * pos.last_close
            gross += abs(pos.qty) * pos.last_close
        self._gross = gross
        return book.cash + total

    def _check(self, equity: float, close_of_day: bool) -> None:
        book = self.book
        assert book.cash >= -0.01, f"negative cash {book.cash} on {book.date}"
        for pos in book.positions.values():
            assert not (pos.product == "delivery" and pos.qty < 0), f"negative delivery qty {pos}"
            if close_of_day:
                assert pos.product != "intraday", f"intraday position survived the close {pos}"
        again = book.cash + sum(ps.qty * ps.last_close for ps in book.positions.values())
        assert abs(equity - again) < 0.01

    def run(self) -> BacktestResult:
        book, p = self.book, self.p
        rows: list[dict] = []
        prev_equity = self.capital
        for i, d in enumerate(self.sessions):
            book.date = d
            self.sold_today = set()
            self.d_costs = self.d_slip = self.d_traded = self.d_div = 0.0
            self._corporate_actions(d, i)
            auction = getattr(self.policy, "decide_auction", None)
            sl = p.day(d)
            if auction is not None and sl is not None:
                lo, hi = sl
                view = AuctionView(
                    d, p.sid[lo:hi], p.open[lo:hi], p.prev_base[lo:hi],
                    p.open[lo:hi] / p.prev_base[lo:hi] - 1.0, p.med_turn_20[lo:hi], p.rank[lo:hi],
                )  # fmt: skip
                book.equity = prev_equity
                for o in auction(d, book, prev_equity, view):
                    self._queue(o, d)
            self._run_phase(d, i, "open")
            self._run_phase(d, i, "close")
            self._squareoff(d, i)
            equity = self._mark(d, i)
            book.equity = equity
            if self.debug:
                self._check(equity, True)
            rows.append(
                {
                    "date": d, "equity": equity, "nav": equity / self.capital * 100.0,
                    "cash_frac": book.cash / equity if equity else float("nan"),
                    "gross": self._gross / equity if equity else float("nan"), "n_pos": len(book.positions),
                    "turnover": self.d_traded / equity if equity else float("nan"),
                    "costs_frac": self.d_costs / equity if equity else float("nan"),
                    "day_ret": equity / prev_equity - 1.0,
                    "ret_gross": (equity + self.d_costs + self.d_slip) / prev_equity - 1.0,
                    "costs_inr": self.d_costs, "slip_inr": self.d_slip, "dividends_inr": self.d_div,
                    "traded_inr": self.d_traded,
                }
            )  # fmt: skip
            prev_equity = equity
            for o in self.policy.decide(d, book, equity):
                self._queue(o, d, strict=True)
        return self._result(rows)

    def _result(self, rows: list[dict]) -> BacktestResult:
        daily = pl.DataFrame(rows, schema=_DAILY_SCHEMA) if rows else pl.DataFrame(schema=_DAILY_SCHEMA)
        trades = pl.DataFrame(self.trades, schema=_TRADE_SCHEMA)
        rejects = pl.DataFrame(self.rejects, schema=_REJECT_SCHEMA)
        fills = pl.DataFrame(self.fills, schema=_FILL_SCHEMA)
        summary: dict[str, Any] = {"capital_inr": self.capital}
        if daily.height:
            summary.update(perf_stats(daily["day_ret"].to_numpy(), None, dates=daily["date"].to_list()))
            summary.update(
                {
                    "final_nav": float(daily["nav"][-1]),
                    "total_costs_inr": float(daily["costs_inr"].sum()),
                    "total_slip_inr": float(daily["slip_inr"].sum()),
                    "total_dividends_inr": float(daily["dividends_inr"].sum()),
                    "mean_turnover": float(daily["turnover"].mean()),
                    "cost_drag_frac_of_capital": float(
                        (daily["costs_inr"].sum() + daily["slip_inr"].sum()) / self.capital
                    ),
                    "n_fills": fills.height,
                    "n_rejects": rejects.height,
                    "n_trades": trades.height,
                }
            )
        return BacktestResult(daily, trades, rejects, summary, fills)


_DAILY_SCHEMA: dict[str, Any] = {
    "date": pl.Date, "equity": pl.Float64, "nav": pl.Float64, "cash_frac": pl.Float64, "gross": pl.Float64,
    "n_pos": pl.Int64, "turnover": pl.Float64, "costs_frac": pl.Float64, "day_ret": pl.Float64,
    "ret_gross": pl.Float64, "costs_inr": pl.Float64, "slip_inr": pl.Float64, "dividends_inr": pl.Float64,
    "traded_inr": pl.Float64,
}  # fmt: skip
_TRADE_SCHEMA: dict[str, Any] = {
    "sid": pl.Int64, "product": pl.Utf8, "direction": pl.Utf8, "entry_date": pl.Date, "exit_date": pl.Date,
    "qty": pl.Int64, "entry_price": pl.Float64, "exit_price": pl.Float64, "net_pnl_inr": pl.Float64,
    "ret": pl.Float64, "hold_sessions": pl.Int64, "cost_frac": pl.Float64, "reason": pl.Utf8,
}  # fmt: skip
_REJECT_SCHEMA: dict[str, Any] = {
    "date": pl.Date, "sid": pl.Int64, "side": pl.Utf8, "product": pl.Utf8, "when": pl.Utf8, "qty": pl.Int64,
    "reason": pl.Utf8,
}  # fmt: skip
_FILL_SCHEMA: dict[str, Any] = {
    "date": pl.Date, "sid": pl.Int64, "side": pl.Utf8, "product": pl.Utf8, "when": pl.Utf8, "qty": pl.Int64,
    "price": pl.Float64, "ref_price": pl.Float64, "slip": pl.Float64, "cost_inr": pl.Float64,
    "value_inr": pl.Float64, "reason": pl.Utf8,
}  # fmt: skip


def run_backtest(
    panel_access: PanelData,
    calendar: TradingCalendar,
    policy: Policy,
    start: date,
    end: date,
    capital_inr: float,
    costs: Costs,
    slippage: SlippageTable,
    risk: Risk,
    *,
    debug: bool = True,
) -> BacktestResult:
    """Replay `policy` over the sessions in [start, end]; equity in rupees, returns as fractions."""
    return _Engine(
        panel_access, calendar, policy, start, end, capital_inr, costs, slippage, risk, debug
    ).run()


def run_grid(
    panel_access: PanelData,
    calendar: TradingCalendar,
    policy_factory: Callable[[], Policy],
    start: date,
    end: date,
    costs: Costs,
    slippage: SlippageTable,
    risk: Risk,
    capitals: Iterable[float] | None = None,
    *,
    debug: bool = False,
) -> dict[float, BacktestResult]:
    """run_backtest at each capital (default costs.grid.capital_inr), a fresh policy per run."""
    caps = [float(c) for c in (capitals if capitals is not None else costs.grid.capital_inr)]
    return {
        c: run_backtest(
            panel_access, calendar, policy_factory(), start, end, c, costs, slippage, risk, debug=debug
        )
        for c in caps
    }


# ---- labels for the clairvoyant policy -------------------------------------------------------------------


def c2o_forward_labels(daily: pl.DataFrame, sessions: Sequence[date], h: int) -> pl.DataFrame:
    """(date, sid, label): adjusted return from the close of session t+1 to the open of session t+1+h
    (the S1 round trip decided at t). Null rows are dropped."""
    sess = pl.DataFrame({"date": list(sessions), "idx": list(range(len(sessions)))})
    d = daily.join(sess, on="date", how="inner")
    a = d.select("sid", (pl.col("idx") - 1).alias("t"), pl.col("adj_close").alias("c"))
    b = d.select("sid", (pl.col("idx") - 1 - h).alias("t"), pl.col("adj_open").alias("o"))
    j = a.join(b, on=["sid", "t"], how="inner").filter(pl.col("t") >= 0)
    t_date = pl.DataFrame({"t": list(range(len(sessions))), "date": list(sessions)})
    return (
        j.join(t_date, on="t")
        .select("date", "sid", (pl.col("o") / pl.col("c") - 1.0).alias("label"))
        .drop_nulls()
        .sort("date", "sid")
    )


# ---- reference policies ----------------------------------------------------------------------------------


class BuyAndHold:
    """Buy `sids` in equal weight at the open after the first decision and hold (the rest is cash)."""

    def __init__(self, sids: Sequence[int]) -> None:
        self.sids = list(sids)
        self._done: set[int] = set()

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        out: list[Order] = []
        nxt = book.next_session(t)
        per = capital_inr / len(self.sids)
        for sid in self.sids:
            if sid in self._done:
                continue
            px = book.close(sid)
            if px is None or px <= 0:
                continue
            qty = math.floor(per / px)
            if qty > 0:
                out.append(Order(sid, "BUY", "delivery", "open", qty, trade_date=nxt, reason="buy_and_hold"))
            self._done.add(sid)
        return out


class EqualWeightUniverse:
    """Every `rebalance_every` decisions, trade to equal weight in all stocks that have a row (optionally the
    `max_names` most liquid), at the next open; only names off target by more than `band` are traded."""

    def __init__(
        self,
        rebalance_every: int = 21,
        max_names: int | None = None,
        invest_frac: float = 0.98,
        band: float = 0.25,
    ) -> None:
        self.every, self.max_names, self.invest, self.band = rebalance_every, max_names, invest_frac, band
        self._n = 0

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        n = self._n
        self._n += 1
        if n % self.every != 0:
            return []
        sl = book.panel.day(t)
        if sl is None:
            return []
        lo, hi = sl
        pan = book.panel
        sids = pan.sid[lo:hi]
        closes = pan.close[lo:hi]
        if self.max_names is not None and len(sids) > self.max_names:
            keep = np.argsort(pan.rank[lo:hi], kind="stable")[: self.max_names]
            sids, closes = sids[keep], closes[keep]
        nxt = book.next_session(t)
        per = capital_inr * self.invest / max(len(sids), 1)
        target = {int(s): math.floor(per / c) for s, c in zip(sids, closes, strict=True) if c > 0}
        out: list[Order] = []
        for sid, pos in book.positions.items():
            if sid not in target and pos.product == "delivery":
                out.append(
                    Order(sid, "SELL", "delivery", "open", pos.qty, trade_date=nxt, reason="rebalance")
                )
        for sid, tq in target.items():
            held = book.qty(sid)
            diff = tq - held
            if abs(diff) <= self.band * tq and held > 0:
                continue
            if diff > 0:
                out.append(Order(sid, "BUY", "delivery", "open", diff, trade_date=nxt, reason="rebalance"))
            elif diff < 0:
                out.append(Order(sid, "SELL", "delivery", "open", -diff, trade_date=nxt, reason="rebalance"))
        return out


class _SleevePolicy:
    """Every decision date t: buy k names at the close of t+1 with 1/h of equity, sell them at the open of
    session t+1+h. Subclasses choose the names."""

    def __init__(self, k: int, h: int) -> None:
        self.k, self.h = k, h

    def _pick(self, t: date, book: BookState) -> list[int]:
        raise NotImplementedError

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        names = self._pick(t, book)
        if not names:
            return []
        buy_day = book.next_session(t, 1)
        sell_day = book.next_session(t, 1 + self.h)
        per = capital_inr / self.h / len(names)
        out: list[Order] = []
        for sid in names:
            px = book.close(sid)
            if px is None or px <= 0:
                continue
            qty = math.floor(per / px)
            if qty <= 0:
                continue
            out.append(Order(sid, "BUY", "delivery", "close", qty, trade_date=buy_day, reason="sleeve_entry"))
            out.append(Order(sid, "SELL", "delivery", "open", qty, trade_date=sell_day, reason="sleeve_exit"))
        return out


class Clairvoyant(_SleevePolicy):
    """Buys the true top-k by `label_frame` (date, sid, label) at the close, sells at the open h later.

    A deliberately unfair policy used to prove the engine does not lose returns."""

    def __init__(self, label_frame: pl.DataFrame, k: int = 20, h: int = 5) -> None:
        super().__init__(k, h)
        top = (
            label_frame.sort(["date", "label"], descending=[False, True])
            .group_by("date", maintain_order=True)
            .head(k)
        )
        self._top: dict[date, list[int]] = {}
        for d, s in zip(top["date"].to_list(), top["sid"].to_list(), strict=True):
            self._top.setdefault(d, []).append(int(s))

    def _pick(self, t: date, book: BookState) -> list[int]:
        return self._top.get(t, [])


class RandomPolicy(_SleevePolicy):
    """Buys k random names (seeded) trading at t at the close, sells at the open h sessions later."""

    def __init__(self, seed: int, k: int = 20, h: int = 5) -> None:
        super().__init__(k, h)
        self._rng = np.random.default_rng(seed)

    def _pick(self, t: date, book: BookState) -> list[int]:
        sl = book.panel.day(t)
        if sl is None:
            return []
        sids = book.panel.sid[sl[0] : sl[1]]
        if len(sids) == 0:
            return []
        n = min(self.k, len(sids))
        return [int(s) for s in self._rng.choice(sids, size=n, replace=False)]
