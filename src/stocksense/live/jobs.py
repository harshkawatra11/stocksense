"""The daemon's jobs (T10.5): one function per scheduled activity, all `(ctx, trade_date) -> detail`.

A job returns a short detail string; a string starting with `skipped` is recorded as status `skipped`.
`DataNotAvailable` means "not yet, retry"; any other exception is a failure recorded by the daemon.
Jobs that need a model or a policy that does not exist yet are named stubs (`STUB_JOBS`, `STUB_STAGES`)
recording `skipped: not implemented`. Nothing here sends an order to a broker: only the paper ledger.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import httpx
import polars as pl

from stocksense.brokers.quotes import QuoteHub
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import BrokerError, DataNotAvailable, StockSenseError
from stocksense.core.log import configure_logging, get_logger
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data import io
from stocksense.live.notify import Notifier
from stocksense.paper.broker import CorpAction, PaperBroker
from stocksense.paper.ledger import Ledger, NewOrder, OrderRow

log = get_logger(__name__)

NOT_IMPLEMENTED = "skipped: not implemented"
OPEN_WAIT_UNTIL = time(9, 25)
CLOSE_WAIT_UNTIL = time(15, 28)
CAPITAL_FALLBACK_AT = time(9, 3)
S2_BOOKS = ("auction", "auction_strict")
S2_STOP = 0.03
DEFAULT_REQUIRED_FRESH = (
    "bhavcopy_eq",
    "bhavcopy_delivery",
    "fo_bhavcopy",
    "index_daily",
    "corporate_actions",
)
ANNOUNCEMENT_POLL_S = 600
CACHE_KEEP_DAYS = 400

Hook = Callable[..., Any]


class StageFailed(StockSenseError):
    """An evening-pipeline stage failed (the stage rows in job_runs say which)."""


# ---- context ---------------------------------------------------------------------------------


class JobContext:
    """Everything a job touches. Real objects are built lazily; tests pass fakes (and `hooks`)."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        state: State,
        calendar: TradingCalendar,
        *,
        hub: Any = None,
        broker: Any = None,
        instruments: Any = None,
        nse: Any = None,
        notifier: Notifier | None = None,
        steps: list | None = None,
        hooks: dict[str, Hook] | None = None,
        required_fresh: tuple[str, ...] = DEFAULT_REQUIRED_FRESH,
        log_console: bool | None = None,
    ) -> None:
        self.settings, self.clock, self.state, self.calendar = settings, clock, state, calendar
        self.ledger = Ledger(state, clock)
        self._hub, self._broker, self._instruments, self._nse = hub, broker, instruments, nse
        self.notifier = notifier if notifier is not None else Notifier(settings)
        self.steps = steps  # None -> the real catch-up steps
        self.hooks: dict[str, Hook] = dict(hooks or {})
        self.required_fresh = required_fresh
        self.log_console = log_console  # None: leave logging alone (tests)
        self.degraded = False
        self.current_job: str | None = None
        self.last_ok_job: str | None = None

    # lazily built live objects
    @property
    def instruments(self) -> Any:
        if self._instruments is None:
            from stocksense.brokers.instruments import InstrumentMap

            self._instruments = InstrumentMap(self.settings, self.clock)
        return self._instruments

    @property
    def broker(self) -> Any:
        if self._broker is None:
            from stocksense.brokers.angel import AngelClient

            self._broker = AngelClient(self.settings, self.clock)
        return self._broker

    @property
    def hub(self) -> Any:
        if self._hub is None:
            from stocksense.brokers.quotes import AngelQuotes, NseMcpQuotes, UpstoxQuotes

            sec = self.settings.secrets
            sources: list = []
            if sec.angel_api_key and sec.angel_client_id and sec.angel_pin and sec.angel_totp_secret:
                sources.append(AngelQuotes(self.broker, self.instruments))
            if sec.upstox_analytics_token:
                from stocksense.brokers.upstox import UpstoxClient

                sources.append(UpstoxQuotes(UpstoxClient(self.settings, self.clock), self.instruments))
            sources.append(NseMcpQuotes(self.clock))
            self._hub = QuoteHub(sources, self.clock)
        return self._hub

    @property
    def nse(self) -> Any:
        if self._nse is None:
            from stocksense.data.nse_http import NseClient

            self._nse = NseClient(self.clock)
        return self._nse

    def ingest_ctx(self) -> Any:
        from stocksense.data.ingest_common import IngestContext

        return IngestContext(
            settings=self.settings, clock=self.clock, calendar=self.calendar, nse=self.nse, state=self.state
        )

    def paper(self) -> PaperBroker:
        from stocksense.costs.slippage import SlippageTable

        return PaperBroker(
            self.ledger, self.hub, self.settings.costs, SlippageTable.load_latest(self.settings),
            self.clock, self.calendar,
        )  # fmt: skip

    def hook(self, name: str) -> Hook | None:
        return self.hooks.get(name)

    def scrub(self, text: str) -> str:
        """Text with every secret value masked (job detail is stored in SQLite)."""
        for v in sorted(self.settings.secrets.secret_values(), key=len, reverse=True):
            text = text.replace(v, "***")
        return text

    def catch_up(self, only: Iterable[str] | None = None) -> list:
        """Run the ingest catch-up (all steps or the named ones); failed steps come back as results."""
        from stocksense.data.ingest_all import catch_up, default_steps

        steps = self.steps if self.steps is not None else default_steps()
        if only is not None:
            want = set(only)
            steps = [s for s in steps if s[0] in want]
        return catch_up(self.ingest_ctx(), steps)


# ---- helpers ---------------------------------------------------------------------------------


def _capital(ctx: JobContext) -> float:
    """Rupees = available cash + market value of holdings, read from the broker (never a constant)."""
    f = ctx.broker.funds()
    held = sum(h.qty * h.ltp for h in ctx.broker.holdings())
    return float(f.available_cash) + float(held)


def capital_ref(ctx: JobContext) -> tuple[float | None, bool]:
    """(today's capital reference or the last known one, stale flag) from kv."""
    raw = ctx.state.kv_get("capital_ref_inr")
    stale = ctx.state.kv_get("capital_stale") == "1"
    return (float(raw) if raw is not None else None), stale


def active_books(ctx: JobContext, d: date) -> list[str]:
    """Books whose first decision is on or before d (a later book is not marked for an earlier day)."""
    con = ctx.state.connect()
    try:
        rows = con.execute(
            "SELECT book FROM decisions GROUP BY book HAVING MIN(trade_date) <= ? ORDER BY book",
            (d.isoformat(),),
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _watch(ctx: JobContext, d: date) -> tuple[list[str], list[str]]:
    """(symbols to quote, symbols held) = holdings of every book plus today's pending orders."""
    held = sorted({p.symbol for b in ctx.ledger.books() for p in ctx.ledger.positions(b)})
    pend = {o.symbol for o in ctx.ledger.orders_for_date(d) if o.status == "pending"}
    return sorted(set(held) | pend), held


def _set_degraded(ctx: JobContext, d: date, hub: Any) -> None:
    ctx.degraded = bool(getattr(hub, "degraded", False))
    if ctx.degraded:
        key, errors = f"quotes_degraded:{d.isoformat()}", dict(getattr(hub, "last_errors", {}))
        if errors or ctx.state.kv_get(key) is None:  # keep the first error text of the day
            ctx.state.kv_set(key, json.dumps(errors))


def _is_s1_exit(o: OrderRow) -> bool:
    return o.side == "SELL" and o.product == "delivery" and o.book not in S2_BOOKS


def roll_or_cancel(ctx: JobContext, orders: list[OrderRow], target: date, why: str) -> tuple[int, int]:
    """Move pending S1 exits to `target`'s open (cancel + re-issue); cancel everything else with `why`."""
    rolled = cancelled = 0
    for o in orders:
        cap = ctx.ledger.decision_capital(o.book, o.trade_date) or capital_ref(ctx)[0]
        if _is_s1_exit(o) and cap:
            dec = ctx.ledger.record_decision(
                o.book, "roll", target, cap, True, {"from": str(o.trade_date)}, {}
            )
            ctx.ledger.add_orders(
                dec, o.book,
                [NewOrder(o.sid, o.symbol, "SELL", "delivery", "open", target, o.qty, o.ref_price, "rolled")],
            )  # fmt: skip
            ctx.ledger.cancel_pending(o.order_id, "rolled")
            rolled += 1
        else:
            ctx.ledger.cancel_pending(o.order_id, why)
            cancelled += 1
    return rolled, cancelled


def late_open_fills(ctx: JobContext, d: date) -> str:
    """The open window was missed: pending open orders are cancelled `late_start`, S1 exits roll on."""
    orders = ctx.ledger.pending_orders(d, "open")
    r, c = roll_or_cancel(ctx, orders, ctx.calendar.next_session(d), "late_start")
    return f"skipped: late (rolled={r} cancelled={c})"


def roll_missed_orders(ctx: JobContext, today: date) -> tuple[int, int]:
    """Orders still pending from an earlier day (the daemon was off): exits roll to the next open."""
    old = ctx.ledger.pending_before(today)
    if not old:
        return 0, 0
    now = ctx.clock.now()
    can_open = (
        ctx.calendar.is_session(today) and ctx.calendar.is_tradable(today) and now.time() <= OPEN_WAIT_UNTIL
    )
    target = today if can_open else ctx.calendar.next_session(today)
    return roll_or_cancel(ctx, old, target, "missed_session")


# ---- morning jobs ----------------------------------------------------------------------------


def preflight(ctx: JobContext, d: date) -> str:
    """Rebuild the calendar, catch up data, roll stale orders, run missed sessions' evening pipelines."""
    notes: list[str] = []
    try:
        ctx.calendar = TradingCalendar.from_store(ctx.settings, ctx.clock)
        notes.append("calendar")
    except DataNotAvailable:
        notes.append("calendar:kept")
    results = ctx.catch_up()
    bad = [r.dataset for r in results if r.status == "failed"]
    notes.append(f"ingest results={len(results)} failed={len(bad)}")
    r, c = roll_missed_orders(ctx, d)
    if r or c:
        notes.append(f"stale orders rolled={r} cancelled={c}")
    last = ctx.state.kv_get("last_eod_pipeline")
    if last is not None:
        todo = [s for s in ctx.calendar.sessions(date.fromisoformat(last) + timedelta(days=1), d) if s < d]
    else:
        todo = []
    done = 0
    for s in todo:
        res = run_eod_pipeline(ctx, s, catchup=True)
        if any(v == "failed" for v in res.values()):
            notes.append(f"catch-up {s} failed")
            break  # keep the order: a later session needs this one
        done += 1
    if todo:
        notes.append(f"catch-up pipelines={done}/{len(todo)}")
    return "; ".join(notes)


def broker_login(ctx: JobContext, d: date) -> str:
    ctx.broker.login()
    ctx.instruments.refresh()
    return "angel login ok; instruments refreshed"


def overnight_data(ctx: JobContext, d: date) -> str:
    res = ctx.catch_up(only=("macro_misc", "corporate_actions", "announcements"))
    return f"results={len(res)} failed={sum(r.status == 'failed' for r in res)}"


def _read_capital_job(ctx: JobContext, d: date, *, final_at: time | None) -> str:
    try:
        cap, stale = _capital(ctx), False
    except (BrokerError, httpx.HTTPError, OSError) as e:
        prev = ctx.state.kv_get("capital_ref_inr")
        if final_at is None or ctx.clock.now().time() < final_at or prev is None:
            raise
        log.warning("capital_stale", error=type(e).__name__)
        cap, stale = float(prev), True
    ctx.state.kv_set("capital_ref_inr", repr(cap))
    ctx.state.kv_set("capital_stale", "1" if stale else "0")
    ctx.state.kv_set("capital_date", d.isoformat())
    return f"capital read stale={int(stale)}"  # the amount is never written to a log or detail


def capital_am(ctx: JobContext, d: date) -> str:
    return _read_capital_job(ctx, d, final_at=CAPITAL_FALLBACK_AT)


def capital_pm(ctx: JobContext, d: date) -> str:
    out = _read_capital_job(ctx, d, final_at=None)
    for b in ctx.ledger.books():
        ctx.state.kv_set(f"free_cash_frac:{b}", repr(ctx.ledger.cash_frac(b)))
    return out + f" books={len(ctx.ledger.books())}"


def preopen_capture(ctx: JobContext, d: date) -> str:
    fn = ctx.hook("capture_preopen")
    if fn is not None:
        return f"rows={fn(ctx)}"
    from stocksense.data.ingest_misc import capture_preopen

    return f"rows={capture_preopen(ctx.ingest_ctx())}"


def _decide_with_hook(ctx: JobContext, d: date, hook_name: str, strategy: str) -> str:
    fn = ctx.hook(hook_name)
    if fn is None:
        return f"{NOT_IMPLEMENTED} (no {strategy} policy wired)"
    cap, stale = capital_ref(ctx)
    if cap is None:
        raise DataNotAvailable("capital not read yet")
    n = 0
    for book, orders, payload in fn(ctx, d, cap):
        dec = ctx.ledger.record_decision(book, strategy, d, cap, stale, {}, payload)
        n += ctx.ledger.add_orders(dec, book, orders)
    return f"orders={n}"


def s2_decide(ctx: JobContext, d: date) -> str:
    return _decide_with_hook(ctx, d, "s2_orders", "s2")


def s1_entries(ctx: JobContext, d: date) -> str:
    out = _decide_with_hook(ctx, d, "s1_orders", "s1")
    if not out.startswith("skipped"):
        ctx.notifier.send(f"s1_entries {d}: {out}")
    return out


# ---- the market day --------------------------------------------------------------------------


def open_fills(ctx: JobContext, d: date) -> str:
    acts: list[CorpAction] = list((ctx.hook("corp_actions") or (lambda c, x: []))(ctx, d))
    r = ctx.paper().process_open(d, acts)
    _set_degraded(ctx, d, ctx.hub)
    if r.pending and ctx.clock.now().time() < OPEN_WAIT_UNTIL:
        raise DataNotAvailable(f"{len(r.pending)} open orders still pending")
    return f"filled={len(r.filled)} cancelled={len(r.cancelled)} pending={len(r.pending)}"


def close_fills(ctx: JobContext, d: date) -> str:
    r = ctx.paper().process_close(d)
    _set_degraded(ctx, d, ctx.hub)
    if r.pending and ctx.clock.now().time() < CLOSE_WAIT_UNTIL:
        raise DataNotAvailable(f"{len(r.pending)} close orders still pending")
    return f"filled={len(r.filled)} cancelled={len(r.cancelled)} pending={len(r.pending)}"


def s2_squareoff(ctx: JobContext, d: date) -> str:
    r = ctx.paper().squareoff_intraday(d)
    return f"filled={len(r.filled)} cancelled={len(r.cancelled)}"


def s2_hard_stops(ctx: JobContext, d: date, quotes: dict) -> int:
    """Intraday positions 3% adverse to their fill get a `now` exit order (live only)."""
    n = 0
    for book in S2_BOOKS:
        cap = ctx.ledger.decision_capital(book, d)
        if cap is None:
            continue
        for p in ctx.ledger.positions(book):
            q = quotes.get(p.symbol)
            if p.product != "intraday" or q is None or not q.ltp:
                continue
            adverse = (
                q.ltp <= p.last_price * (1 - S2_STOP) if p.qty > 0 else q.ltp >= p.last_price * (1 + S2_STOP)
            )
            if not adverse:
                continue
            dec = ctx.ledger.record_decision(book, "stop", d, cap, True, {}, {})
            side = "SELL" if p.qty > 0 else "BUY"
            qty = int(round(abs(p.qty)))
            exit_order = NewOrder(p.sid, p.symbol, side, "intraday", "now", d, qty, p.last_price, "stop")
            n += ctx.ledger.add_orders(dec, book, [exit_order])
    return n


def market_loop(ctx: JobContext, d: date) -> str:
    """One minute of the session: quotes -> quotes_1m, pending fills, S2 stops, announcement poll."""
    syms, held = _watch(ctx, d)
    parts: list[str] = []
    quotes: dict = {}
    if syms:
        from stocksense.data.ingest_misc import capture_quotes

        quotes = ctx.hub.get(syms, held=held)
        _set_degraded(ctx, d, ctx.hub)
        parts.append(f"quotes={capture_quotes(ctx.ingest_ctx(), list(quotes.values()))}")
        stops = s2_hard_stops(ctx, d, quotes)
        pb = ctx.paper()
        a, b = pb.process_open(d), pb.process_now(d)
        parts.append(f"fills={len(a.filled) + len(b.filled)} stops={stops}")
    now = ctx.clock.now()
    last = ctx.state.kv_get(f"ann_poll:{d.isoformat()}")
    if last is None or (now - datetime.fromisoformat(last)).total_seconds() >= ANNOUNCEMENT_POLL_S:
        ctx.state.kv_set(f"ann_poll:{d.isoformat()}", now.isoformat(timespec="seconds"))
        ctx.catch_up(only=("announcements",))
        parts.append("ann_poll")
    return " ".join(parts) or "idle"


def u500_symbols(ctx: JobContext) -> list[str]:
    """Symbols of the newest u500 members (the spread-capture list)."""
    uni = io.read(ctx.settings, "panel_universe", columns=["date", "sid", "u500"]).collect()
    if uni.is_empty():
        raise DataNotAvailable("panel_universe is empty")
    last = uni["date"].max()
    sids = uni.filter((pl.col("date") == last) & pl.col("u500"))["sid"]
    daily = io.read(ctx.settings, "panel_daily", start=last, end=last, columns=["sid", "symbol"]).collect()
    return sorted(daily.filter(pl.col("sid").is_in(sids.implode()))["symbol"].to_list())


def spread_capture(ctx: JobContext, d: date) -> str:
    from stocksense.data.ingest_misc import capture_spreads

    syms = ctx.hook("u500_symbols")(ctx) if ctx.hook("u500_symbols") else u500_symbols(ctx)
    n = capture_spreads(ctx.ingest_ctx(), syms, lambda s: list(ctx.hub.get(s).values()))
    _set_degraded(ctx, d, ctx.hub)
    return f"symbols={len(syms)} rows={n}"


def _last_prices(ctx: JobContext, d: date) -> dict[str, float]:
    q = io.read(ctx.settings, "quotes_1m", start=d, end=d, columns=["symbol", "ts", "ltp"]).collect()
    if q.is_empty():
        return {}
    q = q.drop_nulls("ltp").sort("ts").group_by("symbol").agg(pl.col("ltp").last())
    return dict(zip(q["symbol"].to_list(), q["ltp"].to_list(), strict=True))


def eod_mark(ctx: JobContext, d: date) -> str:
    """Provisional marks for every book from the last stored quotes (the first mark of a day wins)."""
    px = _last_prices(ctx, d)
    n = 0
    for b in active_books(ctx, d):
        prices = {p.sid: px[p.symbol] for p in ctx.ledger.positions(b) if p.symbol in px}
        ctx.ledger.mark(b, d, prices, ctx.ledger.decision_capital(b, d))
        n += 1
    return f"marked books={n}"


# ---- the evening -----------------------------------------------------------------------------


def eod_ingest(ctx: JobContext, d: date) -> str:
    from stocksense.data.status import status_table

    res = ctx.catch_up()
    tab = status_table(ctx.settings, ctx.calendar, ctx.clock, ctx.required_fresh)
    fresh = {r["dataset"]: bool(r["fresh"]) for r in tab.to_dicts()}
    stale = [n for n in ctx.required_fresh if not fresh.get(n, False)]
    if stale:
        raise DataNotAvailable("not fresh yet: " + ",".join(stale))
    return f"fresh results={len(res)}"


def _stage_build_panel(ctx: JobContext, d: date) -> str:
    from stocksense.panel import build_all

    build_all(ctx.settings)
    return "panel built"


def official_rows(ctx: JobContext, d: date) -> pl.DataFrame:
    return io.read(ctx.settings, "panel_daily", start=d, end=d).collect()


def _stage_reconcile(ctx: JobContext, d: date) -> str:
    if not ctx.ledger.fills_for_date(d, reconciled=False):
        return "nothing to reconcile"
    rows = official_rows(ctx, d)
    if rows.is_empty():
        raise DataNotAvailable(f"no official rows for {d}")
    rep = ctx.paper().reconcile(d, rows)
    return (
        f"repriced={len(rep.repriced)} reversed={len(rep.reversed)} clamped={len(rep.clamped)} "
        f"kept={len(rep.kept)} no_bar={len(rep.no_bar)}"
    )


def _stage_marks(ctx: JobContext, d: date) -> str:
    """Official-close marks for books with no mark yet (catch-up days; today's provisional ones stand)."""
    rows = official_rows(ctx, d)
    books = active_books(ctx, d)
    if rows.is_empty() and any(ctx.ledger.positions(b) for b in books):
        raise DataNotAvailable(f"no official rows for {d}: cannot mark held positions")
    px = {int(r["sid"]): float(r["close"]) for r in rows.to_dicts()}
    n = 0
    for b in books:
        prices = {p.sid: px[p.sid] for p in ctx.ledger.positions(b) if p.sid in px}
        ctx.ledger.mark(b, d, prices, ctx.ledger.decision_capital(b, d))
        n += 1
    return f"marked books={n}"


def _stage_build_features(ctx: JobContext, d: date) -> str:
    from stocksense.features.build import build_features

    v = build_features(ctx.settings, start=d, clock=ctx.clock)
    return f"features v{v}"


def _stage_s1_plan(ctx: JobContext, d: date) -> str:
    fn = ctx.hook("s1_plan")
    if fn is None:
        return f"{NOT_IMPLEMENTED} (no S1 planner wired)"
    return f"planned={fn(ctx, d)}"


def _stage_report(ctx: JobContext, d: date, catchup: bool = False) -> str:
    """Minimal daily report (reports/<d>.md and .json) until T12.2 replaces it."""
    paths = Paths(ctx.settings)
    paths.reports.mkdir(parents=True, exist_ok=True)
    books = {}
    for b in ctx.ledger.books():
        s = ctx.ledger.nav_series(b).filter(pl.col("date") == d)
        if s.height:
            r = s.row(0, named=True)
            books[b] = {"nav": r["nav"], "day_ret": r["day_ret"], "n_pos": r["n_pos"], "gross": r["gross"]}
    fills = ctx.ledger.fills_for_date(d)
    con = ctx.state.connect()
    try:
        failed = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT job FROM job_runs WHERE trade_date=? AND status='failed' "
                "AND COALESCE(detail,'') NOT LIKE 'retry:%'",
                (d.isoformat(),),
            )
        ]
    finally:
        con.close()
    body = {
        "date": d.isoformat(), "catch_up": catchup, "books": books, "fills": len(fills),
        "reconciled": sum(f.reconciled for f in fills), "failed_jobs": failed,
        "degraded": ctx.state.kv_get(f"quotes_degraded:{d.isoformat()}") is not None,
    }  # fmt: skip
    (paths.reports / f"{d.isoformat()}.json").write_text(
        json.dumps(body, indent=1), encoding="utf-8", newline="\n"
    )
    lines = [f"# StockSense report {d}" + (" (catch-up)" if catchup else ""), ""]
    for b, v in books.items():
        lines.append(f"- {b}: NAV {v['nav']:.4f}, day {v['day_ret'] * 100:+.3f}%, positions {v['n_pos']}")
    lines += [
        f"- fills {body['fills']} (reconciled {body['reconciled']})",
        f"- failed jobs: {failed or 'none'}",
    ]
    (paths.reports / f"{d.isoformat()}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    nav = ", ".join(f"{b} {v['nav']:.2f}" for b, v in books.items()) or "no books"
    ctx.notifier.send(f"report {d}: {nav}")
    return f"report written books={len(books)}"


def _stub(reason: str = NOT_IMPLEMENTED) -> Callable[[JobContext, date], str]:
    return lambda ctx, d: reason


#: evening stages in order: (name, implementation). Stubs record `skipped: not implemented`.
STAGES: list[tuple[str, Callable[[JobContext, date], str]]] = [
    ("build_panel", _stage_build_panel),
    ("reconcile", _stage_reconcile),
    ("marks", _stage_marks),
    ("build_features", _stage_build_features),
    ("kronos_features", _stub()),
    ("predict", _stub()),
    ("s1_plan", _stage_s1_plan),
    ("nightly_learn", _stub()),
    ("drift", _stub()),
    ("promote", _stub()),
    ("report", _stage_report),
]
STUB_STAGES = ("kronos_features", "predict", "s1_plan", "nightly_learn", "drift", "promote")


def run_eod_pipeline(ctx: JobContext, d: date, catchup: bool = False) -> dict[str, str]:
    """Run the evening stages for session d in order, resuming past stages already done or skipped.

    Returns {stage: done|skipped|failed}. Each stage is one job_runs row named `eod_pipeline.<stage>`.
    """
    out: dict[str, str] = {}
    for name, impl in STAGES:
        key = f"eod_pipeline.{name}"
        prior = ctx.state.job_status(key, d)
        if prior in ("done", "skipped"):
            out[name] = prior
            continue
        fn = ctx.hook(f"stage:{name}") or impl
        attempt = ctx.state.job_start(key, d)
        try:
            detail = fn(ctx, d, catchup) if name == "report" and fn is _stage_report else fn(ctx, d)
            status = "skipped" if detail.startswith("skipped") else "done"
        except DataNotAvailable as e:
            status, detail = "failed", f"retry: {e}"
        except Exception as e:  # noqa: BLE001 - stage boundary: record and stop this pipeline run
            log.exception("stage_failed", stage=name, date=str(d))
            status, detail = "failed", ctx.scrub(repr(e))[:500]
        ctx.state.job_finish(key, d, attempt, status, detail)
        out[name] = status
        if status == "failed":
            break
    if all(out.get(n) in ("done", "skipped") for n, _ in STAGES):
        prev = ctx.state.kv_get("last_eod_pipeline")
        if prev is None or date.fromisoformat(prev) < d:
            ctx.state.kv_set("last_eod_pipeline", d.isoformat())
    return out


def eod_pipeline(ctx: JobContext, d: date) -> str:
    res = run_eod_pipeline(ctx, d)
    failed = [n for n, s in res.items() if s == "failed"]
    if failed:
        con = ctx.state.connect()
        try:
            row = con.execute(
                "SELECT detail FROM job_runs WHERE job=? AND trade_date=? ORDER BY attempt DESC LIMIT 1",
                (f"eod_pipeline.{failed[0]}", d.isoformat()),
            ).fetchone()
        finally:
            con.close()
        detail = (row[0] if row else "") or ""
        if detail.startswith("retry:"):
            raise DataNotAvailable(f"{failed[0]}: {detail[6:].strip()}")
        raise StageFailed(f"{failed[0]} failed: {detail}")
    return " ".join(f"{n}={s}" for n, s in res.items())


# ---- housekeeping and the rest ---------------------------------------------------------------


def housekeeping(ctx: JobContext, d: date) -> str:
    notes = []
    if ctx.log_console is not None:
        configure_logging(ctx.settings, ctx.clock, console=ctx.log_console)
        notes.append("log reconfigured")
    if d.weekday() == 6:
        con = ctx.state.connect()
        try:
            con.execute("VACUUM")
        finally:
            con.close()
        notes.append("vacuum")
    cache = Paths(ctx.settings).cache
    cutoff = ctx.clock.now().timestamp() - CACHE_KEEP_DAYS * 86400
    removed = 0
    if cache.is_dir():
        for p in cache.rglob("*"):
            if p.is_file() and "bhav" not in str(p.relative_to(cache)).lower() and p.stat().st_mtime < cutoff:
                os.remove(p)
                removed += 1
    notes.append(f"cache removed={removed}")
    return "; ".join(notes)


def collect_only(ctx: JobContext, d: date) -> str:
    res = ctx.catch_up(only=("announcements",))
    return f"results={len(res)} failed={sum(r.status == 'failed' for r in res)}"


#: named stubs: recorded as `skipped: not implemented` (listed in docs/DECISIONS.md)
STUB_JOBS = ("plan_open", "weekly_research", "monthly_kronos")

JOBS: dict[str, Callable[[JobContext, date], str]] = {
    "preflight": preflight,
    "broker_login": broker_login,
    "overnight_data": overnight_data,
    "capital_am": capital_am,
    "plan_open": _stub(),
    "preopen_capture": preopen_capture,
    "s2_decide": s2_decide,
    "open_fills": open_fills,
    "market_loop": market_loop,
    "spread_capture": spread_capture,
    "capital_pm": capital_pm,
    "s1_entries": s1_entries,
    "s2_squareoff": s2_squareoff,
    "close_fills": close_fills,
    "eod_mark": eod_mark,
    "eod_ingest": eod_ingest,
    "eod_pipeline": eod_pipeline,
    "housekeeping": housekeeping,
    "weekly_research": _stub(),
    "monthly_kronos": _stub(),
    "collect_only": collect_only,
}

#: what runs when a job is skipped for being late (besides recording `skipped: late`)
LATE_HANDLERS: dict[str, Callable[[JobContext, date], str]] = {"open_fills": late_open_fills}


@dataclass
class JobResult:
    key: str
    status: str
    detail: str
    attempt: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def execute_job(ctx: JobContext, key: str, fn: str, d: date, *, notify: bool = True) -> JobResult:
    """Run one job through the recording wrapper (shared by the daemon and `stocksense job`)."""
    impl = JOBS[fn]
    attempt = ctx.state.job_start(key, d)
    ctx.current_job = key
    try:
        detail = impl(ctx, d)
        status = "skipped" if detail.startswith("skipped") else "done"
    except DataNotAvailable as e:
        status, detail = "failed", f"retry: {ctx.scrub(str(e))}"
    except Exception as e:  # noqa: BLE001 - the job boundary: log with traceback, record, notify, continue
        log.exception("job_failed", job=key, date=str(d))
        status, detail = "failed", ctx.scrub(repr(e))[:500]
        if notify:
            ctx.notifier.send(f"job failed: {key} {d}: {detail[:200]}")
    finally:
        ctx.current_job = None
    ctx.state.job_finish(key, d, attempt, status, detail)
    if status != "failed":
        ctx.last_ok_job = key
    log.info("job_done", job=key, date=str(d), status=status, detail=detail[:200])
    return JobResult(key, status, detail, attempt)


def record_skip(ctx: JobContext, key: str, d: date, detail: str) -> None:
    attempt = ctx.state.job_start(key, d)
    ctx.state.job_finish(key, d, attempt, "skipped", detail)
