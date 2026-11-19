"""The daemon over whole days on the synthetic market (T10.5): fake brokers, SteppingClock, real SQLite state.

The market is the synthetic one; its last sessions are held back from the store and "published" by a fake
ingest step on the clock the real NSE would (after 18:00). Quotes are derived from the held-back bars.
"""

from __future__ import annotations

import dataclasses
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest

from stocksense.brokers.quotes import QuoteHub
from stocksense.brokers.types import Funds, Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, SteppingClock
from stocksense.core.errors import BrokerError
from stocksense.core.state import State
from stocksense.data import io
from stocksense.data.ingest_common import IngestResult
from stocksense.live.daemon import run
from stocksense.live.jobs import JobContext
from stocksense.live.preflight import daemon_lock, read_heartbeat
from stocksense.panel import build_all
from stocksense.paper.ledger import Ledger, NewOrder
from tests.fixtures.synth import _settings_for, make_market, write_market

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

PUBLISH_AT = time(18, 50)
CAPITAL = 1_000_000.0


class World:
    """Synthetic market with its last six sessions held back."""

    def __init__(self, root: Path) -> None:
        self.root = root
        m = make_market(n=30, n_sessions=300)
        self.sessions = m.sessions
        self.base_last = m.sessions[-7]
        self.held = {
            "bhavcopy_eq": m.eq.filter(pl.col("date") > self.base_last),
            "bhavcopy_delivery": m.delivery.filter(pl.col("date") > self.base_last),
        }
        base = dataclasses.replace(
            m,
            eq=m.eq.filter(pl.col("date") <= self.base_last),
            delivery=m.delivery.filter(pl.col("date") <= self.base_last),
        )
        write_market(base, root)
        s = _settings_for(root)
        build_all(s)
        self.eq_all = m.eq
        self.skipped = self.sessions[-6:-3]  # three sessions nobody ran the daemon on
        self.day = self.sessions[-3]  # the day the daemon runs


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    root = tmp_path_factory.mktemp("full_day_base")
    return World(root)


def publish(settings, world: World, upto: date) -> int:
    """Write held-back sessions <= upto into the store (what a successful NSE ingest would add)."""
    n = 0
    for name, df in world.held.items():
        part = df.filter(pl.col("date") <= upto)
        if part.height:
            io.upsert(settings, name, part)
            n += part.height
    return n


class FakeSource:
    """Quotes interpolated along the held-back daily bar (open at 09:15, close at 15:30)."""

    held_only = False

    def __init__(self, clock, world: World, name="angel", fail=False) -> None:
        self.clock, self.name, self.fail, self.calls = clock, name, fail, 0
        self._bars: dict[tuple[date, str], dict] = {
            (r["date"], r["symbol"]): r for r in world.eq_all.filter(pl.col("series") == "EQ").to_dicts()
        }

    def quotes(self, symbols):
        self.calls += 1
        if self.fail:
            raise BrokerError("login failed")
        now = self.clock.now()
        o = now.replace(hour=9, minute=15, second=0, microsecond=0)
        c = now.replace(hour=15, minute=30, second=0, microsecond=0)
        frac = min(max((now - o) / (c - o), 0.0), 1.0)
        out = []
        for s in symbols:
            b = self._bars.get((now.date(), s))
            if b is None or now < o:
                continue
            ltp = round((b["open"] + (b["close"] - b["open"]) * frac) / 0.05) * 0.05
            out.append(
                Quote(
                    symbol=s, ts=now, ltp=ltp, open=b["open"], high=max(b["open"], ltp),
                    low=min(b["open"], ltp), prev_close=b["prev_close"], volume=1e5,
                    bid=ltp - 0.05, ask=ltp + 0.05, bid_qty=500, ask_qty=500,
                    upper_circuit=b["prev_close"] * 1.2,
                    lower_circuit=b["prev_close"] * 0.8, source=self.name,
                )
            )  # fmt: skip
        return out


class FakeBroker:
    def __init__(self) -> None:
        self.logins = 0

    def login(self):
        self.logins += 1

    def funds(self):
        return Funds(available_cash=CAPITAL, net=CAPITAL, utilised=0.0)

    def holdings(self):
        return []


class FakeInstruments:
    def refresh(self):
        pass


class Env:
    def __init__(self, tmp_path, world: World, start: datetime, *, fail_first_source=False, hooks=None):
        self.world = world
        self.root = tmp_path / "data_store"
        shutil.copytree(world.root, self.root)
        self.settings = _settings_for(self.root)
        self.clock = SteppingClock(start)
        self.state = State(self.settings, self.clock)
        self.state.migrate()
        self.broker = FakeBroker()
        srcs = [
            FakeSource(self.clock, world, "angel", fail=fail_first_source),
            FakeSource(self.clock, world, "upstox"),
        ]
        self.hub = QuoteHub(srcs, self.clock)
        self.published: list[date] = []
        self.stage_log: list[tuple[str, date]] = []
        self.hooks = {"capture_preopen": lambda ctx: 3, "stage:build_features": self._stage("build_features")}
        self.hooks.update(hooks or {})
        self.ctx = self.new_ctx()

    def _stage(self, name):
        def fn(ctx, d):
            self.stage_log.append((name, d))
            return f"fake {name}"

        return fn

    def fake_ingest(self, ictx):
        now = ictx.clock.now()
        upto = now.date() if now.time() >= PUBLISH_AT else now.date() - timedelta(days=1)
        n = publish(self.settings, self.world, upto)
        if n:
            self.published.append(upto)
        return [IngestResult("bhavcopy_eq", upto, n, "ok")]

    def new_ctx(self) -> JobContext:
        cal = TradingCalendar.from_store(self.settings, self.clock)
        return JobContext(
            self.settings, self.clock, self.state, cal, hub=self.hub, broker=self.broker,
            instruments=FakeInstruments(), nse=object(), steps=[("fake_ingest", self.fake_ingest)],
            hooks=self.hooks, required_fresh=("bhavcopy_eq", "bhavcopy_delivery"),
        )  # fmt: skip

    def go(self, until: datetime, once=False) -> int:
        return run(self.settings, self.clock, once=once, ctx=self.ctx, poll_s=10, stop_at=until)

    # ---- queries ----
    def job_rows(self, like="%"):
        con = self.state.connect()
        try:
            return con.execute(
                "SELECT job, trade_date, attempt, status, detail FROM job_runs "
                "WHERE job LIKE ? ORDER BY rowid",
                (like,),
            ).fetchall()
        finally:
            con.close()

    def sql(self, q, *a):
        con = self.state.connect()
        try:
            return con.execute(q, a).fetchall()
        finally:
            con.close()


def at(d: date, h: int, m: int = 0, s: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=IST)


def picks(env: Env, k: int = 3):
    """(sid, symbol, price) of k names from the newest stored panel day."""
    row = io.read(env.settings, "panel_daily", start=env.world.base_last, end=env.world.base_last).collect()
    row = row.filter(pl.col("series") == "EQ").sort("sid").head(k)
    return [(int(r["sid"]), r["symbol"], float(r["close"])) for r in row.to_dicts()]


def order_hooks(env: Env):
    chosen = picks(env)

    def s2(ctx, d, cap):
        sid, sym, px = chosen[0]
        qty = max(int(0.02 * cap / px), 1)
        o = NewOrder(sid, sym, "BUY", "intraday", "open", d, qty, px, "s2")
        return [("auction", [o], {"why": "test"}), ("auction_strict", [o], {"why": "test"})]

    def s1(ctx, d, cap):
        orders = [
            NewOrder(sid, sym, "BUY", "delivery", "close", d, max(int(0.03 * cap / px), 1), px, "s1")
            for sid, sym, px in chosen[1:]
        ]
        return [("main", orders, {"why": "test"})]

    return {"s2_orders": s2, "s1_orders": s1}


def make_env(tmp_path, world, start, **kw) -> Env:
    e = Env(tmp_path, world, start, **kw)
    e.hooks.update(order_hooks(e))
    e.ctx = e.new_ctx()  # the context copies the hooks, so build it after they are complete
    return e


def prime(e: Env, upto: date) -> None:
    """Sessions before the daemon's day are already in the store and already processed."""
    publish(e.settings, e.world, upto)
    e.state.kv_set("last_eod_pipeline", upto.isoformat())


# ---- 1: a full session -----------------------------------------------------------------------


def test_full_session_runs_every_job_in_order_and_ends_with_report_marks_and_reconciled_fills(
    tmp_path, world
):
    d = world.day
    e = make_env(tmp_path, world, at(d, 7, 30))
    prime(e, world.sessions[-4])
    assert e.go(at(d, 23, 59)) == 0

    seen: list[str] = []
    for job, trade_date, _a, _s, _x in e.job_rows():
        if trade_date == d.isoformat() and job not in seen:
            seen.append(job)
    expected = [
        "preflight_start", "preflight", "broker_login", "overnight_data", "capital_am", "plan_open",
        "preopen_capture", "s2_decide", "open_fills", "market_loop", "spread_capture@09:20",
        "spread_capture@11:00", "spread_capture@13:00", "spread_capture@15:00", "capital_pm",
        "s1_entries", "s2_squareoff", "close_fills", "eod_mark", "eod_ingest", "eod_pipeline",
        "eod_pipeline.build_panel", "eod_pipeline.reconcile", "eod_pipeline.marks",
        "eod_pipeline.build_features", "eod_pipeline.kronos_features", "eod_pipeline.predict",
        "eod_pipeline.s1_plan", "eod_pipeline.nightly_learn", "eod_pipeline.drift",
        "eod_pipeline.promote", "eod_pipeline.report", "housekeeping",
    ]  # fmt: skip
    assert seen == expected
    final = {}
    for job, _td, _a, status, detail in e.job_rows():
        final[job] = (status, detail)
    unresolved = [j for j, (s, x) in final.items() if s == "failed" and not (x or "").startswith("retry:")]
    assert unresolved == [], unresolved
    assert final["eod_ingest"][0] == "done" and final["eod_pipeline"][0] == "done"
    stubs = {j for j, (s, x) in final.items() if s == "skipped" and x == "skipped: not implemented"}
    assert {"plan_open", "eod_pipeline.predict", "eod_pipeline.drift"} <= stubs
    # eod_ingest waited for the 18:50 publication: 18:30 and 18:45 retried, 19:00 succeeded
    ing = [r for r in e.job_rows("eod_ingest")]
    assert [r[3] for r in ing] == ["failed", "failed", "done"] and ing[0][4].startswith("retry:")

    led = Ledger(e.state, e.clock)
    assert set(led.books()) == {"main", "auction", "auction_strict"}
    for b in led.books():
        assert d in led.nav_series(b)["date"].to_list(), b
    fills = led.fills_for_date(d)
    assert len(fills) >= 4 and all(f.reconciled for f in fills)
    assert any(r[0] for r in e.sql("SELECT 1 FROM corrections WHERE reason='official_reprice'"))
    assert not led.pending_orders(d, "open") and not led.pending_orders(d, "close")
    assert not any(p.product == "intraday" for b in led.books() for p in led.positions(b))
    rep = (e.root / "reports" / f"{d}.md").read_text(encoding="utf-8")
    assert "main" in rep and "failed jobs: none" in rep
    assert (e.root / "reports" / f"{d}.json").is_file()
    assert e.state.kv_get("last_eod_pipeline") == d.isoformat()
    assert read_heartbeat(e.root / "state" / "heartbeat.json")["pid"] > 0


# ---- 2: restart after s1_entries ------------------------------------------------------------


def test_restart_after_s1_entries_inserted_orders_makes_no_duplicates_and_the_day_completes(tmp_path, world):
    d = world.day
    e = make_env(tmp_path, world, at(d, 7, 30))
    prime(e, world.sessions[-4])
    e.go(at(d, 15, 12))
    n_orders = e.sql("SELECT COUNT(*) FROM orders WHERE book='main'")[0][0]
    assert n_orders == 2
    # the process dies mid-job: the s1_entries attempt is left `running`, as if killed after the insert
    e.sql("UPDATE job_runs SET status='running', finished_at=NULL WHERE job='s1_entries'")
    e2_ctx = e.new_ctx()  # a brand-new daemon over the same state
    e.ctx = e2_ctx
    e.clock.advance(timedelta(seconds=90))
    e.go(at(d, 23, 59))
    assert e.sql("SELECT COUNT(*) FROM orders WHERE book='main'")[0][0] == n_orders
    runs = e.job_rows("s1_entries")
    assert [r[3] for r in runs] == ["failed", "done"] and runs[0][4] == "interrupted"
    assert e.sql("SELECT COUNT(*) FROM fills f JOIN orders o USING(order_id) WHERE o.book='main'")[0][0] == 2
    assert (e.root / "reports" / f"{d}.md").is_file()


# ---- 3: a holiday ---------------------------------------------------------------------------


def test_holiday_runs_only_preflight_collect_only_and_housekeeping(tmp_path, world):
    d = world.day
    e = make_env(tmp_path, world, at(d, 7, 30))
    io.upsert(
        e.settings, "holidays",
        pl.DataFrame({"date": [d], "description": ["Test holiday"], "fetched_at": [datetime(2026, 1, 1)]}),
    )  # fmt: skip
    e.ctx = e.new_ctx()
    e.ctx.steps.append(("announcements", lambda ictx: []))
    e.go(at(d, 23, 59))
    names = {r[0] for r in e.job_rows() if r[1] == d.isoformat()}
    assert names == {"preflight_start", "preflight", "collect_only", "housekeeping"}
    assert len([r for r in e.job_rows("collect_only")]) == 1  # one summary row for the day
    assert "runs=" in e.job_rows("collect_only")[0][4]
    assert e.sql("SELECT COUNT(*) FROM orders")[0][0] == 0


# ---- 4: a late start -----------------------------------------------------------------------


def test_late_start_skips_s2_and_open_fills_with_reasons_and_rolls_s1_exits_to_the_next_open(tmp_path, world):
    d = world.day
    e = make_env(tmp_path, world, at(d, 10, 40))
    prime(e, world.sessions[-4])
    led = Ledger(e.state, e.clock)
    (sid, sym, px), (sid2, sym2, px2) = picks(e)[:2]
    prev = world.sessions[-4]
    d_main = led.record_decision("main", "s1", prev, CAPITAL, False, {}, {})
    led.add_orders(d_main, "main", [NewOrder(sid, sym, "SELL", "delivery", "open", d, 7, px, "horizon")])
    d_a = led.record_decision("auction", "s2", d, CAPITAL, False, {}, {})
    led.add_orders(d_a, "auction", [NewOrder(sid2, sym2, "BUY", "intraday", "open", d, 5, px2, "s2")])
    e.go(at(d, 23, 59))
    s2 = e.job_rows("s2_decide")
    of = e.job_rows("open_fills")
    assert s2 == [("s2_decide", d.isoformat(), 1, "skipped", "skipped: late")]
    assert (
        of[0][3] == "skipped" and of[0][4].startswith("skipped: late") and "rolled=1 cancelled=1" in of[0][4]
    )
    nxt = e.ctx.calendar.next_session(d)
    rows = e.sql(
        "SELECT book, side, status, cancel_reason, trade_date FROM orders ORDER BY book, side, trade_date"
    )
    assert ("auction", "BUY", "cancelled", "late_start", d.isoformat()) in rows
    assert ("main", "SELL", "cancelled", "rolled", d.isoformat()) in rows
    assert ("main", "SELL", "pending", None, nxt.isoformat()) in rows
    assert e.sql("SELECT COUNT(*) FROM fills")[0][0] == 2  # only s1 entries (3:10 pm) traded
    assert e.job_rows("s1_entries")[-1][3] == "done"
    assert e.job_rows("eod_pipeline")[-1][3] == "done"


# ---- 5: three skipped sessions -------------------------------------------------------------


def test_three_missed_sessions_are_caught_up_in_order_then_a_normal_day_runs(tmp_path, world):
    d = world.day
    first, *_ = world.skipped
    e = make_env(tmp_path, world, at(d, 7, 30))
    e.state.kv_set("last_eod_pipeline", world.base_last.isoformat())
    log_stage = e._stage("build_panel")

    def panel(ctx, d_):  # missed days only log (fast); the day that traded builds the real panel
        out = log_stage(ctx, d_)
        if d_ == world.day:
            from stocksense.live.jobs import _stage_build_panel

            return _stage_build_panel(ctx, d_)
        return out

    e.hooks["stage:build_panel"] = panel
    e.ctx = e.new_ctx()
    e.go(at(d, 23, 59))
    panel_days = [x for n, x in e.stage_log if n == "build_panel"]
    assert panel_days == [*world.skipped, d]
    for s in world.skipped:
        md = (e.root / "reports" / f"{s}.md").read_text(encoding="utf-8")
        assert "(catch-up)" in md
        assert e.state.job_status("eod_pipeline.report", s) == "done"
    assert "(catch-up)" not in (e.root / "reports" / f"{d}.md").read_text(encoding="utf-8")
    assert e.state.kv_get("last_eod_pipeline") == d.isoformat()
    # no orders were invented for the missed days; the normal day traded as usual
    dates = {r[0] for r in e.sql("SELECT DISTINCT trade_date FROM orders")}
    assert dates == {d.isoformat()} and first.isoformat() not in dates
    assert e.job_rows("s1_entries")[-1][3] == "done"


# ---- 6: first quote source fails -----------------------------------------------------------


def test_first_quote_source_failing_uses_the_fallback_and_records_degraded(tmp_path, world):
    d = world.day
    e = make_env(tmp_path, world, at(d, 7, 30), fail_first_source=True)
    prime(e, world.sessions[-4])
    e.go(at(d, 15, 40))
    led = Ledger(e.state, e.clock)
    assert len(led.fills_for_date(d)) >= 4  # the fallback served every fill
    raw = e.state.kv_get(f"quotes_degraded:{d.isoformat()}")
    assert raw is not None and "angel" in raw and "login failed" in raw
    assert e.ctx.degraded is True
    assert read_heartbeat(e.root / "state" / "heartbeat.json")["degraded"] is True
    assert e.hub.degraded is True


# ---- 7: single instance --------------------------------------------------------------------


def test_second_instance_exits_zero_saying_already_running(tmp_path, world, capsys):
    d = world.day
    e = make_env(tmp_path, world, at(d, 7, 30))
    lock = daemon_lock(e.settings)
    assert lock.acquire()
    try:
        assert e.go(at(d, 8, 0), once=True) == 0
    finally:
        lock.release()
    assert "already running" in capsys.readouterr().out
    assert e.job_rows() == []  # the second instance did nothing
    assert e.go(at(d, 7, 31), once=True) == 0  # and the lock is free again afterwards
    assert e.job_rows() != []
