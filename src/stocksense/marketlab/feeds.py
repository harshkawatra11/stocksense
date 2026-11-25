"""Background feed threads of the dashboard server. Later cards add the tape, sector and flow loops."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import threading
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.marketlab.store import LabStore

log = get_logger(__name__)


def run_loop(
    name: str, store: LabStore, clock: Clock, stop: threading.Event, step: Callable[[], float]
) -> None:
    """Run `step()` until `stop` is set; `step` returns the seconds to wait before its next call."""
    while not stop.is_set():
        try:
            wait = step()
            store.ok(name, clock.now())
        except Exception as e:  # feed boundary: log, count, continue (PLAN-M Part 0 rule 11)
            log.exception("feed_error", feed=name)
            store.fail(name, clock.now(), e)
            wait = 30.0
        stop.wait(wait)


class FeedManager:
    """Owns the feed threads; `stop()` returns promptly because every wait is an Event wait."""

    def __init__(self, settings: Settings, clock: Clock, store: LabStore) -> None:
        self.settings = settings
        self.clock = clock
        self.store = store
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def add(self, name: str, step: Callable[[], float], interval_s: float) -> None:
        """Register a feed loop (started by `start`)."""
        self.store.set_interval(name, interval_s)
        t = threading.Thread(
            target=run_loop,
            args=(name, self.store, self.clock, self.stop_event, step),
            name=f"feed-{name}",
            daemon=True,
        )
        self.threads.append(t)

    def start(self) -> None:
        """Start every registered feed thread."""
        for t in self.threads:
            t.start()

    def stop(self) -> None:
        """Signal every feed to stop and join the threads (5 s cap each)."""
        self.stop_event.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=5.0)


def build_tape(settings: Settings, clock: Clock, store: LabStore):  # noqa: ANN201 (lazy import)
    """Real TapeEngine from stored data and live sources, or None (logged) when calendar data is missing."""
    from stocksense.core.calendar import TradingCalendar
    from stocksense.core.errors import DataNotAvailable
    from stocksense.marketlab.sources import Sources
    from stocksense.marketlab.tape import TapeEngine

    try:
        cal = TradingCalendar.from_store(settings, clock)
    except DataNotAvailable:
        log.warning("tape_disabled_no_calendar")
        return None
    return TapeEngine(settings, clock, cal, Sources(settings, clock, store), store)


# ---- start-up catch-up (PLAN-M M2.7) -------------------------------------------------------------------

MARKET_HOURS = (9 * 60, 15 * 60 + 45)  # minutes since midnight: panel builds wait outside this window
MIN_FREE_RAM_GB = 3.0
RETRY_S = 900.0
REQUIRED_FRESH = ("bhavcopy_eq", "bhavcopy_delivery", "fo_bhavcopy")
Step = tuple[str, Callable[[], str]]


def free_ram_gb() -> float | None:
    """Free physical memory in GB (psutil when installed, else PowerShell), or None when unknown."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1024**3
    except ImportError:
        pass
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return float(out.stdout.strip()) / 1024**2  # the value is in KB
    except (OSError, ValueError, subprocess.SubprocessError):
        log.warning("free_ram_unknown")
        return None


def _calendar(settings: Settings, clock: Clock):  # noqa: ANN202
    from stocksense.core.calendar import TradingCalendar

    return TradingCalendar.from_store(settings, clock)


def make_ingest_step(
    settings: Settings,
    clock: Clock,
    *,
    is_fresh: Callable[[], bool] | None = None,
    run: Callable[[], Iterable[Any]] | None = None,
) -> Callable[[], str]:
    """Step "ingest": run the NSE catch-up when equity, delivery or F&O files are behind; else "fresh"."""

    def fresh() -> bool:
        from stocksense.data.status import status_table

        t = status_table(settings, _calendar(settings, clock), clock, REQUIRED_FRESH)
        return t.height == len(REQUIRED_FRESH) and bool(t["fresh"].fill_null(False).all())

    def catch() -> list[Any]:
        from stocksense.core.state import State
        from stocksense.data.ingest_all import catch_up
        from stocksense.data.ingest_common import IngestContext
        from stocksense.data.nse_http import NseClient

        state = State(settings, clock)
        state.migrate()
        nse = NseClient(clock)
        try:
            ctx = IngestContext(
                settings=settings, clock=clock, calendar=_calendar(settings, clock), nse=nse, state=state
            )
            return catch_up(ctx)
        finally:
            nse.close()

    def step() -> str:
        if (is_fresh or fresh)():
            return "fresh"
        res = list((run or catch)())
        failed = sum(1 for r in res if r.status == "failed")
        return f"{len(res) - failed} ok, {failed} failed"

    return step


def make_panel_step(
    settings: Settings,
    clock: Clock,
    *,
    is_fresh: Callable[[], bool] | None = None,
    build: Callable[[], None] | None = None,
    free_ram_gb: Callable[[], float | None] | None = None,
) -> Callable[[], str]:
    """Step "panel": rebuild the panel when stale, outside 09:00-15:45 IST and with at least 3 GB free."""
    ram = free_ram_gb if free_ram_gb is not None else globals()["free_ram_gb"]

    def fresh() -> bool:
        from stocksense.data.status import status_table

        t = status_table(settings, _calendar(settings, clock), clock, ["panel_daily"])
        return t.height == 1 and bool(t["fresh"].fill_null(False).all())

    def do_build() -> None:
        from stocksense.panel import build_all

        build_all(settings)

    def step() -> str:
        if (is_fresh or fresh)():
            return "fresh"
        now = clock.now()
        minute = now.hour * 60 + now.minute
        if MARKET_HOURS[0] <= minute < MARKET_HOURS[1]:
            return "deferred: market hours"
        free = ram()
        if free is not None and free < MIN_FREE_RAM_GB:
            return "deferred: low memory"
        (build or do_build)()
        return "built"

    return step


def make_scan_step(
    settings: Settings,
    clock: Clock,
    *,
    asof: Callable[[], date] | None = None,
    exists: Callable[[date], bool] | None = None,
    run: Callable[[date], Path] | None = None,
) -> Callable[[], str]:
    """Step "scan": run the radar scan for the last completed session unless its parquet exists."""

    def step() -> str:
        from stocksense.strategies import scan

        d = (asof or (lambda: _calendar(settings, clock).last_completed_session(clock.now())))()
        if (exists or (lambda x: scan.scan_path(settings, x).exists()))(d):
            return f"skipped: scan exists for {d}"
        out = (run or (lambda x: scan.run_scan(settings, x, clock=clock)))(d)
        return f"scanned {d}: {out}"

    return step


def make_flow_step(settings: Settings, clock: Clock) -> Callable[[], str]:
    """Step "flow_eod": build the money-flow EOD tables (idempotent; nothing missing writes 0 rows)."""

    def step() -> str:
        from stocksense.flow.composite import build_eod

        cal = _calendar(settings, clock)
        done = build_eod(settings, cal.last_completed_session(clock.now()), cal)
        return f"{sum(done.values())} rows"

    return step


def make_picks_step(settings: Settings, clock: Clock) -> Callable[[], str]:
    """Step "picks": call `stocksense.strategies.picks.picks_step(settings, asof)` if it exists, else skip."""

    def step() -> str:
        if importlib.util.find_spec("stocksense.strategies.picks") is None:
            return "skipped: picks not built"
        fn = getattr(importlib.import_module("stocksense.strategies.picks"), "picks_step", None)
        if fn is None:
            return "skipped: no picks_step"
        asof = _calendar(settings, clock).last_completed_session(clock.now())
        return str(fn(settings, asof))

    return step


def default_steps(settings: Settings, clock: Clock) -> list[Step]:
    """The five start-up steps in order: ingest, panel, scan, flow_eod, picks."""
    return [
        ("ingest", make_ingest_step(settings, clock)),
        ("panel", make_panel_step(settings, clock)),
        ("scan", make_scan_step(settings, clock)),
        ("flow_eod", make_flow_step(settings, clock)),
        ("picks", make_picks_step(settings, clock)),
    ]


class CatchUp:
    """Runs the start-up steps once, in order; progress goes to `store.catchup` (shown by /api/status)."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        store: LabStore,
        steps: list[Step] | None = None,
        retry_s: float = RETRY_S,
    ) -> None:
        self.settings, self.clock, self.store = settings, clock, store
        self.steps = steps if steps is not None else default_steps(settings, clock)
        self.retry_s = retry_s
        with store.lock:
            store.catchup["state"] = "pending"
            store.catchup["steps"] = [{"name": n, "state": "pending", "detail": ""} for n, _ in self.steps]

    def _set(self, i: int, state: str, detail: str) -> None:
        with self.store.lock:
            self.store.catchup["steps"][i].update(state=state, detail=detail[:300])

    def _run_step(self, i: int) -> str:
        name, fn = self.steps[i]
        self._set(i, "running", "")
        try:
            detail = fn()
        except Exception as e:  # noqa: BLE001 - step boundary: record the failure, the next step still runs
            log.exception("catchup_step_failed", step=name)
            self._set(i, "failed", f"{type(e).__name__}: {str(e)[:120]}")
            return "failed"
        state = next((s for s in ("deferred", "skipped") if detail.startswith(s)), "done")
        self._set(i, state, detail)
        log.info("catchup_step", step=name, state=state, detail=detail)
        return detail

    def run(self, stop: threading.Event | None = None) -> None:
        """Run every step once; steps deferred for market hours are retried every `retry_s` until `stop`."""
        with self.store.lock:
            self.store.catchup["state"] = "running"
        retry = [i for i in range(len(self.steps)) if self._run_step(i) == "deferred: market hours"]
        while retry and stop is not None and not stop.wait(self.retry_s):
            retry = [i for i in retry if self._run_step(i) == "deferred: market hours"]
        with self.store.lock:
            self.store.catchup["state"] = "done"

    def start(self, stop: threading.Event) -> threading.Thread:
        """Start `run` in a daemon thread named "catchup"."""
        t = threading.Thread(target=self.run, args=(stop,), name="catchup", daemon=True)
        t.start()
        return t
