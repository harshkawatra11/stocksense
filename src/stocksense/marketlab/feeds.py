"""Background feed threads of the dashboard server. Later cards add the tape, sector and flow loops."""

from __future__ import annotations

import threading
from collections.abc import Callable

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
