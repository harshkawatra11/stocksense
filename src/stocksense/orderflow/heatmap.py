"""Depth heatmap: forward-only capture of the top-5 order book (Angel FULL quotes) and its price-time matrix.

The book is a snapshot, not a tape: it shows resting size, never executed trades. Capture runs only while the
server is up in market hours; history exists only for days it ran (no backfill is possible).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.marketlab.clock import market_status
from stocksense.marketlab.store import LabStore

log = get_logger(__name__)

DATASET = "depth_snaps"
RING_SECONDS = 7200
CAPTURE_EVERY_S = 5.0
IDLE_WAIT_S = 60.0
FLUSH_EVERY_S = 60.0
WATCH_EACH_SIDE = 10
MAX_WATCH = 25
TOKENS_PER_CALL = 50
_COLS = ["symbol", "ts", "side", "level", "price", "qty"]
_SCHEMA = {
    "symbol": pl.Utf8,
    "ts": pl.Datetime("us", "Asia/Kolkata"),
    "side": pl.Utf8,
    "level": pl.Int32,
    "price": pl.Float64,
    "qty": pl.Float64,
}


class DepthRing:
    """Thread-safe per-symbol buffer of depth rows; each add drops rows older than `max_age_s` (2 h)."""

    def __init__(self, max_age_s: float = RING_SECONDS) -> None:
        self.max_age_s = float(max_age_s)
        self._lock = threading.Lock()
        self._rows: dict[str, deque[dict[str, Any]]] = {}

    def add(self, rows: list[dict[str, Any]]) -> None:
        """Append rows (keys symbol, ts, side, level, price, qty) and evict those older than the window."""
        with self._lock:
            for r in rows:
                self._rows.setdefault(r["symbol"], deque()).append(r)
            for sym, q in list(self._rows.items()):
                newest = q[-1]["ts"]
                while q and (newest - q[0]["ts"]).total_seconds() > self.max_age_s:
                    q.popleft()
                if not q:
                    del self._rows[sym]

    def symbols(self) -> list[str]:
        """Symbols currently held, sorted."""
        with self._lock:
            return sorted(self._rows)

    def frame(self, symbol: str) -> pl.DataFrame:
        """The held rows of one symbol as a frame (empty with the right columns when none)."""
        with self._lock:
            rows = list(self._rows.get(symbol, ()))
        if not rows:
            return pl.DataFrame(schema=_SCHEMA)
        return pl.DataFrame(rows, schema=_SCHEMA)


def heatmap(snaps: pl.DataFrame, bin: float, step_s: int = 30) -> dict[str, Any]:  # noqa: A002
    """Maximum resting quantity (shares) per (time bucket of `step_s` s, price bin of `bin` rupees).

    Returns {times: bucket starts in epoch seconds, prices: bin lower edges in rupees ascending,
    qty: matrix [time][price]}; empty lists when `snaps` has no rows.
    """
    if snaps.height == 0:
        return {"times": [], "prices": [], "qty": []}
    t = snaps["ts"].dt.epoch("s").to_numpy().astype(np.int64)
    tb = (t // step_s) * step_s
    pb = np.floor(snaps["price"].to_numpy() / bin + 1e-9).astype(np.int64)
    q = snaps["qty"].to_numpy().astype(float)
    times, ti = np.unique(tb, return_inverse=True)
    prices, pi = np.unique(pb, return_inverse=True)
    m = np.zeros((len(times), len(prices)))
    np.maximum.at(m, (ti, pi), q)
    return {
        "times": [int(x) for x in times],
        "prices": [round(float(p) * bin, 6) for p in prices],
        "qty": m.tolist(),
    }


class DepthCapture:
    """Capture loop body: `step()` snapshots the watch list every 5 s in market hours, flushes each 60 s."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        calendar: TradingCalendar,
        store: LabStore,
        angel: Any,
        token_of: Callable[[str], str | None],
    ) -> None:
        self.settings, self.clock, self.cal, self.store = settings, clock, calendar, store
        self.angel, self.token_of = angel, token_of
        self.ring = DepthRing()
        self._pending: list[dict[str, Any]] = []
        self._last_flush: datetime | None = None

    def watchlist(self) -> list[str]:
        """Top 10 bulls and top 10 bears of the tape that have an Angel token (at most 25 symbols)."""
        with self.store.lock:
            stocks = dict(self.store.tape.get("stocks", {}))
        out: list[str] = []
        for side in ("bull", "bear"):
            ranked = sorted((r.get("rank", 10**6), s) for s, r in stocks.items() if r.get("side") == side)
            out += [s for _, s in ranked if self.token_of(s)][:WATCH_EACH_SIDE]
        return out[:MAX_WATCH]

    def step(self) -> float:
        """One iteration: seconds to wait before the next (60 s idle outside the open market, else 5 s)."""
        now = self.clock.now()
        if market_status(self.clock, self.cal) != "open":
            return IDLE_WAIT_S
        syms = self.watchlist()
        if syms:
            tokens = [t for t in (self.token_of(s) for s in syms) if t]
            rows: list[dict[str, Any]] = []
            for q in self.angel.quote_full(tokens[:TOKENS_PER_CALL]):
                for side, levels in (("bid", q.depth_bid), ("ask", q.depth_ask)):
                    rows += [
                        {"symbol": q.symbol, "ts": q.ts, "side": side, "level": i, "price": p, "qty": v}
                        for i, (p, v) in enumerate(levels)
                    ]
            self.ring.add(rows)
            self._pending += rows
        if self._last_flush is None:
            self._last_flush = now
        elif (now - self._last_flush).total_seconds() >= FLUSH_EVERY_S:
            self.flush()
            self._last_flush = now
        return CAPTURE_EVERY_S

    def flush(self) -> int:
        """Write pending rows to the `depth_snaps` dataset (keyed upsert, so a rerun is harmless)."""
        if not self._pending:
            return 0
        rows, self._pending = self._pending, []
        df = pl.DataFrame(rows, schema=_SCHEMA).unique(["symbol", "ts", "side", "level"])
        io.upsert(self.settings, DATASET, df, clock=self.clock)
        log.info("depth_flushed", rows=df.height)
        return df.height


def build_depth_capture(settings: Settings, clock: Clock, store: LabStore) -> DepthCapture | None:
    """The real capture loop (Angel read-only quotes), or None when Angel secrets or calendar are missing."""
    from stocksense.brokers.angel import AngelClient
    from stocksense.brokers.instruments import InstrumentMap
    from stocksense.core.errors import ConfigError, DataNotAvailable

    try:
        settings.require("angel_api_key", "angel_client_id", "angel_pin", "angel_totp_secret")
        cal = TradingCalendar.from_store(settings, clock)
    except (ConfigError, DataNotAvailable):
        log.warning("depth_capture_disabled")
        return None
    inst = InstrumentMap(settings, clock)
    loaded = threading.Event()

    def token_of(symbol: str) -> str | None:
        if not loaded.is_set():
            inst.refresh()
            loaded.set()
        return inst.angel_token(symbol)

    return DepthCapture(settings, clock, cal, store, AngelClient(settings, clock), token_of)
