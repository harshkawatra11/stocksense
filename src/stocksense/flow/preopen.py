"""Pre-open board: indicative open gap, buy/sell imbalance and whether the gap held at 09:30.

Timing is unverified: NSE reportedly changed the pre-open phases from 2026-09-07 but its own page could not
be read, so nothing is hard-coded at 09:08. The loop polls 09:00:30-09:14:30 and snapshots the day once, at
the first poll where NIFTY constituents carry a final price (or at 09:12:30 if none ever did). The observed
times are kept in the store so they can be written to docs/DECISIONS.md. Units: gaps in percent.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import time
from typing import Any

import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.log import get_logger
from stocksense.data.ingest_misc import parse_preopen
from stocksense.marketlab.store import LabStore

log = get_logger(__name__)

WINDOW_START = time(9, 0, 30)
WINDOW_END = time(9, 14, 30)
FALLBACK_CAPTURE = time(9, 12, 30)
OPEN_TIME = time(9, 15)
CHECK_MINUTE = 15  # minutes after 09:15 = 09:30
NIFTY_NAMES = ("NIFTY", "NIFTY 50")


def _secs(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def board_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Per symbol: iep, iep_gap_pct (percent vs previous close), imb (-1..+1) and total_qty (shares)."""
    out = []
    for r in df.iter_rows(named=True):
        prev, iep = r["prev_close"], r["indicative_price"]
        buy, sell = r["buy_qty"] or 0.0, r["sell_qty"] or 0.0
        tot = buy + sell
        out.append(
            {
                "symbol": r["symbol"],
                "iep": iep,
                "iep_gap_pct": (iep / prev - 1.0) * 100.0 if iep and prev else None,
                "imb": (buy - sell) / tot if tot > 0 else None,
                "total_qty": tot,
            }
        )
    return out


def follow_through(rows: list[dict[str, Any]], tape: dict[str, dict[str, Any]]) -> dict[str, bool | None]:
    """Per symbol: sign(iep gap) == sign(price vs open) at 09:30 from the tape path; None when unknown."""
    out: dict[str, bool | None] = {}
    for r in rows:
        gap = r.get("iep_gap_pct")
        pts = [p for p in (tape.get(r["symbol"]) or {}).get("path") or [] if p[0] <= CHECK_MINUTE]
        if gap is None or not pts or pts[-1][0] < CHECK_MINUTE - 2 or pts[-1][1] == 0 or gap == 0:
            out[r["symbol"]] = None
            continue
        out[r["symbol"]] = (gap > 0) == (pts[-1][1] > 0)
    return out


class PreopenEngine:
    """Polls the pre-open board in its window (one `step()` per iteration) and serves the stored board."""

    def __init__(
        self,
        clock: Clock,
        cal: TradingCalendar,
        sources: Any,
        store: LabStore,
        capture: Callable[[], int] | None = None,
    ) -> None:
        self.clock, self.cal, self.src, self.store, self.capture = clock, cal, sources, store, capture

    def in_window(self) -> bool:
        """True on a session between 09:00:30 and 09:14:30."""
        now = self.clock.now()
        return self.cal.is_session(now.date()) and WINDOW_START <= now.time() <= WINDOW_END

    def _state(self, today: str) -> dict[str, Any]:
        with self.store.lock:
            st = self.store.flow.get("preopen")
            if not st or st.get("day") != today:
                st = {
                    "day": today,
                    "rows": {},
                    "nifty": None,
                    "observed": {"first_row": None, "first_final": None, "captured_at": None},
                    "captured": False,
                }
                self.store.flow["preopen"] = st
            return st

    def step(self) -> float:
        """Poll once if inside 09:00:30-09:14:30 on a session; returns the seconds to wait."""
        now = self.clock.now()
        t = now.time()
        if not self.cal.is_session(now.date()) or t > WINDOW_END:
            return 60.0
        if t < WINDOW_START:
            return float(max(1, min(30, _secs(WINDOW_START) - _secs(t)))) if t >= time(8, 30) else 60.0
        st = self._state(now.date().isoformat())
        fo = parse_preopen(self.src.nse_preopen("FO"), now)
        nf = parse_preopen(self.src.nse_preopen("NIFTY"), now)
        rows = {r["symbol"]: r for r in board_rows(fo)}
        idx = next((r for r in board_rows(nf) if r["symbol"] in NIFTY_NAMES), None)
        stamp = now.strftime("%H:%M:%S")
        with self.store.lock:
            st["rows"] = rows
            st["nifty"] = idx and {"iep": idx["iep"], "gap_pct": idx["iep_gap_pct"]}
            st["updated"] = now.isoformat(timespec="seconds")
            obs = st["observed"]
            if obs["first_row"] is None and rows:
                obs["first_row"] = stamp
            if obs["first_final"] is None and nf["final_price"].drop_nulls().len() > 0:
                obs["first_final"] = stamp
        due = obs["first_final"] is not None or t >= FALLBACK_CAPTURE
        if due and not st["captured"] and self.capture is not None:
            n = self.capture()
            with self.store.lock:
                st["captured"], obs["captured_at"] = True, stamp
            log.info("preopen_captured", rows=n, at=stamp)
        return 30.0

    def payload(self) -> dict[str, Any]:
        """The /api/flow/preopen body: phase, NIFTY indicative open, board rows, breadth, follow-through."""
        now = self.clock.now()
        t = now.time()
        with self.store.lock:
            st = self.store.flow.get("preopen") or {}
            rows = list(st.get("rows", {}).values()) if st.get("day") == now.date().isoformat() else []
            nifty = st.get("nifty") if rows else None
            tape = dict(self.store.tape.get("stocks", {}))
        phase = "before" if t < time(9, 0) else "live" if t < OPEN_TIME else "after"
        gaps = [r["iep_gap_pct"] for r in rows if r["iep_gap_pct"] is not None]
        out: dict[str, Any] = {
            "descriptive": True,
            "phase": phase,
            "nifty": nifty,
            "rows": [dict(r) for r in rows],
            "breadth": {"adv": sum(g > 0 for g in gaps), "dec": sum(g < 0 for g in gaps)},
        }
        if phase == "after":
            ft = follow_through(rows, tape)
            for r in out["rows"]:
                r["follow_through"] = ft[r["symbol"]]
        return out


def observed_times(store: LabStore) -> dict[str, Any]:
    """The pre-open times seen today (first row, first final price, capture); for docs/DECISIONS.md."""
    with store.lock:
        return dict((store.flow.get("preopen") or {}).get("observed", {}))
