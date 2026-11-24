"""Append-only paper journal of the radar's signals (PLAN-M M5.8). A paper record only: no broker, no orders.

A row is written once, when a scan finds a SIGNAL: the signal session (`trade_date`), strategy, stock, side,
reference price (the signal close), stop and target. SQLite refuses any later change to those fields (a
trigger), so the history cannot be edited. Only `status`, `closed_ts`, `close_price` and `reason` move:
planned -> open at the next open, open -> closed when the strategy's exit rule hits (or planned -> expired
when the trade would not have been taken).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any

import numpy as np

from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.state import State
from stocksense.strategies import base
from stocksense.strategies.data import nifty_table
from stocksense.strategies.exits import follow_1d
from stocksense.strategies.explore import HOLD_DAYS, MAX_RISK, load_ctx, make_prep

log = get_logger(__name__)

COLUMNS = (
    "id",
    "created_ts",
    "trade_date",
    "strat",
    "sym",
    "side",
    "ref_price",
    "stop",
    "target",
    "status",
    "closed_ts",
    "close_price",
    "reason",
)
FALLBACK_EXIT = "trail3"  # strategies that are not in the registry (for example picks:<trial>)
EXPIRE_AFTER_DAYS = 10  # a planned row with no bar after this many calendar days never entered


def _signal_date(row: Mapping[str, Any], trade_date: date | None) -> date:
    """Signal session of a scan row: given, carried in the row, or t_in (the nominal next day) minus one."""
    if trade_date is not None:
        return trade_date
    if row.get("trade_date"):
        return date.fromisoformat(str(row["trade_date"]))
    t_in = int(row["t_in"])
    return date(1970, 1, 1) + timedelta(seconds=t_in - 86400)


def record_signals(
    state: State,
    scan_rows: Iterable[Mapping[str, Any]],
    clock: Clock,
    *,
    trade_date: date | None = None,
    strat: str | None = None,
) -> int:
    """Insert the SIGNAL rows of a scan as `planned`; returns how many were new.

    Idempotent by (trade_date, strat, sym): `INSERT OR IGNORE`. `trade_date` defaults to the signal session
    derived from the row's `t_in`; `strat` overrides the row's strategy (the picks use "picks:<trial>").
    Rows whose state is not SIGNAL are skipped.
    """
    now = clock.now().isoformat(timespec="seconds")
    new = 0
    con = state.connect()
    try:
        for r in scan_rows:
            if r.get("state", "SIGNAL") != "SIGNAL":
                continue
            cur = con.execute(
                "INSERT OR IGNORE INTO lab_journal(created_ts, trade_date, strat, sym, side, ref_price, stop,"
                " target, status, reason) VALUES(?,?,?,?,?,?,?,?,'planned',?)",
                (
                    now,
                    _signal_date(r, trade_date).isoformat(),
                    strat or r["strat"],
                    r["sym"],
                    int(r.get("side") or 1),
                    float(r["p_in"]),
                    None if r.get("stop") is None else float(r["stop"]),
                    None if r.get("target") is None else float(r["target"]),
                    r.get("reason") or "signal at the close; enter at the next open",
                ),
            )
            new += cur.rowcount
    finally:
        con.close()
    return new


def _exit_for(strat: str) -> str:
    try:
        return base.by_name_or_key(strat).default_exit
    except KeyError:
        return FALLBACK_EXIT


def update_marks(state: State, settings: Settings, asof: date) -> dict[str, int]:
    """Advance open journal rows using completed bars up to `asof`; returns {opened, closed, expired}.

    planned -> open at the first session after the signal (entry at its open; expired instead when the stop
    is at or above that open or the risk exceeds 25%, as the engine skips such trades). open -> closed when
    `follow_1d` on the strategy's default exit ends before the last bar. Only status fields change.
    """
    base.load_library()
    con = state.connect()
    con.row_factory = sqlite3.Row
    counts = {"opened": 0, "closed": 0, "expired": 0}
    try:
        rows = con.execute(
            "SELECT * FROM lab_journal WHERE status IN('planned','open') AND trade_date <= ? ORDER BY id",
            (asof.isoformat(),),
        ).fetchall()
        if not rows:
            return counts
        ctxs = load_ctx(settings, sorted({r["sym"] for r in rows}), asof)
        nifty = nifty_table(settings, end=asof)
        preps: dict[str, Any] = {}
        for r in rows:
            ctx = ctxs.get(r["sym"])
            if ctx is None:
                _expire_if_stale(con, r, asof, "no bars for the symbol", counts)
                continue
            if r["sym"] not in preps:
                preps[r["sym"]] = make_prep(settings, ctx, nifty)
            f = preps[r["sym"]].f
            dates = ctx.frame.date
            ie = int(np.searchsorted(dates, np.datetime64(r["trade_date"]), side="right"))
            if ie >= f["n"]:  # the entry session has not completed yet
                _expire_if_stale(con, r, asof, "no bar after the signal", counts)
                continue
            stop = r["stop"]
            e = float(f["o"][ie])
            if r["status"] == "planned":
                if stop is None or e - stop <= 0 or (e - stop) / e > MAX_RISK:
                    _set(con, r["id"], "expired", None, None, f"skipped: open {e:.2f} against stop {stop}")
                    counts["expired"] += 1
                    continue
                _set(con, r["id"], "open", None, None, f"entered at the open {e:.2f} on {dates[ie]}")
                counts["opened"] += 1
            if stop is None:
                continue
            xk = _exit_for(r["strat"])
            j, px, why = follow_1d(f, ie, float(stop), xk)
            still_open = j == f["n"] - 1 and why == "time" and j - ie + 1 < HOLD_DAYS[xk]
            if not still_open:
                _set(con, r["id"], "closed", str(dates[j]), float(px), why)
                counts["closed"] += 1
    finally:
        con.close()
    return counts


def _expire_if_stale(
    con: sqlite3.Connection, r: sqlite3.Row, asof: date, why: str, counts: dict[str, int]
) -> None:
    if r["status"] == "planned" and date.fromisoformat(r["trade_date"]) < asof - timedelta(
        days=EXPIRE_AFTER_DAYS
    ):
        _set(con, r["id"], "expired", None, None, f"expired: {why}")
        counts["expired"] += 1


def _set(
    con: sqlite3.Connection,
    row_id: int,
    status: str,
    closed_ts: str | None,
    close_price: float | None,
    reason: str,
) -> None:
    con.execute(
        "UPDATE lab_journal SET status=?, closed_ts=COALESCE(?, closed_ts),"
        " close_price=COALESCE(?, close_price), reason=? WHERE id=?",
        (status, closed_ts, close_price, reason, row_id),
    )


def read(settings: Settings, days: int = 30, clock: Clock | None = None) -> dict[str, Any]:
    """Market Lab journal shape {"trades": [rows], "paper": True} for the last `days` days, newest first."""
    clk = clock or SystemClock()
    con = State(settings, clk).connect()
    con.row_factory = sqlite3.Row
    try:
        has = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='lab_journal'").fetchone()
        if has is None:
            return {"trades": [], "paper": True}
        cut = (clk.today() - timedelta(days=days)).isoformat()
        rows = con.execute(
            "SELECT * FROM lab_journal WHERE trade_date >= ? ORDER BY trade_date DESC, id DESC", (cut,)
        ).fetchall()
    finally:
        con.close()
    return {"trades": [dict(zip(COLUMNS, tuple(r), strict=True)) for r in rows], "paper": True}
