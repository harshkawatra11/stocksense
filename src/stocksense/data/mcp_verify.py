"""Nightly verification of our raw data against the official NSE bhavcopy MCP server (G2-H5).

Three read-only checks, all in one MCP session: (1) `n` random closes of day d (`get_ltp_by_date`; the bulk
tool takes no date) against `bhavcopy_eq` within 0.01; (2) corporate actions of the last 30 days
(`get_corporate_actions`) against our table; (3) the last five NIFTY 50 closes (`get_index_history`)
against `index_daily`.
The result never blocks anything: it is data for the report and an alert.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl

from stocksense.core.config import Settings
from stocksense.data.io import read

Batch = Callable[[list[tuple[str, dict[str, Any]]]], list[Any]]

TOLERANCE = 0.01
WINDOW_DAYS = 30
INDEX = "NIFTY 50"
INDEX_DAYS = 5
MAX_ACTION_SYMBOLS = 15
#: server action type (upper case) -> the one of ours it must match; our other types match anything
_PRICE_TYPES = {"SPLIT": "split", "BONUS": "bonus", "DIVIDEND": "dividend"}


@dataclass
class VerifyResult:
    status: str  # ok | mismatch | unreachable
    mismatch: int = 0
    closes_checked: int = 0
    actions_checked: int = 0
    index_checked: int = 0
    lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mismatch": self.mismatch,
            "closes_checked": self.closes_checked,
            "actions_checked": self.actions_checked,
            "index_checked": self.index_checked,
            "lines": self.lines[:20],
        }


def _json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return raw


def _number(raw: Any) -> float | None:
    v = _json(raw)
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, dict):
        for k in ("close", "ltp", "price"):
            if isinstance(v.get(k), int | float):
                return float(v[k])
    return None


def _compatible(ours: str, theirs: str) -> bool:
    """Our action type against the server's (SPLIT, BONUS, DIVIDEND, OTHER...) on the same ex-date."""
    if theirs in _PRICE_TYPES:
        return ours == _PRICE_TYPES[theirs] or (theirs == "DIVIDEND" and ours in ("distribution", "noise"))
    return ours not in _PRICE_TYPES.values()  # the server files rights, buybacks etc. under OTHER


def _action_symbols(ours: pl.DataFrame, extra: Sequence[str], seed: int) -> list[str]:
    """Symbols to ask about: every split/bonus first (the price-affecting ones), then random others."""
    syms = sorted(set(ours["symbol"].to_list()))
    key = set(ours.filter(pl.col("action_type").is_in(["split", "bonus"]))["symbol"].to_list())
    rest = [s for s in syms if s not in key]
    random.Random(seed).shuffle(rest)
    picked = (sorted(key) + rest)[:MAX_ACTION_SYMBOLS]
    return list(dict.fromkeys([*picked, *extra]))


def verify(
    settings: Settings,
    d: date,
    batch: Batch,
    *,
    n: int = 20,
    extra_symbols: Sequence[str] = (),
) -> VerifyResult:
    """Run the three checks for day d. `batch(calls)` answers a list of (tool, args) in one session (None for
    a call that failed); an exception from it, or no answer at all, gives status 'unreachable'."""
    from stocksense.data.ingest_eq import sample_pairs

    seed = d.toordinal()
    lo = d - timedelta(days=WINDOW_DAYS)
    pairs = sample_pairs(settings, [d], n, seed=seed)
    ours_ca = (
        read(settings, "corporate_actions", start=lo, end=d)
        .select("symbol", "ex_date", "action_type")
        .collect()
    )
    syms = _action_symbols(ours_ca, extra_symbols, seed)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_ltp_by_date", {"symbol": s, "date": d.isoformat()}) for s, _ in pairs
    ]
    calls += [
        ("get_corporate_actions", {"symbol": s, "fromDate": lo.isoformat(), "toDate": d.isoformat()})
        for s in syms
    ]
    calls.append(("get_index_history", {"indexName": INDEX, "months": 1, "endDate": d.isoformat()}))
    try:
        answers = batch(calls)
    except Exception:  # noqa: BLE001 - an unreachable server is a skip, not a failure
        return VerifyResult("unreachable")
    if len(answers) != len(calls) or all(a is None for a in answers):
        return VerifyResult("unreachable")
    res = VerifyResult("ok")
    close_ans = answers[: len(pairs)]
    ca_ans = answers[len(pairs) : len(pairs) + len(syms)]
    idx_ans = answers[-1]

    # (1) closes
    stored = (
        read(settings, "bhavcopy_eq", start=d, end=d, columns=["symbol", "series", "date", "close"])
        .filter(pl.col("series") == "EQ")
        .collect()
    )
    mine = dict(zip(stored["symbol"].to_list(), stored["close"].to_list(), strict=True))
    for (sym, _), raw in zip(pairs, close_ans, strict=True):
        theirs, ours = _number(raw), mine.get(sym)
        if theirs is None or ours is None:
            continue
        res.closes_checked += 1
        if abs(theirs - ours) > TOLERANCE:
            res.mismatch += 1
            res.lines.append(f"close {sym} {d}: ours {ours:.2f}, NSE {theirs:.2f}")

    # (2) corporate actions
    for sym, raw in zip(syms, ca_ans, strict=True):
        j = _json(raw)
        if not isinstance(j, dict) or not isinstance(j.get("actions"), list):
            continue
        res.actions_checked += 1
        theirs: dict[date, str] = {}
        for a in j["actions"]:
            try:
                ex = date.fromisoformat(str(a["exDate"])[:10])
            except (KeyError, ValueError):
                continue
            if lo <= ex <= d:
                theirs[ex] = str(a.get("actionType", "")).upper()
        mine_a = {
            r["ex_date"]: str(r["action_type"]) for r in ours_ca.filter(pl.col("symbol") == sym).to_dicts()
        }
        for ex, typ in sorted(mine_a.items()):
            if ex not in theirs:
                res.mismatch += 1
                res.lines.append(f"corporate action {sym} ex {ex} ({typ}): missing at NSE")
            elif not _compatible(typ, theirs[ex]):
                res.mismatch += 1
                res.lines.append(f"corporate action {sym} ex {ex}: ours {typ}, NSE {theirs[ex].lower()}")
        for ex, typ in sorted(theirs.items()):
            if typ in _PRICE_TYPES and ex not in mine_a:
                res.mismatch += 1
                res.lines.append(f"corporate action {sym} ex {ex} ({typ.lower()}): missing in our table")

    # (3) index history
    ij = _json(idx_ans)
    rows = ij.get("data") if isinstance(ij, dict) else None
    if isinstance(rows, list):
        last: list[tuple[date, float]] = []
        for r in rows:
            try:
                last.append((date.fromisoformat(str(r["date"])[:10]), float(r["close"])))
            except (KeyError, ValueError, TypeError):
                continue
        last = sorted(x for x in last if x[0] <= d)[-INDEX_DAYS:]
        idx = (
            read(settings, "index_daily", start=last[0][0], end=d, columns=["index_name", "date", "close"])
            .filter(pl.col("index_name") == INDEX)
            .collect()
            if last
            else pl.DataFrame({"date": [], "close": []})
        )
        mine_i = dict(zip(idx["date"].to_list(), idx["close"].to_list(), strict=True))
        for day, theirs_c in last:
            ours_c = mine_i.get(day)
            if ours_c is None:
                res.mismatch += 1
                res.lines.append(f"index {INDEX} {day}: missing in index_daily")
                continue
            res.index_checked += 1
            if abs(ours_c - theirs_c) > TOLERANCE:
                res.mismatch += 1
                res.lines.append(f"index {INDEX} {day}: ours {ours_c:.2f}, NSE {theirs_c:.2f}")
    if res.mismatch:
        res.status = "mismatch"
    return res
