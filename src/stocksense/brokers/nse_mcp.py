"""Read-only helpers for the two official NSE MCP servers (G2-H4, G2-H5).

`cm-market` serves live cash-market data (indices, call-auction list, crawl status); `nse-bhavcopy` serves
end-of-day history. Everything here is read-only and failure-tolerant: a caller treats any exception as
"source unavailable" and carries on. No secrets are involved (both servers are public).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Any

from stocksense.core.clock import IST
from stocksense.core.errors import DataNotAvailable

CM_URL = "https://mcp.nseindia.in/cmmkt/mcp"
BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"
#: both servers' data older than this inside market hours is treated as not available
STALE_AFTER = timedelta(minutes=15)
MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)
#: the server returns at most 500 rows per call whatever `limit` says (probe 2026-11-20)
CALL_AUCTION_LIMIT = 500

#: (tool, arguments) -> raw text (or an already decoded object) of the tool answer
McpCall = Callable[[str, dict[str, Any]], "str | dict[str, Any] | list[Any]"]


class McpError(Exception):
    """The MCP server was unreachable or answered with an error."""


def _text(res: Any) -> str:
    return "".join(getattr(c, "text", "") for c in res.content)


async def _call_async(url: str, tool: str, args: dict[str, Any]) -> str:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        streamable_http_client(url) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        res = await session.call_tool(tool, args)
        txt = _text(res)
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise McpError(f"{tool}: {txt[:120]}")
        return txt


def make_call(url: str, attempts: int = 2) -> McpCall:
    """A real MCP caller: one short session per call, one retry (`initialize` fails now and then)."""

    def call(tool: str, args: dict[str, Any]) -> str:
        last: BaseException | None = None
        for _ in range(attempts):
            try:
                return asyncio.run(_call_async(url, tool, args))
            except BaseException as e:  # noqa: BLE001 - includes anyio ExceptionGroups from the transport
                if isinstance(e, KeyboardInterrupt):
                    raise
                last = e
        raise McpError(f"{tool}: {type(last).__name__}") from last

    return call


async def _batch_async(url: str, calls: list[tuple[str, dict[str, Any]]]) -> list[str | None]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    out: list[str | None] = []
    async with (
        streamable_http_client(url) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        for tool, args in calls:
            try:
                res = await session.call_tool(tool, args)
            except Exception:  # noqa: BLE001 - one failed call is one missing answer
                out.append(None)
                continue
            bad = getattr(res, "is_error", False) or getattr(res, "isError", False)
            out.append(None if bad else _text(res))
    return out


def call_many(url: str, calls: list[tuple[str, dict[str, Any]]], attempts: int = 2) -> list[str | None]:
    """Answers of several tool calls in ONE session (None for a call that failed); McpError if the
    session cannot be opened after `attempts` tries."""
    last: BaseException | None = None
    for _ in range(attempts):
        try:
            return asyncio.run(_batch_async(url, calls))
        except BaseException as e:  # noqa: BLE001 - includes anyio ExceptionGroups from the transport
            if isinstance(e, KeyboardInterrupt):
                raise
            last = e
    raise McpError(f"session failed: {type(last).__name__}") from last


def as_json(raw: str | dict[str, Any] | list[Any]) -> Any:
    """Decode a tool answer; an `{"error": ...}` payload or non-JSON text raises McpError."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as e:
            raise McpError(f"non-JSON answer: {raw[:80]!r}") from e
    if isinstance(raw, dict) and raw.get("error"):
        raise McpError(str(raw["error"])[:120])
    return raw


# ---- freshness ----------------


def in_market_hours(now: datetime) -> bool:
    n = now.astimezone(IST)
    return n.weekday() < 5 and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def _utc_iso(v: Any) -> datetime | None:
    if not v:
        return None
    s = re.sub(r"(\.\d{6})\d+", r"\1", str(v).strip().replace("Z", "+00:00"))  # nanoseconds -> micro
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=IST)


def last_crawled(status: dict[str, Any]) -> datetime | None:
    """The OLDEST `lastCrawled` of a `cm_get_allstocks_status` (flat) or `cm_get_data_status` answer."""
    stamps: list[datetime] = []
    if "lastCrawled" in status:
        stamps.append(_utc_iso(status["lastCrawled"]))  # type: ignore[arg-type]
    for v in (status.get("indices") or {}).values():
        if isinstance(v, dict) and "lastCrawled" in v:
            stamps.append(_utc_iso(v["lastCrawled"]))  # type: ignore[arg-type]
    good = [s for s in stamps if s is not None]
    return min(good) if good else None


def assert_fresh(status: Any, now: datetime, *, max_age: timedelta = STALE_AFTER) -> None:
    """Raise DataNotAvailable when the last crawl is older than `max_age` inside market hours."""
    if not in_market_hours(now):
        return
    st = as_json(status)
    ts = last_crawled(st) if isinstance(st, dict) else None
    if ts is None:
        raise DataNotAvailable("cm-market status has no crawl time")
    age = now - ts
    if age > max_age:
        mins = int(age.total_seconds() // 60)
        raise DataNotAvailable(f"cm-market data is stale: last crawl {mins} min ago")


def check_fresh(call: McpCall, now: datetime) -> None:
    """`assert_fresh` over `cm_get_allstocks_status` (the crawl behind the stock tools)."""
    if in_market_hours(now):
        assert_fresh(call("cm_get_allstocks_status", {}), now)


# ---- live indices ----------------

LIVE_INDICES = ("NIFTY 50", "INDIA VIX")


def parse_index_quote(raw: Any) -> dict[str, Any]:
    """{name, last, prev_close, change_pct, open, high, low, ts} from a `cm_get_index_quote` answer."""
    j = as_json(raw)
    ix = j.get("index") if isinstance(j, dict) else None
    if not isinstance(ix, dict) or ix.get("last") is None:
        raise McpError("index quote without a last value")
    return {
        "name": ix.get("indexName"),
        "last": float(ix["last"]),
        "prev_close": ix.get("previousClose"),
        "change_pct": ix.get("percChange"),
        "open": ix.get("open"),
        "high": ix.get("high"),
        "low": ix.get("low"),
        "ts": j.get("nseTimestamp") or ix.get("timeVal"),
    }


def _nse_ts(v: Any) -> datetime | None:
    try:
        return datetime.strptime(str(v), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
    except ValueError:
        try:
            return datetime.strptime(str(v), "%d-%b-%Y %H:%M").replace(tzinfo=IST)
        except ValueError:
            return None


def fetch_live_index(
    call: McpCall, now: datetime, names: tuple[str, ...] = LIVE_INDICES
) -> dict[str, dict[str, Any]]:
    """Live values of the named indices; a quote older than 15 minutes in market hours is DataNotAvailable."""
    out: dict[str, dict[str, Any]] = {}
    for n in names:
        q = parse_index_quote(call("cm_get_index_quote", {"indexName": n}))
        ts = _nse_ts(q["ts"])
        if in_market_hours(now) and ts is not None and now - ts > STALE_AFTER:
            raise DataNotAvailable(f"{n} quote is stale ({q['ts']})")
        out[n] = q
    return out


# ---- call auction ----------------


def parse_call_auction(raw: Any) -> list[str]:
    """Sorted unique symbols of a `cm_get_call_auction_stocks` answer."""
    j = as_json(raw)
    rows = j.get("stocks") if isinstance(j, dict) else None
    if not isinstance(rows, list):
        raise McpError("call-auction answer without a stocks list")
    return sorted({str(r["symbol"]) for r in rows if isinstance(r, dict) and r.get("symbol")})


def fetch_call_auction(call: McpCall) -> list[str]:
    args = {"limit": CALL_AUCTION_LIMIT, "symbolFilter": ""}
    return parse_call_auction(call("cm_get_call_auction_stocks", args))


def call_auction_symbols(state: Any, d: date) -> frozenset[str]:
    """The stored call-auction list of day d (kv `call_auction:{d}`); empty when absent or unreadable."""
    try:
        raw = state.kv_get(f"call_auction:{d.isoformat()}")
        return frozenset(str(s) for s in json.loads(raw)) if raw else frozenset()
    except (ValueError, TypeError):
        return frozenset()
