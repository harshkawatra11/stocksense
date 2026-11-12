"""Quote sources and the fallback hub (T10.2). Read-only: nothing here sends an order."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any, Protocol

from stocksense.brokers.types import Quote, to_float
from stocksense.core.clock import IST, Clock
from stocksense.core.log import get_logger

log = get_logger(__name__)

NSE_MCP_URL = "https://mcp.nseindia.in/cmmkt/mcp"
NSE_MCP_TOOL = "cm_get_stock_quote"
NSE_MCP_SOURCE = "nse_mcp_delayed"
BREAKER = timedelta(minutes=5)
STALE_AFTER = timedelta(minutes=3)


class QuoteSource(Protocol):
    name: str
    #: only asked for symbols the caller holds (a slow one-symbol-per-call source)
    held_only: bool

    def quotes(self, symbols: list[str]) -> list[Quote]: ...


class AngelQuotes:
    """FULL-mode quotes through AngelClient.quote_full (symbols without a token are skipped)."""

    name = "angel"
    held_only = False

    def __init__(self, client: Any, instruments: Any) -> None:
        self._c, self._i = client, instruments

    def quotes(self, symbols: list[str]) -> list[Quote]:
        tokens = [t for s in symbols if (t := self._i.angel_token(s))]
        return self._c.quote_full(tokens) if tokens else []


class UpstoxQuotes:
    name = "upstox"
    held_only = False

    def __init__(self, client: Any, instruments: Any) -> None:
        self._c, self._i = client, instruments

    def quotes(self, symbols: list[str]) -> list[Quote]:
        keys = [k for s in symbols if (k := self._i.upstox_key(s))]
        return self._c.quotes(keys) if keys else []


def parse_nse_mcp_quote(payload: dict[str, Any] | str, fallback_ts: datetime | None = None) -> Quote:
    """Quote from a `cm_get_stock_quote` answer (shape observed 2026-11-12: `{"stock": {...}}`)."""
    j = json.loads(payload) if isinstance(payload, str) else payload
    s = j.get("stock") or j
    ts = fallback_ts
    raw = s.get("latestTimestamp")
    if raw:
        with contextlib.suppress(ValueError):
            ts = datetime.fromisoformat(str(raw).strip()).replace(tzinfo=IST)
    if ts is None:
        raise ValueError("quote without a timestamp")
    return Quote(
        symbol=str(s.get("symbol") or ""),
        ts=ts,
        ltp=to_float(s.get("lastTradedPrice")),
        open=to_float(s.get("openPrice")),
        high=to_float(s.get("highPrice")),
        low=to_float(s.get("lowPrice")),
        prev_close=to_float(s.get("preClosePrice")),
        volume=to_float(s.get("volume")),
        bid=None,
        ask=None,
        bid_qty=None,
        ask_qty=None,
        upper_circuit=None,
        lower_circuit=None,
        source=NSE_MCP_SOURCE,
    )


async def _mcp_quotes(url: str, symbols: list[str]) -> list[str]:
    """Raw answers of the stock-quote tool, one call per symbol (argument name read from list_tools)."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    out: list[str] = []
    async with (
        streamable_http_client(url) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        tools = {t.name: t for t in (await session.list_tools()).tools}
        tool = tools.get(NSE_MCP_TOOL)
        if tool is None:
            raise RuntimeError(f"{NSE_MCP_TOOL} not offered; tools: {sorted(tools)}")
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        props = schema.get("properties", {})
        arg = next((p for p in props if p.lower() in ("symbol", "ticker", "stock")), None)
        if arg is None:
            arg = (schema.get("required") or list(props) or ["symbol"])[0]
        for sym in symbols:
            res = await session.call_tool(tool.name, {arg: sym})
            out.append("".join(getattr(c, "text", "") for c in res.content))
    return out


class NseMcpQuotes:
    """Delayed LTP from the official NSE MCP server; one symbol per call, so held names only."""

    name = NSE_MCP_SOURCE
    held_only = True

    def __init__(
        self, clock: Clock, url: str = NSE_MCP_URL, call: Callable[[str], str | dict[str, Any]] | None = None
    ) -> None:
        self._clock, self._url, self._call = clock, url, call

    def quotes(self, symbols: list[str]) -> list[Quote]:
        if self._call is not None:
            raws = [self._call(s) for s in symbols]
        else:
            raws = asyncio.run(_mcp_quotes(self._url, symbols))
        out = []
        for sym, raw in zip(symbols, raws, strict=False):
            qt = parse_nse_mcp_quote(raw, self._clock.now())
            out.append(qt if qt.symbol else _with_symbol(qt, sym))
        return out


def _with_symbol(qt: Quote, sym: str) -> Quote:
    from dataclasses import replace

    return replace(qt, symbol=sym)


class QuoteHub:
    """Ordered fallback over quote sources with a 5-minute circuit breaker and a 3-minute staleness drop."""

    def __init__(self, sources: list[QuoteSource], clock: Clock) -> None:
        if not sources:
            raise ValueError("QuoteHub needs at least one source")
        self._sources = list(sources)
        self._clock = clock
        self._open_until: dict[str, datetime] = {}
        self.degraded = False
        self.last_errors: dict[str, str] = {}

    def _fresh(self, qt: Quote) -> bool:
        return self._clock.now() - qt.ts <= STALE_AFTER

    def get(self, symbols: Iterable[str], held: Iterable[str] | None = None) -> dict[str, Quote]:
        """Best available quote per symbol. `held` names may be asked of `held_only` sources."""
        want = list(dict.fromkeys(symbols))
        held_set = set(held or ())
        got: dict[str, Quote] = {}
        self.last_errors = {}
        first_served = False
        for i, src in enumerate(self._sources):
            missing = [s for s in want if s not in got]
            if src.held_only:
                missing = [s for s in missing if s in held_set]
            if not missing:
                if i == 0:
                    first_served = True
                continue
            until = self._open_until.get(src.name)
            if until is not None and self._clock.now() < until:
                continue
            try:
                res = src.quotes(missing)
            except Exception as e:  # a failing source is skipped for 5 minutes
                self._open_until[src.name] = self._clock.now() + BREAKER
                self.last_errors[src.name] = f"{type(e).__name__}: {e}"
                log.warning("quote_source_failed", source=src.name, error=str(e))
                continue
            wanted = set(missing)
            served = 0
            for qt in res:
                if qt.symbol in wanted and qt.symbol not in got and self._fresh(qt):
                    got[qt.symbol] = qt
                    served += 1
            if i == 0 and (served > 0 or not want):
                first_served = True
        self.degraded = bool(want) and not first_served
        return got
