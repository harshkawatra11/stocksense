from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from stocksense.brokers.types import Quote, to_float
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import BrokerError
from stocksense.core.log import get_logger

log = get_logger(__name__)

BASE = "https://api.upstox.com"
SPACING_S = 1.0  # 2,000 requests per 30 minutes
QUOTE_CHUNK = 500
BACKOFF_S = (30.0, 120.0)
_UNIT_DAYS = {"days": 3650, "weeks": 3650, "months": 3650}
_SCHEMA = {
    "ts": pl.Datetime("us", "Asia/Kolkata"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "oi": pl.Float64,
}


def request_windows(unit: str, interval: int, start: date, end: date) -> list[tuple[date, date]]:
    """Inclusive (from, to) windows within the API's span limit: minutes <= 15: 31 days (probed 2026-11-11),
    longer minutes and hours: 90 days, days and up: ten years."""
    if unit == "minutes":
        span = 31 if interval <= 15 else 90
    elif unit == "hours":
        span = 90
    else:
        span = _UNIT_DAYS.get(unit, 3650)
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        last = min(cur + timedelta(days=span - 1), end)
        out.append((cur, last))
        cur = last + timedelta(days=1)
    return out


def parse_candles(rows: list[list[Any]]) -> pl.DataFrame:
    """Candle arrays `[ts, o, h, l, c, v, oi]` (newest first in the API) as an ascending frame."""
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(
        {
            "ts": [datetime.fromisoformat(r[0]) for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
            "oi": [float(r[6]) if len(r) > 6 and r[6] is not None else 0.0 for r in rows],
        }
    ).with_columns(pl.col("ts").dt.convert_time_zone("Asia/Kolkata").dt.cast_time_unit("us"))
    return df.unique(subset=["ts"], keep="last").sort("ts")


class UpstoxClient:
    """Read-only Upstox client authenticated with the long-lived Analytics Token."""

    def __init__(
        self, settings: Settings, clock: Clock, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._token = settings.secrets.upstox_analytics_token
        self._clock = clock
        self._http = httpx.Client(timeout=httpx.Timeout(60, connect=15), transport=transport)
        self._last: datetime | None = None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        if not self._token:
            raise BrokerError("upstox token not set")
        waits = list(BACKOFF_S)
        while True:
            if self._last is not None:
                wait = SPACING_S - (self._clock.now() - self._last).total_seconds()
                if wait > 0:
                    self._clock.sleep(wait)
            self._last = self._clock.now()
            r = self._http.get(
                BASE + path,
                params=params,
                headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
            )
            if r.status_code == 429 or r.status_code >= 500:
                if not waits:
                    raise BrokerError(f"upstox {path}: HTTP {r.status_code} after retries")
                self._clock.sleep(waits.pop(0))
                continue
            try:
                j = r.json()
            except ValueError:
                raise BrokerError(f"upstox {path}: non-JSON response (HTTP {r.status_code})") from None
            if r.status_code != 200 or j.get("status") != "success":
                code = ((j.get("errors") or [{}])[0]).get("errorCode") or r.status_code
                raise BrokerError(f"upstox {path}: error {code}")
            return j

    def candles(self, instrument_key: str, unit: str, interval: int, start: date, end: date) -> pl.DataFrame:
        """Candles (ts IST, open, high, low, close, volume, oi) for [start, end], ascending; the range
        is split into requests the API accepts."""
        rows: list[list[Any]] = []
        for a, b in request_windows(unit, interval, start, end):
            path = f"/v3/historical-candle/{instrument_key}/{unit}/{interval}/{b:%Y-%m-%d}/{a:%Y-%m-%d}"
            rows += (self._get(path).get("data") or {}).get("candles") or []
        return parse_candles(rows)

    def quotes(self, instrument_keys: list[str]) -> list[Quote]:
        """Full market quotes (documented shape: data keyed `NSE_EQ:SYMBOL`).

        2026-11-11 probe: the Analytics Token is refused here (HTTP 401, UDAPI100050), so this raises
        BrokerError with the token the project has; kept for a token that is allowed.
        """
        out: list[Quote] = []
        for i in range(0, len(instrument_keys), QUOTE_CHUNK):
            j = self._get(
                "/v2/market-quote/quotes", {"instrument_key": ",".join(instrument_keys[i : i + QUOTE_CHUNK])}
            )
            for k, q in (j.get("data") or {}).items():
                out.append(parse_quote(k, q, self._clock.now()))
        return out

    def close(self) -> None:
        self._http.close()


def parse_quote(key: str, q: dict[str, Any], now: datetime) -> Quote:
    """Quote from one `data` entry of the market-quote response."""
    ohlc = q.get("ohlc") or {}
    depth = q.get("depth") or {}
    buy = (depth.get("buy") or [{}])[0]
    sell = (depth.get("sell") or [{}])[0]
    ts = now
    if q.get("timestamp"):
        try:
            ts = datetime.fromisoformat(str(q["timestamp"]).replace("Z", "+00:00"))
        except ValueError:
            ts = now
    return Quote(
        symbol=str(q.get("symbol") or key.split(":", 1)[-1]),
        ts=ts,
        ltp=to_float(q.get("last_price")),
        open=to_float(ohlc.get("open")),
        high=to_float(ohlc.get("high")),
        low=to_float(ohlc.get("low")),
        prev_close=to_float(ohlc.get("close")),
        volume=to_float(q.get("volume")),
        bid=to_float(buy.get("price")),
        ask=to_float(sell.get("price")),
        bid_qty=to_float(buy.get("quantity")),
        ask_qty=to_float(sell.get("quantity")),
        upper_circuit=to_float(q.get("upper_circuit_limit")),
        lower_circuit=to_float(q.get("lower_circuit_limit")),
        source="upstox",
    )
