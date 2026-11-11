from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import httpx
import polars as pl
import pyotp

from stocksense.brokers.types import Funds, Holding, Position, Quote, to_float
from stocksense.core.clock import IST, Clock
from stocksense.core.config import Settings
from stocksense.core.errors import BrokerError
from stocksense.core.log import get_logger

log = get_logger(__name__)

BASE = "https://apiconnect.angelone.in"
_LOGIN = "/rest/auth/angelbroking/user/v1/loginByPassword"
_RMS = "/rest/secure/angelbroking/user/v1/getRMS"
_HOLD = "/rest/secure/angelbroking/portfolio/v1/getAllHolding"
_POS = "/rest/secure/angelbroking/order/v1/getPosition"
_QUOTE = "/rest/secure/angelbroking/market/v1/quote/"
_CANDLE = "/rest/secure/angelbroking/historical/v1/getCandleData"

QUOTE_CHUNK = 50
QUOTE_SPACING_S = 1.0
CALL_SPACING_S = 0.4
BACKOFF_S = (60.0, 300.0)
#: maximum span of one getCandleData request, in days
CANDLE_SPAN_DAYS = {"ONE_MINUTE": 30, "FIVE_MINUTE": 100, "ONE_DAY": 2000}
_AUTH_CODES = {"AG8001", "AG8002", "AB8050", "AB8051"}
_RATE_CODES = {"AB1021"}
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _feed_ts(v: object, fallback: datetime) -> datetime:
    """`11-Nov-2026 16:08:11` as IST; the fallback when absent or unparsable."""
    try:
        day, mon, rest = str(v).strip().split("-", 2)
        year, _, clk = rest.partition(" ")
        hh, mm, ss = (clk.split(":") + ["0", "0"])[:3]
        return datetime(
            int(year), _MONTHS.index(mon.lower()) + 1, int(day), int(hh), int(mm), int(ss), tzinfo=IST
        )
    except (ValueError, IndexError):
        return fallback


def parse_quote(q: dict[str, Any], now: datetime) -> Quote:
    """Quote from one `data.fetched[]` entry of a FULL market quote (prices in rupees)."""
    depth = q.get("depth") or {}
    buy = (depth.get("buy") or [{}])[0]
    sell = (depth.get("sell") or [{}])[0]
    sym = str(q.get("tradingSymbol") or "")
    return Quote(
        symbol=sym[:-3] if sym.endswith("-EQ") else sym,
        ts=_feed_ts(q.get("exchFeedTime"), now),
        ltp=to_float(q.get("ltp")),
        open=to_float(q.get("open")),
        high=to_float(q.get("high")),
        low=to_float(q.get("low")),
        prev_close=to_float(q.get("close")),
        volume=to_float(q.get("tradeVolume")),
        bid=to_float(buy.get("price")),
        ask=to_float(sell.get("price")),
        bid_qty=to_float(buy.get("quantity")),
        ask_qty=to_float(sell.get("quantity")),
        upper_circuit=to_float(q.get("upperCircuit")),
        lower_circuit=to_float(q.get("lowerCircuit")),
        source="angel",
    )


def candle_windows(start: datetime, end: datetime, span_days: int) -> list[tuple[datetime, datetime]]:
    """Consecutive (from, to) request windows, each at most `span_days` long, covering [start, end]."""
    out: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=span_days), end)
        out.append((cur, nxt))
        cur = nxt
    return out or [(start, end)]


class AngelClient:
    """Read-only Angel One SmartAPI client (login, funds, holdings, positions, quotes, candles)."""

    def __init__(
        self, settings: Settings, clock: Clock, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._s = settings.secrets
        self._clock = clock
        self._http = httpx.Client(timeout=httpx.Timeout(30, connect=15), transport=transport)
        self._jwt: str | None = None
        self._last: datetime | None = None

    # ---- plumbing ------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1",
            "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "00:00:00:00:00:00",
            "X-PrivateKey": self._s.angel_api_key or "",
        }
        if self._jwt:
            h["Authorization"] = f"Bearer {self._jwt}"
        return h

    def _space(self, seconds: float) -> None:
        if self._last is not None:
            wait = seconds - (self._clock.now() - self._last).total_seconds()
            if wait > 0:
                self._clock.sleep(wait)
        self._last = self._clock.now()

    def _send(self, method: str, path: str, body: dict | None, spacing: float) -> tuple[int, dict | None]:
        self._space(spacing)
        resp = self._http.request(method, BASE + path, headers=self._headers(), json=body)
        try:
            j = resp.json()
        except ValueError:
            j = None
        return resp.status_code, j if isinstance(j, dict) else None

    def _call(
        self, method: str, path: str, body: dict | None = None, spacing: float = CALL_SPACING_S
    ) -> dict:
        """JSON of a call; backs off 60 s then 300 s on rate limits (AB1021, 403, 429, empty body),
        re-logs in once on an auth error, raises BrokerError otherwise."""
        relogged = False
        waits = list(BACKOFF_S)
        while True:
            status, j = self._send(method, path, body, spacing)
            code = str((j or {}).get("errorcode") or "")
            if status == 401 or code in _AUTH_CODES:
                if relogged:
                    raise BrokerError(f"angel auth failed on {path} ({code or status})")
                relogged = True
                self.login()
                continue
            if status in (403, 429) or j is None or code in _RATE_CODES:
                if not waits:
                    raise BrokerError(f"angel rate limit on {path} ({code or status})")
                self._clock.sleep(waits.pop(0))
                continue
            if not j.get("status"):
                raise BrokerError(f"angel {path}: {code} {j.get('message')}")
            return j

    # ---- public --------------------------------------------------------------------------------

    def login(self) -> None:
        """Log in with client code, PIN and a fresh TOTP; one retry at the next 30 s boundary."""
        s = self._s
        if not (s.angel_api_key and s.angel_client_id and s.angel_pin and s.angel_totp_secret):
            raise BrokerError("angel credentials not set")
        self._jwt = None
        for attempt in (1, 2):
            body = {
                "clientcode": s.angel_client_id,
                "password": s.angel_pin,
                "totp": pyotp.TOTP(s.angel_totp_secret).at(self._clock.now()),
            }
            _, j = self._send("POST", _LOGIN, body, CALL_SPACING_S)
            if j and j.get("status") and isinstance(j.get("data"), dict) and j["data"].get("jwtToken"):
                self._jwt = j["data"]["jwtToken"]
                return
            msg = str((j or {}).get("message") or "")
            if attempt == 1 and "totp" in msg.lower():
                t = self._clock.now().timestamp()
                self._clock.sleep(30 - (t % 30) + 1)
                continue
            raise BrokerError(f"angel login failed: {(j or {}).get('errorcode')} {msg}")

    def _ensure(self) -> None:
        if self._jwt is None:
            self.login()

    def funds(self) -> Funds:
        self._ensure()
        d = self._call("GET", _RMS).get("data") or {}
        return Funds(
            available_cash=to_float(d.get("availablecash")) or 0.0,
            net=to_float(d.get("net")) or 0.0,
            utilised=to_float(d.get("utiliseddebits")) or 0.0,
        )

    def holdings(self) -> list[Holding]:
        self._ensure()
        d = self._call("GET", _HOLD).get("data") or {}
        rows = d.get("holdings") if isinstance(d, dict) else d
        return [
            Holding(
                str(h.get("tradingsymbol", "")).removesuffix("-EQ"),
                to_float(h.get("quantity")) or 0.0,
                to_float(h.get("averageprice")) or 0.0,
                to_float(h.get("ltp")) or 0.0,
            )
            for h in (rows or [])
        ]

    def positions(self) -> list[Position]:
        self._ensure()
        rows = self._call("GET", _POS).get("data") or []
        return [
            Position(
                str(p.get("tradingsymbol", "")).removesuffix("-EQ"),
                to_float(p.get("netqty")) or 0.0,
                to_float(p.get("avgnetprice")) or 0.0,
                to_float(p.get("ltp")) or 0.0,
                str(p.get("producttype", "")),
            )
            for p in rows
        ]

    def quote_full(self, tokens: list[str]) -> list[Quote]:
        """FULL quotes for NSE tokens, 50 per call, one call per second."""
        self._ensure()
        out: list[Quote] = []
        for i in range(0, len(tokens), QUOTE_CHUNK):
            chunk = tokens[i : i + QUOTE_CHUNK]
            j = self._call(
                "POST", _QUOTE, {"mode": "FULL", "exchangeTokens": {"NSE": chunk}}, spacing=QUOTE_SPACING_S
            )
            now = self._clock.now()
            out += [parse_quote(q, now) for q in (j.get("data") or {}).get("fetched") or []]
        return out

    def candles(self, token: str, interval: str, start_dt: datetime, end_dt: datetime) -> pl.DataFrame:
        """Candles (ts IST, open, high, low, close, volume) between two datetimes, ascending."""
        if interval not in CANDLE_SPAN_DAYS:
            raise BrokerError(f"unsupported angel interval {interval!r}")
        self._ensure()
        rows: list[list[Any]] = []
        for a, b in candle_windows(start_dt, end_dt, CANDLE_SPAN_DAYS[interval]):
            body = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": a.strftime("%Y-%m-%d %H:%M"),
                "todate": b.strftime("%Y-%m-%d %H:%M"),
            }
            rows += self._call("POST", _CANDLE, body).get("data") or []
        schema = {
            "ts": pl.Datetime("us", "Asia/Kolkata"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        df = pl.DataFrame(
            {
                "ts": [datetime.fromisoformat(r[0]) for r in rows],
                "open": [float(r[1]) for r in rows],
                "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows],
                "close": [float(r[4]) for r in rows],
                "volume": [float(r[5]) for r in rows],
            }
        ).with_columns(pl.col("ts").dt.convert_time_zone("Asia/Kolkata").dt.cast_time_unit("us"))
        return df.unique(subset=["ts"], keep="last").sort("ts")

    def close(self) -> None:
        self._http.close()


__all__ = ["AngelClient", "candle_windows", "parse_quote"]
