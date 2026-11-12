from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import BrokerError

ANGEL_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
UPSTOX_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"


def angel_equity_tokens(items: list[dict]) -> dict[str, str]:
    """symbol -> token for NSE `-EQ` rows of the Angel scrip master."""
    return {
        str(x["symbol"]).removesuffix("-EQ"): str(x["token"])
        for x in items
        if x.get("exch_seg") == "NSE" and str(x.get("symbol", "")).endswith("-EQ")
    }


def upstox_equity_keys(items: list[dict]) -> dict[str, str]:
    """trading_symbol -> instrument_key for NSE_EQ / EQ rows of the Upstox master."""
    return {
        str(x["trading_symbol"]): str(x["instrument_key"])
        for x in items
        if x.get("segment") == "NSE_EQ" and x.get("instrument_type") == "EQ"
    }


class InstrumentMap:
    """Symbol -> Angel token / Upstox key, refreshed once a day (raw files cached under cache/instruments)."""

    def __init__(
        self, settings: Settings, clock: Clock, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._dir = Path(settings.paths.data_root) / "cache" / "instruments"
        self._clock = clock
        self._http = httpx.Client(timeout=httpx.Timeout(180, connect=15), transport=transport)
        self._angel: dict[str, str] | None = None
        self._upstox: dict[str, str] | None = None

    def _fetch(self, url: str, cache: Path, gz: bool) -> list[dict]:
        if cache.is_file():
            raw = cache.read_bytes()
        else:
            r = self._http.get(url)
            if r.status_code != 200:
                raise BrokerError(f"instrument master {url}: HTTP {r.status_code}")
            raw = r.content
            self._dir.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        return json.loads(gzip.decompress(raw) if gz else raw)

    def refresh(self) -> None:
        """Load both masters (today's cache file if present, else download)."""
        d = self._clock.today().isoformat()
        self._angel = angel_equity_tokens(self._fetch(ANGEL_URL, self._dir / f"angel-{d}.json", False))
        try:
            self._upstox = upstox_equity_keys(
                self._fetch(UPSTOX_URL, self._dir / f"upstox-{d}.json.gz", True)
            )
        except (BrokerError, httpx.HTTPError):
            self._upstox = {}

    def angel_token(self, symbol: str) -> str | None:
        if self._angel is None:
            self.refresh()
        return (self._angel or {}).get(symbol)

    def upstox_key(self, symbol: str) -> str | None:
        if self._upstox is None:
            self.refresh()
        return (self._upstox or {}).get(symbol)

    def close(self) -> None:
        self._http.close()
