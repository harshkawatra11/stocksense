from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import httpx

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import BrokerError
from stocksense.core.log import get_logger

log = get_logger(__name__)

ANGEL_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
UPSTOX_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
SERIES = ("EQ", "BE", "BZ")  # EQ is preferred when a symbol trades in several


def _prefer(out: dict[str, tuple[str, str]], symbol: str, value: str, series: str) -> None:
    cur = out.get(symbol)
    if cur is None or (series == "EQ" and cur[1] != "EQ"):
        out[symbol] = (value, series)


def angel_equity_rows(items: list[dict]) -> dict[str, tuple[str, str]]:
    """symbol -> (token, series) for NSE `-EQ`, `-BE`, `-BZ` rows of the Angel scrip master."""
    out: dict[str, tuple[str, str]] = {}
    for x in items:
        sym = str(x.get("symbol", ""))
        if x.get("exch_seg") != "NSE" or "-" not in sym:
            continue
        base, series = sym.rsplit("-", 1)
        if series in SERIES:
            _prefer(out, base, str(x["token"]), series)
    return out


def upstox_equity_rows(items: list[dict]) -> dict[str, tuple[str, str]]:
    """trading_symbol -> (instrument_key, series) for NSE_EQ rows of type EQ / BE / BZ."""
    out: dict[str, tuple[str, str]] = {}
    for x in items:
        if x.get("segment") == "NSE_EQ" and x.get("instrument_type") in SERIES:
            _prefer(out, str(x["trading_symbol"]), str(x["instrument_key"]), x["instrument_type"])
    return out


def angel_equity_tokens(items: list[dict]) -> dict[str, str]:
    """symbol -> token for NSE EQ/BE/BZ rows of the Angel scrip master."""
    return {k: v[0] for k, v in angel_equity_rows(items).items()}


def upstox_equity_keys(items: list[dict]) -> dict[str, str]:
    """trading_symbol -> instrument_key for NSE_EQ rows of type EQ/BE/BZ of the Upstox master."""
    return {k: v[0] for k, v in upstox_equity_rows(items).items()}


def _write_atomic(path: Path, raw: bytes) -> None:
    """Write via `<name>.tmp` then os.replace, so a crash never leaves a half-written cache file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class InstrumentMap:
    """Symbol -> Angel token / Upstox key, refreshed once a day (raw files cached under cache/instruments)."""

    def __init__(
        self, settings: Settings, clock: Clock, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._dir = Path(settings.paths.data_root) / "cache" / "instruments"
        self._clock = clock
        self._http = httpx.Client(timeout=httpx.Timeout(180, connect=15), transport=transport)
        self._angel: dict[str, tuple[str, str]] | None = None
        self._upstox: dict[str, tuple[str, str]] | None = None

    @staticmethod
    def _parse(raw: bytes, gz: bool) -> list[dict]:
        items = json.loads(gzip.decompress(raw) if gz else raw)
        if not isinstance(items, list):
            raise ValueError("instrument master is not a list")
        return items

    def _fetch(self, url: str, cache: Path, gz: bool) -> list[dict]:
        """Today's cache file if present, else download; a payload is cached only after it parsed."""
        if cache.is_file():
            return self._parse(cache.read_bytes(), gz)
        r = self._http.get(url)
        if r.status_code != 200:
            raise BrokerError(f"instrument master {url}: HTTP {r.status_code}")
        items = self._parse(r.content, gz)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(cache, r.content)
        except OSError as e:
            log.warning("instruments_cache_write_failed", error=type(e).__name__)
        return items

    def _previous_cache(self, prefix: str, suffix: str, today: str, gz: bool) -> list[dict] | None:
        """Newest parseable cache file older than today's, or None."""
        for f in sorted(self._dir.glob(f"{prefix}-*{suffix}"), reverse=True):
            if f.name == f"{prefix}-{today}{suffix}":
                continue
            try:
                return self._parse(f.read_bytes(), gz)
            except (OSError, ValueError):
                continue
        return None

    def refresh(self) -> None:
        """Load both masters (today's cache file if present, else download)."""
        d = self._clock.today().isoformat()
        self._angel = angel_equity_rows(self._fetch(ANGEL_URL, self._dir / f"angel-{d}.json", False))
        try:
            self._upstox = upstox_equity_rows(
                self._fetch(UPSTOX_URL, self._dir / f"upstox-{d}.json.gz", True)
            )
        except (BrokerError, httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as e:
            log.warning("instruments_upstox_failed", error=type(e).__name__)
            items = self._previous_cache("upstox", ".json.gz", d, True)
            try:
                self._upstox = upstox_equity_rows(items) if items is not None else {}
            except (KeyError, TypeError):
                self._upstox = {}

    def angel_token(self, symbol: str) -> str | None:
        if self._angel is None:
            self.refresh()
        row = (self._angel or {}).get(symbol)
        return row[0] if row else None

    def upstox_key(self, symbol: str) -> str | None:
        if self._upstox is None:
            self.refresh()
        row = (self._upstox or {}).get(symbol)
        return row[0] if row else None

    def series_of(self, symbol: str) -> str | None:
        """Trading series (EQ, BE or BZ) of a symbol, from the Angel master first, else Upstox."""
        if self._angel is None:
            self.refresh()
        row = (self._angel or {}).get(symbol) or (self._upstox or {}).get(symbol)
        return row[1] if row else None

    def close(self) -> None:
        self._http.close()
