from __future__ import annotations

import random
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from stocksense.core.clock import Clock
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger

log = get_logger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
)
_BASE = "https://www.nseindia.com"
_WARMUP_URLS = (f"{_BASE}/", f"{_BASE}/market-data/live-equity-market")
_MAX_WARMUPS = 2
_MIN_SPACING_S = 0.6
_JITTER_S = 0.2
_ATTEMPTS = 5


class _RetryableStatus(Exception):
    """Internal: a 429/5xx response that should be retried."""


class NseClient:
    """Polite HTTP client for NSE archives (bytes) and JSON APIs; spacing and sleeps use the Clock."""

    def __init__(self, clock: Clock, transport: httpx.BaseTransport | None = None, seed: int = 0) -> None:
        self._clock = clock
        self._rng = random.Random(seed)
        self._last = None
        self._http = httpx.Client(
            http2=False,
            timeout=httpx.Timeout(30, connect=15),
            follow_redirects=True,
            transport=transport,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            },
        )

    # ---- plumbing ---------------------------------------------------------------------------

    def _space(self) -> None:
        """Block (via the clock) until at least 0.6 s + jitter has passed since the last request."""
        now = self._clock.now()
        if self._last is not None:
            gap = _MIN_SPACING_S + self._rng.uniform(0.0, _JITTER_S)
            wait = gap - (now - self._last).total_seconds()
            if wait > 0:
                self._clock.sleep(wait)
        self._last = self._clock.now()

    def _once(self, url: str, headers: dict[str, str], params: dict[str, Any] | None) -> httpx.Response:
        self._space()
        resp = self._http.get(url, headers=headers, params=params)
        if resp.status_code == 429 or 500 <= resp.status_code <= 599:
            raise _RetryableStatus(f"{resp.status_code} for {url}")
        return resp

    def _get(self, url: str, headers: dict[str, str], params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries (5 attempts, exponential 2 -> 60 s) on transport errors, 429 and 5xx."""
        retrying = Retrying(
            stop=stop_after_attempt(_ATTEMPTS),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatus)),
            sleep=self._clock.sleep,
            reraise=True,
        )
        try:
            return retrying(self._once, url, headers, params)
        except _RetryableStatus as e:
            raise DataNotAvailable(f"giving up after {_ATTEMPTS} attempts: {e}") from e

    # ---- public -----------------------------------------------------------------------------

    def get_archive(self, url: str) -> bytes | None:
        """Raw bytes of an NSE archive file; None if it does not exist (404 or NSE's HTML not-found page)."""
        resp = self._get(url, {"Accept": "*/*", "Referer": f"{_BASE}/"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.content
        head = body[:64].lstrip().lower()
        if head.startswith((b"<!doctype html", b"<html")):
            return None
        return body

    def get_api(self, path: str, params: dict[str, Any] | None, referer_path: str) -> Any:
        """Parsed JSON of an NSE API call; warms cookies on 401/403 (max 2 warm-ups)."""
        url = f"{_BASE}{path}"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{_BASE}/{referer_path.lstrip('/')}",
        }
        warmups = 0
        while True:
            resp = self._get(url, headers, params)
            if resp.status_code in (401, 403):
                if warmups >= _MAX_WARMUPS:
                    raise DataNotAvailable(f"{path}: {resp.status_code} after {warmups} warm-ups")
                warmups += 1
                for w in _WARMUP_URLS:
                    try:
                        self._get(w, {"Accept": "text/html,*/*", "Referer": f"{_BASE}/"})
                    except (httpx.HTTPError, DataNotAvailable):
                        log.warning("nse_warmup_failed", url=w)
                self._clock.sleep(1.0)
                continue
            if resp.status_code == 404:
                raise DataNotAvailable(f"{path}: 404")
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as e:
                raise DataNotAvailable(f"{path}: response is not JSON") from e

    def close(self) -> None:
        """Close the underlying connection pool (idempotent)."""
        self._http.close()
