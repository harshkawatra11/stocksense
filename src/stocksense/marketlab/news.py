"""Ported from market-lab sx/news.py (MIT, ba3f771); changes: headlines come from Sources.google_news, the
cache uses the Clock, an expired entry is served stale when the source fails, and each returned list is a copy
so callers can add fields (`since`) without touching the cache.

Headlines for a sector or a stock (Google News, India edition), tagged as results, policy or news, cached for
15 minutes per query.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from typing import Any

import httpx
import polars as pl

from stocksense.core.clock import IST, Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataValidationError
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.marketlab.sources import SourceError

log = get_logger(__name__)

TTL_S = 900.0
RESULTS = re.compile(r"\b(q[1-4]|quarter|results?|profit|earnings|revenue|ebitda|net income|fy2\d)\b", re.I)
POLICY = re.compile(
    r"\b(rbi|sebi|government|govt|ministry|policy|budget|tariff|gst|duty|cabinet|regulat|ban|subsid"
    r"|import|export curb|centre|minister)\b",
    re.I,
)
QUERY = {
    "Railways": "railway stocks OR IRCTC OR RVNL OR IRFC",
    "Energy": "power stocks OR oil gas stocks india OR coal india OR ntpc",
    "Infra": "infrastructure stocks india OR L&T order",
    "Cons Durables": "consumer durables stocks india OR dixon OR voltas OR havells",
    "Metal": "metal stocks india OR tata steel OR hindalco OR jsw steel",
    "Tourism": "hotel stocks india OR indigo OR tourism stocks",
    "Defence": "defence stocks india OR HAL OR BEL OR mazagon dock",
    "Auto": "auto stocks india OR maruti OR tata motors sales",
    "Chemicals": "chemical stocks india",
    "PSU Bank": "psu bank stocks OR SBI OR bank of baroda",
    "Cap Markets": "BSE share OR CDSL OR MCX OR capital market stocks",
    "Realty": "realty stocks india OR DLF OR godrej properties",
    "FMCG": "fmcg stocks india OR HUL OR ITC OR nestle india",
    "Fin Services": "nbfc stocks india OR bajaj finance",
    "Healthcare": "hospital stocks india OR apollo hospitals OR max healthcare",
    "IT": "IT stocks india OR infosys OR tcs OR wipro",
    "Bank": "bank stocks india OR HDFC bank OR ICICI bank",
    "Pharma": "pharma stocks india OR sun pharma OR usfda",
    "Media": "media stocks india OR zee OR sun tv OR PVR",
    "Cement": "cement stocks india OR ultratech OR ambuja",
}
_SUFFIX = re.compile(r"\b(limited|ltd\.?|corporation|corp\.?|india|industries)\b", re.I)


def tag(title: str) -> str:
    """RESULTS, POLICY or NEWS for a headline (results wins when both patterns match)."""
    return "RESULTS" if RESULTS.search(title) else "POLICY" if POLICY.search(title) else "NEWS"


class NewsService:
    """Tagged headline search with a 15-minute cache; `sector(name, wait=False)` reads the warmed cache."""

    def __init__(self, sources: Any, clock: Clock, settings: Settings | None = None) -> None:
        self.src, self.clock, self.settings = sources, clock, settings
        self._lock = threading.Lock()
        self._rec_lock = threading.Lock()
        self._cache: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}

    def _get(self, key: tuple[str, int]) -> tuple[float, list[dict[str, Any]]] | None:
        with self._lock:
            return self._cache.get(key)

    def _record(self, q: str, items: list[dict[str, Any]], symbol: str) -> None:
        """Append every fetched headline to the `news_headlines` dataset (idempotent on ts, query, title).

        This is the forward-only log that k151_nb_news needs (PLAN-M M5.4). A write failure is logged and
        never breaks the dashboard.
        """
        if self.settings is None or not items:
            return
        rows = []
        for h in items:
            try:
                ts = datetime.fromisoformat(h["time"]).astimezone(IST)
            except (KeyError, ValueError):
                continue
            rows.append((ts, q, symbol, h["title"], h.get("source", ""), h.get("link", "")))
        if not rows:
            return
        cols = list(zip(*rows, strict=True))
        df = pl.DataFrame(
            {
                "ts": pl.Series(cols[0], dtype=pl.Datetime("us", "Asia/Kolkata")),
                "query": list(cols[1]),
                "symbol": list(cols[2]),
                "title": list(cols[3]),
                "source": list(cols[4]),
                "link": list(cols[5]),
            }
        )
        try:
            with self._rec_lock:
                io.upsert(self.settings, "news_headlines", df, clock=self.clock)
        except (OSError, DataValidationError):
            log.exception("news_headlines_write_failed", query=q[:40])

    def search(self, q: str, days: int = 30, limit: int = 20, symbol: str = "") -> list[dict[str, Any]]:
        """Headlines for a query, newest first, each with a tag; a failing source returns the stale copy."""
        key = (q, days)
        now = self.clock.now().timestamp()
        hit = self._get(key)
        if hit is not None and now - hit[0] < TTL_S:
            return [dict(h) for h in hit[1]]
        try:
            items = self.src.google_news(q, days, limit)
        except (SourceError, httpx.HTTPError):
            log.warning("news_fetch_failed", query=q[:40])
            return [dict(h) for h in (hit[1] if hit else [])]
        tagged = [{**h, "tag": tag(h["title"])} for h in items]
        with self._lock:
            self._cache[key] = (now, tagged)
        self._record(q, tagged, symbol)
        return [dict(h) for h in tagged]

    def sector(self, name: str, wait: bool = True) -> list[dict[str, Any]]:
        """Headlines for a map sector; `wait=False` returns whatever the background warmer already holds."""
        q = QUERY.get(name, name + " stocks india")
        if not wait:
            hit = self._get((q, 30))
            return [dict(h) for h in (hit[1] if hit else [])]
        return self.search(q)

    def stock(self, sym: str, company: str | None) -> list[dict[str, Any]]:
        """Headlines for one company over 60 days (name without Ltd/India/Industries, or the symbol)."""
        short = _SUFFIX.sub("", company or sym).strip(" .,")
        return self.search(f'"{short}" OR {sym}', days=60, limit=12, symbol=sym)

    def market(self) -> list[dict[str, Any]]:
        """Market-wide headlines (Sensex, Nifty) over 30 days."""
        return self.search('sensex OR nifty OR "stock market" india', days=30, limit=30)

    def warm(self, names: list[str], stop: threading.Event) -> list[dict[str, Any]]:
        """Fetch every sector's headlines half a second apart, then the market's; returns the market list."""
        for n in names:
            if stop.is_set():
                break
            self.sector(n)
            stop.wait(0.5)
        return self.market()
