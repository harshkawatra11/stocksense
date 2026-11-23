"""Live data sources of the dashboard: NSE website JSON, Yahoo Finance, Google News RSS, Wikipedia.

Ported from market-lab sx/live.py, sx/universe.py, sx/news.py and lab/research.py (MIT, ba3f771); changes:
NSE calls go through NseClient (cookie warm-up, spacing, retries), the clock replaces time.time, a Yahoo 429
sets a back-off in the LabStore, and pandas is replaced by plain lists. Values keep Market Lab's units
(percent as percent). These feeds serve the dashboard only; research uses the stored NSE files.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.nse_http import NseClient
from stocksense.marketlab.store import LabStore

log = get_logger(__name__)

SPARK = "https://query2.finance.yahoo.com/v7/finance/spark"
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"
NIFTY500_CSV = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
THROTTLE_S = 120.0
_UA = "Mozilla/5.0"
_WIKI_UA = "StockSense-local-dashboard/0.1 (personal research tool; low volume)"


class SourceError(Exception):
    """A source answered with something unusable."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


class SourceThrottled(SourceError):
    """A source answered 429 (or is inside its back-off window): skip it until the window passes."""

    def __init__(self, source: str) -> None:
        super().__init__(source, "throttled")


def to_float(x: Any) -> float | None:
    """float(x) for numbers and numeric strings (commas allowed); None otherwise."""
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ---- pure parsers (tested on saved payloads) ---------------------------------------------------------


def parse_all_indices(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """allIndices payload to {index name: row}; w1 = percent change against the one-week-ago level."""
    out: dict[str, dict[str, Any]] = {}
    for r in d["data"]:
        last, wk = to_float(r.get("last")), to_float(r.get("oneWeekAgoVal"))
        out[r["index"]] = {
            "name": r["index"],
            "last": last,
            "pct": to_float(r.get("percentChange")),
            "chg": to_float(r.get("variation")),
            "prev": to_float(r.get("previousClose")),
            "w1": round((last / wk - 1) * 100, 2) if last and wk else None,
            "m1": to_float(r.get("perChange30d")),
            "y1": to_float(r.get("perChange365d")),
            "pe": to_float(r.get("pe")),
            "adv": to_float(r.get("advances")),
            "dec": to_float(r.get("declines")),
            "high52": to_float(r.get("yearHigh")),
            "low52": to_float(r.get("yearLow")),
            "open": to_float(r.get("open")),
        }
    return out


def parse_watch(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Market-watch rows (EQ series only) to {symbol: row}; value_cr in crore rupees."""
    return {
        r["symbol"]: {
            "sym": r["symbol"],
            "ltp": r.get("lastPrice"),
            "pct": r.get("pChange"),
            "prev": r.get("previousClose"),
            "open": r.get("open"),
            "high": r.get("dayHigh"),
            "low": r.get("dayLow"),
            "volume": r.get("totalTradedVolume") or 0,
            "value_cr": round((r.get("totalTradedValue") or 0) / 1e7, 2),
            "ffmc": r.get("ffmc") or 0,
            "high52": r.get("yearHigh"),
            "low52": r.get("yearLow"),
            "name": (r.get("companyName") or r["symbol"]).title(),
        }
        for r in rows
        if r.get("series") == "EQ"
    }


def parse_symbol_data(d: dict[str, Any], fetched: float) -> dict[str, Any]:
    """getSymbolData payload to the classification and quote fields Market Lab uses."""
    e = d["equityResponse"][0]
    si, ti, pi, md = (
        e.get("secInfo") or {},
        e.get("tradeInfo") or {},
        e.get("priceInfo") or {},
        e.get("metaData") or {},
    )
    return {
        "macro": si.get("macro"),
        "sector": si.get("sector"),
        "industry": si.get("industryInfo") or si.get("industry"),
        "basic": si.get("basicIndustry"),
        "index": si.get("index"),
        "pe": to_float(si.get("pdSymbolPe")),
        "sector_pe": to_float(si.get("pdSectorPe")),
        "mcap": ti.get("totalMarketCap"),
        "ffmc": ti.get("ffmc"),
        "delivery": to_float(ti.get("deliveryToTradedQuantity")),
        "volume": ti.get("totalTradedVolume"),
        "high52": pi.get("yearHigh"),
        "low52": pi.get("yearLow"),
        "name": md.get("companyName"),
        "ltp": md.get("lastPrice") or ti.get("lastPrice"),
        "pct": md.get("pChange"),
        "fetched": fetched,
    }


def parse_spark(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Yahoo spark payload to {yahoo symbol: {ltp, prev, time, ts[], close[]}}; null closes are dropped."""
    out: dict[str, dict[str, Any]] = {}
    for res in payload["spark"]["result"]:
        r = (res.get("response") or [None])[0]
        if not r or not r.get("timestamp"):
            continue
        m = r["meta"]
        pairs = [
            (t, c)
            for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"], strict=False)
            if c is not None
        ]
        out[res["symbol"]] = {
            "ltp": m.get("regularMarketPrice"),
            "prev": m.get("chartPreviousClose"),
            "time": m.get("regularMarketTime"),
            "ts": [t for t, _ in pairs],
            "close": [c for _, c in pairs],
        }
    return out


def parse_google_news(xml_bytes: bytes, now: datetime, days: int, limit: int) -> list[dict[str, Any]]:
    """Google News RSS to headlines, newest first; source suffix stripped, duplicate titles dropped."""
    root = ET.fromstring(xml_bytes)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    cutoff = now.astimezone(UTC) - timedelta(days=days)
    for it in root.iter("item"):
        src, title = it.findtext("source", ""), it.findtext("title", "")
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        k = title.lower()[:60]
        if k in seen:
            continue
        seen.add(k)
        try:
            t = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if t < cutoff:
            continue
        out.append({"time": t.isoformat(), "title": title, "source": src, "link": it.findtext("link", "")})
    out.sort(key=lambda x: x["time"], reverse=True)
    return out[:limit]


# ---- the source object ------------------------------------------------------------------------------


class Sources:
    """All outside data the dashboard reads; every method raises SourceError/DataNotAvailable on failure."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        store: LabStore,
        nse: NseClient | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.store = store
        self.nse = nse or NseClient(clock)
        self.http = http or httpx.Client(
            timeout=httpx.Timeout(20, connect=10), follow_redirects=True, headers={"User-Agent": _UA}
        )
        self.calls: dict[str, int] = {}

    def _count(self, source: str) -> None:
        self.calls[source] = self.calls.get(source, 0) + 1

    # NSE website JSON ------------------------------------------------------------------------------
    def nse_all_indices(self) -> dict[str, dict[str, Any]]:
        """Every NSE index: level, percent move, week/month/year change, PE, advances/declines."""
        self._count("nse")
        return parse_all_indices(self.nse.get_api("/api/allIndices", None, "market-data/live-market-indices"))

    def nse_watch(self, index_name: str) -> dict[str, dict[str, Any]]:
        """Members of an index with live fields (for example 'NIFTY 500' or 'SECURITIES IN F&O')."""
        self._count("nse")
        ref = "market-data/live-equity-market"
        try:
            d = self.nse.get_api(
                "/api/NextApi/apiClient/marketWatchApi",
                {"functionName": "getIndicesData", "symbol": index_name},
                ref,
            )
            rows = d["data"]["data"]
        except (DataNotAvailable, KeyError, TypeError):
            log.warning("nse_watch_fallback", index=index_name)
            d = self.nse.get_api("/api/equity-stockIndices", {"index": index_name}, ref)
            rows = d["data"]
        return parse_watch(rows)

    def nse_symbol_data(self, sym: str) -> dict[str, Any]:
        """Classification, PE, market cap, delivery and 52-week range of one stock."""
        self._count("nse")
        d = self.nse.get_api(
            "/api/NextApi/apiClient/GetQuoteApi",
            {"functionName": "getSymbolData", "marketType": "N", "series": "EQ", "symbol": sym},
            f"get-quotes/equity?symbol={sym}",
        )
        return parse_symbol_data(d, self.clock.now().timestamp())

    def nse_integrated_filing(self, sym: str) -> list[dict[str, Any]]:
        """Raw quarterly filing rows of one stock (amounts in rupees lakh)."""
        self._count("nse")
        d = self.nse.get_api(
            "/api/NextApi/apiClient/GetQuoteApi",
            {"functionName": "getIntegratedFilingData", "symbol": sym},
            f"get-quotes/equity?symbol={sym}",
        )
        return d or []

    def nse_results_comparison(self, sym: str) -> dict[str, Any]:
        """Raw results-comparison payload (debt/equity, interest cover)."""
        self._count("nse")
        return self.nse.get_api(
            "/api/results-comparision", {"symbol": sym}, f"get-quotes/equity?symbol={sym}"
        )

    def nse_option_expiries(self, symbol: str) -> list[str]:
        """Listed option expiry dates ('DD-Mon-YYYY', nearest first) of an index or stock."""
        self._count("nse")
        d = self.nse.get_api("/api/option-chain-contract-info", {"symbol": symbol}, "option-chain")
        return list(d.get("expiryDates", []))

    def nse_option_chain(self, symbol: str, is_index: bool, expiry: str | None = None) -> dict[str, Any]:
        """Live option-chain `records` (rows with CE/PE and underlyingValue) for one expiry (nearest if None).

        Probe 2026-11-23: the legacy /api/option-chain-indices path is gone (404) and the equities path
        returns an empty object; the working pair is option-chain-contract-info (expiries) and
        option-chain-v3 with type, symbol and an exact expiry.
        """
        if expiry is None:
            expiries = self.nse_option_expiries(symbol)
            if not expiries:
                raise SourceError("nse", f"no option expiries for {symbol}")
            expiry = expiries[0]
        self._count("nse")
        d = self.nse.get_api(
            "/api/option-chain-v3",
            {"type": "Indices" if is_index else "Equity", "symbol": symbol, "expiry": expiry},
            "option-chain",
        )
        return d.get("records", {})

    def nse_oi_spurts(self) -> list[dict[str, Any]]:
        """Raw rows of NSE's OI spurts list (underlyings)."""
        self._count("nse")
        d = self.nse.get_api("/api/live-analysis-oi-spurts-underlyings", None, "market-data/oi-spurts")
        return d.get("data", []) if isinstance(d, dict) else list(d)

    def nse_preopen(self, key: str) -> dict[str, Any]:
        """Raw pre-open payload for a key such as 'NIFTY', 'FO' or 'ALL'."""
        self._count("nse")
        return self.nse.get_api(
            "/api/market-data-pre-open", {"key": key}, "market-data/pre-open-market-cm-and-emerge-market"
        )

    def nifty500_list(self) -> dict[str, dict[str, str]]:
        """NIFTY 500 constituents {symbol: {name, industry}} from the NSE archive CSV."""
        self._count("nse")
        raw = self.nse.get_archive(NIFTY500_CSV)
        if raw is None:
            raise SourceError("nse", "nifty500 list not available")
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        return {r["Symbol"]: {"name": r["Company Name"], "industry": r["Industry"]} for r in rows}

    # Yahoo ------------------------------------------------------------------------------------------
    def _yahoo_guard(self) -> None:
        if self.store.slow_until.get("yahoo", 0.0) > self.clock.now().timestamp():
            raise SourceThrottled("yahoo")

    def _yahoo_get(self, url: str, params: dict[str, str]) -> Any:
        self._yahoo_guard()
        self._count("yahoo")
        resp = self.http.get(url, params=params)
        if resp.status_code == 429:
            self.store.slow_until["yahoo"] = self.clock.now().timestamp() + THROTTLE_S
            raise SourceThrottled("yahoo")
        if resp.status_code != 200:
            raise SourceError("yahoo", f"status {resp.status_code}")
        try:
            return resp.json()
        except ValueError as e:
            raise SourceError("yahoo", "response is not JSON") from e

    def yahoo_spark(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Latest price, previous close and today's 1-minute closes for Yahoo symbols (20 per call)."""
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(symbols), 20):
            chunk = symbols[i : i + 20]
            try:
                payload = self._yahoo_get(
                    SPARK, {"symbols": ",".join(chunk), "range": "1d", "interval": "1m"}
                )
            except SourceThrottled:
                if out:
                    break
                raise
            out.update(parse_spark(payload))
            self.clock.sleep(0.08)
        return out

    def yahoo_chart(self, ysym: str, interval: str, rng: str) -> dict[str, Any]:
        """Raw chart result (timestamps and OHLCV) of one Yahoo symbol."""
        payload = self._yahoo_get(CHART + urllib.parse.quote(ysym), {"interval": interval, "range": rng})
        try:
            return payload["chart"]["result"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise SourceError("yahoo", f"no chart result for {ysym}") from e

    # Google News and Wikipedia ----------------------------------------------------------------------
    def google_news(self, q: str, days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
        """Headlines for a query from Google News (India edition), newest first."""
        self._count("google")
        url = "https://news.google.com/rss/search"
        params = {"q": f"{q} when:{days}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
        resp = self.http.get(url, params=params)
        if resp.status_code != 200:
            raise SourceError("google", f"status {resp.status_code}")
        try:
            return parse_google_news(resp.content, self.clock.now(), days, limit)
        except ET.ParseError as e:
            raise SourceError("google", "bad RSS") from e

    def wiki_summary(self, name: str) -> dict[str, Any] | None:
        """Two or three sentences about a company from Wikipedia, or None when no page matches its name."""
        if not name:
            return None
        self._count("wikipedia")
        short = re.sub(r"\b(limited|ltd\.?)$", "", name.strip(), flags=re.IGNORECASE).strip()
        hdr = {"User-Agent": _WIKI_UA}
        r = self.http.get(
            "https://en.wikipedia.org/w/api.php",
            headers=hdr,
            params={
                "action": "query",
                "list": "search",
                "format": "json",
                "srlimit": "3",
                "srsearch": f"{short} company India",
            },
        )
        if r.status_code != 200:
            return None
        words = [w for w in re.findall(r"[a-z]+", short.lower()) if len(w) > 2][:3]
        for hit in r.json().get("query", {}).get("search", []):
            title = hit["title"]
            if not any(w in title.lower() for w in words):
                continue
            p = self.http.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(title.replace(" ", "_")),
                headers=hdr,
            )
            if p.status_code != 200:
                continue
            j = p.json()
            ext = j.get("extract") or ""
            if ext:
                sent = re.split(r"(?<=[.!?])\s+", ext)
                return {
                    "text": " ".join(sent[:3]),
                    "url": (j.get("content_urls") or {}).get("desktop", {}).get("page"),
                    "title": title,
                }
        return None


def dumps_probe(obj: Any, rows: int = 5) -> str:
    """Compact JSON of a payload trimmed to `rows` list items per list (for saved test fixtures)."""

    def trim(x: Any) -> Any:
        if isinstance(x, list):
            return [trim(i) for i in x[:rows]]
        if isinstance(x, dict):
            return {k: trim(v) for k, v in x.items()}
        return x

    return json.dumps(trim(obj), indent=1, default=str)
