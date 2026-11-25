"""Ported from market-lab sx/universe.py, sx/live.py and the sector functions of app.py (MIT, ba3f771);
changes: numpy and plain lists instead of pandas, the Clock and calendar instead of wall time, Sources
(NseClient/Yahoo) for every call, a LabStore instead of module globals, a year of daily history read from our
stored panel (completed sessions only, split-adjusted) instead of 200 Yahoo downloads, sector index histories
from `index_daily` for all 20 sectors, a stock without enough history no longer breaks a sector view.

The Sector Map: 20 sectors (NSE index plus the Nifty 500 stocks NSE classifies into each), the stocks inside,
one stock up close. All percentages are Market Lab's percent units.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.marketlab import smc
from stocksense.marketlab.clock import market_status, minute_now
from stocksense.marketlab.sources import SourceError, Sources, SourceThrottled
from stocksense.marketlab.store import LabStore, TTLCache
from stocksense.marketlab.tape import chart_to_bars

log = get_logger(__name__)

# the 20 sectors on the map: NSE index, group column, and which stocks belong (NSE classification)
SECTORS: list[dict[str, Any]] = [
    # NSE's market-watch lists no thematic index members: Railways is the railway PSUs and wagon makers
    {
        "name": "Railways",
        "index": "NIFTY INDIA RAILWAYS PSU",
        "group": "Industrials & PSU",
        "symbols": [
            "IRCTC",
            "IRFC",
            "RVNL",
            "RAILTEL",
            "IRCON",
            "RITES",
            "CONCOR",
            "BEML",
            "TITAGARH",
            "JWL",
            "TEXRAIL",
            "RKFORGE",
            "HBLENGINE",
        ],
    },
    {
        "name": "Energy",
        "index": "NIFTY ENERGY",
        "group": "Commodities & energy",
        "sector": ["Oil Gas & Consumable Fuels", "Power"],
    },
    {
        "name": "Infra",
        "index": "NIFTY INFRASTRUCTURE",
        "group": "Industrials & PSU",
        "sector": ["Construction"],
    },
    {
        "name": "Cons Durables",
        "index": "NIFTY CONSUMER DURABLES",
        "group": "Consumption",
        "sector": ["Consumer Durables"],
    },
    {"name": "Metal", "index": "NIFTY METAL", "group": "Commodities & energy", "sector": ["Metals & Mining"]},
    {
        "name": "Tourism",
        "index": "NIFTY INDIA TOURISM",
        "group": "Consumption",
        "basic": [
            "Hotels & Resorts",
            "Airline",
            "Tour, Travel Related Services",
            "Restaurants",
            "Airport & Airport services",
        ],
    },
    {
        "name": "Defence",
        "index": "NIFTY INDIA DEFENCE",
        "group": "Industrials & PSU",
        "basic": ["Aerospace & Defense", "Ship Building & Allied Services", "Explosives"],
    },
    {
        "name": "Auto",
        "index": "NIFTY AUTO",
        "group": "Consumption",
        "sector": ["Automobile and Auto Components"],
    },
    {
        "name": "Chemicals",
        "index": "NIFTY CHEMICALS",
        "group": "Commodities & energy",
        "sector": ["Chemicals"],
    },
    {"name": "PSU Bank", "index": "NIFTY PSU BANK", "group": "Financials", "basic": ["Public Sector Bank"]},
    {
        "name": "Cap Markets",
        "index": "NIFTY CAPITAL MARKETS",
        "group": "Financials",
        "industry": ["Capital Markets"],
    },
    {"name": "Realty", "index": "NIFTY REALTY", "group": "Consumption", "sector": ["Realty"]},
    {"name": "FMCG", "index": "NIFTY FMCG", "group": "Consumption", "sector": ["Fast Moving Consumer Goods"]},
    {
        "name": "Fin Services",
        "index": "NIFTY FINANCIAL SERVICES",
        "group": "Financials",
        "sector": ["Financial Services"],
        "not_basic": ["Private Sector Bank", "Public Sector Bank"],
    },
    {
        "name": "Healthcare",
        "index": "NIFTY HEALTHCARE INDEX",
        "group": "Healthcare",
        "sector": ["Healthcare"],
        "not_basic": ["Pharmaceuticals"],
    },
    {"name": "IT", "index": "NIFTY IT", "group": "Tech & media", "sector": ["Information Technology"]},
    {"name": "Bank", "index": "NIFTY BANK", "group": "Financials", "industry": ["Banks"]},
    {"name": "Pharma", "index": "NIFTY PHARMA", "group": "Healthcare", "basic": ["Pharmaceuticals"]},
    {
        "name": "Media",
        "index": "NIFTY MEDIA",
        "group": "Tech & media",
        "sector": ["Media Entertainment & Publication"],
    },
    {
        "name": "Cement",
        "index": "NIFTY CEMENT",
        "group": "Commodities & energy",
        "sector": ["Construction Materials"],
    },
]
GROUPS = [
    "Tech & media",
    "Financials",
    "Healthcare",
    "Consumption",
    "Commodities & energy",
    "Industrials & PSU",
]
YAHOO = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRASTRUCTURE": "^CNXINFRA",
    "NIFTY FINANCIAL SERVICES": "NIFTY_FIN_SERVICE.NS",
    "NIFTY HEALTHCARE INDEX": "NIFTY_HEALTHCARE.NS",
    "NIFTY CONSUMER DURABLES": "NIFTY_CONSR_DURBL.NS",
}
_RANGE_N = {"w1": 5, "m1": 21, "m3": 63}
_CHART_RANGE = {"1m": "5d", "5m": "1mo", "15m": "60d", "60m": "2y", "1d": "5y"}
_QUOTE_KEYS = ("delivery", "volume", "pe", "high52", "low52", "mcap", "name", "basic", "industry", "sector")
_SOFT = (SourceError, DataNotAvailable, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError)


# ---- series types and pure helpers --------------------------------------------------------------------


@dataclass(frozen=True)
class Hist:
    """Daily bars of one symbol (completed sessions, ascending): date, close, high, volume."""

    dates: np.ndarray  # datetime64[D]
    close: np.ndarray
    high: np.ndarray
    volume: np.ndarray
    low: np.ndarray | None = None  # adjusted lows when stored (the research note's ATR)

    def __len__(self) -> int:
        return int(len(self.dates))

    def series(self) -> Ser:
        """The closes as a dated series."""
        return Ser(self.dates, self.close)


@dataclass(frozen=True)
class Ser:
    """A dated series (datetime64[D] dates ascending, float values)."""

    dates: np.ndarray
    values: np.ndarray


def r2(x: Any) -> float | None:
    """Round to 2 decimals; None for None and NaN."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def ret(series: Ser | None, n: int) -> float | None:
    """Percent change of the last value vs the one n points earlier; None when under n+1 points."""
    if series is None or len(series.values) <= n:
        return None
    v = series.values
    return float(v[-1] / v[-1 - n] - 1) * 100


def stats(h: Hist | None) -> dict[str, float | None]:
    """Week, month, 3-month moves (percent), 50-day average, 60-day mean volume, 52-week high, last close."""
    if h is None or len(h) < 30:
        return {}
    c = h.close

    def r(n: int) -> float | None:
        return float(c[-1] / c[-1 - n] - 1) * 100 if len(c) > n else None

    return {
        "w1": r(5),
        "m1": r(21),
        "m3": r(63),
        "dma50": float(c[-50:].mean()),
        "vol60": float(h.volume[-60:].mean() or 0),
        "high52": float(h.high[-250:].max()),
        "last_close": float(c[-1]),
    }


def _ffill_cols(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    for j in range(out.shape[1]):
        last = np.nan
        for i in range(out.shape[0]):
            if np.isnan(out[i, j]):
                out[i, j] = last
            else:
                last = out[i, j]
    return out


def align_ffill(sers: list[Ser]) -> tuple[np.ndarray, np.ndarray]:
    """Outer-join dated series on date, forward-fill, drop rows that still have a gap: (dates, matrix)."""
    dates = np.unique(np.concatenate([s.dates for s in sers]))
    m = np.full((len(dates), len(sers)), np.nan)
    for j, s in enumerate(sers):
        m[np.searchsorted(dates, s.dates), j] = s.values
    m = _ffill_cols(m)
    keep = ~np.isnan(m).any(axis=1)
    return dates[keep], m[keep]


def sector_series(hists: dict[str, Hist], syms: list[str], days: int = 70) -> Ser | None:
    """Equal-weight index of the sector's stocks (each rebased to 100): median of their last `days` paths."""
    cols = [hists[s] for s in syms if s in hists and len(hists[s]) > days]
    if not cols:
        return None
    dates, m = align_ffill([Ser(h.dates[-days:], h.close[-days:]) for h in cols])
    if not len(dates):
        return None
    return Ser(dates, np.median(m / m[0] * 100, axis=1))


def members(
    sec: dict[str, Any], cls: dict[str, Any], universe: dict[str, Any], index_members: Any = ()
) -> list[str]:
    """Nifty 500 stocks NSE classifies into this sector, plus the sector index's own members."""
    out = [s for s in index_members if s in universe]
    if "symbols" in sec:
        return [s for s in sec["symbols"] if s in universe or s in cls]
    for s, c in cls.items():
        if s not in universe:
            continue
        ok = False
        if "sector" in sec and c.get("sector") in sec["sector"]:
            ok = True
        if "industry" in sec and c.get("industry") in sec["industry"]:
            ok = True
        if "basic" in sec and c.get("basic") in sec["basic"]:
            ok = True
        if c.get("basic") in sec.get("not_basic", []):
            ok = False
        if ok and s not in out:
            out.append(s)
    return out


def ist_date(iso: str) -> date:
    """The Indian calendar day of an ISO timestamp with an offset."""
    return datetime.fromisoformat(iso).astimezone(IST).date()


def pins(items: list[dict[str, Any]], dates: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Headlines bucketed onto the last chart day on or before them: stacks of N / R / P."""
    out: dict[str, list[dict[str, Any]]] = {}
    ds = [date.fromisoformat(d) for d in dates]
    for h in items:
        t = ist_date(h["time"])
        cand = [d for d in ds if d <= t]
        if not cand:
            continue
        out.setdefault(cand[-1].isoformat(), []).append(
            {"tag": h["tag"][0], "title": h["title"], "source": h["source"], "link": h["link"]}
        )
    return out


def rel_chart(a: Ser | None, b: Ser | None, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Two series rebased to 100 over the same last 55 days, with headline pins; None when under 5 days."""
    if a is None or b is None:
        return None
    dates, m = align_ffill([a, b])
    dates, m = dates[-55:], m[-55:]
    if len(dates) < 5:
        return None
    m = m / m[0] * 100
    ds = [str(d) for d in dates]
    return {
        "dates": ds,
        "a": [r2(x) for x in m[:, 0]],
        "b": [r2(x) for x in m[:, 1]],
        "pins": pins(items, ds),
    }


def reindex_ffill(s: Ser, dates: np.ndarray, bfill: bool = False) -> np.ndarray:
    """Values of `s` on exactly `dates` (NaN where absent), forward-filled, optionally back-filled too."""
    out = np.full(len(dates), np.nan)
    pos = np.searchsorted(s.dates, dates)
    ok = (pos < len(s.dates)) & (s.dates[np.minimum(pos, len(s.dates) - 1)] == dates)
    out[ok] = s.values[pos[ok]]
    out = _ffill_cols(out.reshape(-1, 1))[:, 0]
    if bfill:
        out = _ffill_cols(out[::-1].reshape(-1, 1))[::-1, 0]
    return out


def load_hist(
    settings: Settings, symbols: list[str], index_names: list[str], end: date, days: int = 420
) -> dict[str, Hist]:
    """Split-adjusted daily bars of symbols (`SYM`) and indices (`IDX:NAME`) from our panel, to `end`."""
    start = end - timedelta(days=days)
    out: dict[str, Hist] = {}
    df = (
        io.read(
            settings,
            "panel_daily",
            start=start,
            end=end,
            columns=["symbol", "date", "series", "adj_close", "adj_high", "adj_low", "adj_volume"],
        )
        .filter(
            (pl.col("series") == "EQ") & pl.col("symbol").is_in(symbols) & pl.col("adj_close").is_not_null()
        )
        .sort(["symbol", "date"])
        .collect()
    )
    for key, g in df.group_by("symbol", maintain_order=True):
        out[str(key[0])] = Hist(
            np.array(g["date"].to_list(), dtype="datetime64[D]"),
            g["adj_close"].to_numpy().astype(float),
            g["adj_high"].fill_null(strategy="forward").to_numpy().astype(float),
            g["adj_volume"].fill_null(0.0).to_numpy().astype(float),
            g["adj_low"].fill_null(strategy="forward").to_numpy().astype(float),
        )
    ix = (
        io.read(settings, "index_daily", start=start, end=end, columns=["index_name", "date", "close"])
        .filter(pl.col("index_name").is_in(index_names) & pl.col("close").is_not_null())
        .sort(["index_name", "date"])
        .collect()
    )
    for key, g in ix.group_by("index_name", maintain_order=True):
        c = g["close"].to_numpy().astype(float)
        out["IDX:" + str(key[0])] = Hist(
            np.array(g["date"].to_list(), dtype="datetime64[D]"), c, c, np.zeros(len(c))
        )
    return out


# ---- the engine ---------------------------------------------------------------------------------------


class SectorEngine:
    """Sector Map state in the LabStore (`sx` live, `univ` universe) and the payloads of its /api routes."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        cal: TradingCalendar,
        sources: Any,
        store: LabStore,
        news: Any,
        slow_sources: Any = None,
    ) -> None:
        self.settings, self.clock, self.cal, self.src, self.store, self.news = (
            settings,
            clock,
            cal,
            sources,
            store,
            news,
        )
        self.slow_src = slow_sources or sources
        self.cache = TTLCache(clock)
        self.stop = threading.Event()
        self.last: dict[str, Any] = {"idx": 0.0, "watch": 0.0, "members": 0.0, "fast": 0.0, "day": None}
        for k, v in (
            ("indices", {}),
            ("stocks", {}),
            ("members", {}),
            ("spark", {}),
            ("focus", []),
            ("focus_until", 0.0),
            ("day", {"snaps": []}),
            ("time", None),
            ("errors", 0),
            ("mk", []),
        ):
            store.sx.setdefault(k, v)
        for k, v in (("list", {}), ("cls", {}), ("hist", {}), ("progress", "")):
            store.univ.setdefault(k, v)

    @property
    def sx(self) -> dict[str, Any]:
        """Live sector-map state."""
        return self.store.sx

    @property
    def univ(self) -> dict[str, Any]:
        """Universe state (Nifty 500 list, classification, history)."""
        return self.store.univ

    # ---- live reads -----------------------------------------------------------------------------------
    def price(self, sym: str) -> dict[str, Any]:
        """Best live price for a stock: Yahoo (seconds old) if it is being watched, else NSE (a minute)."""
        st = self.sx["stocks"].get(sym) or {}
        y = self.sx["spark"].get(sym + ".NS")
        if y and y["ltp"] and self.clock.now().timestamp() - (y.get("time") or 0) < 120:
            prev = st.get("prev") or y["prev"]
            return {
                **st,
                "ltp": y["ltp"],
                "pct": round((y["ltp"] / prev - 1) * 100, 2) if prev else st.get("pct"),
                "live": True,
            }
        return st

    def index_value(self, name: str) -> dict[str, Any]:
        """An index's level and move, Yahoo-fresh when Yahoo carries it."""
        ix = dict(self.sx["indices"].get(name) or {})
        y = YAHOO.get(name)
        g = self.sx["spark"].get(y) if y else None
        if g and g["ltp"] and g["prev"]:
            ix.update(last=round(g["ltp"], 2), pct=round((g["ltp"] / g["prev"] - 1) * 100, 2), live=True)
        return ix

    def focus(self, syms: list[str]) -> None:
        """The page asks for these stocks to be refreshed every few seconds (the open sector or stock)."""
        self.sx["focus"] = [s for s in syms if s in self.univ["list"]][:120]
        self.sx["focus_until"] = self.clock.now().timestamp() + 60

    def sec_members(self, sec: dict[str, Any]) -> list[str]:
        """Members of a sector: NSE classification plus the sector index's own constituents."""
        idx_members = list((self.sx["members"].get(sec["index"]) or {}).keys())
        return members(sec, self.univ["cls"], self.univ["list"], idx_members)

    # ---- sector series and moves ----------------------------------------------------------------------
    def series_for(self, sec: dict[str, Any]) -> Ser | None:
        """3-4 months of the sector: its NSE index when stored, else the median of its stocks."""
        h = self.univ["hist"].get("IDX:" + sec["index"])
        if h is not None and len(h) > 70:
            return Ser(h.dates[-70:], h.close[-70:] / h.close[-70] * 100)
        return sector_series(self.univ["hist"], self.sec_members(sec))

    def all_series(self) -> dict[str, Ser | None]:
        """Every sector's series, cached for an hour (and rebuilt when the history is replaced)."""
        key = ("series", self.clock.today().isoformat(), id(self.univ["hist"]), len(self.univ["hist"]))
        return self.cache.get(key, 3600, lambda: {s["name"]: self.series_for(s) for s in SECTORS})

    def nifty_hist(self) -> Ser | None:
        """Nifty 50 daily closes."""
        h = self.univ["hist"].get("IDX:NIFTY 50")
        return h.series() if h is not None else None

    def sector_move(self, sec: dict[str, Any], rng: str) -> float | None:
        """Percent move of a sector over today / w1 / m1 / m3 (NSE's own week and month figure if present)."""
        ix = self.index_value(sec["index"])
        if rng == "today":
            return ix.get("pct")
        if rng in ("w1", "m1") and ix.get(rng) is not None:
            return ix.get(rng)
        return ret(self.all_series().get(sec["name"]), _RANGE_N[rng])

    def nifty_move(self, rng: str) -> float | None:
        """Percent move of Nifty 50 over today / w1 / m1 / m3."""
        ix = self.index_value("NIFTY 50")
        if rng == "today":
            return ix.get("pct")
        if rng in ("w1", "m1") and ix.get(rng) is not None:
            return ix[rng]
        return ret(self.nifty_hist(), _RANGE_N[rng])

    def ranks_now(self, rng: str) -> tuple[dict[str, int], dict[str, float | None]]:
        """(rank by move among the 20 sectors, move) for the range."""
        v = {s["name"]: self.sector_move(s, rng) for s in SECTORS}
        order = sorted((k for k in v if v[k] is not None), key=lambda k: -v[k])  # type: ignore[operator]
        return {k: i + 1 for i, k in enumerate(order)}, v

    def rank_history(self, name: str, weeks: int = 10) -> list[dict[str, Any]]:
        """Rank among the 20 by trailing month, at the end of each of the last weeks."""
        ser = self.all_series()
        out = []
        for w in range(weeks, -1, -1):
            end = -1 - w * 5
            vals = {}
            for k, s in ser.items():
                if s is not None and len(s.values) > 22 - end:
                    vals[k] = float(s.values[end] / s.values[end - 21] - 1)
            order = sorted(vals, key=lambda k: -vals[k])
            if name in order:
                out.append({"date": str(ser[name].dates[end]), "rank": order.index(name) + 1})  # type: ignore[union-attr]
        return out

    def themes(self) -> list[dict[str, Any]]:
        """Basic industries with 5+ Nifty 500 stocks: their middle stock today and how many beat Nifty."""
        nifty = (self.index_value("NIFTY 50") or {}).get("pct") or 0
        by: dict[str, list[tuple[str, float, Any]]] = {}
        for s, c in self.univ["cls"].items():
            p = self.price(s)
            if c.get("basic") and p.get("pct") is not None:
                by.setdefault(c["basic"], []).append((s, p["pct"], c.get("sector")))
        out = []
        for b, rows in by.items():
            if len(rows) < 5:
                continue
            mid = float(np.median([r[1] for r in rows]))
            out.append(
                {
                    "name": b,
                    "parent": rows[0][2],
                    "n": len(rows),
                    "beat": sum(r[1] > nifty for r in rows),
                    "pct": round(mid, 2),
                }
            )
        return sorted((t for t in out if t["pct"] > nifty), key=lambda t: -t["pct"])[:8]

    def since(self, sec_name: str, iso: str) -> float | None:
        """How far the sector moved from the day of a headline to now (percent)."""
        s = self.all_series().get(sec_name)
        if s is None:
            return None
        k = int(np.searchsorted(s.dates, np.datetime64(ist_date(iso)), side="left"))
        if k == 0:
            return None
        today = self.index_value(next(x["index"] for x in SECTORS if x["name"] == sec_name)).get("pct") or 0
        return round((float(s.values[-1] / s.values[k - 1]) * (1 + today / 100) - 1) * 100, 2)

    # ---- payloads -------------------------------------------------------------------------------------
    def horizon(self, rng: str) -> dict[str, Any]:
        """/api/horizon: sector moves, ranks, trails, climbs, themes, headlines and the Nifty chart."""
        rk, vals = self.ranks_now(rng)
        snaps = self.sx["day"]["snaps"]
        first = snaps[0]["sec"] if snaps else {}
        rk_open = {k: i + 1 for i, k in enumerate(sorted(first, key=lambda k: -first[k]))} if first else {}
        secs = []
        for sec in SECTORS:
            mem = self.sec_members(sec)
            size = sum((self.sx["stocks"].get(s) or {}).get("ffmc", 0) for s in mem)
            ix = self.index_value(sec["index"])
            trail = snaps[-7]["sec"].get(sec["name"]) if rng == "today" and len(snaps) > 6 else None
            secs.append(
                {
                    "name": sec["name"],
                    "index": sec["index"],
                    "group": sec["group"],
                    "v": r2(vals.get(sec["name"])),
                    "size": size,
                    "n": len(mem),
                    "rank": rk.get(sec["name"]),
                    "rank_open": rk_open.get(sec["name"]),
                    "trail": r2(trail),
                    "live": bool(ix.get("live")),
                    "path": [[s["m"], s["sec"].get(sec["name"])] for s in snaps] if rng == "today" else [],
                }
            )
        climb = [
            {"name": s["name"], "from": s["rank_open"], "to": s["rank"]}
            for s in secs
            if s["rank_open"] and s["rank"]
        ]
        if rng != "today":
            prev = {}
            for sec in SECTORS:
                sr = self.all_series().get(sec["name"])
                n = _RANGE_N[rng]
                if sr is not None and len(sr.values) > n + 6:
                    prev[sec["name"]] = float(sr.values[-6] / sr.values[-6 - n] - 1)
            order = sorted(prev, key=lambda k: -prev[k])
            climb = [{"name": k, "from": order.index(k) + 1, "to": rk[k]} for k in order if k in rk]
        news: list[dict[str, Any]] = []
        for sec in SECTORS:
            for h in self.news.sector(sec["name"], wait=False)[:4]:
                news.append({**h, "sector": sec["name"]})
        news.sort(key=lambda h: h["time"], reverse=True)
        for h in news[:14]:
            h["since"] = self.since(h["sector"], h["time"])
        n = self.nifty_hist()
        chart = None
        if n is not None:
            dates = [str(d) for d in n.dates[-60:]]
            chart = {
                "dates": dates,
                "a": [r2(x) for x in n.values[-60:]],
                "pins": pins(list(self.sx.get("mk", [])) + news, dates),
            }
        return {
            "range": rng,
            "nifty": {"v": r2(self.nifty_move(rng)), "last": self.index_value("NIFTY 50").get("last")},
            "sectors": secs,
            "groups": GROUPS,
            "climb": climb,
            "themes": self.themes() if rng == "today" else [],
            "news": news[:14],
            "chart": chart,
            "snaps": [{"m": s["m"], "nifty": s["nifty"]} for s in snaps],
        }

    def stock_row(self, s: str, c: dict[str, Any] | None = None) -> dict[str, Any]:
        """One stock's row for sector and stock views (percent units; w1/m1/m3 from completed daily bars)."""
        c = c or self.univ["cls"].get(s, {})
        p = self.price(s)
        st = stats(self.univ["hist"].get(s))
        ltp = p.get("ltp")
        vr = (p.get("volume") or 0) / st["vol60"] if st.get("vol60") else None
        return {
            "sym": s,
            "name": c.get("name") or p.get("name") or s,
            "group": c.get("industry") or "Other",
            "basic": c.get("basic"),
            "ltp": r2(ltp),
            "pct": r2(p.get("pct")),
            "mcap": c.get("mcap") or p.get("ffmc") or 0,
            "live": bool(p.get("live")),
            "w1": r2(st.get("w1")),
            "m1": r2(st.get("m1")),
            "m3": r2(st.get("m3")),
            "above50": bool(ltp > st["dma50"]) if ltp and st.get("dma50") else None,
            "near_high": bool(ltp and st.get("high52") and ltp >= st["high52"] * 0.95),
            "vol_ratio": r2(vr),
            "high52": r2(c.get("high52") or st.get("high52")),
            "dma50": r2(st.get("dma50")),
        }

    def sector_view(self, name: str, rng: str) -> dict[str, Any]:
        """/api/sector: one sector's index, rank, stocks grouped by industry, movers, headlines and chart."""
        sec = next((x for x in SECTORS if x["name"].lower() == name.lower()), None)
        if not sec:
            return {"error": "unknown sector"}
        mem = self.sec_members(sec)
        self.focus(mem)
        rows = [r for r in (self.stock_row(s) for s in mem) if r["ltp"]]
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)

        def mid(v: list[dict[str, Any]]) -> float | None:
            return r2(np.median([r["pct"] for r in v if r["pct"] is not None]))

        def n50(v: list[dict[str, Any]]) -> int:
            return sum(bool(r["above50"]) for r in v)

        glist = [
            {
                "name": g,
                "n": len(v),
                "mid": mid(v),
                "above50": n50(v),
                "stocks": sorted(v, key=lambda r: -(r["mcap"] or 0)),
            }
            for g, v in groups.items()
        ]
        glist.sort(key=lambda g: -sum(r["mcap"] or 0 for r in g["stocks"]))
        if len(glist) > 6:  # small groups fold into one
            rest = [r for g in glist[5:] for r in g["stocks"]]
            glist = glist[:5] + [
                {
                    "name": "Other index members",
                    "n": len(rest),
                    "mid": mid(rest),
                    "above50": n50(rest),
                    "stocks": rest,
                }
            ]
        ix = self.index_value(sec["index"])
        rk, _ = self.ranks_now("today")
        snaps = self.sx["day"]["snaps"]
        first = snaps[0]["sec"] if snaps else {}
        rk_open = (
            sorted(first, key=lambda k: -first[k]).index(sec["name"]) + 1 if sec["name"] in first else None
        )
        # stocks the sector index holds that NSE's classification files under another sector
        ix_mem = set((self.sx["members"].get(sec["index"]) or {}).keys())
        own = set(members(sec, self.univ["cls"], self.univ["list"]))
        others = []
        for s in sorted(ix_mem - own):
            o = next(
                (x for x in SECTORS if x is not sec and s in members(x, self.univ["cls"], self.univ["list"])),
                None,
            )
            if o:
                p = self.price(s)
                others.append(
                    {"sym": s, "sector": o["name"], "ltp": r2(p.get("ltp")), "pct": r2(p.get("pct"))}
                )
        ser = self.all_series().get(sec["name"])
        wk_above = 0
        for r in rows:
            h = self.univ["hist"].get(r["sym"])
            if h is not None and len(h) > 56:
                wk_above += int(h.close[-6] > h.close[-56:-6].mean())
        news = self.news.sector(sec["name"])
        for h in news:
            h["since"] = self.since(sec["name"], h["time"])
        nifty_pe = (self.sx["indices"].get("NIFTY 50") or {}).get("pe")
        movable = [r for r in rows if r["w1"] is not None]
        return {
            "name": sec["name"],
            "index": sec["index"],
            "group": sec["group"],
            "ix": {
                "last": ix.get("last"),
                "pct": ix.get("pct"),
                "w1": r2(self.sector_move(sec, "w1")),
                "m1": r2(self.sector_move(sec, "m1")),
                "m3": r2(self.sector_move(sec, "m3")),
                "pe": ix.get("pe"),
                "live": bool(ix.get("live")),
            },
            "nifty": {
                "pct": r2(self.nifty_move("today")),
                "w1": r2(self.nifty_move("w1")),
                "m1": r2(self.nifty_move("m1")),
                "m3": r2(self.nifty_move("m3")),
                "pe": nifty_pe,
            },
            "rank": rk.get(sec["name"]),
            "rank_open": rk_open,
            "n": len(rows),
            "middle": mid(rows) if rows else None,
            "groups": glist,
            "others": others[:14],
            "rank_hist": self.rank_history(sec["name"]),
            "inside": {
                "above50": n50(rows),
                "above50_wk": wk_above,
                "rose_week": sum((r["w1"] or 0) > 0 for r in rows),
                "near_high": sum(r["near_high"] for r in rows),
                "busy": sum((r["vol_ratio"] or 0) > 1.5 for r in rows),
                "n": len(rows),
            },
            "movers": sorted(movable, key=lambda r: -r["w1"])[:6],
            "laggards": sorted(movable, key=lambda r: r["w1"])[:4],
            "news": news[:10],
            "chart": rel_chart(ser, self.nifty_hist(), news),
        }

    def stock_view(self, sym: str) -> dict[str, Any]:
        """/api/stock: one stock vs its sector and Nifty: peers, rank, headlines, volume, 3-month lines."""
        sym = sym.upper()
        c = self.univ["cls"].get(sym) or {}
        try:
            q = self.cache.get(("q", sym), 30, lambda: self.src.nse_symbol_data(sym))
        except _SOFT:
            q = {}
        c = {**c, **{k: v for k, v in q.items() if v is not None and k in _QUOTE_KEYS}}
        row = self.stock_row(sym, c)
        sec = next((x for x in SECTORS if sym in self.sec_members(x)), None)
        peers: list[dict[str, Any]] = []
        if sec:
            peers = [
                p
                for p in (self.stock_row(s) for s in self.sec_members(sec))
                if p["group"] == row["group"] and p["ltp"]
            ]
        self.focus([sym] + [p["sym"] for p in peers])
        st = stats(self.univ["hist"].get(sym))
        h = self.univ["hist"].get(sym)
        sser = self.all_series().get(sec["name"]) if sec else None
        against = []
        if sec:
            sx = self.index_value(sec["index"])
            for k, lab in (("today", "Today"), ("w1", "Week"), ("m1", "1 month"), ("m3", "3 months")):
                a = row["pct"] if k == "today" else row[k]
                b = sx.get("pct") if k == "today" else self.sector_move(sec, k)
                against.append(
                    {
                        "label": lab,
                        "stock": r2(a),
                        "sector": r2(b),
                        "gap": r2(a - b) if a is not None and b is not None else None,
                    }
                )
        rank = None
        if sec and row["w1"] is not None:
            ws = sorted(
                (p for p in (self.stock_row(s) for s in self.sec_members(sec)) if p["w1"] is not None),
                key=lambda p: -p["w1"],
            )
            rank = {"pos": next((i + 1 for i, p in enumerate(ws) if p["sym"] == sym), None), "of": len(ws)}
        lines = None
        if h is not None and len(h) > 63:
            a_d, a = h.dates[-63:], h.close[-63:]
            with np.errstate(invalid="ignore", divide="ignore"):
                lines = {"stock": [r2(x / a[0] * 100) for x in a]}
                n = self.nifty_hist()
                if n is not None:
                    nn = reindex_ffill(n, a_d)
                    lines["nifty"] = [r2(x / nn[0] * 100) for x in nn]
                if sser is not None:
                    ss = reindex_ffill(sser, a_d, bfill=True)
                    lines["sector"] = [r2(x / ss[0] * 100) for x in ss]
            lines["m3"] = {
                "stock": r2(ret(h.series(), 63)),
                "sector": r2(self.sector_move(sec, "m3")) if sec else None,
                "nifty": r2(self.nifty_move("m3")),
            }
        news = self.news.stock(sym, c.get("name"))
        vol60 = st.get("vol60")
        return {
            "sym": sym,
            "row": row,
            "cls": {
                k: c.get(k)
                for k in ("sector", "industry", "basic", "pe", "mcap", "delivery", "high52", "low52", "name")
            },
            "sector": sec and {"name": sec["name"], "index": sec["index"]},
            "peers": sorted(peers, key=lambda p: -(p["pct"] or -99)),
            "against": against,
            "rank": rank,
            "lines": lines,
            "news": news,
            "volume": {
                "today": q.get("volume"),
                "normal": r2(vol60),
                "ratio": r2((q.get("volume") or 0) / vol60) if vol60 else None,
            },
            "delivery": c.get("delivery"),
            "dma50": r2(st.get("dma50")),
            "chart": rel_chart(h.series() if h is not None else None, self.nifty_hist(), news),
        }

    def search(self, q: str) -> list[dict[str, Any]]:
        """/api/search: sectors and stocks whose name or symbol contains the text (at most 15)."""
        q = q.strip().upper()
        out: list[dict[str, Any]] = [
            {"type": "sector", "name": s["name"]} for s in SECTORS if q in s["name"].upper()
        ]
        for s, c in self.univ["cls"].items():
            if q in s or q in (c.get("name") or "").upper():
                out.append(
                    {"type": "stock", "sym": s, "name": c.get("name"), "pct": self.price(s).get("pct")}
                )
            if len(out) > 14:
                break
        return out

    def chart(self, sym: str, tf: str) -> dict[str, Any]:
        """/api/chart: Yahoo candles and the smart-money read for the pop-up chart."""
        rng = _CHART_RANGE[tf]
        y = sym if sym.startswith("^") else sym + ".NS"
        d = self.src.yahoo_chart(y, tf, rng)
        b = chart_to_bars(d)
        rd = smc.read(b, 2 if tf in ("1m", "5m") else 3) if len(b) >= 30 else None
        m = d["meta"]
        return {
            "sym": sym,
            "tf": tf,
            "ltp": m.get("regularMarketPrice"),
            "prev": m.get("chartPreviousClose"),
            "time": m.get("regularMarketTime"),
            "candles": [
                {
                    "time": int(t),
                    "open": float(o),
                    "high": float(h),
                    "low": float(lo),
                    "close": float(c),
                    "volume": float(v),
                }
                for t, o, h, lo, c, v in zip(b.ts, b.open, b.high, b.low, b.close, b.volume, strict=True)
            ],
            "read": rd,
        }

    def tick(self, syms: list[str]) -> dict[str, Any]:
        """/api/tick: live prices for the symbols on screen plus every sector's and Nifty's move."""
        out = {}
        for s in syms:
            p = self.price(s)
            out[s] = {"ltp": r2(p.get("ltp")), "pct": r2(p.get("pct"))}
        ix = {}
        for s in SECTORS:
            v = self.index_value(s["index"])
            ix[s["name"]] = {"v": r2(v.get("pct")), "last": v.get("last")}
        n = self.index_value("NIFTY 50")
        return {
            "stocks": out,
            "sectors": ix,
            "nifty": {"v": r2(n.get("pct")), "last": n.get("last")},
            "time": self.sx["time"],
            "market": market_status(self.clock, self.cal),
        }

    # ---- background loops -----------------------------------------------------------------------------
    def _try(self, fn: Any, *args: Any) -> Any:
        """Call a source; a failure is logged and counted and gives None (the feed keeps its old data)."""
        try:
            return fn(*args)
        except SourceThrottled:
            return None
        except _SOFT as e:
            self.sx["errors"] += 1
            log.warning("sector_source_failed", call=getattr(fn, "__name__", "?"), err=type(e).__name__)
            return None

    def _days_dir(self) -> Path:
        d = Paths(self.settings).lab / "days"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record(self) -> None:
        """Every 5 minutes of the session: each sector's move, for the replay and the 30-minute tails."""
        m = minute_now(self.clock)
        if not (0 <= m <= 375):
            return
        d = self.sx["day"]
        if d["snaps"] and d["snaps"][-1]["m"] > m - 5:
            return
        snap: dict[str, Any] = {"m": m, "nifty": self.index_value("NIFTY 50").get("pct"), "sec": {}}
        for sec in SECTORS:
            v = self.index_value(sec["index"]).get("pct")
            if v is not None:
                snap["sec"][sec["name"]] = v
        d["snaps"].append(snap)
        (self._days_dir() / f"{self.clock.today()}.json").write_text(json.dumps(d), encoding="utf-8")

    def step(self) -> float:
        """One iteration of Market Lab's live loop; returns the seconds to wait before the next call."""
        now = self.clock.now().timestamp()
        today = self.clock.today()
        sx = self.sx
        if self.last["day"] != today:
            p = self._days_dir() / f"{today}.json"
            sx["day"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"snaps": []}
            self.last["day"] = today
        live = market_status(self.clock, self.cal) == "open"
        every = 15 if live else 300
        if now - self.last["idx"] >= every:
            got = self._try(self.src.nse_all_indices)
            if got:
                sx["indices"] = got
                self.last["idx"] = now
        if now - self.last["watch"] >= every:
            got = self._try(self.src.nse_watch, "NIFTY 500")
            if got:
                sx["stocks"] = got
                self.last["watch"] = now
        if now - self.last["members"] >= 3600:
            mem = {}
            for sec in SECTORS:
                got = self._try(self.src.nse_watch, sec["index"])
                if got:
                    mem[sec["index"]] = got
                if self.stop.wait(0.3):
                    return 0.0
            if mem:
                sx["members"] = mem
            self.last["members"] = now
        if now > self.store.slow_until.get("yahoo", 0.0) and (live or now - self.last["fast"] > 300):
            ys = list(dict.fromkeys(YAHOO.values()))
            if now < sx["focus_until"]:
                ys += [s + ".NS" for s in sx["focus"]]
            got = self._try(self.src.yahoo_spark, ys)
            if got:
                with self.store.lock:
                    sx["spark"].update(got)
            self.last["fast"] = now
        self.record()
        sx["time"] = self.clock.now().astimezone(UTC).isoformat(timespec="seconds")
        return 5.0 if live else 30.0

    def _cls_path(self) -> Path:
        p = Paths(self.settings).lab
        p.mkdir(parents=True, exist_ok=True)
        return p / "classification.json"

    def build_classification(self) -> None:
        """NSE classification for every Nifty 500 stock; resumes where it stopped, weekly refresh."""
        path = self._cls_path()
        cls: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.univ["cls"] = dict(cls)
        now = self.clock.now().timestamp()
        todo = [s for s in self.univ["list"] if s not in cls or now - cls[s].get("fetched", 0) > 7 * 86400]
        for i, s in enumerate(todo):
            if self.stop.is_set():
                break
            try:
                cls[s] = self.slow_src.nse_symbol_data(s)
            except _SOFT:
                self.stop.wait(2.0)
                continue
            if i % 25 == 0:
                self.univ["progress"] = f"classifying stocks {len(cls)}/{len(self.univ['list'])}"
                path.write_text(json.dumps(cls), encoding="utf-8")
                self.univ["cls"] = dict(cls)
            self.stop.wait(0.25)
        path.write_text(json.dumps(cls), encoding="utf-8")
        self.univ["cls"] = cls
        self.univ["progress"] = ""
        log.info("classification_done", stocks=len(cls))

    def load_history(self) -> None:
        """A year of split-adjusted daily bars per stock and per sector index, from our stored panel."""
        end = self.cal.last_completed_session(self.clock.now())
        names = sorted({s["index"] for s in SECTORS} | {"NIFTY 50"})
        hist = load_hist(self.settings, list(self.univ["list"]), names, end)
        self.univ["hist"] = hist
        log.info("history_loaded", series=len(hist), end=str(end))

    def slow_step(self) -> float:
        """Hourly: the Nifty 500 list, stored history, then the (resumable, weekly) classification."""
        self.univ["list"] = self.slow_src.nifty500_list()
        self.load_history()
        self.build_classification()
        return 3600.0

    def news_step(self) -> float:
        """Every 10 minutes: refresh every sector's headlines and the market's."""
        self.sx["mk"] = self.news.warm([s["name"] for s in SECTORS], self.stop)
        return 600.0


def build_sector_engine(settings: Settings, clock: Clock, store: LabStore) -> SectorEngine | None:
    """Real SectorEngine with its own NSE sessions, or None (logged) when calendar data is missing."""
    from stocksense.marketlab.news import NewsService

    try:
        cal = TradingCalendar.from_store(settings, clock)
    except DataNotAvailable:
        log.warning("sector_disabled_no_calendar")
        return None
    live, slow = Sources(settings, clock, store), Sources(settings, clock, store)
    return SectorEngine(
        settings, clock, cal, live, store, NewsService(live, clock, settings), slow_sources=slow
    )
