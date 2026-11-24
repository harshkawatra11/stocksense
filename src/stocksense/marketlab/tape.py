"""Ported from market-lab sm/feed.py (MIT, ba3f771); changes: StockSense Clock and calendar instead of wall
time and holiday files, Sources (NseClient/Yahoo) for every call, daily levels and normal range read from our
panel_daily instead of 200 Yahoo downloads, numpy instead of pandas, a LabStore instead of globals.

Live tape for Smart Money: every F&O stock and Nifty, Bank Nifty, Sensex. Prices come from Yahoo's spark
endpoint (1-minute closes since the open); NSE market-watch adds official open, high, low, value, size.
Structure, order blocks and gaps are read from 5-minute bars built out of the 1-minute closes. All percentages
are Market Lab's percent units.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.marketlab import smc
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.clock import market_status, minute_now, session_minute
from stocksense.marketlab.sources import SourceError, Sources, SourceThrottled
from stocksense.marketlab.store import LabStore

log = get_logger(__name__)

INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}

INSURANCE = {
    "SBILIFE",
    "HDFCLIFE",
    "ICICIPRULI",
    "ICICIGI",
    "LICI",
    "POLICYBZR",
    "NIACL",
    "GICRE",
    "STARHEALTH",
    "MAXHEALTH_X",
}
CAP_MKTS = {
    "BSE",
    "MCX",
    "CDSL",
    "CAMS",
    "ANGELONE",
    "KFINTECH",
    "NUVAMA",
    "360ONE",
    "HDFCAMC",
    "NAM-INDIA",
    "MOTILALOFS",
    "IIFL",
    "UTIAMC",
    "ABSLAMC",
}
PSU_BANK = {
    "SBIN",
    "BANKBARODA",
    "PNB",
    "CANBK",
    "UNIONBANK",
    "INDIANB",
    "BANKINDIA",
    "IOB",
    "UCOBANK",
    "CENTRALBK",
    "MAHABANK",
}
DEFENCE = {
    "HAL",
    "BEL",
    "BDL",
    "MAZDOCK",
    "COCHINSHIP",
    "GRSE",
    "BEML",
    "DATAPATTNS",
    "SOLARINDS",
    "ASTRAMICRO",
    "PARAS",
    "MIDHANI",
}
NEW_AGE = {"ETERNAL", "ZOMATO", "PAYTM", "NYKAA", "SWIGGY", "DELHIVERY_X", "IXIGO", "CARTRADE", "NAUKRI"}
LOGISTICS = {"DELHIVERY", "CONCOR", "BLUEDART", "TCI", "ADANIPORTS", "GMRAIRPORT", "JSWINFRA", "INDIGO"}
TELECOM = {"BHARTIARTL", "IDEA", "INDUSTOWER", "TATACOMM", "HFCL", "TEJASNET"}
GROUP = {
    "Oil Gas & Consumable Fuels": "OIL & GAS",
    "Consumer Durables": "CONSUMER",
    "Consumer Services": "CONSUMER",
    "Textiles": "CONSUMER",
    "Capital Goods": "CAP GOODS",
    "Construction": "CAP GOODS",
    "Metals & Mining": "METAL",
    "Construction Materials": "CEMENT",
    "Power": "POWER",
    "Automobile and Auto Components": "AUTO",
    "Healthcare": "HEALTHCARE",
    "Chemicals": "CHEM",
    "Realty": "REALTY",
    "Information Technology": "IT",
    "Fast Moving Consumer Goods": "FMCG",
    "Telecommunication": "TELECOM",
    "Services": "LOGISTICS",
    "Media Entertainment & Publication": "OTHERS",
    "Diversified": "OTHERS",
    "Forest Materials": "OTHERS",
}
PHARMA_HINT = {
    "SUNPHARMA",
    "CIPLA",
    "DRREDDY",
    "DIVISLAB",
    "LUPIN",
    "AUROPHARMA",
    "ZYDUSLIFE",
    "ALKEM",
    "TORNTPHARM",
    "BIOCON",
    "GLENMARK",
    "LAURUSLABS",
    "MANKIND",
    "IPCALAB",
    "GRANULES",
    "PPLPHARMA",
    "AJANTPHARM",
    "NATCOPHARM",
    "ABBOTINDIA",
}
BANKS = {
    "HDFCBANK",
    "ICICIBANK",
    "AXISBANK",
    "KOTAKBANK",
    "INDUSINDBK",
    "FEDERALBNK",
    "IDFCFIRSTB",
    "BANDHANBNK",
    "AUBANK",
    "RBLBANK",
    "YESBANK",
    "CUB",
}


def sector_of(sym: str, industry: str) -> str:
    """Trading group for the sector chips: NSE's industry, split where one industry hides several stories."""
    for members, name in (
        (INSURANCE, "INSURANCE"),
        (CAP_MKTS, "CAP MKTS"),
        (PSU_BANK, "PSU BANK"),
        (DEFENCE, "DEFENCE"),
        (NEW_AGE, "NEW-AGE"),
        (LOGISTICS, "LOGISTICS"),
        (TELECOM, "TELECOM"),
    ):
        if sym in members:
            return name
    if industry == "Financial Services":
        return "BANK" if sym.endswith("BANK") or sym in BANKS else "NBFC"
    if industry == "Healthcare":
        return "PHARMA" if sym in PHARMA_HINT else "HEALTHCARE"
    return GROUP.get(industry, "OTHERS")


def bars5(ts: list[int], close: list[float], open_px: float | None = None) -> Bars | None:
    """5-minute OHLC from 1-minute closes (buckets start at 09:15, 09:20, ...; empty buckets dropped).

    The first bar's open is replaced by the official open when given, widening its high/low to include it.
    Volume is zero (the tape has none). 09:15 IST is a multiple of five minutes, so epoch//300 aligns buckets.
    """
    if not len(ts):
        return None
    t = np.asarray(ts, dtype=np.int64)
    c = np.asarray(close, dtype=float)
    key = t // 300
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(t)]
    bt, o, h, lo, cl = [], [], [], [], []
    for a, z in zip(starts[:-1], starts[1:], strict=True):
        seg = c[a:z]
        bt.append(int(key[a]) * 300)
        o.append(float(seg[0]))
        h.append(float(seg.max()))
        lo.append(float(seg.min()))
        cl.append(float(seg[-1]))
    if open_px:
        o[0] = float(open_px)
        h[0] = max(h[0], float(open_px))
        lo[0] = min(lo[0], float(open_px))
    return Bars.from_arrays(bt, o, h, lo, cl, None)


def path(ts: list[int], close: list[float], base: float | None) -> list[list[float]]:
    """[[minute since 09:15, percent from `base`], ...] for the 1-minute closes of the session (0 to 375)."""
    if not len(ts) or not base:
        return []
    out = []
    for t, c in zip(ts, close, strict=True):
        m = session_minute(int(t))
        if 0 <= m <= 375:
            out.append([m, round(float(c / base - 1) * 100, 3)])
    return out


def read_summary(b: Bars | None, daily: Bars | None = None) -> dict[str, Any] | None:
    """Market Lab's `_read`: compact smart-money read of a tape bar series (None when too short)."""
    if b is None or len(b) < 12:
        return None
    try:
        r = smc.read(b, 2, daily)
    except (IndexError, ValueError, KeyError):
        log.warning("tape_read_failed")
        return None
    if not r:
        return None
    st = r["structure"]["last"]
    first = int(b.ts[0])
    return {
        "struct": st and {"type": st["type"], "dir": st["dir"], "time": st["time"]},
        "ob": (r["order_blocks"] or [None])[0],
        "fvg": (r["fvgs"] or [None])[0],
        "liq": r["liquidity"],
        "ev": [
            [session_minute(e["time"]), 1 if e["dir"] == "bull" else -1]
            for e in r["chart"]["events"]
            if e["time"] >= first
        ],
    }


def stock_row(
    sym: str,
    nse: dict[str, Any],
    got: dict[str, Any] | None,
    dly: dict[str, float] | None,
    industry: str,
    now_m: int,
    old: dict[str, Any],
    structure: bool,
) -> dict[str, Any] | None:
    """One tape row (percent units) for a stock, or None when price, previous close or open is missing."""
    ltp = (got or {}).get("ltp") or nse.get("ltp")
    prev = nse.get("prev") or (got or {}).get("prev")
    ts, cl = (got["ts"], got["close"]) if got else ([], [])
    opn = nse.get("open") or (float(cl[0]) if len(cl) else None)
    if not (ltp and prev and opn):
        return None
    fo = (ltp / opn - 1) * 100
    row: dict[str, Any] = {
        "sym": sym,
        "name": nse["name"],
        "sector": sector_of(sym, industry),
        "ltp": round(ltp, 2),
        "pct": round((ltp / prev - 1) * 100, 2),
        "from_open": round(fo, 2),
        "disp": round(fo / ((dly or {}).get("range") or 2.0), 2),
        "path": path(ts, cl, opn),
        "value_cr": nse.get("value_cr"),
        "ffmc": nse.get("ffmc", 0),
    }
    b = bars5(ts, cl, opn)
    if b is not None:
        row["bars"] = [
            [round(float(x), 2) for x in r] for r in zip(b.open, b.high, b.low, b.close, strict=True)
        ]
    if dly:
        row["pdh"], row["pdl"] = round(dly["pdh"], 2), round(dly["pdl"], 2)
        row["liq"] = [
            k
            for k, hit in (
                ("PDH", (nse.get("high") or 0) > dly["pdh"]),
                ("PDL", 0 < (nse.get("low") or 1e12) < dly["pdl"]),
            )
            if hit
        ]
    if structure or "read" not in old:
        rd = read_summary(b)
        row["read"] = rd and {k: rd[k] for k in ("struct", "ob", "fvg", "ev")}
    else:
        row["read"] = old.get("read")
    p = row["path"]
    row["trail"] = next((v for m, v in reversed(p) if m <= now_m - 15), None) if p else None
    return row


def rank_rows(stocks: dict[str, dict[str, Any]]) -> None:
    """Add rank and side in place: bulls by descending displacement, bears most negative first."""
    bull = sorted((r for r in stocks.values() if r["from_open"] > 0), key=lambda r: -r["disp"])
    bear = sorted((r for r in stocks.values() if r["from_open"] <= 0), key=lambda r: r["disp"])
    for i, r in enumerate(bull):
        r["rank"], r["side"] = i + 1, "bull"
    for i, r in enumerate(bear):
        r["rank"], r["side"] = i + 1, "bear"


def sector_chips(stocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean percent change, up and down counts per trading group, strongest first."""
    secs: dict[str, list[dict[str, Any]]] = {}
    for r in stocks:
        secs.setdefault(r["sector"], []).append(r)
    chips = [
        {
            "name": k,
            "pct": round(float(np.mean([r["pct"] for r in v])), 2),
            "up": sum(r["pct"] > 0 for r in v),
            "down": sum(r["pct"] < 0 for r in v),
            "n": len(v),
        }
        for k, v in secs.items()
    ]
    return sorted(chips, key=lambda s: -s["pct"])


def next_expiry(key: str, today: date, cal: TradingCalendar) -> str | None:
    """Nifty weekly Tuesday, Bank Nifty month-end Tuesday, Sensex weekly Thursday (moved back on holidays)."""
    for k in range(45):
        x = today + timedelta(days=k)
        if x.weekday() != (3 if key == "SENSEX" else 1):
            continue
        if key == "BANKNIFTY" and (x + timedelta(days=7)).month == x.month:
            continue
        e = x
        while not cal.is_session(e):
            e -= timedelta(days=1)
        if e >= today:
            return e.isoformat()
    return None


class TapeEngine:
    """Owns the tape state in the LabStore and the loop that refreshes it (one `step()` per iteration)."""

    def __init__(
        self, settings: Settings, clock: Clock, cal: TradingCalendar, sources: Sources, store: LabStore
    ) -> None:
        self.settings, self.clock, self.cal, self.src, self.store = settings, clock, cal, sources, store
        self.last: dict[str, Any] = {"day": None, "nse": 0.0, "stocks": 0.0, "struct": 0.0}
        t = store.tape
        for k, v in (
            ("stocks", {}),
            ("indices", {}),
            ("meta", {}),
            ("daily", {}),
            ("daily_idx", {}),
            ("industry", {}),
            ("got", {}),
            ("time", None),
            ("errors", 0),
        ):
            t.setdefault(k, v)

    # ---- data loading -------------------------------------------------------------------------
    def _fno_meta(self) -> dict[str, dict[str, Any]]:
        rows = self.src.nse_watch("SECURITIES IN F&O")
        return {
            s: {
                "name": r["name"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "prev": r["prev"],
                "ltp": r["ltp"],
                "value_cr": round(r["value_cr"], 1),
                "volume": r.get("volume"),
                "ffmc": r["ffmc"],
            }
            for s, r in rows.items()
        }

    def load_daily_levels(self, symbols: list[str], today: date) -> dict[str, dict[str, float]]:
        """Previous day's high/low/close and the 20-session mean of (high-low)/close in percent, per symbol.

        Read from our stored raw daily bars (panel_daily) for completed sessions only.
        """
        start = today - timedelta(days=60)
        lf = io.read(
            self.settings,
            "panel_daily",
            start=start,
            end=today - timedelta(days=1),
            columns=["symbol", "date", "series", "high", "low", "close"],
        )
        df = (
            lf.filter((pl.col("series") == "EQ") & pl.col("symbol").is_in(symbols))
            .sort(["symbol", "date"])
            .collect()
        )
        out: dict[str, dict[str, float]] = {}
        for sym, g in df.group_by("symbol", maintain_order=True):
            if g.height < 15:
                continue
            hi, lo, cl = g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
            rng = (hi - lo) / cl * 100
            out[str(sym[0])] = {
                "pdh": float(hi[-1]),
                "pdl": float(lo[-1]),
                "pdc": float(cl[-1]),
                "range": float(rng[-20:].mean()),
            }
        return out

    def _daily_idx(self) -> dict[str, Bars]:
        out = {}
        for k, y in INDICES.items():
            try:
                out[k] = self._yahoo_daily(y)
            except (SourceError, KeyError, TypeError, ValueError):
                log.warning("tape_daily_idx_failed", index=k)
        return out

    def _yahoo_daily(self, ysym: str, rng: str = "3mo") -> Bars:
        d = self.src.yahoo_chart(ysym, "1d", rng)
        return chart_to_bars(d)

    def rebuild_day(self) -> None:
        """New session: industry map, F&O list, daily index bars and daily levels (from our panel)."""
        t = self.store.tape
        inds = {s: v["industry"] for s, v in self.src.nifty500_list().items()}
        meta = self._fno_meta()
        with self.store.lock:
            t["industry"], t["meta"], t["daily_idx"] = inds, meta, self._daily_idx()
        self.cycle(True, True)  # show prices before the slower daily levels arrive
        daily = self.load_daily_levels(list(meta), self.clock.today())
        with self.store.lock:
            t["daily"] = daily

    # ---- one refresh ----------------------------------------------------------------------------
    def fetch(self, ysyms: list[str]) -> None:
        """Pull Yahoo spark for the symbols into the store's `got` cache; a throttle just skips this round."""
        t = self.store.tape
        try:
            got = self.src.yahoo_spark(ysyms)
        except SourceThrottled:
            return
        except SourceError:
            t["errors"] += 1
            return
        t["errors"] = 0
        t["got"].update(got)

    def cycle(self, stocks_too: bool, structure: bool) -> None:
        """Refresh prices; when asked also every stock row and structure read; rebuild ranks and indices."""
        t = self.store.tape
        meta = t["meta"]
        syms = list(meta)
        self.fetch(([s + ".NS" for s in syms] if stocks_too else []) + list(INDICES.values()))
        now_m = minute_now(self.clock)
        if stocks_too:
            stocks: dict[str, dict[str, Any]] = {}
            for s in syms:
                row = stock_row(
                    s,
                    meta[s],
                    t["got"].get(s + ".NS"),
                    t["daily"].get(s),
                    t["industry"].get(s, ""),
                    now_m,
                    t["stocks"].get(s, {}),
                    structure,
                )
                if row:
                    stocks[s] = row
            rank_rows(stocks)
        else:
            stocks = t["stocks"]
        idx = {}
        today = self.clock.today()
        for k, y in INDICES.items():
            g = t["got"].get(y)
            if not g or not g["ltp"]:
                continue
            ts, cl = g["ts"], g["close"]
            base = float(cl[0]) if len(cl) else g["ltp"]
            b = bars5(ts, cl, base)
            old = t["indices"].get(k, {})
            rd = (
                read_summary(b, t["daily_idx"].get(k))
                if structure or not old.get("read")
                else old.get("read")
            )
            idx[k] = {
                "ltp": round(g["ltp"], 2),
                "pct": round((g["ltp"] / g["prev"] - 1) * 100, 2) if g["prev"] else None,
                "from_open": round((g["ltp"] / base - 1) * 100, 2),
                "path": path(ts, cl, base),
                "bars": [
                    [round(float(x), 2) for x in r] for r in zip(b.open, b.high, b.low, b.close, strict=True)
                ]
                if b is not None
                else [],
                "read": rd,
                "expiry": next_expiry(k, today, self.cal),
                "delay": int(self.clock.now().timestamp() - g["time"]) if g.get("time") else None,
            }
        with self.store.lock:
            t["stocks"], t["indices"], t["time"] = stocks, idx, self.clock.now().isoformat(timespec="seconds")

    def step(self) -> float:
        """One iteration of Market Lab's run loop; returns the seconds to wait before the next call."""
        today = self.clock.today()
        now = self.clock.now().timestamp()
        if self.last["day"] != today:
            self.rebuild_day()
            self.last.update(day=today, nse=now)
        live = market_status(self.clock, self.cal) == "open"
        t = self.store.tape
        if now - self.last["nse"] >= (60 if live else 1800):
            try:
                t["meta"] = self._fno_meta() or t["meta"]
            except SourceError:
                log.warning("tape_nse_list_failed")
            self.last["nse"] = now
        slow = self.store.slow_until.get("yahoo", 0.0) > now
        every = 12 if live and not slow else 60 if live else 300
        if now - self.last["stocks"] >= every:
            st = now - self.last["struct"] >= 60
            self.cycle(True, st)
            self.last["stocks"] = now
            if st:
                self.last["struct"] = now
        elif live and not slow:
            self.cycle(False, False)
        return 4.0 if live else 60.0

    # ---- API payloads -----------------------------------------------------------------------------
    def snapshot(self, paths: bool = True) -> dict[str, Any]:
        """Market Lab's /api/sm/live payload; `paths=False` drops each stock's 1-minute path to save bytes."""
        with self.store.lock:
            t = self.store.tape
            stocks = list(t["stocks"].values())
            idx = dict(t["indices"])
            time_, errors = t["time"], t["errors"]
        if not paths:
            stocks = [{k: v for k, v in r.items() if k != "path"} for r in stocks]
        ordered = sorted(stocks, key=lambda r: r.get("rank", 999))
        return {
            "time": time_,
            "market": market_status(self.clock, self.cal),
            "errors": errors,
            "indices": idx,
            "stocks": stocks,
            "sectors": sector_chips(stocks),
            "bull": [r["sym"] for r in ordered if r.get("side") == "bull"],
            "bear": [r["sym"] for r in ordered if r.get("side") == "bear"],
        }

    def quote(self, sym: str) -> dict[str, Any]:
        """Fresh last price of one symbol for the pop-up chart's live candle."""
        y = INDICES.get(sym, sym if sym.startswith("^") else sym + ".NS")
        m = self.src.yahoo_chart(y, "1m", "1d")["meta"]
        return {
            "ltp": m.get("regularMarketPrice"),
            "time": m.get("regularMarketTime"),
            "prev": m.get("chartPreviousClose"),
        }

    def chart(self, sym: str, tf: str = "5m") -> dict[str, Any]:
        """Candles and the smart-money read for the pop-up chart (Yahoo; 1d needs no daily frame)."""
        rng = {"1m": "5d", "5m": "1mo", "15m": "60d", "60m": "2y", "1d": "5y"}[tf]
        y = INDICES.get(sym, sym if sym.startswith("^") else sym + ".NS")
        d = self.src.yahoo_chart(y, tf, rng)
        b = chart_to_bars(d)
        daily = None
        if tf != "1d":
            try:
                daily = self._yahoo_daily(y)
            except (SourceError, KeyError, TypeError, ValueError):
                daily = None
        rd = smc.read(b, 2 if tf in ("1m", "5m") else 3, daily) if len(b) >= 30 else None
        m = d["meta"]
        return {
            "sym": sym,
            "tf": tf,
            "ltp": m.get("regularMarketPrice"),
            "prev": m.get("chartPreviousClose"),
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


def chart_to_bars(d: dict[str, Any]) -> Bars:
    """Bars from a Yahoo chart result; rows with a missing price are dropped and missing volume becomes 0."""
    q = d["indicators"]["quote"][0]
    cols = {
        k: np.array([np.nan if x is None else x for x in q[k]], dtype=float)
        for k in ("open", "high", "low", "close", "volume")
    }
    ts = np.asarray(d["timestamp"], dtype=np.int64)
    ok = ~(np.isnan(cols["open"]) | np.isnan(cols["high"]) | np.isnan(cols["low"]) | np.isnan(cols["close"]))
    vol = np.nan_to_num(cols["volume"], nan=0.0)
    return Bars.from_arrays(
        ts[ok], cols["open"][ok], cols["high"][ok], cols["low"][ok], cols["close"][ok], vol[ok]
    )
