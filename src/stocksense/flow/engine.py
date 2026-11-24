"""Flow engine: the live loop (pre-open, relative-volume curves, CVD slope, sector value history) and the
read side behind `/api/flow/*`. Everything it serves is descriptive, never a forecast."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.flow.composite import (
    EodFlow,
    build_eod,
    cvd_slope,
    minute_of,
    rows_of,
    score_table,
    stock_card,
    vol_curve,
)
from stocksense.flow.preopen import PreopenEngine
from stocksense.flow.sector_flow import (
    HIST_MIN,
    ValueHistory,
    live_sector_flow,
    rotation,
)
from stocksense.marketlab.clock import market_status
from stocksense.marketlab.sources import SourceError, Sources, SourceThrottled
from stocksense.marketlab.store import LabStore

log = get_logger(__name__)

INDEX_UNDERLYINGS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")
STOCK_SORTS = ("flow_score", "disp", "rvol", "oi_chg_pct")
RANGE_DAYS = {"w1": 5, "m1": 21}
CVD_EVERY_S = 60.0
CVD_SYMBOLS = 20  # top and bottom this many tape ranks
VOL_LOOKBACK_DAYS = 45


def _last_n_dates(df: pl.DataFrame, n: int) -> list[date]:
    return sorted(df["date"].unique().to_list())[-n:]


class FlowEngine:
    """Owns the flow state in the LabStore; `step()` is one feed iteration, the view methods feed the API."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        cal: TradingCalendar,
        sources: Any,
        store: LabStore,
        capture: Any = None,
    ) -> None:
        self.settings, self.clock, self.cal, self.src, self.store = settings, clock, cal, sources, store
        self.preopen = PreopenEngine(clock, cal, sources, store, capture)
        self.history = ValueHistory(Paths(settings).lab / "flow_val")
        self.eod_rows = EodFlow(settings)
        self.last: dict[str, Any] = {"eod_day": None, "curve_day": None, "cvd": -1e9, "val_minute": -99}

    # ---- EOD cache ---------------------------------------------------------------------------------
    def asof(self) -> date | None:
        """Last completed session, or None when the calendar has none."""
        try:
            return self.cal.last_completed_session(self.clock.now())
        except ValueError:
            return None

    def eod(self) -> EodFlow:
        """The EOD rows cache, loaded for the latest completed session."""
        a = self.asof()
        if a is not None:
            self.eod_rows.load(a)
        return self.eod_rows

    # ---- feed iteration ------------------------------------------------------------------------------
    def step(self) -> float:
        """One iteration: pre-open poll, then (outside the pre-open window) the slower flow jobs."""
        wait = self.preopen.step()
        if self.preopen.in_window():
            return wait
        today = self.clock.today()
        if self.last["eod_day"] != today and (a := self.asof()) is not None:
            build_eod(self.settings, a, self.cal)
            self.eod()
            self.last["eod_day"] = today
        live = market_status(self.clock, self.cal) == "open"
        if self.store.tape.get("meta") and self.last["curve_day"] != today:
            self.load_curves(today)
        if live:
            self.record_values()
            self.refresh_cvd()
        return 30.0 if live else 120.0

    def load_curves(self, today: date) -> None:
        """Cache each F&O symbol's 20-session volume curve (5-minute bars) once per day."""
        syms = list(self.store.tape["meta"])
        from stocksense.data.ingest_intraday import load_bars

        bars = load_bars(
            self.settings, syms, today - timedelta(days=VOL_LOOKBACK_DAYS), today, "5m", "raw_ratio"
        )
        curves = {s: None for s in syms}
        if bars.height:
            for key, g in bars.partition_by("symbol", as_dict=True).items():
                curves[str(key[0])] = vol_curve(g, today)
        with self.store.lock:
            self.store.flow["vol_curve"] = curves
        self.last["curve_day"] = today

    def record_values(self) -> None:
        """Every 5 minutes: each sector's traded value so far, kept to compare future sessions."""
        m = minute_of(self.store)
        if not (0 <= m <= 375) or m - self.last["val_minute"] < 5:
            return
        vals: dict[str, float] = {}
        with self.store.lock:
            for r in self.store.tape.get("stocks", {}).values():
                vals[r["sector"]] = vals.get(r["sector"], 0.0) + (r.get("value_cr") or 0.0)
        if vals:
            self.history.record(self.clock.today(), m, vals)
            self.last["val_minute"] = m

    def refresh_cvd(self) -> None:
        """Every 60 s: CVD slope (1-minute bars, bulk-volume classification) for the top and bottom 20."""
        now = self.clock.now().timestamp()
        if now - self.last["cvd"] < CVD_EVERY_S:
            return
        self.last["cvd"] = now
        from stocksense.marketlab.tape import chart_to_bars

        with self.store.lock:
            ranked = sorted(self.store.tape.get("stocks", {}).values(), key=lambda r: r.get("rank", 999))
        bull = [r["sym"] for r in ranked if r.get("side") == "bull"][:CVD_SYMBOLS]
        bear = [r["sym"] for r in ranked if r.get("side") == "bear"][:CVD_SYMBOLS]
        out: dict[str, float | None] = {}
        for s in bull + bear:
            try:
                out[s] = cvd_slope(chart_to_bars(self.src.yahoo_chart(f"{s}.NS", "1m", "1d")))
            except SourceThrottled:
                break
            except SourceError:
                log.warning("flow_cvd_failed", sym=s)
        with self.store.lock:
            self.store.flow["cvd"] = out

    # ---- API views --------------------------------------------------------------------------------------
    def _tape_rows(self) -> list[dict[str, Any]]:
        with self.store.lock:
            return list(self.store.tape.get("stocks", {}).values())

    def _read(self, dataset: str, start: date | None, end: date | None) -> pl.DataFrame:
        return io.read(self.settings, dataset, start, end).collect()

    def stocks(self, sort: str) -> dict[str, Any]:
        """Tape rows with their flow card, best first by `sort` (missing values last)."""
        eod = self.eod()
        table = score_table(self.store, eod)
        out = []
        for r in self._tape_rows():
            card = stock_card(r["sym"], self.store, eod, table)
            if card is None:
                continue
            out.append(
                {
                    "sym": r["sym"],
                    "name": r["name"],
                    "sector": r["sector"],
                    "ltp": r["ltp"],
                    "pct": r["pct"],
                    **card,
                }
            )
        out.sort(key=lambda x: (x[sort] is None, -(x[sort] or 0.0)))
        return {"descriptive": True, "sort": sort, "rows": out}

    def oi(self, scope: str) -> dict[str, Any]:
        """Latest futures OI buildup rows, index futures or stock futures."""
        a = self.asof()
        df = self._read("flow_oi_daily", a - timedelta(days=15), a) if a else pl.DataFrame()
        rows: list[dict[str, Any]] = []
        if df.height:
            last = df.filter(pl.col("date") == df["date"].max())
            idx = pl.col("symbol").is_in(INDEX_UNDERLYINGS)
            last = last.filter(idx if scope == "index" else ~idx)
            cols = ["symbol", "quad_label", "oi_chg_pct", "price_chg_pct", "basis_pct", "rollover_pct"]
            rows = rows_of(last.select(cols).sort("symbol"))
        return {"descriptive": True, "scope": scope, "rows": rows}

    def participants(self, days: int) -> dict[str, Any]:
        """Last `days` sessions of participant positioning and FII/DII cash."""
        a = self.asof()
        cols = ["date", "fii_ratio", "dii_ratio", "pro_ratio", "client_ratio", "fii_cash_net", "dii_cash_net"]
        df = self._read("flow_participants_daily", a - timedelta(days=days * 2 + 10), a) if a else None
        rows = rows_of(df.select(cols).sort("date").tail(days)) if df is not None and df.height else []
        return {"descriptive": True, "rows": rows}

    def deals(self, days: int) -> dict[str, Any]:
        """Bulk and block deal summaries of the last `days` sessions."""
        a = self.asof()
        df = self._read("flow_deals_daily", a - timedelta(days=days * 2 + 5), a) if a else None
        if df is None or not df.height:
            return {"descriptive": True, "rows": []}
        keep = _last_n_dates(df, days)
        rows = rows_of(
            df.filter(pl.col("date").is_in(keep)).sort(["date", "symbol"], descending=[True, False])
        )
        return {"descriptive": True, "rows": rows}

    def options(self, und: str, expiry: str | None) -> dict[str, Any] | None:
        """Option chain strikes and positioning totals of one underlying; None when it has no data."""
        a = self.asof()
        if a is None:
            return None
        daily = self._read("flow_options_daily", a - timedelta(days=120), a)
        daily = daily.filter(pl.col("underlying") == und).sort("date")
        if not daily.height:
            return None
        last = daily.row(-1, named=True)
        st = self._read("flow_options_strikes", last["date"], last["date"]).filter(
            pl.col("underlying") == und
        )
        exps = sorted(st["expiry"].unique().to_list())
        chosen = (
            date.fromisoformat(expiry) if expiry else last.get("near_expiry") or (exps[0] if exps else None)
        )
        cols = ["strike", "call_oi", "put_oi", "call_doi", "put_doi", "call_iv", "put_iv", "gex_cr"]
        strikes = st.filter(pl.col("expiry") == chosen).select(cols).sort("strike") if chosen else st.head(0)
        keys = ("gex_total_cr", "zero_gamma", "call_wall", "put_wall", "max_pain", "pcr_oi", "pcr_doi")
        return {
            "descriptive": True,
            "und": und,
            "spot": last["spot"],
            "expiry": chosen.isoformat() if chosen else None,
            "expiries": [e.isoformat() for e in exps],
            "strikes": rows_of(strikes),
            "totals": {k: rows_of(daily.tail(1).select(k))[0][k] for k in keys},
            "history": rows_of(daily.select("date", "gex_total_cr", "pcr_oi", "spot")),
        }

    def sectors(self, rng: str) -> dict[str, Any]:
        """Sector money flow: live from the tape (`today`) or aggregated from flow_sector_daily (w1, m1)."""
        if rng == "today":
            eod = self.eod()
            quads = {s: v.get("quad_label") for s, v in eod.oi.items() if v.get("quad_label")}
            spikes = {s for s, v in eod.deliv.items() if v.get("deliv_spike")}
            base = self.history.mean_share(self.clock.today(), minute_of(self.store))
            out = live_sector_flow(self._tape_rows(), base, quads, spikes)
            return {"descriptive": True, "range": rng, "rows": out}
        a = self.asof()
        n = RANGE_DAYS[rng]
        df = self._read("flow_sector_daily", a - timedelta(days=200), a) if a else pl.DataFrame()
        if not df.height:
            return {"descriptive": True, "range": rng, "rows": []}
        dates = sorted(df["date"].unique().to_list())
        cur_d, base_d = dates[-n:], dates[-n - 20 : -n]
        cur = df.filter(pl.col("date").is_in(cur_d))
        base = df.filter(pl.col("date").is_in(base_d))
        base_share = (
            base.group_by("sector").agg(pl.col("turnover_share").mean()) if len(base_d) >= HIST_MIN else None
        )
        bs = dict(base_share.iter_rows()) if base_share is not None else {}
        out = []
        for (sector,), g in cur.group_by("sector", maintain_order=True):
            val = g["value_cr"].sum()
            fd = (g["flow_dir"].fill_null(0.0) * g["value_cr"]).sum() / val if val and val > 0 else None
            share = g["turnover_share"].mean()
            ratio = share / bs[sector] if bs.get(sector) else None
            out.append(
                {
                    "name": sector,
                    "flow_dir": None if fd is None else round(float(fd), 4),
                    "turnover_share": round(float(share), 4),
                    "share_ratio": None if ratio is None else round(float(ratio), 3),
                    "rotation": rotation(ratio, fd),
                    "up": int(g["up"].sum()),
                    "down": int(g["down"].sum()),
                    "oi_quads": {},
                    "deliv_spikes": int(g["deliv_spikes"].sum()),
                }
            )
        out.sort(key=lambda s: -(s["flow_dir"] if s["flow_dir"] is not None else -9))
        return {"descriptive": True, "range": rng, "rows": out}

    def market(self) -> dict[str, Any]:
        """Participants, index option positioning, VIX, breadth and pre-open summary."""
        a = self.asof()
        out: dict[str, Any] = {
            "descriptive": True,
            "asof": a.isoformat() if a else None,
            "live": market_status(self.clock, self.cal) == "open",
        }
        part = (
            self._read("flow_participants_daily", a - timedelta(days=30), a).sort("date")
            if a
            else pl.DataFrame()
        )
        for c in ("fii", "dii", "pro", "client"):
            ratio, chg = f"{c}_ratio", f"{c}_ratio_chg5"
            cash = f"{c}_cash_net"
            have = part.height > 0 and ratio in part.columns
            net = (
                part[cash].drop_nulls() if have and cash in part.columns else pl.Series([], dtype=pl.Float64)
            )
            out[c] = {
                "cash_net": _clean_f(net[-1]) if net.len() else None,
                "cash_net_5d": _clean_f(net.tail(5).sum()) if net.len() else None,
                "idx_fut_ratio": _clean_f(part[ratio][-1]) if have else None,
                "ratio_chg5": _clean_f(part[chg][-1]) if have else None,
            }
        opt = self._read("flow_options_daily", a - timedelta(days=15), a) if a else pl.DataFrame()
        indices: dict[str, Any] = {}
        for und in ("NIFTY", "BANKNIFTY"):
            g = opt.filter(pl.col("underlying") == und).sort("date") if opt.height else opt
            if not g.height:
                indices[und] = None
                continue
            r = rows_of(g.tail(1))[0]
            gex = r["gex_total_cr"]
            indices[und] = {
                "spot": r["spot"],
                "pcr_oi": r["pcr_oi"],
                "gex_total_cr": gex,
                "regime": None if gex is None else "positive" if gex >= 0 else "negative",
                "zero_gamma": r["zero_gamma"],
                "call_wall": r["call_wall"],
                "put_wall": r["put_wall"],
                "max_pain": r["max_pain"],
                "atm_iv": r["atm_iv"],
            }
        out["indices"] = indices
        vix = self._read("india_vix", a - timedelta(days=15), a).sort("date") if a else pl.DataFrame()
        out["vix"] = (
            {
                "close": vix["close"][-1],
                "chg1": vix["close"][-1] - vix["close"][-2] if vix.height > 1 else None,
            }
            if vix.height
            else None
        )
        rows = self._tape_rows()
        out["breadth"] = {"adv": sum(r["pct"] > 0 for r in rows), "dec": sum(r["pct"] < 0 for r in rows)}
        pre = self.preopen.payload()
        out["preopen"] = (
            {
                "adv": pre["breadth"]["adv"],
                "dec": pre["breadth"]["dec"],
                "nifty_iep": (pre["nifty"] or {}).get("iep"),
            }
            if pre["rows"]
            else None
        )
        return out


def _clean_f(v: Any) -> float | None:
    return None if v is None or v != v else float(v)


def build_flow_engine(settings: Settings, clock: Clock, store: LabStore, tape: Any) -> FlowEngine | None:
    """Real FlowEngine sharing the tape engine's calendar and sources, or None without a tape engine."""
    if tape is None:
        return None
    from stocksense.core.state import State
    from stocksense.data.ingest_common import IngestContext
    from stocksense.data.ingest_misc import capture_preopen

    src: Sources = tape.src
    ctx = IngestContext(settings, clock, tape.cal, src.nse, State(settings, clock))
    return FlowEngine(settings, clock, tape.cal, src, store, capture=lambda: capture_preopen(ctx))
