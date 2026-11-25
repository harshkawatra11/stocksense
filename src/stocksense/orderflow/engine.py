"""Order-flow service behind `/api/of/*`: picks the bar source for a symbol and day and runs the pure tools.

Every number is an estimate from OHLCV bars (no tick data exists for NSE equities here); responses say so.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.clock import IST, Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.data.ingest_intraday import load_bars
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.tape import chart_to_bars
from stocksense.orderflow import bigtrades, delta, footprint, heatmap, profile, timescales, vwap

log = get_logger(__name__)

YAHOO_INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
HIST_DAYS = 45  # calendar days of same-interval history for the big-trade baseline
DAILY_DAYS = 500
INTRADAY_DAYS = 14


def jsonable(x: Any) -> Any:
    """Recursively convert numpy values to plain Python (NaN and inf become None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _day_bounds(d: date) -> tuple[int, int]:
    a = int(datetime.combine(d, time(0), IST).timestamp())
    return a, a + 86400


def _frame_to_bars(df: pl.DataFrame) -> Bars:
    return Bars.from_polars(df.sort("ts")) if df.height else Bars.from_arrays([], [], [], [], [], [])


class OrderflowEngine:
    """Bars for a symbol and day (live Yahoo, Angel, or stored) and the order-flow tools over them."""

    def __init__(
        self, settings: Settings, clock: Clock, sources: Any = None, angel_candles: Any = None
    ) -> None:
        self.settings, self.clock = settings, clock
        self.sources = sources
        self.angel_candles = angel_candles  # callable(sym, start_dt, end_dt) -> polars frame, or None

    # ---- bar sources ------------------------------------------------------------------------------
    def _live(self, sym: str, d: date) -> tuple[Bars, str] | None:
        if self.sources is not None:
            ysym = YAHOO_INDEX.get(sym, f"{sym}.NS")
            try:
                b = chart_to_bars(self.sources.yahoo_chart(ysym, "1m", "1d"))
                lo, hi = _day_bounds(d)
                b = b.take((b.ts >= lo) & (b.ts < hi))
                if len(b):
                    return b, "yahoo_1m"
            except Exception as e:  # noqa: BLE001 - a dead live source falls through to the next one
                log.warning("of_yahoo_failed", sym=sym, error=type(e).__name__)
        if self.angel_candles is not None:
            try:
                start = datetime.combine(d, time(9, 15), IST)
                df = self.angel_candles(sym, start, datetime.combine(d, time(15, 30), IST))
                if df.height:
                    return _frame_to_bars(df), "angel_1m"
            except Exception as e:  # noqa: BLE001
                log.warning("of_angel_failed", sym=sym, error=type(e).__name__)
        return None

    def bars(self, sym: str, d: date) -> tuple[Bars, str]:
        """(1-minute-or-5-minute bars of `sym` on `d`, data_source); raises DataNotAvailable."""
        if d == self.clock.today():
            live = self._live(sym, d)
            if live is not None:
                return live
        for interval in ("1m", "5m"):
            df = load_bars(self.settings, [sym], d, d, interval=interval)
            if df.height:
                return _frame_to_bars(df), f"stored_{interval}"
        raise DataNotAvailable(f"no intraday bars for {sym} on {d}")

    def history(self, sym: str, d: date, source: str) -> Bars:
        """Stored bars of the same interval for the sessions before `d` (the big-trade baseline)."""
        interval = "1m" if source.endswith("1m") else "5m"
        df = load_bars(
            self.settings, [sym], d - timedelta(days=HIST_DAYS), d - timedelta(days=1), interval=interval
        )
        return _frame_to_bars(df)

    def daily(self, sym: str, end: date) -> Bars:
        """Split-adjusted daily bars up to `end` (ts = midnight IST of the session)."""
        df = (
            io.read(
                self.settings,
                "panel_daily",
                start=end - timedelta(days=DAILY_DAYS),
                end=end,
                columns=["symbol", "date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"],
            )
            .filter(pl.col("symbol") == sym)
            .sort("date")
            .collect()
        )
        if df.height == 0:
            return Bars.from_arrays([], [], [], [], [], [])
        ts = [int(datetime.combine(x, time(0), IST).timestamp()) for x in df["date"].to_list()]
        return Bars.from_arrays(
            ts, df["adj_open"], df["adj_high"], df["adj_low"], df["adj_close"], df["adj_volume"]
        )

    def intraday_5m(self, sym: str, end: date) -> Bars:
        """Stored 5-minute bars of the last two weeks up to `end`."""
        df = load_bars(self.settings, [sym], end - timedelta(days=INTRADAY_DAYS), end, interval="5m")
        return _frame_to_bars(df)

    # ---- tools -------------------------------------------------------------------------------------
    def profile(self, b: Bars, kind: str) -> dict[str, Any]:
        """Volume or TPO profile of the day (rupee prices, shares)."""
        if kind == "tpo":
            return jsonable(profile.tpo_profile(b))
        return jsonable(profile.volume_profile(b))

    def delta(self, b: Bars, method: str) -> dict[str, Any]:
        """Per-bar delta, CVD and CVD divergence (shares)."""
        d = delta.delta_bvc(b) if method == "bvc" else delta.delta_clv(b)
        c = delta.cvd(d, b.day_index())
        return jsonable(
            {"ts": b.ts, "delta": d, "cvd": c, "divergence": delta.divergence(b, c), "close": b.close}
        )

    def footprint(self, b: Bars, tf: int) -> dict[str, Any]:
        """Footprint ladders per `tf`-minute bar."""
        bin_ = profile.bin_size(float(np.mean(b.close)))
        return jsonable({"bin": bin_, "tf": tf, "bars": footprint.footprint(b, tf, bin_)})

    def bigtrades(self, sym: str, d: date, b: Bars, source: str) -> dict[str, Any]:
        """Volume spikes against the same-minute median of the previous 20 sessions."""
        hist = self.history(sym, d, source)
        return jsonable({"trades": bigtrades.big_bars(b, hist), "history_bars": len(hist)})

    def depth(self, sym: str, d: date) -> dict[str, Any]:
        """Depth heatmap of a captured day, or {empty: true, reason}."""
        df = io.read(self.settings, heatmap.DATASET, start=d, end=d).filter(pl.col("symbol") == sym).collect()
        if df.height == 0:
            return {"empty": True, "reason": "depth not captured for this date"}
        bin_ = profile.bin_size(float(df["price"].mean()))
        return {"empty": False, "bin": bin_, **heatmap.heatmap(df, bin_, 30)}

    def timescales(self, sym: str) -> dict[str, Any]:
        """Alignment table across 5m, 15m, 60m, 1d, 1w from stored bars up to today."""
        frames: dict[str, Bars] = {}
        five = self.intraday_5m(sym, self.clock.today())
        if len(five):
            frames["5m"] = five
        daily = self.daily(sym, self.clock.today())
        if len(daily):
            frames["1d"] = daily
        if not frames:
            raise DataNotAvailable(f"no stored bars for {sym}")
        return jsonable(timescales.alignment(frames))

    def candles(
        self, b: Bars, tf: int, with_vwap: bool, with_bands: bool, anchor: int | None
    ) -> dict[str, Any]:
        """Candles of `tf` minutes with optional session VWAP, sigma bands and an anchored VWAP."""
        if tf == 5:
            b = timescales.resample(b, "5m")
        out: dict[str, Any] = {
            "candles": [
                {"time": int(t), "open": o, "high": h, "low": lo, "close": c, "volume": v}
                for t, o, h, lo, c, v in zip(b.ts, b.open, b.high, b.low, b.close, b.volume, strict=True)
            ]
        }
        if with_vwap or with_bands:
            vw, sig = vwap.session_vwap(b)
            out["vwap"] = vw
            if with_bands:
                out["bands"] = vwap.bands(vw, sig)
        if anchor is not None:
            out["anchored"] = vwap.anchored_vwap(b, anchor)
        return jsonable(out)


def build_orderflow_engine(settings: Settings, clock: Clock, store: Any) -> OrderflowEngine:
    """Engine with live Yahoo (Angel candles only when its secrets are set; read-only)."""
    from stocksense.marketlab.sources import Sources

    return OrderflowEngine(settings, clock, Sources(settings, clock, store))
