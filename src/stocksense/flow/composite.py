"""Flow composite: one descriptive card per F&O stock, the EOD flow build and the cached EOD rows.

Nothing here forecasts. `flow_score` is the mean of the available components, each z-scored across today's
F&O rows and clipped to +-3, so it only says which stocks look unusually one-sided right now.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import SystemClock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.flow.deals import build_deals_daily
from stocksense.flow.delivery import build_delivery_daily
from stocksense.flow.oi import build_oi_daily
from stocksense.flow.options import build_options_daily
from stocksense.flow.participants import build_participants_daily
from stocksense.flow.sector_flow import build_sector_flow_daily
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.store import LabStore
from stocksense.orderflow.delta import cvd, delta_bvc

log = get_logger(__name__)

CLIP = 3.0
MIN_Z_ROWS = 3
CVD_BARS = 30
CURVE_SESSIONS = 20
CURVE_MIN_SESSIONS = 10
CURVE_GRID = np.arange(0, 376, 5, dtype=float)  # minutes since 09:15 at 5-minute bar ends
OPEN_MINUTE = 555  # 09:15 in minutes of the day

# EOD jobs: (name, source dataset whose dates define "available", derived dataset, builder)
_EOD_JOBS = (
    ("oi", "fo_bhavcopy", "flow_oi_daily"),
    ("delivery", "bhavcopy_delivery", "flow_delivery_daily"),
    ("participants", "participant_oi", "flow_participants_daily"),
    ("options", "fo_bhavcopy", "flow_options_daily"),
    ("deals", "deals", "flow_deals_daily"),
    ("sector", "panel_universe", "flow_sector_daily"),
)


# ---- cross-sectional score -------------------------------------------------------------------------
def zscores(values: dict[str, float | None]) -> dict[str, float | None]:
    """Cross-sectional z-score (population sd) of finite values, clipped to +-3; None under 3 rows."""
    ok = {k: float(v) for k, v in values.items() if v is not None and math.isfinite(v)}
    out: dict[str, float | None] = dict.fromkeys(values)
    if len(ok) < MIN_Z_ROWS:
        return out
    arr = np.array(list(ok.values()))
    sd = float(arr.std())
    if sd <= 0:
        return out
    mu = float(arr.mean())
    for k, v in ok.items():
        out[k] = float(np.clip((v - mu) / sd, -CLIP, CLIP))
    return out


def combine(z: dict[str, float | None]) -> tuple[float | None, list[str]]:
    """Mean of the components that exist, and their names (None when none exist)."""
    used = [k for k, v in z.items() if v is not None]
    if not used:
        return None, []
    return float(np.mean([z[k] for k in used])), used


# ---- relative volume --------------------------------------------------------------------------------
def vol_curve(bars: pl.DataFrame, today: date) -> dict[str, list[float]] | None:
    """Mean cumulative volume by minute since 09:15 over the last 20 full sessions before `today`.

    `bars` are 5-minute rows (ts in IST, volume). Returns {"x": minutes, "y": shares} or None when fewer
    than 10 sessions have a nearly complete day (70+ bars).
    """
    if not bars.height:
        return None
    b = bars.with_columns(
        pl.col("ts").dt.date().alias("_d"),
        ((pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute() - OPEN_MINUTE) // 5).alias(
            "_k"
        ),
    ).filter((pl.col("_d") < today) & pl.col("_k").is_between(0, 74))
    curves = []
    for d in sorted(b["_d"].unique().to_list())[-CURVE_SESSIONS:]:
        day = b.filter(pl.col("_d") == d)
        if day.height < 70:
            continue
        per = np.zeros(75)
        per[day["_k"].to_numpy()] = day["volume"].fill_null(0).to_numpy().astype(float)
        curves.append(np.r_[0.0, np.cumsum(per)])
    if len(curves) < CURVE_MIN_SESSIONS:
        return None
    return {"x": CURVE_GRID.tolist(), "y": np.mean(curves, axis=0).tolist()}


def rvol(volume: float | None, curve: dict[str, list[float]] | None, minute: int) -> float | None:
    """Volume so far today over the 20-session mean at the same minute (interpolated); None if unknown."""
    if volume is None or curve is None or minute <= 0:
        return None
    exp = float(np.interp(min(minute, 375), curve["x"], curve["y"]))
    return float(volume) / exp if exp > 0 else None


def cvd_slope(b: Bars, n: int = CVD_BARS) -> float | None:
    """OLS slope of the bulk-volume CVD over the last `n` bars divided by their mean volume (per bar)."""
    if len(b) < n:
        return None
    return slope_per_volume(cvd(delta_bvc(b), b.day_index())[-n:], b.volume[-n:])


def slope_per_volume(series: np.ndarray, volume: np.ndarray) -> float | None:
    """OLS slope of `series` against its index, divided by the mean of `volume`; None if no volume."""
    mv = float(np.mean(volume))
    if mv <= 0:
        return None
    x = np.arange(len(series), dtype=float)
    return float(np.polyfit(x, np.asarray(series, dtype=float), 1)[0] / mv)


# ---- EOD rows -----------------------------------------------------------------------------------------
def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, date | datetime):
        return v.isoformat()
    return v


def rows_of(df: pl.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe row dicts (NaN to None, dates to ISO strings)."""
    return [{k: _clean(v) for k, v in r.items()} for r in df.iter_rows(named=True)]


def _latest(df: pl.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    if not df.height:
        return {}
    last = df.sort("date").group_by(key, maintain_order=True).last()
    return {str(r[key]): r for r in last.iter_rows(named=True)}


class EodFlow:
    """Latest EOD flow rows per symbol, held in memory and reloaded when `asof` changes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.asof: date | None = None
        self.oi: dict[str, dict[str, Any]] = {}
        self.deliv: dict[str, dict[str, Any]] = {}
        self.options: dict[str, dict[str, Any]] = {}

    def load(self, asof: date) -> None:
        """Read the newest row per symbol up to `asof` (15 calendar days back); no-op if already loaded."""
        if asof == self.asof:
            return
        lo = asof - timedelta(days=15)
        s = self.settings
        self.oi = _latest(io.read(s, "flow_oi_daily", lo, asof).collect(), "symbol")
        self.options = _latest(io.read(s, "flow_options_daily", lo, asof).collect(), "underlying")
        dl = _latest(io.read(s, "flow_delivery_daily", lo, asof).collect(), "sid")
        pn = io.read(s, "panel_daily", lo, asof, ["sid", "symbol", "date", "series"]).collect()
        names = _latest(pn.filter(pl.col("series") == "EQ"), "sid") if pn.height else {}
        self.deliv = {names[k]["symbol"]: v for k, v in dl.items() if k in names}
        self.asof = asof


# ---- the card -----------------------------------------------------------------------------------------
def minute_of(store: LabStore) -> int:
    """Minutes since 09:15 at the tape's last refresh (from its ISO time); 0 when unknown."""
    t = store.tape.get("time")
    if not t:
        return 0
    d = datetime.fromisoformat(t)
    return d.hour * 60 + d.minute - OPEN_MINUTE


def _components(sym: str, row: dict[str, Any], store: LabStore, eod: EodFlow) -> dict[str, float | None]:
    fl = store.flow
    rv = rvol(
        (store.tape.get("meta", {}).get(sym) or {}).get("volume"),
        (fl.get("vol_curve") or {}).get(sym),
        minute_of(store),
    )
    oi = eod.oi.get(sym) or {}
    return {
        "disp": row.get("disp"),
        "rvol": rv,
        "oi_quad": oi.get("quad"),
        "depth_imb5": ((fl.get("depth") or {}).get(sym) or {}).get("imb5"),
        "cvd_slope_30m": (fl.get("cvd") or {}).get(sym),
    }


def score_table(store: LabStore, eod: EodFlow) -> dict[str, dict[str, Any]]:
    """Per tape symbol: raw components, z-scored mean `flow_score` and `components_used` (computed once)."""
    with store.lock:
        rows = dict(store.tape.get("stocks", {}))
    raw = {s: _components(s, r, store, eod) for s, r in rows.items()}
    cols = {
        "disp": {s: c["disp"] for s, c in raw.items()},
        "rvol": {s: None if c["rvol"] is None else c["rvol"] - 1.0 for s, c in raw.items()},
        "oi_quad": {s: c["oi_quad"] for s, c in raw.items()},
        "depth_imb5": {s: c["depth_imb5"] for s, c in raw.items()},
        "cvd_slope_30m": {s: c["cvd_slope_30m"] for s, c in raw.items()},
    }
    z = {k: zscores(v) for k, v in cols.items()}
    out = {}
    for s in raw:
        score, used = combine({k: z[k][s] for k in z})
        out[s] = {"raw": raw[s], "flow_score": None if score is None else round(score, 3), "used": used}
    return out


def stock_card(
    sym: str, store: LabStore, eod: EodFlow, table: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Descriptive flow card of one F&O stock (not a forecast); None when the tape has no row for it."""
    with store.lock:
        row = store.tape.get("stocks", {}).get(sym)
        pre = ((store.flow.get("preopen") or {}).get("rows") or {}).get(sym) or {}
        bsr = (store.flow.get("bsr") or {}).get(sym)
    if row is None:
        return None
    t = (table or score_table(store, eod)).get(sym)
    if t is None:
        return None
    raw, oi, dl, op = t["raw"], eod.oi.get(sym) or {}, eod.deliv.get(sym) or {}, eod.options.get(sym)
    gex = op.get("gex_total_cr") if op else None
    return {
        "disp": row.get("disp"),
        "rvol": None if raw["rvol"] is None else round(raw["rvol"], 2),
        "oi_quad": oi.get("quad"),
        "oi_quad_label": oi.get("quad_label"),
        "oi_chg_pct": _clean(oi.get("oi_chg_pct")),
        "deliv_z20_prev": _clean(dl.get("deliv_z20")),
        "depth_imb5": raw["depth_imb5"],
        "buy_sell_ratio": bsr,
        "preopen_imb": pre.get("imb"),
        "cvd_slope_30m": raw["cvd_slope_30m"],
        "options": None
        if not op
        else {
            "pcr": _clean(op.get("pcr_oi")),
            "gex_sign": None if gex is None or gex != gex else (gex > 0) - (gex < 0),
            "call_wall": _clean(op.get("call_wall")),
            "put_wall": _clean(op.get("put_wall")),
        },
        "flow_score": t["flow_score"],
        "components_used": t["used"],
        "descriptive": True,
    }


def attach_flow(stocks: list[dict[str, Any]], store: LabStore, eod: EodFlow | None) -> list[dict[str, Any]]:
    """Copies of tape rows with a `flow` card (None when it cannot be computed, e.g. no EOD engine)."""
    if eod is None:
        return [{**r, "flow": None} for r in stocks]
    table = score_table(store, eod)
    return [{**r, "flow": stock_card(r["sym"], store, eod, table)} for r in stocks]


# ---- EOD build ----------------------------------------------------------------------------------------
def _dates(settings: Settings, dataset: str, end: date) -> set[date]:
    col = "date"
    df = io.read(settings, dataset, None, end, [col]).select(col).unique().collect()
    return set(df[col].to_list())


def build_eod(settings: Settings, asof: date, calendar: TradingCalendar | None = None) -> dict[str, int]:
    """Build every flow dataset for the dates the source has but the flow table lacks, up to `asof`.

    Idempotent: a second call finds nothing missing (and upserts are keyed, so a rerun is harmless).
    Returns rows written per job.
    """
    done: dict[str, int] = {}
    for name, source, target in _EOD_JOBS:
        have = _dates(settings, target, asof)
        floor = min(have) if have else None  # dates before the first derived row never produce one
        missing = sorted(d for d in _dates(settings, source, asof) - have if floor is None or d > floor)
        if not missing:
            done[name] = 0
            continue
        a = missing[0]
        if name == "oi":
            cal = calendar or TradingCalendar.from_store(settings, SystemClock())
            done[name] = build_oi_daily(settings, a, asof, cal)
        elif name == "delivery":
            done[name] = build_delivery_daily(settings, a, asof)
        elif name == "participants":
            done[name] = build_participants_daily(settings, a, asof)
        elif name == "options":
            done[name] = build_options_daily(settings, a, asof)
        elif name == "deals":
            done[name] = build_deals_daily(settings, a, asof)
        else:
            done[name] = build_sector_flow_daily(settings, a, asof)
        log.info("flow_eod_built", job=name, start=str(a), rows=done[name])
    return done
