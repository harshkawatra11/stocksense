"""Order-flow routes (`/api/of/*`): bar-based estimates of profile, delta, footprint, big trades. GET only."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from stocksense.api.deps import get_clock, get_orderflow
from stocksense.core.clock import Clock
from stocksense.core.errors import DataNotAvailable
from stocksense.marketlab.bars import Bars
from stocksense.orderflow.engine import OrderflowEngine

router = APIRouter(prefix="/api/of")
Engine = Annotated[OrderflowEngine, Depends(get_orderflow)]
Sym = Annotated[str, Query(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9&_\-]+$")]
Day = Annotated[date | None, Query()]


def _meta(method: str, source: str) -> dict[str, Any]:
    return {"method": method, "data_source": source, "estimated": True}


def _load(eng: OrderflowEngine, clock: Clock, sym: str, d: date | None) -> tuple[Bars, str, date, str]:
    day = d or clock.today()
    sym = sym.upper()
    b, source = eng.bars(sym, day)
    return b, source, day, sym


def _missing(e: DataNotAvailable) -> JSONResponse:
    return JSONResponse({"error": str(e)}, status_code=404)


@router.get("/profile", response_model=None)
def profile(
    eng: Engine,
    clock: Annotated[Clock, Depends(get_clock)],
    sym: Sym,
    date: Day = None,  # noqa: A002 - the query parameter is named `date`
    kind: Literal["volume", "tpo"] = "volume",
) -> dict[str, Any] | JSONResponse:
    """Volume profile (bars spread uniformly over their range) or TPO profile of one session."""
    try:
        b, src, day, s = _load(eng, clock, sym, date)
    except DataNotAvailable as e:
        return _missing(e)
    method = "volume_profile_uniform_spread" if kind == "volume" else "tpo_30min_letters"
    return {"sym": s, "date": day.isoformat(), "kind": kind, **_meta(method, src), **eng.profile(b, kind)}


@router.get("/delta", response_model=None)
def delta(
    eng: Engine,
    clock: Annotated[Clock, Depends(get_clock)],
    sym: Sym,
    date: Day = None,  # noqa: A002
    method: Literal["bvc", "clv"] = "bvc",
) -> dict[str, Any] | JSONResponse:
    """Estimated per-bar delta and cumulative delta (bulk-volume or close-location-value)."""
    try:
        b, src, day, s = _load(eng, clock, sym, date)
    except DataNotAvailable as e:
        return _missing(e)
    name = "bulk_volume_classification" if method == "bvc" else "close_location_value"
    return {"sym": s, "date": day.isoformat(), **_meta(name, src), **eng.delta(b, method)}


@router.get("/footprint", response_model=None)
def footprint(
    eng: Engine,
    clock: Annotated[Clock, Depends(get_clock)],
    sym: Sym,
    date: Day = None,  # noqa: A002
    tf: Annotated[int, Query(ge=1, le=60)] = 5,
) -> dict[str, Any] | JSONResponse:
    """Footprint ladders (estimated buy and sell volume per price bin) per `tf`-minute bar."""
    try:
        b, src, day, s = _load(eng, clock, sym, date)
    except DataNotAvailable as e:
        return _missing(e)
    return {
        "sym": s,
        "date": day.isoformat(),
        **_meta("bvc_split_over_bar_range", src),
        **eng.footprint(b, tf),
    }


@router.get("/bigtrades", response_model=None)
def bigtrades(
    eng: Engine,
    clock: Annotated[Clock, Depends(get_clock)],
    sym: Sym,
    date: Day = None,  # noqa: A002
) -> dict[str, Any] | JSONResponse:
    """Bars whose volume exceeds 5x the same-minute median of the previous 20 sessions."""
    try:
        b, src, day, s = _load(eng, clock, sym, date)
        out = eng.bigtrades(s, day, b, src)
    except DataNotAvailable as e:
        return _missing(e)
    return {"sym": s, "date": day.isoformat(), **_meta("volume_spike_vs_same_minute_median", src), **out}


@router.get("/heatmap")
def heatmap(
    eng: Engine,
    clock: Annotated[Clock, Depends(get_clock)],
    sym: Sym,
    date: Day = None,  # noqa: A002
) -> dict[str, Any]:
    """Resting depth (top 5 levels, 30 s buckets) for a day the capture loop ran, else {empty: true}."""
    day = date or clock.today()
    return {
        "sym": sym.upper(),
        "date": day.isoformat(),
        **_meta("depth_snapshot_max_qty", "depth_snaps"),
        **eng.depth(sym.upper(), day),
    }


@router.get("/timescales", response_model=None)
def timescales(eng: Engine, sym: Sym) -> dict[str, Any] | JSONResponse:
    """Multi-timeframe alignment: trend, value, value-area and momentum signs and their mean score."""
    try:
        out = eng.timescales(sym.upper())
    except DataNotAvailable as e:
        return _missing(e)
    return {"sym": sym.upper(), **_meta("structure_vwap_va_roc_signs", "stored_5m+panel_1d"), **out}


@router.get("/bars", response_model=None)
def bars(
    eng: Engine,
    clock: Annotated[Clock, Depends(get_clock)],
    sym: Sym,
    date: Day = None,  # noqa: A002
    tf: Annotated[int, Query(ge=1, le=5)] = 1,
    vwap: bool = False,
    bands: bool = False,
    anchor: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any] | JSONResponse:
    """Candles for the workbench chart; optional session VWAP, sigma bands and anchored VWAP (epoch s)."""
    if tf not in (1, 5):
        return JSONResponse({"error": "tf must be 1 or 5"}, status_code=400)
    try:
        b, src, day, s = _load(eng, clock, sym, date)
    except DataNotAvailable as e:
        return _missing(e)
    return {
        "sym": s,
        "date": day.isoformat(),
        "tf": tf,
        **_meta("ohlcv_bars", src),
        **eng.candles(b, tf, vwap, bands, anchor),
    }
