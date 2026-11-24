"""Sector Map routes (port of market-lab's /api/horizon, sector, stock, search, chart, tick)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from stocksense.api.deps import get_sector
from stocksense.marketlab.sectors import SectorEngine
from stocksense.marketlab.sources import SourceError

router = APIRouter(prefix="/api")
RANGES = {"today", "w1", "m1", "m3"}
TFS = {"1m", "5m", "15m", "60m", "1d"}
Eng = Annotated[SectorEngine, Depends(get_sector)]


@router.get("/horizon")
def horizon(eng: Eng, range: str = "today") -> dict[str, Any]:  # noqa: A002 (the query parameter is `range`)
    """The 20 sectors over today / w1 / m1 / m3: ranks, trails, climbs, themes, headlines, Nifty chart."""
    if range not in RANGES:
        raise HTTPException(status_code=400, detail="unknown range")
    return eng.horizon(range)


@router.get("/sector")
def sector(eng: Eng, name: str = "Energy", range: str = "today") -> dict[str, Any]:  # noqa: A002
    """One sector: index, rank, stocks by industry, movers, headlines and the sector-vs-Nifty chart."""
    if range not in RANGES:
        raise HTTPException(status_code=400, detail="unknown range")
    out = eng.sector_view(name, range)
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@router.get("/stock")
def stock(eng: Eng, sym: str = "RELIANCE") -> dict[str, Any]:
    """One stock against its sector and Nifty: peers, rank, lines, headlines, volume, delivery."""
    return eng.stock_view(sym)


@router.get("/search")
def search(eng: Eng, q: str = "") -> list[dict[str, Any]]:
    """Sectors and stocks matching the text (at most 15)."""
    return eng.search(q)


@router.get("/chart")
def chart(eng: Eng, sym: str = "RELIANCE", tf: str = "5m") -> dict[str, Any]:
    """Candles and the smart-money read of one symbol for the pop-up chart."""
    if tf not in TFS:
        raise HTTPException(status_code=400, detail="unknown timeframe")
    try:
        return eng.chart(sym.upper(), tf)
    except SourceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/tick")
def tick(eng: Eng, syms: str = "") -> dict[str, Any]:
    """Live prices of the (at most 150) symbols on screen plus every sector's and Nifty's move."""
    return eng.tick([s for s in syms.split(",") if s][:150])
