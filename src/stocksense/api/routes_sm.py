"""Smart Money tape routes (port of market-lab's /api/sm/*)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from stocksense.api.deps import get_tape
from stocksense.marketlab.sources import SourceError
from stocksense.marketlab.tape import TapeEngine

router = APIRouter(prefix="/api/sm")
TFS = {"1m", "5m", "15m", "60m", "1d"}


@router.get("/live")
def live(tape: Annotated[TapeEngine, Depends(get_tape)], paths: str = "1") -> dict[str, Any]:
    """The tape: indices, every F&O stock row, sector chips and the bull/bear orders."""
    return tape.snapshot(paths == "1")


@router.get("/chart")
def chart(
    tape: Annotated[TapeEngine, Depends(get_tape)], sym: str, tf: Annotated[str, Query()] = "5m"
) -> dict[str, Any]:
    """Candles and the smart-money read of one symbol for the pop-up chart."""
    if tf not in TFS:
        raise HTTPException(status_code=400, detail="unknown timeframe")
    try:
        return tape.chart(sym.upper(), tf)
    except SourceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/quote")
def quote(tape: Annotated[TapeEngine, Depends(get_tape)], sym: str) -> dict[str, Any]:
    """Fresh last price of one symbol for the pop-up chart's live candle."""
    try:
        return tape.quote(sym.upper())
    except SourceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
