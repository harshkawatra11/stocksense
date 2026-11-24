"""Smart Money tape routes (port of market-lab's /api/sm/*)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from stocksense.api.deps import get_tape
from stocksense.flow.composite import attach_flow
from stocksense.marketlab.sources import SourceError
from stocksense.marketlab.tape import TapeEngine

router = APIRouter(prefix="/api/sm")
TFS = {"1m", "5m", "15m", "60m", "1d"}


@router.get("/live")
def live(
    request: Request, tape: Annotated[TapeEngine, Depends(get_tape)], paths: str = "1"
) -> dict[str, Any]:
    """The tape: indices, every F&O stock row (with its `flow` card, null if unavailable), sector chips."""
    snap = tape.snapshot(paths == "1")
    flow = request.app.state.flow
    snap["stocks"] = attach_flow(snap["stocks"], request.app.state.store, flow.eod() if flow else None)
    return snap


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
