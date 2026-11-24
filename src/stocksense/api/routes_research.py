"""Research-note route (port of market-lab's /api/research)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from stocksense.api.deps import get_research
from stocksense.core.errors import DataNotAvailable
from stocksense.marketlab.research import INDEX_SYMBOLS, ResearchEngine
from stocksense.marketlab.sources import SourceError

router = APIRouter(prefix="/api")


@router.get("/research")
def research(
    eng: Annotated[ResearchEngine, Depends(get_research)], sym: str = "", fresh: str = "0"
) -> dict[str, Any]:
    """The one-page research note of a stock (cached 12 hours; `fresh=1` rebuilds)."""
    sym = sym.upper()
    if not sym or sym in INDEX_SYMBOLS:
        return {"error": "indices have no company report"}
    try:
        return eng.report(sym, fresh=fresh == "1")
    except (SourceError, DataNotAvailable) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
