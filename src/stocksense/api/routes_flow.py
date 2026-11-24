"""Money-flow routes (`/api/flow/*`): descriptive positioning and flow, never a forecast. GET only."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from stocksense.api.deps import get_flow
from stocksense.flow.engine import STOCK_SORTS, FlowEngine

router = APIRouter(prefix="/api/flow")
Flow = Annotated[FlowEngine, Depends(get_flow)]


@router.get("/market")
def market(flow: Flow) -> dict[str, Any]:
    """Participants, index option positioning, VIX, breadth and the pre-open summary."""
    return flow.market()


@router.get("/stocks")
def stocks(flow: Flow, sort: str = "flow_score") -> dict[str, Any]:
    """F&O stocks with their flow card, ordered by flow_score, disp, rvol or oi_chg_pct."""
    if sort not in STOCK_SORTS:
        raise HTTPException(status_code=400, detail="unknown sort")
    return flow.stocks(sort)


@router.get("/oi")
def oi(flow: Flow, scope: Literal["stocks", "index"] = "stocks") -> dict[str, Any]:
    """Futures OI buildup rows of the latest session."""
    return flow.oi(scope)


@router.get("/options")
def options(
    flow: Flow,
    und: str = "NIFTY",
    expiry: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
) -> dict[str, Any]:
    """Strikes, totals and history of one underlying's option chain."""
    out = flow.options(und.upper(), expiry)
    if out is None:
        raise HTTPException(status_code=404, detail="no option data for this underlying")
    return out


@router.get("/sectors")
def sectors(
    flow: Flow, rng: Annotated[Literal["today", "w1", "m1"], Query(alias="range")] = "today"
) -> dict[str, Any]:
    """Sector money flow for today (live) or the last week or month (EOD)."""
    return flow.sectors(rng)


@router.get("/preopen")
def preopen(flow: Flow) -> dict[str, Any]:
    """The pre-open board with the 09:30 follow-through once the market is open."""
    return flow.preopen.payload()


@router.get("/deals")
def deals(flow: Flow, days: Annotated[int, Query(ge=1, le=60)] = 5) -> dict[str, Any]:
    """Bulk and block deal summaries of the last `days` sessions."""
    return flow.deals(days)


@router.get("/participants")
def participants(flow: Flow, days: Annotated[int, Query(ge=1, le=500)] = 60) -> dict[str, Any]:
    """Participant positioning and FII/DII cash flow over the last `days` sessions."""
    return flow.participants(days)
