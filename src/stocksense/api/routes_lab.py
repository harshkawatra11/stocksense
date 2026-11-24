"""Lab routes on the exploratory engine (PLAN-M M2.6): trades, swing summary, matrix, journal, strategies.

Every number here is exploratory (`exploratory: true`): only the gated trials of M5.7 are evidence.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from stocksense.api.deps import get_clock, get_settings
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.marketlab.research import INDEX_SYMBOLS
from stocksense.strategies import explore, journal, scan
from stocksense.strategies.base import load_library

router = APIRouter(prefix="/api/lab")
TFS = {"1d", "5m", "60m"}


def _bad(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


@router.get("/trades", response_model=None)
def trades(
    settings: Annotated[Settings, Depends(get_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
    sym: str = "NIFTY",
    tf: str = "1d",
    strat: str = "",
    exit: str = "",  # noqa: A002 (the query parameter is named `exit`)
    trend: str = "",
) -> dict[str, Any] | JSONResponse:
    """Exploratory trades of one symbol for the chart (Market Lab's /api/lab/trades shape, plus flags)."""
    sym = sym.upper()
    if tf not in TFS:
        return _bad("unknown timeframe")
    if sym in INDEX_SYMBOLS and tf == "1d":
        return {
            "trades": [],
            "note": "Swing rules are for stocks; pick 5m or 60m for the index rules.",
            "exploratory": True,
        }
    try:
        return explore.trades_for(sym, strat or None, exit or None, settings, clock, tf, trend=trend or None)
    except KeyError:
        return _bad("unknown strategy")
    except ValueError as e:
        return _bad("unknown exit" if "exit" in str(e) else str(e))
    except DataNotAvailable as e:
        return _bad(str(e), 503)


@router.get("/swing")
def swing(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """The swing system's research summary and radar (Market Lab's /api/lab/swing key tree)."""
    return scan.summary(settings)


@router.get("/summary")
def summary(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """The intraday library: {list: [names], strategies: {name: {...}}} with the latest gate verdicts."""
    verdicts = explore.gated_verdicts(settings)
    specs = [s for s in load_library().values() if s.timeframe in ("5m", "60m")]
    return {
        "list": [s.name for s in specs],
        "strategies": {
            s.name: {
                "key": s.key,
                "tier": s.tier,
                "source": s.source,
                "kind": s.kind,
                "direction": s.direction,
                "exits": list(s.exits),
                "default_exit": s.default_exit,
                "gated": verdicts.get(s.key),
            }
            for s in specs
        },
        "exploratory": True,
    }


@router.get("/journal")
def lab_journal(
    settings: Annotated[Settings, Depends(get_settings)], clock: Annotated[Clock, Depends(get_clock)]
) -> dict[str, Any]:
    """The append-only paper journal of the last 30 days."""
    return journal.read(settings, days=30, clock=clock)


@router.get("/matrix", response_model=None)
def matrix(
    settings: Annotated[Settings, Depends(get_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
    sym: str = "",
    tf: str = "1d",
) -> dict[str, Any] | JSONResponse:
    """Every strategy x every allowed exit on one stock, with the expected best-of-N Sharpe under no skill."""
    if not sym or tf not in TFS:
        return _bad("sym and a known tf are required")
    try:
        return explore.matrix(sym.upper(), settings, clock, tf)
    except DataNotAvailable as e:
        return _bad(str(e), 503)


@router.get("/strategies")
def strategies(settings: Annotated[Settings, Depends(get_settings)]) -> list[dict[str, Any]]:
    """The strategy registry with the latest gate verdict of each."""
    verdicts = explore.gated_verdicts(settings)
    return [
        {
            "key": s.key,
            "name": s.name,
            "source": s.source,
            "tier": s.tier,
            "timeframe": s.timeframe,
            "kind": s.kind,
            "direction": s.direction,
            "exits": list(s.exits),
            "description": s.description,
            "reference": s.reference,
            "gated": verdicts.get(s.key),
        }
        for s in load_library().values()
    ]
