"""Daily Picks routes (PLAN-M M5.9): `/api/picks` and `/api/picks/history`.

Read-only and paper-only. The response carries paper quantities and rupee notionals, never the broker's capital,
funds, client codes or tokens. When no picks file exists for the date, a preview is computed without capital and
is neither stored nor journalled (so it can never block the real, sized picks of the evening).
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from stocksense.api.deps import get_clock, get_settings
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.state import State
from stocksense.strategies import picks as picks_mod

router = APIRouter(prefix="/api/picks")


def _bad(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


@router.get("", response_model=None)
def picks(
    settings: Annotated[Settings, Depends(get_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
    asof: str = "",
) -> dict[str, Any] | JSONResponse:
    """The Daily Picks of `asof` (default: the latest scan date): gated, exploratory, the note and the counts."""
    try:
        day = date.fromisoformat(asof) if asof else picks_mod.latest_scan_asof(settings)
    except ValueError:
        return _bad("asof must be YYYY-MM-DD")
    if day is None:
        return {"error": "run: stocksense lab scan"}
    path = picks_mod.picks_path(settings, day)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        return picks_mod.build_picks(settings, State(settings, clock), day, None, clock=clock, persist=False)
    except DataNotAvailable as e:
        return _bad(str(e), 404)
    except sqlite3.OperationalError:
        return _bad("the state database is not initialised: run stocksense status", 503)


@router.get("/history")
def history(
    settings: Annotated[Settings, Depends(get_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
    days: int = 30,
) -> dict[str, Any]:
    """Past picks of the last `days` days with their realised outcome from the paper journal."""
    return {"days": days, "rows": picks_mod.history(settings, clock, days), "paper": True}
