"""FastAPI dependencies that read shared objects from `app.state`."""

from __future__ import annotations

from fastapi import HTTPException, Request

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.marketlab.sectors import SectorEngine
from stocksense.marketlab.store import LabStore
from stocksense.marketlab.tape import TapeEngine


def get_store(request: Request) -> LabStore:
    """The app's shared LabStore."""
    return request.app.state.store


def get_settings(request: Request) -> Settings:
    """The app's Settings."""
    return request.app.state.settings


def get_clock(request: Request) -> Clock:
    """The app's Clock (a fixed clock in tests)."""
    return request.app.state.clock


def get_tape(request: Request) -> TapeEngine:
    """The Smart Money tape engine; 503 when the server runs without feeds."""
    eng = request.app.state.tape
    if eng is None:
        raise HTTPException(status_code=503, detail="tape not running")
    return eng


def get_sector(request: Request) -> SectorEngine:
    """The Sector Map engine; 503 when the server runs without feeds."""
    eng = request.app.state.sector
    if eng is None:
        raise HTTPException(status_code=503, detail="sector map not running")
    return eng
