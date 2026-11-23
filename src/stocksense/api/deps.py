"""FastAPI dependencies that read shared objects from `app.state`."""

from __future__ import annotations

from fastapi import Request

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.marketlab.store import LabStore


def get_store(request: Request) -> LabStore:
    """The app's shared LabStore."""
    return request.app.state.store


def get_settings(request: Request) -> Settings:
    """The app's Settings."""
    return request.app.state.settings


def get_clock(request: Request) -> Clock:
    """The app's Clock (a fixed clock in tests)."""
    return request.app.state.clock
