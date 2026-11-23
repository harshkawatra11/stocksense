"""Health and status routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from stocksense import __version__
from stocksense.api.deps import get_clock, get_store
from stocksense.core.clock import Clock
from stocksense.marketlab.store import LabStore

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"ok": True, "version": __version__}


@router.get("/api/status")
def status(
    store: Annotated[LabStore, Depends(get_store)], clock: Annotated[Clock, Depends(get_clock)]
) -> dict[str, Any]:
    """Feed health, catch-up progress, server time and market status (market is 'closed' until M2.1)."""
    out = store.snapshot_status()
    out["time"] = clock.now().isoformat()
    out["market"] = "closed"
    return out
