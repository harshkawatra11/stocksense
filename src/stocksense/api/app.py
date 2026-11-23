"""Read-only HTTP API and dashboard server (PLAN.md T12.3, PLAN-M M2)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stocksense import __version__
from stocksense.api import routes_status
from stocksense.api.readonly import ReadOnlyMiddleware
from stocksense.api.static import mount_web
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.marketlab.feeds import FeedManager
from stocksense.marketlab.store import LabStore

WEB_OUT = Path(__file__).resolve().parents[3] / "web" / "out"


def create_app(
    settings: Settings,
    *,
    start_feeds: bool = True,
    clock: Clock | None = None,
    store: LabStore | None = None,
    web_out: Path = WEB_OUT,
) -> FastAPI:
    """FastAPI app: GET-only JSON under /api, /health, and the static dashboard at /."""
    clock = clock or SystemClock()
    store = store or LabStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        mgr = FeedManager(settings, clock, store)
        if start_feeds:
            mgr.start()
        app.state.feeds = mgr
        yield
        mgr.stop()

    app = FastAPI(title="StockSense", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings, app.state.clock, app.state.store = settings, clock, store
    app.state.catchup_enabled = True
    app.add_middleware(ReadOnlyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(routes_status.router)
    # later cards add their own include_router line here, before the static mount
    mount_web(app, web_out)
    return app
