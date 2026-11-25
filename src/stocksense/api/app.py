"""Read-only HTTP API and dashboard server (PLAN.md T12.3, PLAN-M M2)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stocksense import __version__
from stocksense.api import (
    routes_flow,
    routes_lab,
    routes_orderflow,
    routes_picks,
    routes_research,
    routes_sector,
    routes_sm,
    routes_status,
)
from stocksense.api.readonly import ReadOnlyMiddleware
from stocksense.api.static import mount_web
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.flow.engine import FlowEngine, build_flow_engine
from stocksense.flow.oi import build_futures_oi_live
from stocksense.marketlab.feeds import CatchUp, FeedManager, build_tape
from stocksense.marketlab.research import ResearchEngine, build_research_engine
from stocksense.marketlab.sectors import SectorEngine, build_sector_engine
from stocksense.marketlab.sources import Sources
from stocksense.marketlab.store import LabStore
from stocksense.marketlab.tape import TapeEngine
from stocksense.orderflow.engine import OrderflowEngine, build_orderflow_engine
from stocksense.orderflow.heatmap import build_depth_capture

WEB_OUT = Path(__file__).resolve().parents[3] / "web" / "out"


def create_app(
    settings: Settings,
    *,
    start_feeds: bool = True,
    clock: Clock | None = None,
    store: LabStore | None = None,
    web_out: Path = WEB_OUT,
    tape: TapeEngine | None = None,
    sector: SectorEngine | None = None,
    research: ResearchEngine | None = None,
    flow: FlowEngine | None = None,
    orderflow: OrderflowEngine | None = None,
) -> FastAPI:
    """FastAPI app: GET-only JSON under /api, /health, and the static dashboard at /."""
    clock = clock or SystemClock()
    store = store or LabStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        mgr = FeedManager(settings, clock, store)
        if start_feeds:
            if app.state.tape is None:
                app.state.tape = build_tape(settings, clock, store)
            if app.state.tape is not None:
                mgr.add("tape", app.state.tape.step, 4.0)
            if app.state.sector is None:
                app.state.sector = build_sector_engine(settings, clock, store)
            eng = app.state.sector
            if eng is not None:
                eng.stop = mgr.stop_event
                mgr.add("sector", eng.step, 5.0)
                mgr.add("sector_slow", eng.slow_step, 3600.0)
                mgr.add("news", eng.news_step, 600.0)
            if app.state.research is None:
                app.state.research = build_research_engine(
                    settings, clock, store, Sources(settings, clock, store)
                )
            if app.state.flow is None:
                app.state.flow = build_flow_engine(settings, clock, store, app.state.tape)
            if app.state.flow is not None:
                mgr.add("flow", app.state.flow.step, 30.0)
            if app.state.orderflow is None:
                app.state.orderflow = build_orderflow_engine(settings, clock, store)
            if app.state.oi_live is None:
                app.state.oi_live = build_futures_oi_live(settings, clock, store)
            if app.state.oi_live is not None:
                mgr.add("oi_live", app.state.oi_live.step, 60.0)
            else:
                store.set_interval("oi_live", 0.0)
                store.feeds["oi_live"].last_err = "skipped: no angel secrets"
            if app.state.depth is None:
                app.state.depth = build_depth_capture(settings, clock, store)
            if app.state.depth is not None:
                mgr.add("depth", app.state.depth.step, 60.0)
            mgr.start()
            if app.state.catchup_enabled:
                CatchUp(settings, clock, store).start(mgr.stop_event)
        app.state.feeds = mgr
        yield
        mgr.stop()

    app = FastAPI(title="StockSense", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings, app.state.clock, app.state.store = settings, clock, store
    app.state.catchup_enabled = True
    app.state.tape = tape
    app.state.sector = sector
    app.state.research = research
    app.state.flow = flow
    app.state.depth = None
    app.state.oi_live = None
    app.state.orderflow = orderflow
    app.add_middleware(ReadOnlyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(routes_status.router)
    app.include_router(routes_sm.router)
    app.include_router(routes_sector.router)
    app.include_router(routes_research.router)
    app.include_router(routes_flow.router)
    app.include_router(routes_lab.router)
    app.include_router(routes_orderflow.router)
    app.include_router(routes_picks.router)
    # later cards add their own include_router line here, before the static mount
    mount_web(app, web_out)
    return app
