import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from data.db.database import init_db
from backend.routers import signals, portfolio, logs, accuracy, market_data, ohlcv, market_overview, live, brain, providers, system_health, ws_prices, confirmations
from backend.services.quote_cache import start_quote_cache_feed
from intelligence.intraday_stops import start_intraday_stop_monitor
from data.pipeline.nse_ticker_loader import FALLBACK_TICKERS as NSE_TICKERS
from data.pipeline.upstox_client import resolve_instrument_keys
from intelligence.portfolio_guard import get_portfolio_tickers as get_held_tickers

# Index instrument keys aren't ISIN-based like equities — hardcode the two
# indices the frontend tracks (useMarketIndices.ts expects exactly these
# symbol strings). See docs/UPSTOX_API_NOTES.md §4 for the key format.
_INDEX_SYMBOL_MAP = {
    "NSE_INDEX|Nifty 50": "Nifty 50",
    "NSE_INDEX|Sensex": "Sensex",
}


async def _build_live_watchlist() -> tuple[list[str], dict[str, str]]:
    """
    Instrument keys to subscribe on the live feed: the two tracked indices
    plus every ticker currently held in the paper portfolio (the set that
    actually needs live P&L). Falls back to just the indices if DB/ticker
    resolution isn't ready yet — never blocks startup.
    """
    instrument_keys = list(_INDEX_SYMBOL_MAP.keys())
    symbol_map = dict(_INDEX_SYMBOL_MAP)
    try:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        try:
            held = await get_held_tickers(conn)
        finally:
            await conn.close()
        resolved = await resolve_instrument_keys(list(held))
        for ticker, key in resolved.items():
            instrument_keys.append(key)
            symbol_map[key] = ticker
    except Exception:
        log.exception("Could not resolve held-ticker instrument keys — live feed will cover indices only")
    return instrument_keys, symbol_map
from intelligence.signal_pipeline import (
    run_pipeline_batch,
    run_pipeline_batch_streaming,
    save_signal,
)
from intelligence.claude_cli import intraday_signal_check
from intelligence.portfolio_guard import get_portfolio_tickers
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Live quote cache feed: indices + currently-held tickers. Held-ticker
    # coverage is deliberately narrow at startup — watchlist/top-signal
    # tickers can be added to the live set later without changing this
    # wiring (quote_cache.update tolerates new symbols showing up anytime).
    instrument_keys, symbol_map = await _build_live_watchlist()
    asyncio.create_task(start_quote_cache_feed(instrument_keys, symbol_map=symbol_map))

    # Fast intraday stop/target monitor: polls quote_cache every few seconds
    # and exits breached positions immediately, independent of the 30-min
    # position_review scheduler cycle. See intelligence/intraday_stops.py.
    asyncio.create_task(start_intraday_stop_monitor())

    # Telegram approve/reject bot: long-polls the Bot API so the human can
    # decide each queued SANDBOX trade from their phone. Clean no-op (logs
    # once, returns) when TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are unset.
    from backend.services.telegram_bot import start_telegram_bot
    asyncio.create_task(start_telegram_bot())

    # Scheduler-death watchdog: alerts Telegram the moment job_runs goes
    # stale, instead of the mid-June incident where the scheduler was down
    # for weeks with nothing surfacing it. See intelligence/scheduler_watchdog.py.
    from intelligence.scheduler_watchdog import start_scheduler_watchdog
    asyncio.create_task(start_scheduler_watchdog())

    log.info("StockSense backend started")
    yield
    log.info("StockSense backend shutting down")


app = FastAPI(title="StockSense API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(logs.router, prefix="/api/logs", tags=["logs"])
app.include_router(accuracy.router, prefix="/api/accuracy", tags=["accuracy"])
app.include_router(market_data.router, prefix="/api/market", tags=["market"])
app.include_router(ohlcv.router, prefix="/api/ohlcv", tags=["ohlcv"])
app.include_router(market_overview.router, prefix="/api/market", tags=["market"])
app.include_router(live.router, prefix="/api/live", tags=["live"])
app.include_router(brain.router, prefix="/api/brain", tags=["brain"])
app.include_router(providers.router, prefix="/api/providers", tags=["providers"])
app.include_router(system_health.router, prefix="/api/system", tags=["system"])
app.include_router(ws_prices.router, prefix="/api/ws", tags=["ws"])
app.include_router(confirmations.router, prefix="/api/confirmations", tags=["confirmations"])

try:
    from backend.routers import upstox_auth
    app.include_router(upstox_auth.router, prefix="/api/upstox", tags=["upstox"])
except ImportError:
    log.info("backend.routers.upstox_auth not present yet — skipping mount (producer-side stub pending)")
    # TODO: once the parallel producer agent adds backend/routers/upstox_auth.py, this will auto-mount.


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "StockSense"}


@app.get("/api/stream/signals")
async def stream_signals():
    """
    Server-Sent Events endpoint.
    Yields per-stage events (ml_result, slm_result) as each model
    completes so the frontend terminals update in real time — not all at once.
    """

    async def generate() -> AsyncGenerator[str, None]:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        portfolio_tickers = await get_portfolio_tickers(conn)
        from intelligence.signal_pipeline import fetch_sector_map
        sector_map = await fetch_sector_map(conn)

        # Prefer the active tickers loaded in the DB; fall back to the static list.
        db_rows = await conn.fetch(
            "SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker LIMIT $1",
            settings.MAX_TICKERS_PER_RUN,
        )
        if db_rows:
            tickers = [r["ticker"] for r in db_rows]
        else:
            tickers = [t["ticker"] for t in NSE_TICKERS][: settings.MAX_TICKERS_PER_RUN]

        claude_candidates = []
        # track final slm_result events for DB save and Claude batching
        final_events: dict[str, dict] = {}

        try:
            for ticker in tickers:
                from intelligence.signal_pipeline import run_single_ticker_streaming

                last_slm_event = None
                async for event in run_single_ticker_streaming(conn, ticker, portfolio_tickers, sector_map):
                    yield f"data: {json.dumps(event)}\n\n"

                    if event.get("type") == "slm_result":
                        last_slm_event = event

                # Persist the final slm_result to DB
                if last_slm_event:
                    try:
                        signal_for_db = {
                            "ticker": last_slm_event["ticker"],
                            "price": last_slm_event.get("price", 0),
                            "signal": last_slm_event.get("signal", "HOLD"),
                            "confidence": last_slm_event.get("confidence", 0.5),
                            "stop_loss": last_slm_event.get("stop_loss"),
                            "target": last_slm_event.get("target"),
                            "ml_confidence": last_slm_event.get("ml_confidence"),
                            "slm_confidence": last_slm_event.get("slm_confidence"),
                            "ml_reasoning": last_slm_event.get("ml_reasoning", ""),
                            "slm_reasoning": last_slm_event.get("slm_reasoning", ""),
                            "combined_reasoning": last_slm_event.get("combined_reasoning", ""),
                        }
                        signal_id = await save_signal(conn, signal_for_db)
                        last_slm_event["id"] = signal_id
                    except Exception as e:
                        log.error(f"DB save error for {ticker}: {e}")

                    if last_slm_event.get("pass_to_claude"):
                        claude_candidates.append(last_slm_event)

            # Claude batch after all tickers processed
            if claude_candidates:
                yield f"data: {json.dumps({'type': 'claude_checking', 'count': len(claude_candidates)})}\n\n"
                enriched = intraday_signal_check(claude_candidates)
                for sig in enriched:
                    yield f"data: {json.dumps({**sig, 'type': 'claude_enriched'})}\n\n"

            yield f"data: {json.dumps({'type': 'batch_complete', 'total': len(tickers)})}\n\n"

        finally:
            await conn.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stream/activity")
async def stream_activity():
    """
    SSE feed of new activity_log events (fills, stops, learnings, param changes)
    as they happen — pushed via backend/services/activity_bus by log_activity.
    Frontend keeps a slow poll of /api/live/activity as fallback for events
    logged from other processes.
    """
    from backend.services import activity_bus

    async def generate() -> AsyncGenerator[str, None]:
        q = activity_bus.subscribe()
        try:
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps({'type': 'activity', **event}, default=str)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat comment keeps proxies/browsers from timing out.
                    yield ": keepalive\n\n"
        finally:
            activity_bus.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----- serve the built frontend (single-process local deployment) ----- #
# Must stay at the very end: the catch-all would shadow any route defined after it.
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = os.path.join(_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_DIST, "index.html"))
