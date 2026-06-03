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
from backend.routers import signals, portfolio, logs, accuracy
from data.pipeline.nse_ticker_loader import FALLBACK_TICKERS as NSE_TICKERS
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "StockSense"}


@app.get("/api/stream/signals")
async def stream_signals():
    """
    Server-Sent Events endpoint.
    Yields per-stage events (ml_result, kronos_result, slm_result) as each model
    completes so the frontend terminals update in real time — not all at once.
    """

    async def generate() -> AsyncGenerator[str, None]:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        portfolio_tickers = await get_portfolio_tickers(conn)
        tickers = NSE_TICKERS[: settings.MAX_TICKERS_PER_RUN]

        claude_candidates = []
        # track final slm_result events for DB save and Claude batching
        final_events: dict[str, dict] = {}

        try:
            for ticker in tickers:
                from intelligence.signal_pipeline import run_single_ticker_streaming

                last_slm_event = None
                async for event in run_single_ticker_streaming(conn, ticker, portfolio_tickers):
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
                            "kronos_confidence": last_slm_event.get("kronos_confidence"),
                            "slm_confidence": last_slm_event.get("slm_confidence"),
                            "ml_reasoning": last_slm_event.get("ml_reasoning", ""),
                            "kronos_reasoning": last_slm_event.get("kronos_reasoning", ""),
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
