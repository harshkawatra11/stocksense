import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from data.db.database import init_db, get_session
from backend.routers import signals, portfolio, logs, accuracy
from data.pipeline.nse_ticker_loader import FALLBACK_TICKERS as NSE_TICKERS
from intelligence.signal_pipeline import run_pipeline_batch, save_signal
from intelligence.claude_cli import intraday_signal_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = __import__("os").getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")


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
    Frontend connects here to receive live signal stream with full reasoning.
    """

    async def generate() -> AsyncGenerator[str, None]:
        tickers = NSE_TICKERS[:50]  # Start with top 50, expand gradually
        batch_signals = []

        async for signal in run_pipeline_batch(tickers):
            # Save to DB
            try:
                conn = await asyncpg.connect(DB_DSN)
                signal["id"] = await save_signal(conn, signal)
                await conn.close()
            except Exception as e:
                log.error(f"DB save error: {e}")

            batch_signals.append(signal)
            # Stream each signal immediately
            yield f"data: {json.dumps(signal)}\n\n"

        # After all processed, run Claude batch check on top signals
        claude_candidates = [s for s in batch_signals if s.get("pass_to_claude")]
        if claude_candidates:
            yield f"data: {json.dumps({'type': 'claude_checking', 'count': len(claude_candidates)})}\n\n"
            enriched = intraday_signal_check(claude_candidates)
            for signal in enriched:
                yield f"data: {json.dumps({**signal, 'type': 'claude_enriched'})}\n\n"

        yield f"data: {json.dumps({'type': 'batch_complete', 'total': len(batch_signals)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
