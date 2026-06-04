"""
Full signal pipeline orchestrator.
Runs every 30 min during market hours for all NSE tickers.
Supports both batch (non-streaming) and per-stage SSE streaming modes.
"""
import asyncio
import asyncpg
import pandas as pd
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

from data.pipeline.feature_engineering import compute_features
from models.ml.predict import predict_with_reasoning
from models.kronos.integration import forecast as kronos_forecast
from models.kronos.combine import combine_signals
from models.slm.infer import slm_enrich
from intelligence.portfolio_guard import get_portfolio_tickers, is_held
from config import settings

log = logging.getLogger(__name__)


async def fetch_ohlcv(conn, ticker: str, limit: int = 300) -> pd.DataFrame:
    rows = await conn.fetch(
        """
        SELECT time, open, high, low, close, volume
        FROM ohlcv_daily
        WHERE ticker = $1 AND close IS NOT NULL
        ORDER BY time DESC LIMIT $2
        """,
        ticker, limit,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """14-day Average True Range in absolute price terms. None if insufficient data."""
    if len(df) < period + 1:
        return None
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


async def fetch_sector_map(conn) -> dict[str, str]:
    """ticker -> sector from the stocks DB table. Loaded once per run."""
    rows = await conn.fetch("SELECT ticker, sector FROM stocks WHERE sector IS NOT NULL")
    return {r["ticker"]: r["sector"] for r in rows}


async def fetch_recent_learnings(conn, limit: int = 20) -> str:
    rows = await conn.fetch(
        """
        SELECT title, body FROM learnings
        ORDER BY created_at DESC LIMIT $1
        """,
        limit,
    )
    if not rows:
        return ""
    return "\n".join(f"- {r['title']}: {r['body'][:200]}" for r in rows)


async def run_single_ticker_streaming(
    conn, ticker: str, portfolio_tickers: set, sector_map: dict[str, str] | None = None
) -> AsyncGenerator[dict, None]:
    """
    Async generator. Yields one dict per model stage as it completes.
    Frontend can render each stage immediately via SSE.
    """
    df = await fetch_ohlcv(conn, ticker)
    if df.empty or len(df) < 60:
        return

    current_price = float(df["close"].iloc[-1])
    held = ticker in portfolio_tickers
    sector = (sector_map or {}).get(ticker)

    # Stage 1: ML LightGBM
    try:
        ml_result = predict_with_reasoning(df, ticker, sector=sector)
        yield {
            "type": "ml_result",
            "ticker": ticker,
            "signal": ml_result.get("signal", "HOLD"),
            "confidence": ml_result.get("confidence", 0.5),
            "ml_reasoning": ml_result.get("reasoning", ""),
            "price": current_price,
        }
    except Exception as e:
        log.warning(f"ML predict failed for {ticker}: {e}")
        ml_result = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reasoning": f"ML unavailable: {e}",
        }
        yield {
            "type": "ml_result",
            "ticker": ticker,
            "signal": "HOLD",
            "confidence": 0.5,
            "ml_reasoning": f"ML error: {e}",
            "price": current_price,
        }

    # Stage 2: Kronos time-series foundation model
    try:
        kronos_result = kronos_forecast(df, steps=5)
        yield {
            "type": "kronos_result",
            "ticker": ticker,
            "signal": kronos_result.get("signal", "HOLD"),
            "confidence": kronos_result.get("confidence", 0.5),
            "kronos_reasoning": kronos_result.get("reasoning", ""),
        }
    except Exception as e:
        log.warning(f"Kronos forecast failed for {ticker}: {e}")
        kronos_result = {
            "signal": ml_result.get("signal", "HOLD"),
            "confidence": ml_result.get("confidence", 0.5),
            "reasoning": "Kronos unavailable — using ML signal only",
            "predicted_close": current_price,
        }
        yield {
            "type": "kronos_result",
            "ticker": ticker,
            "signal": kronos_result["signal"],
            "confidence": kronos_result["confidence"],
            "kronos_reasoning": kronos_result["reasoning"],
        }

    # Stage 3: Combine signals + SLM enrichment
    combined = combine_signals(ml_result, kronos_result)

    # Skip weak signals early
    if combined["confidence"] < 0.55 and combined["signal"] != "HOLD":
        return

    learnings = await fetch_recent_learnings(conn)
    atr = compute_atr(df)
    try:
        slm_result = slm_enrich(
            ticker=ticker,
            price=current_price,
            combined_signal=combined,
            ml_reasoning=ml_result.get("reasoning", ""),
            kronos_reasoning=kronos_result.get("reasoning", ""),
            portfolio_held=held,
            learnings_context=learnings,
            atr=atr,
        )
    except Exception as e:
        log.warning(f"SLM enrich failed for {ticker}: {e}")
        slm_result = {
            "signal": combined.get("signal", "HOLD"),
            "confidence": combined.get("confidence", 0.5),
            "reasoning": f"SLM unavailable: {e}",
            "stop_loss": current_price * 0.97,
            "target": current_price * 1.03,
            "pass_to_claude": False,
        }

    if slm_result.get("signal") == "SKIP":
        return

    yield {
        "type": "slm_result",
        "ticker": ticker,
        "signal": slm_result.get("signal", combined["signal"]),
        "confidence": slm_result.get("confidence", combined["confidence"]),
        "slm_reasoning": slm_result.get("reasoning", ""),
        "stop_loss": slm_result.get("stop_loss", current_price * 0.97),
        "target": slm_result.get("target", current_price * 1.03),
        "pass_to_claude": slm_result.get("pass_to_claude", False),
        "price": current_price,
        "ml_confidence": ml_result.get("confidence"),
        "kronos_confidence": kronos_result.get("confidence"),
        "slm_confidence": slm_result.get("confidence"),
        "ml_reasoning": ml_result.get("reasoning", ""),
        "kronos_reasoning": kronos_result.get("reasoning", ""),
        "combined_reasoning": combined.get("combined_reasoning", ""),
        "fired_at": datetime.now(timezone.utc).isoformat(),
    }


async def run_single_ticker(
    conn, ticker: str, portfolio_tickers: set, sector_map: dict[str, str] | None = None
) -> dict | None:
    """Full pipeline for one ticker (non-streaming). Returns final signal dict or None."""
    df = await fetch_ohlcv(conn, ticker)
    if df.empty or len(df) < 60:
        return None

    current_price = float(df["close"].iloc[-1])
    held = ticker in portfolio_tickers
    sector = (sector_map or {}).get(ticker)

    # Step 1: LightGBM
    try:
        ml_result = predict_with_reasoning(df, ticker, sector=sector)
    except Exception as e:
        log.warning(f"ML predict failed for {ticker}: {e}")
        return None

    # Step 2: Kronos
    try:
        kronos_result = kronos_forecast(df, steps=5)
    except Exception as e:
        log.warning(f"Kronos forecast failed for {ticker}: {e}")
        kronos_result = {
            "signal": ml_result["signal"],
            "confidence": ml_result["confidence"],
            "reasoning": "Kronos unavailable — using ML signal only",
            "predicted_close": current_price,
        }

    # Step 3: Combine
    combined = combine_signals(ml_result, kronos_result)

    # Reject weak signals early
    if combined["confidence"] < 0.55 and combined["signal"] != "HOLD":
        return None

    # Step 4: SLM enrichment + portfolio guard
    learnings = await fetch_recent_learnings(conn)
    atr = compute_atr(df)
    slm_result = slm_enrich(
        ticker=ticker,
        price=current_price,
        combined_signal=combined,
        ml_reasoning=ml_result.get("reasoning", ""),
        kronos_reasoning=kronos_result.get("reasoning", ""),
        portfolio_held=held,
        learnings_context=learnings,
        atr=atr,
    )

    if slm_result.get("signal") == "SKIP":
        return None

    signal_out = {
        "ticker": ticker,
        "price": current_price,
        "signal": slm_result.get("signal", combined["signal"]),
        "confidence": slm_result.get("confidence", combined["confidence"]),
        "stop_loss": slm_result.get("stop_loss", current_price * 0.97),
        "target": slm_result.get("target", current_price * 1.03),
        "ml_confidence": ml_result.get("confidence"),
        "kronos_confidence": kronos_result.get("confidence"),
        "slm_confidence": slm_result.get("confidence"),
        "ml_reasoning": ml_result.get("reasoning", ""),
        "kronos_reasoning": kronos_result.get("reasoning", ""),
        "slm_reasoning": slm_result.get("reasoning", ""),
        "pass_to_claude": slm_result.get("pass_to_claude", False),
        "combined_reasoning": combined.get("combined_reasoning", ""),
        "fired_at": datetime.now(timezone.utc).isoformat(),
    }

    return signal_out


async def run_pipeline_batch(tickers: list[str]) -> AsyncGenerator[dict, None]:
    """
    Run full pipeline for all tickers (non-streaming, batch mode).
    Used by scheduler. Yields individual final signals as they complete.
    Uses a connection pool so concurrent tasks each get their own connection
    (a single shared asyncpg connection raises InterfaceError under concurrency).
    """
    pool = await asyncpg.create_pool(settings.DATABASE_DSN, min_size=2, max_size=12)

    # Load shared context once using a dedicated pool connection
    async with pool.acquire() as meta_conn:
        portfolio_tickers = await get_portfolio_tickers(meta_conn)
        sector_map = await fetch_sector_map(meta_conn)

    semaphore = asyncio.Semaphore(10)

    async def bounded(ticker):
        async with semaphore:
            async with pool.acquire() as conn:
                return await run_single_ticker(conn, ticker, portfolio_tickers, sector_map)

    tasks = [bounded(t) for t in tickers]
    try:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                yield result
    finally:
        await pool.close()


async def run_pipeline_batch_streaming(
    tickers: list[str],
) -> AsyncGenerator[dict, None]:
    """
    SSE-compatible streaming pipeline.
    Yields per-stage events for each ticker in sequence.
    Use this from the /api/stream/signals endpoint.
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    portfolio_tickers = await get_portfolio_tickers(conn)
    sector_map = await fetch_sector_map(conn)

    try:
        for ticker in tickers:
            async for event in run_single_ticker_streaming(conn, ticker, portfolio_tickers, sector_map):
                yield event
    finally:
        await conn.close()


async def run_pipeline():
    """
    Scheduler entry point. Runs full pipeline batch and saves all signals to DB.
    Called by market_runner.py every 30 min during market hours.
    """
    save_conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        # Load active tickers from DB; fall back to static list
        rows = await save_conn.fetch(
            "SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker LIMIT $1",
            settings.MAX_TICKERS_PER_RUN,
        )
        if rows:
            tickers = [r["ticker"] for r in rows]
        else:
            from data.pipeline.nse_ticker_loader import FALLBACK_TICKERS
            tickers = [t["ticker"] for t in FALLBACK_TICKERS][:settings.MAX_TICKERS_PER_RUN]

        async for signal in run_pipeline_batch(tickers):
            try:
                await save_signal(save_conn, signal)
                log.info(f"Saved signal for {signal['ticker']} ({signal.get('signal')})")
            except Exception as e:
                log.warning(f"Failed to save signal for {signal.get('ticker')}: {e}")
    finally:
        await save_conn.close()


async def save_signal(conn, signal: dict) -> int:
    """Persist signal + reasoning to DB. Returns signal ID."""
    signal_id = await conn.fetchval(
        """
        INSERT INTO signals (
            ticker, signal_type, timeframe, price_at_signal,
            target_price, stop_loss,
            ml_confidence, kronos_confidence, slm_confidence,
            final_confidence, fired_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        RETURNING id
        """,
        signal["ticker"],
        signal.get("signal", "HOLD"),
        "intraday",
        signal["price"],
        signal.get("target"),
        signal.get("stop_loss"),
        signal.get("ml_confidence"),
        signal.get("kronos_confidence"),
        signal.get("slm_confidence"),
        signal.get("confidence"),
        datetime.now(timezone.utc),
    )

    for model_name, reasoning_key in [
        ("lgbm", "ml_reasoning"),
        ("kronos", "kronos_reasoning"),
        ("slm", "slm_reasoning"),
        ("combined", "combined_reasoning"),
    ]:
        reasoning = signal.get(reasoning_key, "")
        if reasoning:
            await conn.execute(
                """
                INSERT INTO signal_reasoning (signal_id, model_name, reasoning)
                VALUES ($1, $2, $3)
                """,
                signal_id, model_name, reasoning,
            )

    return signal_id
