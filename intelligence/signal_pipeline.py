"""
Full signal pipeline orchestrator.
Runs every 30 min during market hours for all NSE tickers.
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

log = logging.getLogger(__name__)

DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")


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


async def run_single_ticker(conn, ticker: str, portfolio_tickers: set) -> dict | None:
    """Full pipeline for one ticker. Returns signal dict or None if SKIP."""
    df = await fetch_ohlcv(conn, ticker)
    if df.empty or len(df) < 60:
        return None

    current_price = float(df["close"].iloc[-1])
    held = ticker in portfolio_tickers

    # Step 1: LightGBM
    try:
        ml_result = predict_with_reasoning(df, ticker)
    except Exception as e:
        log.warning(f"ML predict failed for {ticker}: {e}")
        return None

    # Step 2: Kronos
    try:
        kronos_result = kronos_forecast(df, steps=5)
    except Exception as e:
        log.warning(f"Kronos forecast failed for {ticker}: {e}")
        kronos_result = {"signal": ml_result["signal"], "confidence": ml_result["confidence"],
                         "reasoning": "Kronos unavailable — using ML signal only", "predicted_close": current_price}

    # Step 3: Combine
    combined = combine_signals(ml_result, kronos_result)

    # Reject weak signals early
    if combined["confidence"] < 0.55 and combined["signal"] != "HOLD":
        return None

    # Step 4: SLM enrichment + portfolio guard
    learnings = await fetch_recent_learnings(conn)
    slm_result = slm_enrich(
        ticker=ticker,
        price=current_price,
        combined_signal=combined,
        ml_reasoning=ml_result.get("reasoning", ""),
        kronos_reasoning=kronos_result.get("reasoning", ""),
        portfolio_held=held,
        learnings_context=learnings,
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
    Run full pipeline for all tickers.
    Yields individual signals as they complete (for SSE streaming).
    """
    conn = await asyncpg.connect(DB_DSN)
    portfolio_tickers = await get_portfolio_tickers(conn)

    semaphore = asyncio.Semaphore(10)

    async def bounded(ticker):
        async with semaphore:
            return await run_single_ticker(conn, ticker, portfolio_tickers)

    tasks = [bounded(t) for t in tickers]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            yield result

    await conn.close()


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
        signal["signal"],
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
