"""
End-of-day review job. Runs at 15:45 IST.
Fetches all today's signals, compares to actual closes, calls Claude Opus, logs learnings.
"""
import asyncpg
import asyncio
import yfinance as yf
import logging
import os
from datetime import date, datetime, timezone

from intelligence.claude_cli import eod_review_call
from data.pipeline.fetch_historical import ticker_to_yf

log = logging.getLogger(__name__)
DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")


async def fetch_todays_signals(conn) -> list[dict]:
    today = date.today()
    rows = await conn.fetch(
        """
        SELECT s.id, s.ticker, s.signal_type, s.price_at_signal,
               s.target_price, s.stop_loss, s.final_confidence,
               s.fired_at,
               ml.reasoning as ml_reasoning,
               kr.reasoning as kronos_reasoning,
               sl.reasoning as slm_reasoning,
               cl.reasoning as claude_reasoning
        FROM signals s
        LEFT JOIN signal_reasoning ml ON ml.signal_id = s.id AND ml.model_name = 'lgbm'
        LEFT JOIN signal_reasoning kr ON kr.signal_id = s.id AND kr.model_name = 'kronos'
        LEFT JOIN signal_reasoning sl ON sl.signal_id = s.id AND sl.model_name = 'slm'
        LEFT JOIN signal_reasoning cl ON cl.signal_id = s.id AND cl.model_name = 'claude'
        WHERE s.fired_at::date = $1
        """,
        today,
    )
    return [dict(r) for r in rows]


async def fetch_actual_closes(tickers: list[str]) -> dict[str, float]:
    closes = {}
    for ticker in set(tickers):
        try:
            df = yf.download(ticker_to_yf(ticker), period="2d", auto_adjust=True, progress=False)
            if not df.empty:
                closes[ticker] = float(df["Close"].iloc[-1])
        except Exception as e:
            log.warning(f"Could not fetch actual close for {ticker}: {e}")
    return closes


async def update_signal_actuals(conn, signal_id: int, actual_close: float, status: str):
    await conn.execute(
        """
        UPDATE signals SET actual_close = $1, status = $2, resolved_at = NOW()
        WHERE id = $3
        """,
        actual_close, status, signal_id,
    )


async def save_learnings(conn, review_result: dict, today: date):
    for learning in review_result.get("learnings", []):
        await conn.execute(
            """
            INSERT INTO learnings (learning_date, learning_type, ticker, title, body, tags, raw_claude_output)
            VALUES ($1, 'eod_review', $2, $3, $4, $5, $6)
            """,
            today,
            learning.get("ticker"),
            learning.get("title", "")[:300],
            learning.get("body", ""),
            learning.get("tags", []),
            review_result.get("raw", ""),
        )
    log.info(f"Saved {len(review_result.get('learnings', []))} learnings for {today}")


async def run_eod_review():
    conn = await asyncpg.connect(DB_DSN)
    today = date.today()
    log.info(f"Starting EOD review for {today}")

    signals = await fetch_todays_signals(conn)
    if not signals:
        log.info("No signals today — skipping EOD review")
        await conn.close()
        return

    log.info(f"Reviewing {len(signals)} signals from today")

    tickers = [s["ticker"] for s in signals]
    actual_closes = await fetch_actual_closes(tickers)

    predictions = []
    for s in signals:
        actual = actual_closes.get(s["ticker"])
        if actual is None:
            continue

        predicted_close = float(s.get("target_price") or s.get("price_at_signal", 0))
        signal_type = s.get("signal_type", "BUY")

        # Determine outcome
        if signal_type == "BUY":
            correct = actual > float(s["price_at_signal"])
        elif signal_type == "SELL":
            correct = actual < float(s["price_at_signal"])
        else:
            correct = True

        status = "hit_target" if correct else "expired"
        await update_signal_actuals(conn, s["id"], actual, status)

        predictions.append({
            "ticker": s["ticker"],
            "signal": signal_type,
            "predicted_close": predicted_close,
            "actual_close": actual,
            "ml_reasoning": s.get("ml_reasoning", ""),
            "kronos_reasoning": s.get("kronos_reasoning", ""),
            "slm_reasoning": s.get("slm_reasoning", ""),
            "claude_reasoning": s.get("claude_reasoning", ""),
        })

    log.info(f"Calling Claude Opus for EOD review of {len(predictions)} predictions...")
    review_result = eod_review_call(predictions)

    await save_learnings(conn, review_result, today)

    accuracy = review_result.get("accuracy_today", {})
    log.info(
        f"EOD Summary: {accuracy.get('correct', 0)}/{accuracy.get('total', 0)} correct "
        f"({accuracy.get('pct', 0)*100:.1f}%)"
    )
    log.info(f"Summary: {review_result.get('summary', '')[:300]}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(run_eod_review())
