"""
End-of-day review job. Runs at 15:45 IST.
Fetches all today's signals, compares to actual closes, calls Claude Opus, logs learnings.
"""
import asyncpg
import asyncio
import hashlib
import logging
from datetime import date, datetime, timezone

from intelligence.claude_cli import eod_review_call
from config import settings

log = logging.getLogger(__name__)


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


async def fetch_actual_closes(conn, tickers: list[str]) -> dict[str, float]:
    """
    Actual closes from the NSE-fed ohlcv_daily table (NSE path only — no yfinance).
    Uses the most recent close on/before today per ticker. Tickers whose Bhavcopy
    for the day hasn't landed yet are simply omitted and skipped downstream.
    """
    if not tickers:
        return {}
    rows = await conn.fetch(
        """
        SELECT DISTINCT ON (ticker) ticker, close
        FROM ohlcv_daily
        WHERE ticker = ANY($1::text[]) AND close IS NOT NULL
        ORDER BY ticker, time DESC
        """,
        list(set(tickers)),
    )
    return {r["ticker"]: float(r["close"]) for r in rows}


async def update_signal_actuals(conn, signal_id: int, actual_close: float, status: str):
    await conn.execute(
        """
        UPDATE signals SET actual_close = $1, status = $2, resolved_at = NOW()
        WHERE id = $3
        """,
        actual_close, status, signal_id,
    )


def _learning_hash(title: str, body: str) -> str:
    return hashlib.sha256(f"{title.strip()}|{body.strip()}".encode("utf-8")).hexdigest()


async def save_learnings(conn, review_result: dict, today: date):
    """Insert learnings, skipping duplicates by title+body hash against recent history."""
    # Pull recent learning hashes (last 90 days) to dedup against
    recent = await conn.fetch(
        "SELECT title, body FROM learnings WHERE learning_date >= $1::date - 90",
        today,
    )
    seen = {_learning_hash(r["title"] or "", r["body"] or "") for r in recent}

    saved = 0
    for learning in review_result.get("learnings", []):
        title = learning.get("title", "")[:300]
        body = learning.get("body", "")
        h = _learning_hash(title, body)
        if h in seen:
            continue
        seen.add(h)
        await conn.execute(
            """
            INSERT INTO learnings (learning_date, learning_type, ticker, title, body, tags, raw_claude_output)
            VALUES ($1, 'eod_review', $2, $3, $4, $5, $6)
            """,
            today,
            learning.get("ticker"),
            title,
            body,
            learning.get("tags", []),
            review_result.get("raw", ""),
        )
        saved += 1
    log.info(f"Saved {saved} new learnings for {today} (skipped {len(review_result.get('learnings', [])) - saved} duplicates)")


async def update_model_accuracy(conn, predictions: list[dict], today: date):
    """
    Write per-model rolling accuracy for today into model_accuracy.
    A prediction is 'correct' when the realized move agrees with the signal direction.
    """
    if not predictions:
        return

    # Direction correctness per prediction (shared across model rows for the day)
    correct = 0
    for p in predictions:
        delta = p["actual_close"] - p["predicted_close"]
        sig = p.get("signal", "HOLD")
        if (sig == "BUY" and delta > 0) or (sig == "SELL" and delta < 0):
            correct += 1
    total = len(predictions)
    acc = round(correct / total, 4) if total else 0.0

    # Record one row per model layer so combine.py weight refresh has data to read
    for model_name in ("lgbm", "kronos", "slm", "combined"):
        await conn.execute(
            """
            INSERT INTO model_accuracy
                (model_name, signal_type, timeframe, period_start, period_end,
                 total_signals, correct_signals, accuracy, created_at)
            VALUES ($1, 'all', 'intraday', $2, $2, $3, $4, $5, NOW())
            """,
            model_name, today, total, correct, acc,
        )
    log.info(f"Model accuracy for {today}: {correct}/{total} ({acc*100:.1f}%) written for all layers")


async def run_eod_review():
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    today = date.today()
    log.info(f"Starting EOD review for {today}")

    signals = await fetch_todays_signals(conn)
    if not signals:
        log.info("No signals today — skipping EOD review")
        await conn.close()
        return

    log.info(f"Reviewing {len(signals)} signals from today")

    tickers = [s["ticker"] for s in signals]
    actual_closes = await fetch_actual_closes(conn, tickers)

    predictions = []
    for s in signals:
        actual = actual_closes.get(s["ticker"])
        if actual is None:
            continue

        predicted_close = float(s.get("target_price") or s.get("price_at_signal", 0))
        signal_type = s.get("signal_type", "BUY")
        entry = float(s["price_at_signal"])
        target = float(s["target_price"]) if s.get("target_price") else None
        stop = float(s["stop_loss"]) if s.get("stop_loss") else None

        # Determine outcome using target/stop thresholds (EOD approximation)
        if signal_type == "BUY":
            if target and actual >= target:
                status = "hit_target"
            elif stop and actual <= stop:
                status = "hit_stop"
            else:
                status = "expired"
        elif signal_type == "SELL":
            if stop and actual >= stop:
                status = "hit_stop"
            elif target and actual <= target:
                status = "hit_target"
            else:
                status = "expired"
        else:
            status = "expired"
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
    await update_model_accuracy(conn, predictions, today)

    accuracy = review_result.get("accuracy_today", {})
    log.info(
        f"EOD Summary: {accuracy.get('correct', 0)}/{accuracy.get('total', 0)} correct "
        f"({accuracy.get('pct', 0)*100:.1f}%)"
    )
    log.info(f"Summary: {review_result.get('summary', '')[:300]}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(run_eod_review())
