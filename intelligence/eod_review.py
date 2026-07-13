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
               sl.reasoning as slm_reasoning,
               cl.reasoning as claude_reasoning
        FROM signals s
        LEFT JOIN signal_reasoning ml ON ml.signal_id = s.id AND ml.model_name = 'lgbm'
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

    STRICT same-day match only: a ticker is included only if ohlcv_daily has a
    row dated *today*. This used to be "most recent close on/before today",
    which silently fell back to yesterday's (or older) close whenever today's
    bhavcopy hadn't landed in ohlcv_daily yet — and because that stale close is
    the exact same row used to set price_at_signal earlier in the day, every
    signal resolved under the old query showed a fabricated 0.0% return. Since
    eod_review now runs after incremental_ohlcv (see market_runner.py's
    eod_review job comment), today's row should normally exist; if it doesn't
    (bhavcopy delayed/missing), the ticker is correctly omitted and skipped
    downstream rather than resolved on fake data.

    Timezone note (found live 2026-07-13): ohlcv_daily.time is stored as
    midnight-IST-in-UTC (e.g. trading day 2026-07-13 is stored as
    2026-07-12 18:30:00+00:00). Postgres's default session timezone here is
    UTC, so a bare `time::date` comparison evaluates in UTC and is
    permanently one day behind the IST trading-day date Python's
    date.today() returns on this machine — every row silently missed its
    own trading day forever, which is the real reason the self-learning
    loop (learnings table) had produced zero rows for weeks despite
    eod_review running without error. Comparing in Asia/Kolkata fixes it.
    """
    if not tickers:
        return {}
    today = date.today()
    rows = await conn.fetch(
        """
        SELECT ticker, close
        FROM ohlcv_daily
        WHERE ticker = ANY($1::text[]) AND close IS NOT NULL
          AND (time AT TIME ZONE 'Asia/Kolkata')::date = $2
        """,
        list(set(tickers)), today,
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


LEARNING_TTL_DAYS = 60          # a learning that hasn't been refreshed expires after this
MIN_APPLICATIONS_FOR_JUDGEMENT = 10   # don't judge hit-rate on a small sample
MIN_HIT_RATE = 0.5              # below this hit-rate (after enough applications), kill it


async def save_learnings(conn, review_result: dict, today: date):
    """Insert learnings, skipping duplicates by title+body hash against recent history.

    Every new learning gets expires_at = created_at + LEARNING_TTL_DAYS — nothing
    accumulates forever unless it keeps getting regenerated.
    """
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
            INSERT INTO learnings
                (learning_date, learning_type, ticker, title, body, tags, raw_claude_output, expires_at)
            VALUES ($1, 'eod_review', $2, $3, $4, $5, $6, NOW() + ($7 || ' days')::interval)
            """,
            today,
            learning.get("ticker"),
            title,
            body,
            learning.get("tags", []),
            review_result.get("raw", ""),
            str(LEARNING_TTL_DAYS),
        )
        saved += 1
    log.info(f"Saved {saved} new learnings for {today} (skipped {len(review_result.get('learnings', [])) - saved} duplicates)")


async def get_active_learnings(conn, limit: int = 20) -> list[dict]:
    """
    Non-expired learnings for use in a Claude prompt (synthesis, calibration,
    weekend review), most recent first, capped at `limit`.

    Every row returned counts as "applied" — applies_count is bumped for each
    (see schema_v8_learnings_lifecycle.sql for what applies_count/hit_count
    actually measure and why).
    """
    rows = await conn.fetch(
        """
        SELECT id, ticker, title, body, tags, applies_count, hit_count, created_at
        FROM learnings
        WHERE expires_at IS NULL OR expires_at > NOW()
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )
    rows = [dict(r) for r in rows]
    if rows:
        ids = [r["id"] for r in rows]
        await conn.execute(
            "UPDATE learnings SET applies_count = applies_count + 1 WHERE id = ANY($1::int[])",
            ids,
        )
    return rows


async def record_learning_hits(conn, today: date, day_win_rate: float | None):
    """
    Rough daily proxy for learning usefulness (NOT precise causal attribution
    — see schema_v8_learnings_lifecycle.sql docstring). Once per EOD review,
    every currently-active learning that has been applied at least once gets
    hit_count += 1 if today's SELL win-rate was >= MIN_HIT_RATE.
    """
    if day_win_rate is None:
        return
    if day_win_rate < MIN_HIT_RATE:
        return
    await conn.execute(
        """
        UPDATE learnings
        SET hit_count = hit_count + 1
        WHERE applies_count > 0 AND (expires_at IS NULL OR expires_at > NOW())
        """
    )


async def expire_stale_learnings(conn) -> dict:
    """
    Delete learnings that are hard-expired (expires_at passed) OR that have
    been applied enough times (>= MIN_APPLICATIONS_FOR_JUDGEMENT) with a
    hit-rate below MIN_HIT_RATE. Returns counts for logging.
    """
    hard_expired = await conn.fetch(
        "DELETE FROM learnings WHERE expires_at IS NOT NULL AND expires_at <= NOW() RETURNING id"
    )
    low_hit_rate = await conn.fetch(
        """
        DELETE FROM learnings
        WHERE applies_count >= $1
          AND (hit_count::float / NULLIF(applies_count, 0)) < $2
        RETURNING id
        """,
        MIN_APPLICATIONS_FOR_JUDGEMENT, MIN_HIT_RATE,
    )
    result = {"hard_expired": len(hard_expired), "low_hit_rate": len(low_hit_rate)}
    if result["hard_expired"] or result["low_hit_rate"]:
        log.info("expire_stale_learnings: %s", result)
    return result


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
    for model_name in ("lgbm", "slm", "combined"):
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
            "slm_reasoning": s.get("slm_reasoning", ""),
            "claude_reasoning": s.get("claude_reasoning", ""),
        })

    log.info(f"Calling Claude Opus for EOD review of {len(predictions)} predictions...")
    review_result = eod_review_call(predictions)

    await save_learnings(conn, review_result, today)
    await update_model_accuracy(conn, predictions, today)

    # Daily hit-rate proxy + lifecycle cleanup — see record_learning_hits/
    # expire_stale_learnings docstrings for what's actually being measured.
    day_sell = await conn.fetchrow(
        """
        SELECT COUNT(*) FILTER (WHERE outcome = 'win') AS wins, COUNT(*) AS total
        FROM decisions WHERE action = 'SELL' AND decided_at::date = $1
        """,
        today,
    )
    day_win_rate = (day_sell["wins"] / day_sell["total"]) if day_sell and day_sell["total"] else None
    await record_learning_hits(conn, today, day_win_rate)
    await expire_stale_learnings(conn)

    accuracy = review_result.get("accuracy_today", {})
    log.info(
        f"EOD Summary: {accuracy.get('correct', 0)}/{accuracy.get('total', 0)} correct "
        f"({accuracy.get('pct', 0)*100:.1f}%)"
    )
    log.info(f"Summary: {review_result.get('summary', '')[:300]}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(run_eod_review())
