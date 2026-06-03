"""
Rolling accuracy tracker. Runs nightly after EOD review.
Computes 7-day accuracy per model, updates model_accuracy table,
and adjusts combine weights if one model is significantly better.
"""
import asyncpg
import logging
from datetime import datetime, timezone, timedelta
from config import settings

log = logging.getLogger(__name__)


async def compute_rolling_accuracy(conn, days: int = 7) -> dict:
    """
    Returns {model_name: accuracy} for signals fired in the last `days` days
    where we have actuals (stop_loss/target hit or price moved opposite).
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await conn.fetch(
        """
        SELECT
            sr.model_name,
            COUNT(*) AS total,
            SUM(CASE
                WHEN s.signal_type = 'BUY' AND s.status = 'hit_target' THEN 1
                WHEN s.signal_type = 'SELL' AND s.status = 'hit_target' THEN 1
                ELSE 0
            END) AS correct
        FROM signal_reasoning sr
        JOIN signals s ON sr.signal_id = s.id
        WHERE s.fired_at >= $1
          AND s.status != 'active'
        GROUP BY sr.model_name
        """,
        since,
    )

    result = {}
    for row in rows:
        total = row["total"]
        correct = row["correct"]
        result[row["model_name"]] = round(correct / total, 4) if total > 0 else None

    return result


async def get_current_weights(conn) -> tuple[float, float]:
    """Return (lgbm_weight, kronos_weight) from recent model_accuracy rows."""
    rows = await conn.fetch(
        """
        SELECT model_name, accuracy
        FROM model_accuracy
        WHERE created_at >= NOW() - INTERVAL '7 days'
        ORDER BY created_at DESC
        LIMIT 20
        """,
    )

    lgbm_accs = [r["accuracy"] for r in rows if r["model_name"] == "lgbm" and r["accuracy"]]
    kronos_accs = [r["accuracy"] for r in rows if r["model_name"] == "kronos" and r["accuracy"]]

    if not lgbm_accs or not kronos_accs:
        return 0.40, 0.60  # defaults

    lgbm_avg = sum(lgbm_accs) / len(lgbm_accs)
    kronos_avg = sum(kronos_accs) / len(kronos_accs)

    total = lgbm_avg + kronos_avg
    if total == 0:
        return 0.40, 0.60

    lgbm_w = round(lgbm_avg / total, 2)
    kronos_w = round(1.0 - lgbm_w, 2)
    return lgbm_w, kronos_w


async def run_accuracy_tracker():
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        accuracies = await compute_rolling_accuracy(conn, days=7)
        log.info(f"7-day rolling accuracies: {accuracies}")

        if not accuracies:
            log.info("No signal outcomes available yet — skipping accuracy update")
            return

        for model_name, acc in accuracies.items():
            if acc is None:
                continue
            try:
                await conn.execute(
                    """
                    INSERT INTO model_accuracy (
                        model_name, signal_type, timeframe,
                        period_start, period_end,
                        total_signals, correct_signals, accuracy, avg_confidence, created_at
                    )
                    SELECT
                        $1, 'combined', '7day',
                        NOW() - INTERVAL '7 days', NOW(),
                        COUNT(*), SUM(CASE WHEN status='hit_target' THEN 1 ELSE 0 END),
                        $2, AVG(final_confidence), NOW()
                    FROM signals
                    WHERE fired_at >= NOW() - INTERVAL '7 days'
                      AND status != 'active'
                    """,
                    model_name, acc,
                )
            except Exception as e:
                log.warning(f"Could not update model_accuracy for {model_name}: {e}")

        # Check if combined accuracy is below 52% for alert
        combined_rows = await conn.fetch(
            """
            SELECT accuracy FROM model_accuracy
            WHERE timeframe = '7day'
              AND created_at >= NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 7
            """
        )
        if len(combined_rows) >= 7:
            recent_accs = [r["accuracy"] for r in combined_rows if r["accuracy"] is not None]
            if recent_accs and all(a < 0.52 for a in recent_accs):
                log.warning(
                    "ALERT: Combined accuracy below 52% for 7 consecutive days. "
                    "Consider retraining LightGBM model."
                )

        lgbm_w, kronos_w = await get_current_weights(conn)
        log.info(f"Dynamic weights — LightGBM: {lgbm_w:.2f}, Kronos: {kronos_w:.2f}")

        from models.ml.retrain_trigger import check_and_trigger_retrain
        await check_and_trigger_retrain(conn, auto_retrain=False)

    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_accuracy_tracker())
