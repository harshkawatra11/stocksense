"""
Monitors rolling accuracy. If 7-day accuracy drops below 52% for 7
consecutive days, logs an alert and optionally triggers model retraining.
Called at the end of run_accuracy_tracker().
"""
import asyncpg
import logging
import subprocess
import sys
from config import settings

log = logging.getLogger(__name__)

ACCURACY_FLOOR = 0.52
CONSECUTIVE_DAYS_THRESHOLD = 7


async def check_and_trigger_retrain(conn, auto_retrain: bool = False) -> bool:
    """
    Returns True if retraining was triggered.
    auto_retrain=True: actually launches python -m models.ml.train as subprocess.
    auto_retrain=False (default): only logs the alert.
    """
    rows = await conn.fetch(
        """
        SELECT accuracy, created_at
        FROM model_accuracy
        WHERE timeframe = '7day'
        ORDER BY created_at DESC
        LIMIT $1
        """,
        CONSECUTIVE_DAYS_THRESHOLD,
    )

    if len(rows) < CONSECUTIVE_DAYS_THRESHOLD:
        log.info(
            f"Retrain check: only {len(rows)}/{CONSECUTIVE_DAYS_THRESHOLD} "
            "accuracy records — not enough history yet"
        )
        return False

    accs = [r["accuracy"] for r in rows if r["accuracy"] is not None]
    if len(accs) < CONSECUTIVE_DAYS_THRESHOLD:
        return False

    if not all(a < ACCURACY_FLOOR for a in accs):
        log.info(
            f"Retrain check: accuracy OK (last {CONSECUTIVE_DAYS_THRESHOLD} days: "
            f"{[round(a, 3) for a in accs]})"
        )
        return False

    log.warning(
        f"RETRAIN ALERT: Combined accuracy below {ACCURACY_FLOOR:.0%} for "
        f"{CONSECUTIVE_DAYS_THRESHOLD} consecutive days: {[round(a, 3) for a in accs]}"
    )

    if auto_retrain:
        log.info("Launching LightGBM retraining (python -m models.ml.train)...")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "models.ml.train"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("Retraining process launched in background.")
        except Exception as e:
            log.error(f"Failed to launch retraining: {e}")
        return True

    log.warning(
        "Auto-retraining is disabled. Run manually: python -m models.ml.train"
    )
    return False


async def run_retrain_check(auto_retrain: bool = False):
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        await check_and_trigger_retrain(conn, auto_retrain=auto_retrain)
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_retrain_check())
