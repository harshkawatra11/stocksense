"""
Scheduler-death watchdog.

The mid-June incident: scheduler/market_runner.py died and nothing noticed
for weeks — no jobs ran, no signals queued, no alert fired, because the only
place that knew was backend/routers/system_health.py's /api/system/health
endpoint, and nothing was polling it. This loop closes that gap by running
INSIDE the backend process (a separate process from the scheduler, so a dead
scheduler doesn't also kill the watchdog) and pushing a Telegram alert the
moment the job_runs heartbeat goes stale.

Started from backend/main.py's lifespan, same pattern as quote_cache /
intraday_stops / telegram_bot. Clean no-op when Telegram isn't configured
(logs a warning periodically instead, so it's still visible via /api/system/health).
"""
from __future__ import annotations

import asyncio
import logging

import asyncpg

from config import settings
from backend.routers.system_health import _scheduler_heartbeat

log = logging.getLogger(__name__)

POLL_SECONDS = 900  # 15 min — fine-grained enough to catch a dead scheduler
                     # within one signal-pipeline cycle, cheap enough to run forever


async def start_scheduler_watchdog() -> None:
    from backend.services.telegram_bot import send_message, telegram_configured

    was_down = False
    while True:
        try:
            conn = await asyncpg.connect(settings.DATABASE_DSN)
            try:
                status = await _scheduler_heartbeat(conn)
            finally:
                await conn.close()

            is_down = status["status"] == "unavailable"
            if is_down and not was_down:
                msg = f"⚠️ Scheduler appears DOWN\n{status['detail']}"
                log.warning("scheduler_watchdog: %s", msg)
                if telegram_configured():
                    await send_message(msg)
            elif not is_down and was_down:
                msg = "✅ Scheduler heartbeat recovered"
                log.info("scheduler_watchdog: %s", msg)
                if telegram_configured():
                    await send_message(msg)
            was_down = is_down
        except Exception:
            log.exception("scheduler_watchdog: check failed (will retry)")

        await asyncio.sleep(POLL_SECONDS)
