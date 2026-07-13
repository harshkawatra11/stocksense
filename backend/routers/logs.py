from fastapi import APIRouter, Query
import asyncpg
from datetime import date
from pathlib import Path
from config import settings

router = APIRouter()
DB_DSN = settings.DATABASE_DSN

# logs/ lives at the package root (alongside config.py). Dated app.log files are
# written there by the launch scripts; surface their tail so the user can watch
# raw pipeline/scheduler output in-browser without a terminal.
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"


@router.get("/learnings")
async def get_learnings(limit: int = Query(50, le=200), ticker: str = None):
    conn = await asyncpg.connect(DB_DSN)
    if ticker:
        rows = await conn.fetch(
            "SELECT * FROM learnings WHERE ticker = $1 ORDER BY created_at DESC LIMIT $2",
            ticker, limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM learnings ORDER BY created_at DESC LIMIT $1", limit
        )
    await conn.close()
    return [dict(r) for r in rows]


@router.get("/learnings/today")
async def get_todays_learnings():
    conn = await asyncpg.connect(DB_DSN)
    rows = await conn.fetch(
        # learning_date is written as Python's date.today() (correct IST date,
        # see intelligence/eod_review.py), but Postgres's bare CURRENT_DATE
        # evaluates in the session's UTC timezone — for ~5.5h every evening
        # (IST 00:00-05:30) UTC's calendar date still lags a day behind IST,
        # so this would return nothing even though today's real learnings
        # exist. Same bug class fixed elsewhere in this codebase today.
        "SELECT * FROM learnings WHERE learning_date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date "
        "ORDER BY created_at DESC"
    )
    await conn.close()
    return [dict(r) for r in rows]


@router.get("/files")
async def get_log_files(tail: int = Query(200, le=2000), day: str = None):
    """
    Tail of the app.log for a given day (default: today) plus the list of
    available daily log folders. Lets the Logs panel stream raw run output.
    """
    target_day = day or date.today().isoformat()
    log_path = LOGS_DIR / target_day / "app.log"

    available = []
    if LOGS_DIR.exists():
        available = sorted(
            (p.name for p in LOGS_DIR.iterdir() if p.is_dir()),
            reverse=True,
        )

    lines: list[str] = []
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-tail:]
        except Exception as e:
            lines = [f"<error reading log: {e}>"]
    else:
        lines = [f"<no app.log for {target_day}>"]

    return {
        "day": target_day,
        "path": str(log_path),
        "available_days": available,
        "lines": [ln.rstrip("\n") for ln in lines],
    }
