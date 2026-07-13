"""
Regression guards for the real bugs found and fixed on 2026-07-13. Each test
targets the exact mechanism that was broken, not just "does it run" — a bug
in this class should mechanically fail one of these rather than needing
another live debugging session to rediscover.

DB-touching tests connect to the real database (DATABASE_DSN) and clean up
after themselves with unique ticker names — this project's established
testing philosophy is "run it for real, check real DB state" rather than
mocking the database, since several of today's bugs were exactly the kind
that a mocked DB would have hidden.

Run: pytest tests/test_regression_guards.py  (or python -m tests.test_regression_guards)
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg

from config import settings

_TEST_TICKER_PREFIX = "ZZT"


def _unique_ticker(tag: str) -> str:
    # ticker column is VARCHAR(20) — keep well under that.
    suffix = str(int(datetime.now().timestamp()))[-6:]
    return f"{_TEST_TICKER_PREFIX}{tag[:4]}{suffix}"


async def _connect():
    return await asyncpg.connect(settings.DATABASE_DSN)


# ------------------------------------------------------------------ #
# 1. Timezone off-by-one: (time AT TIME ZONE 'Asia/Kolkata')::date    #
# ------------------------------------------------------------------ #
def test_ist_date_conversion_crosses_utc_midnight():
    """
    A timestamp stored as midnight-IST-in-UTC (trading day D -> D-1
    18:30:00+00:00, the exact ohlcv_daily convention) must convert back to
    calendar day D under (time AT TIME ZONE 'Asia/Kolkata')::date — and a
    bare ::date cast (the bug) must NOT, proving the fix is load-bearing
    rather than a no-op.
    """
    async def run():
        conn = await _connect()
        try:
            # 2026-07-12 18:30:00 UTC == 2026-07-13 00:00:00 IST
            stored_utc = datetime(2026, 7, 12, 18, 30, 0, tzinfo=timezone.utc)
            row = await conn.fetchrow(
                "SELECT $1::timestamptz::date AS bare_date, "
                "($1::timestamptz AT TIME ZONE 'Asia/Kolkata')::date AS ist_date",
                stored_utc,
            )
            assert row["ist_date"].isoformat() == "2026-07-13", (
                "IST-converted date should be 2026-07-13"
            )
            assert row["bare_date"].isoformat() == "2026-07-12", (
                "bare ::date cast should reproduce the historical bug (2026-07-12) "
                "in a UTC session — if this assertion fails, either the DB session "
                "timezone changed or Postgres's cast behavior changed; re-verify "
                "every ::date = CURRENT_DATE comparison in the codebase"
            )
        finally:
            await conn.close()

    asyncio.run(run())


# ------------------------------------------------------------------ #
# 2. _open_position_count() must exclude watch_only rows               #
# ------------------------------------------------------------------ #
def test_open_position_count_excludes_watch_only():
    """
    watch_only=TRUE rows are real external holdings the brain only monitors
    (see intelligence/auto_trader.py's docstrings) — they must never count
    against the paper engine's own position cap. Found live 2026-07-13: this
    was broken for 3 weeks, silently blocking every paper auto-buy.
    """
    async def run():
        conn = await _connect()
        ticker = _unique_ticker("watchonly")
        try:
            from intelligence.auto_trader import _open_position_count

            before = await _open_position_count(conn)

            await conn.execute(
                "INSERT INTO stocks (ticker, name) VALUES ($1, $1) ON CONFLICT DO NOTHING",
                ticker,
            )
            await conn.execute(
                """
                INSERT INTO portfolio (ticker, quantity, avg_price, buy_date, active, watch_only)
                VALUES ($1, 10, 100.0, NOW(), TRUE, TRUE)
                """,
                ticker,
            )

            after_watch_only = await _open_position_count(conn)
            assert after_watch_only == before, (
                f"watch_only row inflated the position count: {before} -> {after_watch_only}"
            )

            await conn.execute(
                "UPDATE portfolio SET watch_only = FALSE WHERE ticker = $1", ticker
            )
            after_real = await _open_position_count(conn)
            assert after_real == before + 1, (
                "a genuine (non-watch_only) position should count toward the cap"
            )
        finally:
            await conn.execute("DELETE FROM portfolio WHERE ticker = $1", ticker)
            await conn.execute("DELETE FROM stocks WHERE ticker = $1", ticker)
            await conn.close()

    asyncio.run(run())


# ------------------------------------------------------------------ #
# 3. _sandbox_net_position() dedup logic                               #
# ------------------------------------------------------------------ #
def test_sandbox_net_position_nets_buy_sell_and_excludes_failed():
    """
    The confirmation queue's dedup must track actual net sandbox exposure,
    not just "is anything currently PENDING" — that check stopped matching
    the instant a row resolved to APPROVED, letting the same trade repeat
    (found live 2026-07-13: ALKALI bought 3x in one session).
    """
    async def run():
        conn = await _connect()
        ticker = _unique_ticker("netpos")
        try:
            from intelligence.live_confirmation import _sandbox_net_position

            assert await _sandbox_net_position(conn, ticker) == 0

            async def insert(action, qty, status, execution_status):
                await conn.execute(
                    """
                    INSERT INTO pending_trade_confirmations
                        (ticker, action, quantity, price, reasoning, status, execution_status)
                    VALUES ($1, $2, $3, 100.0, 'test', $4, $5)
                    """,
                    ticker, action, qty, status, execution_status,
                )

            await insert("BUY", 10, "APPROVED", "PLACED")
            assert await _sandbox_net_position(conn, ticker) == 10

            await insert("BUY", 5, "APPROVED", "FAILED")
            assert await _sandbox_net_position(conn, ticker) == 10, (
                "a FAILED execution must not count toward net position"
            )

            await insert("SELL", 4, "APPROVED", "PLACED")
            assert await _sandbox_net_position(conn, ticker) == 6

            await insert("BUY", 100, "PENDING", None)
            assert await _sandbox_net_position(conn, ticker) == 6, (
                "a PENDING (not yet APPROVED) row must not count toward net position"
            )
        finally:
            await conn.execute(
                "DELETE FROM pending_trade_confirmations WHERE ticker = $1", ticker
            )
            await conn.close()

    asyncio.run(run())


# ------------------------------------------------------------------ #
# 4. Market-close cron boundaries                                      #
# ------------------------------------------------------------------ #
def test_no_trading_job_fires_after_market_close():
    """
    signal_pipeline/position_review must never fire after NSE's 15:30 IST
    close. hour="9-15" naively spans the whole hour-15 block, which used to
    include a 15:45/15:55 firing — found live 2026-07-13, generated real
    decisions off stale post-close prices.
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from scheduler.market_runner import build_scheduler

    # build_scheduler() only registers jobs (add_job), it never calls
    # .start() — there's no event loop attached, so .shutdown() would raise.
    # Nothing to clean up: this scheduler object is never started or shared.
    scheduler: AsyncIOScheduler = build_scheduler()
    trading_job_ids = {
        "signal_pipeline", "signal_pipeline_close",
        "position_review", "position_review_close",
    }
    checked = 0
    for job in scheduler.get_jobs():
        if job.id not in trading_job_ids:
            continue
        checked += 1
        trigger = job.trigger
        fields = {f.name: f for f in trigger.fields}
        hour_field = fields.get("hour")
        minute_field = fields.get("minute")
        assert hour_field is not None and minute_field is not None

        # Reconstruct every concrete (hour, minute) this trigger can fire at
        # and assert none is later than 15:30 IST.
        hours = [e for expr in hour_field.expressions for e in _expand(expr, 0, 23)]
        minutes = [e for expr in minute_field.expressions for e in _expand(expr, 0, 59)]
        for h in hours:
            for m in minutes:
                assert (h, m) <= (15, 30), (
                    f"job {job.id!r} can fire at {h:02d}:{m:02d} IST, after market close"
                )
    assert checked == len(trading_job_ids), "not all expected trading jobs were registered"


def _expand(expr, lo: int, hi: int) -> list[int]:
    """Best-effort expansion of an APScheduler cron field expression to concrete ints."""
    s = str(expr)
    if s == "*":
        return list(range(lo, hi + 1))
    if "-" in s and "/" not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if s.isdigit():
        return [int(s)]
    # */N or other step expressions — fall back to full range so the test
    # stays conservative (never silently skips a value that could violate
    # the assertion) rather than trying to fully replicate cron semantics.
    return list(range(lo, hi + 1))


if __name__ == "__main__":
    test_ist_date_conversion_crosses_utc_midnight()
    test_open_position_count_excludes_watch_only()
    test_sandbox_net_position_nets_buy_sell_and_excludes_failed()
    test_no_trading_job_fires_after_market_close()
    print("All regression guards passed.")
