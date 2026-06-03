"""
Saturday 9:00 AM IST — weekend deep review job.
Aggregates the week's learnings + accuracy stats, calls Claude Opus for
regime assessment, generates next-week sector watchlist, stores in DB.
"""
import asyncpg
import logging
from datetime import datetime, timezone, timedelta
from intelligence.claude_cli import weekend_deep_review
from config import settings

log = logging.getLogger(__name__)


async def build_weekly_summary(conn) -> dict:
    """Pull this week's signals, learnings, and accuracy stats for the review prompt."""
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    signal_rows = await conn.fetch(
        """
        SELECT signal_type, COUNT(*) AS n,
               AVG(final_confidence) AS avg_conf,
               SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses
        FROM signals
        WHERE fired_at >= $1
        GROUP BY signal_type
        """,
        week_ago,
    )

    learning_rows = await conn.fetch(
        """
        SELECT title, body, model_at_fault, tags
        FROM learnings
        WHERE created_at >= $1
        ORDER BY created_at DESC
        LIMIT 50
        """,
        week_ago,
    )

    accuracy_rows = await conn.fetch(
        """
        SELECT model_name, accuracy
        FROM model_accuracy
        WHERE created_at >= $1
        ORDER BY created_at DESC
        LIMIT 20
        """,
        week_ago,
    )

    return {
        "signals": [dict(r) for r in signal_rows],
        "learnings": [dict(r) for r in learning_rows],
        "accuracy": [dict(r) for r in accuracy_rows],
    }


async def save_weekly_review(conn, review_text: str):
    """Persist the weekly review as a learning entry."""
    try:
        await conn.execute(
            """
            INSERT INTO learnings (title, body, learning_type, tags, created_at)
            VALUES ($1, $2, 'weekly_review', ARRAY['weekly', 'regime'], NOW())
            """,
            f"Weekly Review — {datetime.now().strftime('%Y-%m-%d')}",
            review_text[:4000],
        )
        log.info("Weekly review saved to learnings table")
    except Exception as e:
        log.error(f"Could not save weekly review: {e}")


async def run_weekend_review():
    log.info("Starting weekend deep review")
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        summary = await build_weekly_summary(conn)

        signals_text = "\n".join(
            f"  {r['signal_type']}: {r['n']} signals, "
            f"{r['wins']}W/{r['losses']}L, avg conf {r.get('avg_conf', 0):.2f}"
            for r in summary["signals"]
        ) or "  No signals this week"

        learnings_text = "\n".join(
            f"  [{r.get('model_at_fault', 'unknown')}] {r['title']}: {r['body'][:150]}"
            for r in summary["learnings"]
        ) or "  No learnings this week"

        accuracy_text = "\n".join(
            f"  {r['model_name']}: {r['accuracy']:.2%}"
            for r in summary["accuracy"]
            if r.get("accuracy") is not None
        ) or "  No accuracy data yet"

        context = (
            f"=== WEEKLY SIGNAL SUMMARY ===\n{signals_text}\n\n"
            f"=== MODEL ACCURACY ===\n{accuracy_text}\n\n"
            f"=== THIS WEEK'S LEARNINGS ===\n{learnings_text}"
        )

        log.info("Calling Claude Opus for weekend deep review")
        review_results = weekend_deep_review(context)

        for result in review_results:
            body = result.get("regime_assessment", "") or result.get("raw", "")
            if body:
                await save_weekly_review(conn, body)

        log.info("Weekend review complete")

    except Exception as e:
        log.error(f"Weekend review failed: {e}")
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_weekend_review())
