"""
Rolling accuracy tracker. Runs nightly after EOD review.
Computes 7-day accuracy per model, updates model_accuracy table,
and adjusts combine weights if one model is significantly better.

Outcome classification (explicit, consistent everywhere in this module):
  - hit_target -> WIN   (gross return = (target-entry)/entry for BUY, mirrored for SELL)
  - hit_stop   -> LOSS  (gross return = (stop-entry)/entry for BUY, mirrored for SELL)
  - expired    -> mark-to-market against entry using actual_close (NOT ignored/dropped)
  - active/cancelled -> excluded (not yet resolved / never happened)

Every realized/marked outcome has intelligence.costs.net_return applied before
win rate or expectancy is computed, so both numbers are cost-aware, not just
directionally-correct-on-paper.

Epoch filter: both compute_rolling_accuracy() and compute_net_expectancy() only
consider signals fired at/after the current funding epoch's start (see
intelligence.trading_account.get_current_epoch_start) — signals fired against
the old, insolvent ₹500 account are excluded so they never count toward the
live PAPER->LIVE gate (see trading_mode.py).
"""
import asyncpg
import logging
from datetime import datetime, timezone, timedelta
from config import settings
from intelligence.costs import net_return
from intelligence.trading_account import get_current_epoch_start

log = logging.getLogger(__name__)

RESOLVED_STATUSES = ("hit_target", "hit_stop", "expired")


def classify_and_return(signal_type: str, status: str, entry: float,
                         target: float | None, stop: float | None,
                         actual_close: float | None) -> tuple[str, float] | None:
    """
    Given a resolved signal's fields, return (outcome, gross_return_pct) where
    outcome is 'win' | 'loss' | 'flat', and gross_return_pct is a fractional
    return (e.g. 0.012 = +1.2%) BEFORE transaction costs.

    Returns None when the outcome can't be classified (e.g. missing entry).
    """
    if not entry:
        return None

    direction = 1 if signal_type == "BUY" else -1  # SELL profits on price decline

    if status == "hit_target":
        # Win by definition. Use target price as the realized exit if available,
        # else fall back to actual_close.
        exit_price = target if target is not None else actual_close
        if exit_price is None:
            return None
        gross = direction * (exit_price - entry) / entry
        return ("win", gross)

    if status == "hit_stop":
        # Loss by definition. Use stop price as the realized exit if available,
        # else fall back to actual_close.
        exit_price = stop if stop is not None else actual_close
        if exit_price is None:
            return None
        gross = direction * (exit_price - entry) / entry
        return ("loss", gross)

    if status == "expired":
        # Mark-to-market against entry using the actual close — never dropped.
        if actual_close is None:
            return None
        gross = direction * (actual_close - entry) / entry
        outcome = "win" if gross > 0 else ("loss" if gross < 0 else "flat")
        return (outcome, gross)

    return None


async def compute_rolling_accuracy(conn, days: int = 7) -> dict:
    """
    Returns {model_name: accuracy} for signals fired in the last `days` days
    where we have actuals (stop_loss/target hit or price moved opposite).

    "Accuracy" here is win rate under the explicit hit_target=win / hit_stop=loss /
    expired=mark-to-market classification above (still gross/raw win rate — kept
    for backward compatibility with existing consumers). See
    compute_net_expectancy() for the cost-adjusted metric.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    epoch_start = await get_current_epoch_start(conn)
    since = max(since, epoch_start)

    rows = await conn.fetch(
        """
        SELECT
            sr.model_name,
            s.signal_type, s.status, s.price_at_signal, s.target_price,
            s.stop_loss, s.actual_close
        FROM signal_reasoning sr
        JOIN signals s ON sr.signal_id = s.id
        WHERE s.fired_at >= $1
          AND s.status = ANY($2::text[])
        """,
        since, list(RESOLVED_STATUSES),
    )

    totals: dict[str, int] = {}
    corrects: dict[str, int] = {}
    for row in rows:
        outcome_ret = classify_and_return(
            row["signal_type"], row["status"], float(row["price_at_signal"]),
            float(row["target_price"]) if row["target_price"] is not None else None,
            float(row["stop_loss"]) if row["stop_loss"] is not None else None,
            float(row["actual_close"]) if row["actual_close"] is not None else None,
        )
        if outcome_ret is None:
            continue
        outcome, _ = outcome_ret
        model_name = row["model_name"]
        totals[model_name] = totals.get(model_name, 0) + 1
        if outcome == "win":
            corrects[model_name] = corrects.get(model_name, 0) + 1

    result = {}
    for model_name, total in totals.items():
        correct = corrects.get(model_name, 0)
        result[model_name] = round(correct / total, 4) if total > 0 else None

    return result


async def compute_net_expectancy(conn, n: int = 60) -> dict:
    """
    Net (cost-adjusted) expectancy over the trailing `n` resolved signals,
    across all models/combined — i.e. one expectancy number for the strategy,
    not per model layer (model-level attribution isn't meaningful once costs
    are netted against a single realized trade).

    Returns:
        {
          "n": <int, actual sample size used>,
          "win_rate": <float 0..1 | None>,          # cost-agnostic win rate for reference
          "net_win_rate": <float 0..1 | None>,       # win rate after costs (flips near-breakeven wins to losses)
          "net_expectancy_bps": <float | None>,      # mean net return per trade, in basis points
          "gross_expectancy_bps": <float | None>,    # mean gross return per trade, for comparison
        }
    """
    epoch_start = await get_current_epoch_start(conn)
    rows = await conn.fetch(
        """
        SELECT signal_type, status, price_at_signal, target_price, stop_loss, actual_close
        FROM signals
        WHERE status = ANY($1::text[])
          AND fired_at >= $3
        ORDER BY resolved_at DESC NULLS LAST, fired_at DESC
        LIMIT $2
        """,
        list(RESOLVED_STATUSES), n, epoch_start,
    )

    gross_returns: list[float] = []
    net_returns: list[float] = []
    net_wins = 0

    for row in rows:
        outcome_ret = classify_and_return(
            row["signal_type"], row["status"], float(row["price_at_signal"]),
            float(row["target_price"]) if row["target_price"] is not None else None,
            float(row["stop_loss"]) if row["stop_loss"] is not None else None,
            float(row["actual_close"]) if row["actual_close"] is not None else None,
        )
        if outcome_ret is None:
            continue
        _, gross = outcome_ret
        net = net_return(gross)
        gross_returns.append(gross)
        net_returns.append(net)
        if net > 0:
            net_wins += 1

    sample = len(net_returns)
    if sample == 0:
        return {
            "n": 0,
            "win_rate": None,
            "net_win_rate": None,
            "net_expectancy_bps": None,
            "gross_expectancy_bps": None,
        }

    gross_wins = sum(1 for g in gross_returns if g > 0)
    return {
        "n": sample,
        "win_rate": round(gross_wins / sample, 4),
        "net_win_rate": round(net_wins / sample, 4),
        "net_expectancy_bps": round((sum(net_returns) / sample) * 10000, 2),
        "gross_expectancy_bps": round((sum(gross_returns) / sample) * 10000, 2),
    }


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

        net_exp = await compute_net_expectancy(conn, n=60)
        log.info(
            f"Net expectancy (trailing {net_exp['n']} resolved signals): "
            f"{net_exp['net_expectancy_bps']} bps net "
            f"(gross {net_exp['gross_expectancy_bps']} bps), "
            f"net win rate {net_exp['net_win_rate']}"
        )

        from models.ml.retrain_trigger import check_and_trigger_retrain
        # Autonomous: when accuracy degrades for 7 straight days, retrain without asking.
        await check_and_trigger_retrain(conn, auto_retrain=True)

    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_accuracy_tracker())
