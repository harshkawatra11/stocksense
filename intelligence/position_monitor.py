"""
Position re-analysis engine.

When you buy a stock the engine predicted would move ₹300 → ₹310 in ~2.5 days,
this re-checks that exact position against its original prediction as the horizon
approaches: how much of the predicted move has happened, is it on track, and a
fresh forecast → verdict (HOLD / EXIT / ADD). Each review is logged to
position_reviews and surfaced in the activity log (REANALYZED event).

Run:
    python -m intelligence.position_monitor
Schedule alongside the signal pipeline (e.g. every cycle during market hours).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import asyncpg

from config import settings
from intelligence.signal_pipeline import fetch_ohlcv, compute_atr
from models.ml.predict import predict_with_ensemble
from models.ml.combine import quantile_target_stop
from models.ml.train import QUANTILE_HORIZON_DAYS
from intelligence.activity import log_activity

log = logging.getLogger(__name__)


async def _originating_signal(conn, ticker: str) -> dict | None:
    """The signal behind the most recent BUY decision for this ticker."""
    row = await conn.fetchrow(
        """
        SELECT s.id, s.price_at_signal, s.target_price, s.stop_loss,
               s.target_eta_days, s.timeframe, s.fired_at
        FROM decisions d
        JOIN signals s ON s.id = d.signal_id
        WHERE d.ticker = $1 AND d.action = 'BUY'
        ORDER BY d.decided_at DESC
        LIMIT 1
        """,
        ticker,
    )
    return dict(row) if row else None


def _classify(progress_pct: float, days_elapsed: float, eta_days: float | None) -> tuple[str, str]:
    """Return (status, verdict) from progress vs time-elapsed-vs-ETA."""
    if progress_pct >= 100:
        return "target_hit", "EXIT"
    if progress_pct <= -100:          # fell as much as the intended gain — risk side
        return "stopped", "EXIT"

    if eta_days and eta_days > 0:
        time_frac = days_elapsed / eta_days          # how far through the predicted window
        expected = time_frac * 100                    # linear expectation
        if days_elapsed >= eta_days and progress_pct < 80:
            return "expired", "EXIT"                  # window's up, target not met
        if progress_pct >= expected + 20:
            return "ahead", "HOLD"
        if progress_pct <= expected - 20:
            return "behind", "HOLD"
        return "on_track", "HOLD"
    # No ETA — fall back to simple progress.
    return ("on_track", "HOLD") if progress_pct >= 0 else ("behind", "HOLD")


async def _resolve_stop_target(conn, ticker: str, entry: float) -> dict:
    """Stop/target/eta for a ticker's originating signal, with sane defaults
    when no signal is on record. Shared by the slow re-analysis cycle and the
    fast intraday breach check so both agree on the same numbers."""
    sig = await _originating_signal(conn, ticker)
    target = float(sig["target_price"]) if sig and sig["target_price"] else round(entry * 1.03, 2)
    stop = float(sig["stop_loss"]) if sig and sig["stop_loss"] else round(entry * 0.97, 2)
    eta_days = float(sig["target_eta_days"]) if sig and sig.get("target_eta_days") else None
    signal_id = sig["id"] if sig else None
    return {"target": target, "stop": stop, "eta_days": eta_days, "signal_id": signal_id}


async def check_stop_target_breach(conn, pos: dict, current_price: float) -> dict | None:
    """
    Fast breach primitive: literal stop-loss/target check against a live price,
    with none of review_position's richer re-forecast work. Returns None if
    neither has been breached, otherwise a dict shaped like a review_position
    result (ticker/signal_id/status/verdict/progress_pct/current_price/reasoning)
    so it can be fed straight into auto_trader.auto_exit — the same exit path
    the slow 30-min cycle uses. This is the primitive the fast intraday loop
    (intelligence/intraday_stops.py) polls every few seconds; the slow loop
    still owns the full progress/ETA re-analysis via review_position/_classify.
    """
    ticker = pos["ticker"]
    entry = float(pos["avg_price"])
    st = await _resolve_stop_target(conn, ticker, entry)
    target, stop, signal_id = st["target"], st["stop"], st["signal_id"]

    if current_price <= stop:
        status, verdict = "stopped", "EXIT"
    elif current_price >= target:
        status, verdict = "target_hit", "EXIT"
    else:
        return None

    denom = (target - entry) if (target - entry) != 0 else 1e-9
    progress_pct = round((current_price - entry) / denom * 100, 2)
    reasoning = (
        f"{ticker}: fast intraday breach — entry ₹{entry:.1f}, live ₹{current_price:.1f} "
        f"vs stop ₹{stop:.1f} / target ₹{target:.1f}. Status: {status} → {verdict}."
    )
    return {
        "ticker": ticker, "signal_id": signal_id, "status": status, "verdict": verdict,
        "progress_pct": progress_pct, "current_price": current_price, "reasoning": reasoning,
    }


async def review_position(conn, pos: dict) -> dict | None:
    """Re-analyze one active position. Persists a position_reviews row + activity event."""
    ticker = pos["ticker"]
    entry = float(pos["avg_price"])
    buy_date = pos["buy_date"]

    st = await _resolve_stop_target(conn, ticker, entry)
    target, stop, eta_days, signal_id = st["target"], st["stop"], st["eta_days"], st["signal_id"]

    # Current price
    df = await fetch_ohlcv(conn, ticker)
    if df.empty:
        return None
    current = float(df["close"].iloc[-1])

    now = datetime.now(timezone.utc)
    days_elapsed = round((now - buy_date).total_seconds() / 86400.0, 2) if buy_date else 0.0

    denom = (target - entry) if (target - entry) != 0 else 1e-9
    progress_pct = round((current - entry) / denom * 100, 2)
    status, verdict = _classify(progress_pct, days_elapsed, eta_days)

    # Fresh forward look — updated target/ETA from here. Uses the same
    # ensemble+quantile path signal_pipeline.py uses for new signals (Stage
    # 3 — replaces the old Kronos candle-path forecast here too; this call
    # site was still calling kronos_forecast() unconditionally, unlike the
    # signal-generation path's KRONOS_ENABLED gate — see WHAT_TO_DO_NEXT.txt
    # Kronos-drop follow-up).
    fresh_target = fresh_eta = fresh_move = None
    try:
        pred = predict_with_ensemble(df, ticker, sector=pos.get("sector"))
        q10, q50, q90 = pred.get("q10"), pred.get("q50"), pred.get("q90")
        if q10 is not None and q50 is not None and q90 is not None:
            atr = compute_atr(df)
            fresh_stop, fresh_target = quantile_target_stop(
                current, pred.get("signal", "HOLD"), q10, q50, q90, atr=atr,
            )
            fresh_eta = float(QUANTILE_HORIZON_DAYS)
            fresh_move = (fresh_target - current) / current if current else None
    except Exception as e:
        log.debug("Fresh forecast failed for %s: %s", ticker, e)

    reasoning = (
        f"{ticker}: entered ₹{entry:.1f}, now ₹{current:.1f} "
        f"({progress_pct:+.0f}% of the ₹{entry:.1f}→₹{target:.1f} move). "
        f"{days_elapsed:.1f}d elapsed"
        + (f" of ~{eta_days}d predicted" if eta_days else "")
        + f". Status: {status} → {verdict}."
        + (f" Fresh forecast: ₹{fresh_target:.1f} in ~{fresh_eta}d ({fresh_move:+.1f}%)."
           if fresh_target else "")
    )

    review_id = await conn.fetchval(
        """
        INSERT INTO position_reviews (
            portfolio_id, signal_id, ticker, days_elapsed, eta_days,
            entry_price, target_price, current_price, progress_pct,
            status, verdict, reasoning
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        RETURNING id
        """,
        pos.get("id"), signal_id, ticker, days_elapsed, eta_days,
        entry, target, current, progress_pct, status, verdict, reasoning,
    )

    await log_activity(
        conn, event_type="REANALYZED", ticker=ticker, signal_id=signal_id,
        note=f"{status} → {verdict} ({progress_pct:+.0f}%)",
        payload={"progress_pct": progress_pct, "days_elapsed": days_elapsed,
                 "current": current, "target": target, "verdict": verdict},
    )

    log.info("Reviewed %s: %s → %s (%.0f%%)", ticker, status, verdict, progress_pct)
    return {"review_id": review_id, "ticker": ticker, "status": status,
            "verdict": verdict, "progress_pct": progress_pct, "reasoning": reasoning}


async def review_all_positions() -> list[dict]:
    """Re-analyze every active position. Returns the review summaries."""
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        positions = await conn.fetch(
            "SELECT id, ticker, quantity, avg_price, buy_date FROM portfolio WHERE active = TRUE"
        )
        out = []
        for p in positions:
            try:
                r = await review_position(conn, dict(p))
                if r:
                    out.append(r)
            except Exception as e:
                log.warning("Review failed for %s: %s", p["ticker"], e)
        return out
    finally:
        await conn.close()


if __name__ == "__main__":
    import sys
    # Windows consoles default to cp1252 and choke on ₹/→ in reasoning strings.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reviews = asyncio.run(review_all_positions())
    if not reviews:
        print("No active positions to review.")
    for r in reviews:
        print(f"\n{r['ticker']}: {r['status']} → {r['verdict']} ({r['progress_pct']:+.0f}%)")
        print(f"  {r['reasoning']}")
