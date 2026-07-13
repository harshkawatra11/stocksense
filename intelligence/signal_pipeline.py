"""
Full signal pipeline orchestrator.
Runs every 30 min during market hours for all NSE tickers.
Supports both batch (non-streaming) and per-stage SSE streaming modes.
"""
import asyncio
import asyncpg
import json
import math
import pandas as pd
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

from data.pipeline.feature_engineering import compute_features
from models.ml.predict import (
    predict_with_reasoning, get_model_status, predict_with_ensemble, get_ensemble_status,
)
from models.ml.combine import combine_signals, quantile_target_stop
from models.slm.infer import slm_enrich
from intelligence.portfolio_guard import get_portfolio_tickers
from intelligence.macro_context import get_macro_context, MacroContext
from intelligence.activity import log_activity
from intelligence import llm_cli
from config import settings

log = logging.getLogger(__name__)

# Adaptive params snapshot for the current run — refreshed at the top of
# run_pipeline_multi so all tickers in a run see one consistent set.
_run_params: dict[str, float] = {}


# ------------------------------------------------------------------ #
# Live multi-timeframe helpers                                        #
# ------------------------------------------------------------------ #
def annotate_affordability(price: float) -> tuple[bool, int]:
    """How many whole shares the deployable cash buys; affordable = at least 1."""
    cash = settings.CASH_AVAILABLE
    shares = int(cash // price) if price and price > 0 else 0
    return shares >= 1, shares


def apply_macro_nudge(
    confidence: float, signal: str, sector: str | None, macro: MacroContext,
    cap: float = 0.10,
) -> tuple[float, float]:
    """
    Nudge confidence by the sector's macro news sentiment (±cap max — adaptive
    via brain_params). For BUY, a bullish sector raises confidence, a bearish
    one lowers it. Returns (new_confidence, sector_score).
    """
    score = macro.sector_score(sector)  # -1.0 .. +1.0
    delta = cap * score if signal == "BUY" else (-cap * score if signal == "SELL" else 0.0)
    return max(0.0, min(0.97, confidence + delta)), score


def horizon_stops(
    price: float, signal: str, atr: float | None, hold_days: int
) -> tuple[float, float]:
    """
    Volatility + horizon-scaled stop/target. Longer holds get wider bands
    (ATR × sqrt(hold_days)), clamped to [0.5%, 8%]. Target at 2R.
    """
    base = atr if atr and atr > 0 else price * 0.02
    factor = max(1.0, math.sqrt(max(hold_days, 1)))
    stop_dist = 1.5 * base * factor
    stop_dist = max(price * 0.005, min(stop_dist, price * 0.08))
    target_dist = 2.0 * stop_dist
    if signal == "SELL":
        return round(price + stop_dist, 2), round(price - target_dist, 2)
    return round(price - stop_dist, 2), round(price + target_dist, 2)


async def fetch_ohlcv(conn, ticker: str, limit: int = 300) -> pd.DataFrame:
    rows = await conn.fetch(
        """
        SELECT time, open, high, low, close, volume
        FROM ohlcv_daily
        WHERE ticker = $1 AND close IS NOT NULL
        ORDER BY time DESC LIMIT $2
        """,
        ticker, limit,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


async def fetch_fo(conn, ticker: str, limit: int = 300) -> pd.DataFrame:
    """Recent F&O daily aggregates for one ticker, same window/shape convention as
    fetch_ohlcv (most-recent-first then re-sorted ascending), so predict_with_ensemble
    sees the same feature distribution live as it learned on in training (train.py's
    load_fo_data). Returns an empty DataFrame — not a crash — for tickers with no F&O
    coverage (non-F&O-eligible small caps, or dates before fo_daily's 2010-12-16 start);
    compute_features() degrades those rows to its documented sentinel defaults."""
    rows = await conn.fetch(
        """
        SELECT time, total_oi, oi_change, put_oi, call_oi, pcr
        FROM fo_daily
        WHERE ticker = $1
        ORDER BY time DESC LIMIT $2
        """,
        ticker, limit,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "total_oi", "oi_change", "put_oi", "call_oi", "pcr"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    for col in ["total_oi", "oi_change", "put_oi", "call_oi", "pcr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """14-day Average True Range in absolute price terms. None if insufficient data."""
    if len(df) < period + 1:
        return None
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


# ------------------------------------------------------------------ #
# Stage 0 truth layer — component health                              #
# ------------------------------------------------------------------ #
# Shared status contract (see WHAT_TO_DO_NEXT.txt Section 4 / Stage 0):
#   {status: "ok"|"degraded"|"unavailable", detail: <str>, source: <enum>}
# Components: lightgbm (status + stale: bool, no source),
#             llm_synthesis (claude|codex|gemini|skipped),
#             macro (ollama_local|ollama_cloud|neutral_fallback)
#
# Each producer module exposes its own status getter — this is a thin
# passthrough that assembles them into one snapshot dict:
#   models/ml/predict.py       -> get_model_status()   (source: loaded|stale|hot_reload_check_failed|unavailable)
#   intelligence/llm_cli.py    -> get_component_status() (source: claude|codex|gemini|skipped)
#   intelligence/macro_context.py -> MacroContext.component_status/_source/_detail

def _lightgbm_status() -> dict:
    try:
        return get_model_status()
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": f"lightgbm status check failed: {e}", "source": "unavailable"}


def _ensemble_status() -> dict:
    try:
        return get_ensemble_status()
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": f"ensemble status check failed: {e}", "source": "unavailable"}


def _llm_synthesis_status() -> dict:
    try:
        return llm_cli.get_component_status()
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": f"llm_synthesis status check failed: {e}", "source": "skipped"}


def _macro_status(macro: "MacroContext | None" = None) -> dict:
    if macro is None:
        return {"status": "unavailable", "detail": "no macro context produced yet",
                "source": "neutral_fallback"}
    return {
        "status": macro.component_status,
        "detail": macro.component_detail,
        "source": macro.component_source,
    }


def get_component_statuses(macro: "MacroContext | None" = None) -> dict:
    """
    Stage 0 truth-layer snapshot: collects the {status, detail, source} of every
    pipeline component as of "right now" into one dict, ready to be persisted as
    components_json alongside each signal (DB column/migration owned by another
    agent — see save_signal_multi below for the write side already wired up).

    Shape (each component key maps to {status, detail, source}):
        {
          "lightgbm":      {"status": "ok"|"degraded"|"unavailable", "detail": str, "source": "loaded"|"stale"|"hot_reload_check_failed"|"unavailable"},
          "llm_synthesis": {"status": "ok"|"degraded"|"unavailable", "detail": str, "source": "claude"|"codex"|"gemini"|"skipped"},
          "macro":         {"status": "ok"|"degraded"|"unavailable", "detail": str, "source": "ollama_local"|"ollama_cloud"|"neutral_fallback"},
        }

    Called once per signal (cheap — no network calls of its own beyond what each
    component already tracks in-process) and attached to the saved row as
    components_json, and exposed via GET /api/system/health.
    """
    return {
        "ensemble": _ensemble_status(),
        "lightgbm": _lightgbm_status(),
        "llm_synthesis": _llm_synthesis_status(),
        "macro": _macro_status(macro),
    }


async def fetch_sector_map(conn) -> dict[str, str]:
    """ticker -> sector from the stocks DB table. Loaded once per run."""
    rows = await conn.fetch("SELECT ticker, sector FROM stocks WHERE sector IS NOT NULL")
    return {r["ticker"]: r["sector"] for r in rows}


async def fetch_recent_learnings(conn, limit: int = 20) -> str:
    """Active (non-expired, capped) learnings for a Claude prompt. See
    intelligence.eod_review.get_active_learnings for the lifecycle rules."""
    from intelligence.eod_review import get_active_learnings
    rows = await get_active_learnings(conn, limit=limit)
    if not rows:
        return ""
    return "\n".join(f"- {r['title']}: {r['body'][:200]}" for r in rows)


async def run_single_ticker_streaming(
    conn, ticker: str, portfolio_tickers: set, sector_map: dict[str, str] | None = None
) -> AsyncGenerator[dict, None]:
    """
    Async generator. Yields one dict per model stage as it completes.
    Frontend can render each stage immediately via SSE.
    """
    df = await fetch_ohlcv(conn, ticker)
    if df.empty or len(df) < 60:
        return

    fo_df = await fetch_fo(conn, ticker)
    current_price = float(df["close"].iloc[-1])
    held = ticker in portfolio_tickers
    sector = (sector_map or {}).get(ticker)

    # Stage 1: LightGBM ensemble (confidence = cross-seed agreement) + quantiles.
    try:
        ml_result = predict_with_ensemble(df, ticker, sector=sector, fo_df=fo_df)
        yield {
            "type": "ml_result",
            "ticker": ticker,
            "signal": ml_result.get("signal", "HOLD"),
            "confidence": ml_result.get("confidence", 0.5),
            "ml_reasoning": ml_result.get("ml_reasoning", ml_result.get("reasoning", "")),
            "price": current_price,
        }
    except Exception as e:
        log.warning(f"Ensemble predict failed for {ticker}: {e}")
        ml_result = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reasoning": f"Ensemble unavailable: {e}",
            "component_status": "unavailable",
            "component_source": "error",
        }
        yield {
            "type": "ml_result",
            "ticker": ticker,
            "signal": "HOLD",
            "confidence": 0.5,
            "ml_reasoning": f"Ensemble error: {e}",
            "price": current_price,
        }

    # Stage 2: Combine (ensemble-only) + SLM enrichment
    combined = combine_signals(ml_result)

    # Skip weak signals early
    if combined["confidence"] < 0.55 and combined["signal"] != "HOLD":
        return

    learnings = await fetch_recent_learnings(conn)
    atr = compute_atr(df)
    try:
        slm_result = slm_enrich(
            ticker=ticker,
            price=current_price,
            combined_signal=combined,
            ml_reasoning=ml_result.get("reasoning", ml_result.get("ml_reasoning", "")),
            portfolio_held=held,
            learnings_context=learnings,
            atr=atr,
        )
    except Exception as e:
        log.warning(f"SLM enrich failed for {ticker}: {e}")
        slm_result = {
            "signal": combined.get("signal", "HOLD"),
            "confidence": combined.get("confidence", 0.5),
            "reasoning": f"SLM unavailable: {e}",
            "stop_loss": current_price * 0.97,
            "target": current_price * 1.03,
            "pass_to_claude": False,
        }

    if slm_result.get("signal") == "SKIP":
        return

    # Quantile-derived stop/target (falls back to ATR heuristic inside
    # quantile_target_stop when q10/q50/q90 aren't available).
    q_stop, q_target = quantile_target_stop(
        current_price, combined.get("signal", "HOLD"),
        combined.get("q10"), combined.get("q50"), combined.get("q90"), atr=atr,
    )

    yield {
        "type": "slm_result",
        "ticker": ticker,
        "signal": slm_result.get("signal", combined["signal"]),
        "confidence": slm_result.get("confidence", combined["confidence"]),
        "slm_reasoning": slm_result.get("reasoning", ""),
        "stop_loss": slm_result.get("stop_loss", q_stop),
        "target": slm_result.get("target", q_target),
        "pass_to_claude": slm_result.get("pass_to_claude", False),
        "price": current_price,
        "ml_confidence": ml_result.get("confidence"),
        "slm_confidence": slm_result.get("confidence"),
        "ml_reasoning": ml_result.get("reasoning", ml_result.get("ml_reasoning", "")),
        "combined_reasoning": combined.get("combined_reasoning", ""),
        "fired_at": datetime.now(timezone.utc).isoformat(),
    }


async def run_single_ticker(
    conn, ticker: str, portfolio_tickers: set, sector_map: dict[str, str] | None = None
) -> dict | None:
    """Full pipeline for one ticker (non-streaming). Returns final signal dict or None."""
    df = await fetch_ohlcv(conn, ticker)
    if df.empty or len(df) < 60:
        return None

    fo_df = await fetch_fo(conn, ticker)
    current_price = float(df["close"].iloc[-1])
    held = ticker in portfolio_tickers
    sector = (sector_map or {}).get(ticker)

    # Step 1: LightGBM ensemble
    try:
        ml_result = predict_with_ensemble(df, ticker, sector=sector, fo_df=fo_df)
    except Exception as e:
        log.warning(f"Ensemble predict failed for {ticker}: {e}")
        return None

    # Step 2: Combine (ensemble-only)
    combined = combine_signals(ml_result)

    # Reject weak signals early
    if combined["confidence"] < 0.55 and combined["signal"] != "HOLD":
        return None

    # Step 3: SLM enrichment + portfolio guard
    learnings = await fetch_recent_learnings(conn)
    atr = compute_atr(df)
    slm_result = slm_enrich(
        ticker=ticker,
        price=current_price,
        combined_signal=combined,
        ml_reasoning=ml_result.get("reasoning", ml_result.get("ml_reasoning", "")),
        portfolio_held=held,
        learnings_context=learnings,
        atr=atr,
    )

    if slm_result.get("signal") == "SKIP":
        return None

    q_stop, q_target = quantile_target_stop(
        current_price, combined.get("signal", "HOLD"),
        combined.get("q10"), combined.get("q50"), combined.get("q90"), atr=atr,
    )

    signal_out = {
        "ticker": ticker,
        "price": current_price,
        "signal": slm_result.get("signal", combined["signal"]),
        "confidence": slm_result.get("confidence", combined["confidence"]),
        "stop_loss": slm_result.get("stop_loss", q_stop),
        "target": slm_result.get("target", q_target),
        "ml_confidence": ml_result.get("confidence"),
        "slm_confidence": slm_result.get("confidence"),
        "ml_reasoning": ml_result.get("reasoning", ml_result.get("ml_reasoning", "")),
        "slm_reasoning": slm_result.get("reasoning", ""),
        "pass_to_claude": slm_result.get("pass_to_claude", False),
        "combined_reasoning": combined.get("combined_reasoning", ""),
        "fired_at": datetime.now(timezone.utc).isoformat(),
    }

    return signal_out


async def run_pipeline_batch(tickers: list[str]) -> AsyncGenerator[dict, None]:
    """
    Run full pipeline for all tickers (non-streaming, batch mode).
    Used by scheduler. Yields individual final signals as they complete.
    Uses a connection pool so concurrent tasks each get their own connection
    (a single shared asyncpg connection raises InterfaceError under concurrency).
    """
    pool = await asyncpg.create_pool(settings.DATABASE_DSN, min_size=2, max_size=12)

    # Load shared context once using a dedicated pool connection
    async with pool.acquire() as meta_conn:
        portfolio_tickers = await get_portfolio_tickers(meta_conn)
        sector_map = await fetch_sector_map(meta_conn)

    semaphore = asyncio.Semaphore(10)

    async def bounded(ticker):
        async with semaphore:
            async with pool.acquire() as conn:
                return await run_single_ticker(conn, ticker, portfolio_tickers, sector_map)

    tasks = [bounded(t) for t in tickers]
    try:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                yield result
    finally:
        await pool.close()


async def run_pipeline_batch_streaming(
    tickers: list[str],
) -> AsyncGenerator[dict, None]:
    """
    SSE-compatible streaming pipeline.
    Yields per-stage events for each ticker in sequence.
    Use this from the /api/stream/signals endpoint.
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    portfolio_tickers = await get_portfolio_tickers(conn)
    sector_map = await fetch_sector_map(conn)

    try:
        for ticker in tickers:
            async for event in run_single_ticker_streaming(conn, ticker, portfolio_tickers, sector_map):
                yield event
    finally:
        await conn.close()


async def run_pipeline():
    """
    Scheduler entry point. Runs full pipeline batch and saves all signals to DB.
    Called by market_runner.py every 30 min during market hours.
    """
    save_conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        # Load active tickers from DB; fall back to static list
        rows = await save_conn.fetch(
            "SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker LIMIT $1",
            settings.MAX_TICKERS_PER_RUN,
        )
        if rows:
            tickers = [r["ticker"] for r in rows]
        else:
            from data.pipeline.nse_ticker_loader import FALLBACK_TICKERS
            tickers = [t["ticker"] for t in FALLBACK_TICKERS][:settings.MAX_TICKERS_PER_RUN]

        async for signal in run_pipeline_batch(tickers):
            try:
                await save_signal(save_conn, signal)
                log.info(f"Saved signal for {signal['ticker']} ({signal.get('signal')})")
            except Exception as e:
                log.warning(f"Failed to save signal for {signal.get('ticker')}: {e}")
    finally:
        await save_conn.close()


async def save_signal(conn, signal: dict) -> int:
    """Persist signal + reasoning to DB. Returns signal ID."""
    signal_id = await conn.fetchval(
        """
        INSERT INTO signals (
            ticker, signal_type, timeframe, price_at_signal,
            target_price, stop_loss,
            ml_confidence, slm_confidence,
            final_confidence, fired_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        RETURNING id
        """,
        signal["ticker"],
        signal.get("signal", "HOLD"),
        "intraday",
        signal["price"],
        signal.get("target"),
        signal.get("stop_loss"),
        signal.get("ml_confidence"),
        signal.get("slm_confidence"),
        signal.get("confidence"),
        datetime.now(timezone.utc),
    )

    for model_name, reasoning_key in [
        ("lgbm", "ml_reasoning"),
        ("slm", "slm_reasoning"),
        ("combined", "combined_reasoning"),
    ]:
        reasoning = signal.get(reasoning_key, "")
        if reasoning:
            await conn.execute(
                """
                INSERT INTO signal_reasoning (signal_id, model_name, reasoning)
                VALUES ($1, $2, $3)
                """,
                signal_id, model_name, reasoning,
            )

    return signal_id


# ================================================================== #
# LIVE multi-timeframe pipeline                                       #
#   ML ensemble prior → combine → macro nudge → BUY-only filter →     #
#   affordability annotation → save per horizon.                      #
#   The SLM's old per-ticker enrich is replaced by the global macro   #
#   layer (intelligence/macro_context), run once per cycle.           #
# ================================================================== #

async def run_single_ticker_multi(
    conn,
    ticker: str,
    portfolio_tickers: set,
    sector_map: dict[str, str],
    macro: MacroContext,
) -> list[dict]:
    """Produce one signal PER active timeframe for a ticker. Returns a list."""
    df = await fetch_ohlcv(conn, ticker)
    if df.empty or len(df) < 60:
        return []

    fo_df = await fetch_fo(conn, ticker)
    price = float(df["close"].iloc[-1])
    sector = sector_map.get(ticker)
    held = ticker in portfolio_tickers
    atr = compute_atr(df)

    # Ensemble directional prior — computed once, reused across horizons.
    # The ensemble's q10/q50/q90 quantiles (5-day horizon, models/ml/train.py)
    # supply the target/stop role.
    try:
        ml_result = predict_with_ensemble(df, ticker, sector=sector, fo_df=fo_df)
    except Exception as e:
        log.warning("Ensemble predict failed for %s: %s", ticker, e)
        return []

    combined = combine_signals(ml_result)
    signal = combined["signal"]
    base_conf = combined["confidence"]

    out: list[dict] = []
    for tf in settings.ACTIVE_TIMEFRAMES:
        hold = tf["hold_days"]
        conf = base_conf

        # BUY-only phase: drop everything that isn't a BUY.
        if settings.BUY_ONLY and signal != "BUY":
            continue

        # Macro news/sector nudge (cap adaptive via brain_params).
        conf, sector_score = apply_macro_nudge(
            conf, signal, sector, macro, cap=_run_params.get("macro_nudge_cap", 0.10)
        )

        # Confidence gate (per-horizon, adaptive via brain_params).
        gate = _run_params.get("confidence_threshold", settings.CONFIDENCE_THRESHOLD)
        if conf < gate:
            continue

        # Target + stop from the ensemble's q10/q50/q90 forward-return quantiles.
        # The quantile regressors are trained on a fixed QUANTILE_HORIZON_DAYS
        # (5-day) horizon (models/ml/train.py); eta_days is approximated as this
        # timeframe's own hold_days since there's no per-horizon forecast path
        # anymore — an honest simplification, not a fabricated fractional ETA.
        stop, target = quantile_target_stop(
            price, signal, combined.get("q10"), combined.get("q50"), combined.get("q90"), atr=atr,
        )
        move_pct = round((target - price) / price * 100, 2) if target else None
        eta_days = float(hold) if target else None
        affordable, shares = annotate_affordability(price)
        components = get_component_statuses(macro)

        # --- Stage 3: market-reality check (Upstox live-quote sanity check, ---
        # --- not a forecaster) — additive, does not affect signal/gating.  ---
        from intelligence.market_reality import enrich_signal_with_market_reality
        _mr = enrich_signal_with_market_reality({"ticker": ticker, "price": price})
        components["market_reality"] = _mr["component_status"]
        # --- end Stage 3 insertion ---

        out.append({
            "ticker": ticker,
            "timeframe": tf["label"],
            "horizon_days": hold,
            "price": price,
            "signal": signal,
            "confidence": round(conf, 4),
            "stop_loss": stop,
            "target": target,
            "target_eta_days": eta_days,
            "expected_move_pct": move_pct,
            "predicted_path": [],
            "affordable": affordable,
            "shares_affordable": shares,
            "macro_sector_score": round(sector_score, 4),
            "sector": sector,
            "held": held,
            "ml_confidence": ml_result.get("confidence"),
            "ml_reasoning": ml_result.get("reasoning", ""),
            "combined_reasoning": combined.get("combined_reasoning", ""),
            "macro_reasoning": (
                f"Macro: market {macro.market_bias}; sector {sector or 'n/a'} "
                f"score {sector_score:+.2f} — {macro.sectors.get((sector or '').upper(), {}).get('drivers', 'n/a')}"
            ),
            "headline": (
                f"BUY {ticker} @ ₹{price:.1f} → ₹{target:.1f} "
                f"({'+%.1f' % move_pct if move_pct else '?'}%) "
                f"in ~{eta_days}d" if (target and eta_days) else
                f"BUY {ticker} @ ₹{price:.1f}"
            ),
            "fired_at": datetime.now(timezone.utc).isoformat(),
            "components": components,
            "market_reality": _mr["market_reality"],
        })

    return out


async def save_signal_multi(conn, signal: dict) -> int:
    """Persist a multi-timeframe live signal (truthful timeframe + annotations)."""
    signal_id = await conn.fetchval(
        """
        INSERT INTO signals (
            ticker, signal_type, timeframe, horizon_days, price_at_signal,
            target_price, stop_loss,
            ml_confidence, slm_confidence, final_confidence,
            affordable, shares_affordable, macro_sector_score,
            target_eta_days, expected_move_pct, predicted_path, fired_at,
            components_json
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
        RETURNING id
        """,
        signal["ticker"],
        signal.get("signal", "HOLD"),
        signal["timeframe"],
        signal.get("horizon_days"),
        signal["price"],
        signal.get("target"),
        signal.get("stop_loss"),
        signal.get("ml_confidence"),
        None,  # slm_confidence — SLM is now the global macro layer, not per-signal
        signal.get("confidence"),
        signal.get("affordable"),
        signal.get("shares_affordable"),
        signal.get("macro_sector_score"),
        signal.get("target_eta_days"),
        signal.get("expected_move_pct"),
        json.dumps(signal.get("predicted_path")) if signal.get("predicted_path") else None,
        datetime.now(timezone.utc),
        json.dumps(signal.get("components")) if signal.get("components") else None,
    )

    for model_name, key in [
        ("lgbm", "ml_reasoning"),
        ("combined", "combined_reasoning"),
        ("macro", "macro_reasoning"),
    ]:
        reasoning = signal.get(key, "")
        if reasoning:
            await conn.execute(
                "INSERT INTO signal_reasoning (signal_id, model_name, reasoning) VALUES ($1,$2,$3)",
                signal_id, model_name, reasoning,
            )

    # Surface affordable BUYs in the activity feed so the user can rate/act on them.
    if signal.get("affordable") and signal.get("signal") == "BUY":
        await log_activity(
            conn, event_type="SUGGESTED", ticker=signal["ticker"], signal_id=signal_id,
            note=signal.get("headline", ""),
            payload={"timeframe": signal["timeframe"], "confidence": signal.get("confidence"),
                     "target": signal.get("target"), "eta_days": signal.get("target_eta_days"),
                     "shares_affordable": signal.get("shares_affordable")},
        )
    return signal_id


async def run_pipeline_multi(tickers: list[str] | None = None, save: bool = True) -> list[dict]:
    """
    Live multi-timeframe run. Loads macro context ONCE, then produces 1D/3D/5D
    BUY signals across the universe. Returns all signals (also saved unless save=False).
    """
    macro = get_macro_context()  # cached ~30 min
    log.info("Macro: bias=%s, %d headlines, source=%s",
             macro.market_bias, macro.headline_count, macro.source)

    pool = await asyncpg.create_pool(settings.DATABASE_DSN, min_size=2, max_size=8)
    async with pool.acquire() as meta:
        from intelligence.data_freshness import get_data_freshness
        from intelligence.brain_params import get_params
        global _run_params
        _run_params = await get_params(meta, force_refresh=True)
        log.info("Brain params: %s", _run_params)
        freshness = await get_data_freshness(meta)
        log.info("Data freshness: %s", freshness["label"])
        portfolio_tickers = await get_portfolio_tickers(meta)
        sector_map = await fetch_sector_map(meta)
        if tickers is None:
            rows = await meta.fetch(
                "SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker LIMIT $1",
                settings.MAX_TICKERS_PER_RUN,
            )
            tickers = [r["ticker"] for r in rows]

    semaphore = asyncio.Semaphore(8)
    all_signals: list[dict] = []

    async def bounded(t):
        async with semaphore:
            async with pool.acquire() as conn:
                sigs = await run_single_ticker_multi(conn, t, portfolio_tickers, sector_map, macro)
                if save and sigs:
                    for s in sigs:
                        try:
                            s["id"] = await save_signal_multi(conn, s)
                        except Exception as e:
                            log.warning("save failed %s/%s: %s", s["ticker"], s["timeframe"], e)
                return sigs

    try:
        results = await asyncio.gather(*[bounded(t) for t in tickers])
        for r in results:
            all_signals.extend(r)

        # Claude CLI final synthesis on the top signals (best per ticker), with learnings.
        if save and all_signals and settings.CLAUDE_SYNTHESIS_ENABLED:
            async with pool.acquire() as conn:
                await _claude_synthesis(conn, all_signals)
    finally:
        await pool.close()

    log.info("Multi-timeframe run done: %d signals across %d tickers",
             len(all_signals), len(tickers))
    return all_signals


async def _claude_synthesis(conn, all_signals: list[dict]) -> None:
    """
    Send the strongest signals (best timeframe per ticker, capped at
    TOP_SIGNALS_FOR_CLAUDE) to Claude Sonnet with recent learnings, then write the
    verdict back onto the saved rows: claude_confidence, final_confidence, a claude
    reasoning row, and demote REJECTs to HOLD.
    """
    from intelligence.claude_cli import intraday_signal_check

    # Best signal per ticker (highest confidence across its timeframes).
    best: dict[str, dict] = {}
    for s in all_signals:
        cur = best.get(s["ticker"])
        if cur is None or s["confidence"] > cur["confidence"]:
            best[s["ticker"]] = s
    top = sorted(best.values(), key=lambda x: -x["confidence"])[: settings.TOP_SIGNALS_FOR_CLAUDE]
    if not top:
        return

    learnings = await fetch_recent_learnings(conn)
    log.info("Claude synthesis on %d top signals (%d learnings in context)…",
             len(top), learnings.count("\n") + 1 if learnings else 0)
    try:
        enriched = intraday_signal_check(top, learnings_context=learnings)
    except Exception as e:
        log.warning("Claude synthesis failed: %s", e)
        return

    # The synthesis CLI's real status is only known now, after it actually ran —
    # patch it into each saved row's components_json (llm_synthesis was a
    # placeholder/"not yet called" value when the row was first saved).
    synth_status = _llm_synthesis_status()

    for s in enriched:
        sid = s.get("id")
        if not sid:
            continue
        claude_conf = s.get("claude_confidence")
        final_conf = s.get("final_confidence", s.get("confidence"))
        new_type = "HOLD" if s.get("claude_action") == "REJECT" else s.get("signal", "BUY")
        await conn.execute(
            "UPDATE signals SET claude_confidence=$1, final_confidence=$2, signal_type=$3, "
            "components_json = COALESCE(components_json, '{}'::jsonb) || jsonb_build_object('llm_synthesis', $5::jsonb) "
            "WHERE id=$4",
            claude_conf, final_conf, new_type, sid, json.dumps(synth_status),
        )
        if s.get("claude_reasoning"):
            await conn.execute(
                "INSERT INTO signal_reasoning (signal_id, model_name, reasoning) VALUES ($1,'claude',$2)",
                sid, s["claude_reasoning"],
            )
        await log_activity(
            conn, event_type="NOTE", ticker=s["ticker"], signal_id=sid,
            note=f"Claude {s.get('claude_action','CONFIRM')}: {final_conf*100:.0f}% final",
        )
