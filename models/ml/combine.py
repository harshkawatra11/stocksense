"""
Combines the LightGBM ensemble's classification + quantile outputs into a
unified signal with confidence, target and stop-loss.

Derives everything from the LGBM ensemble (models/ml/train.py +
models/ml/predict.py:predict_with_ensemble):
  - confidence: ensemble mean probability, nudged by cross-seed agreement.
  - target / stop-loss: derived from the q10/q50/q90 forward-return quantile
    regressors.
"""
import asyncpg
import logging
from config import settings

log = logging.getLogger(__name__)

# Ensemble sources that indicate a real, informative result. "single_model_fallback"
# (ensemble models missing, degraded to predict_with_reasoning) and "unavailable"
# still produce a usable signal but should be flagged as degraded, not silently
# treated as a full ensemble read.
_DEGRADED_ENSEMBLE_SOURCES = {"single_model_fallback", "partial", "hot_reload_check_failed"}
_UNAVAILABLE_ENSEMBLE_SOURCES = {"unavailable"}


def combine_signals(ensemble_result: dict) -> dict:
    """
    ensemble_result: the dict returned by models.ml.predict.predict_with_ensemble().

    Returns:
        {
          "signal": "BUY"|"SELL"|"HOLD",
          "confidence": float,                  # 0..0.95, agreement-nudged
          "agreement": float | None,             # 0..1 cross-seed agreement
          "confidence_std": float | None,
          "q10": float | None, "q50": float | None, "q90": float | None,
          "combined_reasoning": str,
          "ensemble_component_status": "ok"|"degraded"|"unavailable",
          "ensemble_component_source": str,
          "ml_signal": str, "ml_confidence": float,
        }
    """
    if "error" in ensemble_result:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "agreement": None,
            "confidence_std": None,
            "q10": None, "q50": None, "q90": None,
            "combined_reasoning": f"Ensemble error: {ensemble_result['error']}",
            "ensemble_component_status": "unavailable",
            "ensemble_component_source": "error",
            "ml_signal": "HOLD",
            "ml_confidence": 0.0,
        }

    signal = ensemble_result.get("signal", "HOLD")
    conf = float(ensemble_result.get("confidence", 0.5))
    agreement = ensemble_result.get("agreement")
    conf_std = ensemble_result.get("confidence_std")
    q10 = ensemble_result.get("q10")
    q50 = ensemble_result.get("q50")
    q90 = ensemble_result.get("q90")
    source = ensemble_result.get("component_source", "unavailable")
    status = ensemble_result.get("component_status", "unavailable")

    # Agreement nudge: full-ensemble reads get a small confidence adjustment
    # toward/away from their raw probability based on how much the seeds
    # agree; a single-model fallback (agreement=None) is left untouched.
    # Weights are placeholder-simple (ensemble-agreement-only) — TODO: once
    # backtest/walkforward.py (Stage 2, running separately) produces IC per
    # component, replace this with an IC-weighted blend instead of the flat
    # nudge below. Do not hand-tune these constants further until that lands.
    if agreement is not None and signal != "HOLD":
        nudge = 0.05 * (agreement - 0.5)  # +/-0.025 max
        combined_conf = max(0.0, min(0.95, conf + nudge))
    else:
        combined_conf = max(0.0, min(0.95, conf))

    if source in _UNAVAILABLE_ENSEMBLE_SOURCES:
        combined_conf = min(combined_conf, 0.5)  # never let an unavailable read look confident

    reasoning_lines = [
        f"Ensemble Analysis ({signal} @ {combined_conf*100:.1f}% confidence):",
        f"  • Mean seed probability: {conf*100:.1f}%"
        + (f", agreement {agreement*100:.0f}%" if agreement is not None else " (single-model fallback)"),
    ]
    if q10 is not None:
        reasoning_lines.append(
            f"  • Forward-return quantiles: q10={q10*100:+.1f}% q50={q50*100:+.1f}% q90={q90*100:+.1f}%"
        )
    else:
        reasoning_lines.append("  • Quantile targets unavailable — stop/target fall back to ATR heuristic")
    if status != "ok":
        reasoning_lines.append(f"  • Ensemble component status: {status} ({source})")

    return {
        "signal": signal,
        "confidence": round(combined_conf, 4),
        "agreement": agreement,
        "confidence_std": conf_std,
        "q10": q10, "q50": q50, "q90": q90,
        "combined_reasoning": "\n".join(reasoning_lines),
        "ensemble_component_status": status,
        "ensemble_component_source": source,
        "ml_signal": signal,
        "ml_confidence": round(conf, 4),
    }


def quantile_target_stop(price: float, signal: str, q10: float | None, q50: float | None,
                          q90: float | None, atr: float | None = None) -> tuple[float, float]:
    """
    Derive (stop_loss, target) from the q10/q50/q90 forward-return quantile
    interval. Falls back to a plain ATR heuristic if quantiles are
    unavailable (e.g. ensemble degraded to single-model mode).

    For BUY: target = price*(1+q90) if q90 upside exists, else price*(1+q50);
             stop   = price*(1+q10) if q10 is a real downside, else ATR-based.
    For SELL: mirrored (q10 is upside for a short, q90 is downside).
    """
    if q10 is None or q50 is None or q90 is None:
        base = atr if atr and atr > 0 else price * 0.02
        if signal == "SELL":
            return round(price + 1.5 * base, 2), round(price - 3.0 * base, 2)
        return round(price - 1.5 * base, 2), round(price + 3.0 * base, 2)

    if signal == "SELL":
        target = price * (1 - max(q90, 0.005))
        stop = price * (1 - min(q10, -0.002))
    else:  # BUY (and HOLD, harmless — pipeline won't act on HOLD stop/target)
        target = price * (1 + max(q90, 0.005))
        stop = price * (1 + min(q10, -0.002))

    return round(stop, 2), round(target, 2)


# ------------------------------------------------------------------ #
# Weight refresh — retained for the scheduler job (task_refresh_weights,  #
# scheduler/market_runner.py), tracks LGBM-ensemble accuracy. No-op       #
# placeholder until the walk-forward harness (backtest/walkforward.py)    #
# produces IC to act on.                                                  #
# ------------------------------------------------------------------ #
async def refresh_weights_from_db():
    """
    Kept as a scheduler entry point (see scheduler/market_runner.py
    task_refresh_weights) for interface stability. Logs the rolling LGBM
    accuracy for visibility and returns. TODO: once backtest/walkforward.py's
    IC numbers exist, use this hook to adjust the agreement-nudge constants
    in combine_signals() instead of leaving them hand-set.
    """
    try:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        try:
            avg_acc = await conn.fetchval(
                """
                SELECT AVG(accuracy) FROM model_accuracy
                WHERE created_at >= NOW() - INTERVAL '7 days' AND model_name = 'lgbm'
                """,
            )
        finally:
            await conn.close()
        if avg_acc is not None:
            log.info(f"LGBM ensemble rolling 7-day accuracy: {float(avg_acc):.4f} "
                      "(TODO IC-based nudge tuning)")
    except Exception as e:
        log.warning(f"Could not refresh LGBM accuracy snapshot: {e}")
