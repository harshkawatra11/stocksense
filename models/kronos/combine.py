"""
Combines LightGBM + Kronos signals into a unified signal with weighted confidence.
Weights are read dynamically from model_accuracy DB; fallback to 40/60.
"""
import asyncpg
import logging
from config import settings

log = logging.getLogger(__name__)

# Module-level weight cache — refreshed by scheduler hourly
_lgbm_weight: float = 0.40
_kronos_weight: float = 0.60
_agreement_boost: float = 0.05  # adaptive via brain_params


async def refresh_weights_from_db():
    """Call from scheduler to update weights based on 7-day rolling accuracy."""
    global _lgbm_weight, _kronos_weight, _agreement_boost
    try:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        try:
            rows = await conn.fetch(
                """
                SELECT model_name, AVG(accuracy) AS avg_acc
                FROM model_accuracy
                WHERE created_at >= NOW() - INTERVAL '7 days'
                  AND model_name IN ('lgbm', 'kronos')
                GROUP BY model_name
                """,
            )
            boost = await conn.fetchval(
                "SELECT value FROM brain_params WHERE param_name = 'agreement_boost'"
            )
            if boost is not None:
                _agreement_boost = float(boost)
        finally:
            await conn.close()

        accs = {r["model_name"]: float(r["avg_acc"]) for r in rows if r["avg_acc"]}
        lgbm_acc = accs.get("lgbm")
        kronos_acc = accs.get("kronos")

        if lgbm_acc and kronos_acc:
            total = lgbm_acc + kronos_acc
            _lgbm_weight = round(lgbm_acc / total, 2)
            _kronos_weight = round(1.0 - _lgbm_weight, 2)
            log.info(f"Weights updated — LightGBM: {_lgbm_weight}, Kronos: {_kronos_weight}")
    except Exception as e:
        log.warning(f"Could not refresh weights from DB: {e}. Using {_lgbm_weight}/{_kronos_weight}")


# Kronos sources that must NOT be blended into the weighted combine — a mock
# (random-walk noise) or unavailable model has zero informational value, and
# blending it in at its usual 60% weight would silently corrupt the combined
# signal. "pretrained" is degraded (off-domain weights) but still a real model
# output, so it stays in the blend — just flagged via component_status.
_EXCLUDED_KRONOS_SOURCES = {"mock", "unavailable"}


def combine_signals(ml_result: dict, kronos_result: dict) -> dict:
    ml_conf = ml_result.get("confidence", 0.5)
    kr_conf = kronos_result.get("confidence", 0.5)
    ml_sig = ml_result.get("signal", "HOLD")
    kr_sig = kronos_result.get("signal", "HOLD")

    kronos_source = kronos_result.get("component_source", "unavailable")
    kronos_excluded = kronos_source in _EXCLUDED_KRONOS_SOURCES

    if kronos_excluded:
        # Kronos is noise or absent this cycle — go ML-only rather than blend in
        # random-walk confidence. Renormalize: ML gets full weight, Kronos zero.
        lgbm_w, kronos_w = 1.0, 0.0
        combined_conf = round(ml_conf, 4)
        reasoning = (
            f"Combined Analysis ({ml_sig} @ {combined_conf*100:.1f}% confidence, ML-only):\n"
            f"  • LightGBM: {ml_sig} at {ml_conf*100:.1f}% (weight 100%)\n"
            f"  • Kronos EXCLUDED — component_source={kronos_source} "
            f"({kronos_result.get('component_detail', 'no detail')}); "
            "not blended to avoid mixing in noise."
        )
        return {
            "signal": ml_sig,
            "confidence": combined_conf,
            "ml_signal": ml_sig,
            "kronos_signal": kr_sig,
            "ml_confidence": round(ml_conf, 4),
            "kronos_confidence": round(kr_conf, 4),
            "combined_reasoning": reasoning,
            "agreement": None,
            "kronos_excluded": True,
            "kronos_component_status": kronos_result.get("component_status", "unavailable"),
            "kronos_component_source": kronos_source,
        }

    lgbm_w = _lgbm_weight
    kronos_w = _kronos_weight

    # Both HOLD — neither model sees an edge. Reduce confidence further;
    # this is the weakest possible signal and should not reach Claude.
    if ml_sig == "HOLD" and kr_sig == "HOLD":
        combined_conf = round(min(ml_conf, kr_conf) * 0.8, 4)
        reasoning = (
            f"Combined Analysis (HOLD @ {combined_conf*100:.1f}% confidence):\n"
            f"  • LightGBM: HOLD at {ml_conf*100:.1f}%\n"
            f"  • Kronos: HOLD at {kr_conf*100:.1f}%\n"
            "  • Both models neutral — no edge, confidence reduced."
        )
        return {
            "signal": "HOLD",
            "confidence": combined_conf,
            "ml_signal": ml_sig,
            "kronos_signal": kr_sig,
            "ml_confidence": round(ml_conf, 4),
            "kronos_confidence": round(kr_conf, 4),
            "combined_reasoning": reasoning,
            "agreement": True,
            "kronos_excluded": False,
            "kronos_component_status": kronos_result.get("component_status", "ok"),
            "kronos_component_source": kronos_source,
        }

    # Agreement bonus (adaptive — tuned by nightly calibration within bounds)
    if ml_sig == kr_sig and ml_sig != "HOLD":
        agreement_boost = _agreement_boost
    elif ml_sig != kr_sig and "HOLD" not in (ml_sig, kr_sig):
        # Disagreement — penalize, pass to SLM for arbitration
        reasoning = (
            "⚠️ Model disagreement detected:\n"
            f"  ML model says {ml_sig} ({ml_conf*100:.1f}%)\n"
            f"  Kronos says {kr_sig} ({kr_conf*100:.1f}%)\n"
            "  Holding off — conflicting signals, passing to SLM for arbitration."
        )
        return {
            "signal": "HOLD",
            "confidence": 0.45,
            "ml_signal": ml_sig,
            "kronos_signal": kr_sig,
            "ml_confidence": ml_conf,
            "kronos_confidence": kr_conf,
            "combined_reasoning": reasoning,
            "agreement": False,
            "kronos_excluded": False,
            "kronos_component_status": kronos_result.get("component_status", "ok"),
            "kronos_component_source": kronos_source,
        }
    else:
        agreement_boost = 0.0

    combined_conf = ml_conf * lgbm_w + kr_conf * kronos_w + agreement_boost
    combined_conf = min(combined_conf, 0.95)

    combined_sig = ml_sig if ml_sig == kr_sig else (kr_sig if kr_conf > ml_conf else ml_sig)

    reasoning = (
        f"Combined Analysis ({combined_sig} @ {combined_conf*100:.1f}% confidence):\n"
        f"  • LightGBM: {ml_sig} at {ml_conf*100:.1f}% (weight {lgbm_w*100:.0f}%)\n"
        f"  • Kronos: {kr_sig} at {kr_conf*100:.1f}% (weight {kronos_w*100:.0f}%)\n"
        f"  • {'✅ Models in agreement — confidence boosted' if agreement_boost > 0 else '⚡ Single model dominant'}"
    )

    return {
        "signal": combined_sig,
        "confidence": round(combined_conf, 4),
        "ml_signal": ml_sig,
        "kronos_signal": kr_sig,
        "ml_confidence": round(ml_conf, 4),
        "kronos_confidence": round(kr_conf, 4),
        "combined_reasoning": reasoning,
        "agreement": ml_sig == kr_sig,
        "kronos_excluded": False,
        "kronos_component_status": kronos_result.get("component_status", "ok"),
        "kronos_component_source": kronos_source,
    }
