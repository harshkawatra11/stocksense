"""
Combines LightGBM + Kronos signals into a unified signal with weighted confidence.
"""


def combine_signals(ml_result: dict, kronos_result: dict) -> dict:
    """
    Weight: LightGBM 40%, Kronos 60% for intraday.
    Both must agree for high-confidence signal.
    """
    ml_conf = ml_result.get("confidence", 0.5)
    kr_conf = kronos_result.get("confidence", 0.5)
    ml_sig = ml_result.get("signal", "HOLD")
    kr_sig = kronos_result.get("signal", "HOLD")

    # Agreement bonus
    if ml_sig == kr_sig and ml_sig != "HOLD":
        agreement_boost = 0.05
    elif ml_sig != kr_sig and "HOLD" not in (ml_sig, kr_sig):
        # Disagreement — penalize both
        combined_conf = 0.45
        combined_sig = "HOLD"
        reasoning = (
            "⚠️ Model disagreement detected:\n"
            f"  ML model says {ml_sig} ({ml_conf*100:.1f}%)\n"
            f"  Kronos says {kr_sig} ({kr_conf*100:.1f}%)\n"
            "  Holding off — conflicting signals, passing to SLM for arbitration."
        )
        return {
            "signal": combined_sig,
            "confidence": combined_conf,
            "ml_signal": ml_sig,
            "kronos_signal": kr_sig,
            "ml_confidence": ml_conf,
            "kronos_confidence": kr_conf,
            "combined_reasoning": reasoning,
            "agreement": False,
        }
    else:
        agreement_boost = 0.0

    combined_conf = ml_conf * 0.4 + kr_conf * 0.6 + agreement_boost
    combined_conf = min(combined_conf, 0.95)

    # Dominant signal
    if ml_sig == kr_sig:
        combined_sig = ml_sig
    elif kr_conf > ml_conf:
        combined_sig = kr_sig
    else:
        combined_sig = ml_sig

    reasoning = (
        f"Combined Analysis ({combined_sig} @ {combined_conf*100:.1f}% confidence):\n"
        f"  • LightGBM: {ml_sig} at {ml_conf*100:.1f}%\n"
        f"  • Kronos: {kr_sig} at {kr_conf*100:.1f}%\n"
        f"  • {'✅ Models in agreement — confidence boosted' if agreement_boost > 0 else '⚡ Single model dominant'}\n"
        f"  • Combined weight: LightGBM 40% + Kronos 60%"
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
    }
