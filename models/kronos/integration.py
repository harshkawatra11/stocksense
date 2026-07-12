"""
Kronos foundation model integration — REMOVED, this is a thin stub.

Stage 3 dropped Kronos (WHAT_TO_DO_NEXT.txt Section 2.1/2.2/3, Stage 3 of
quiet-giggling-bumblebee.md): it never ran fine-tuned on NSE data — what was
live was either off-domain pretrained weights or a random-walk mock — and
its confidence score was fabricated by construction (a sigmoid of predicted
move magnitude). The LightGBM 3-seed ensemble + quantile regressors
(models/ml/train.py, models/ml/predict.py:predict_with_ensemble) replaced
its forecasting role entirely; models/ml/combine.py (formerly
models/kronos/combine.py) replaced its combine-weighting role.

The vendored upstream repo (kronos_repo/, ~17MB) and model weights
(weights/, ~31MB pretrained + an empty failed fine-tune attempt) have been
deleted — see the same commit that added this stub. This file is kept only
because intelligence/signal_pipeline.py still imports forecast() and
get_kronos_status() for an opt-in diagnostic path (config.KRONOS_ENABLED,
default false — see intelligence/position_monitor.py's docstring/history
for the real bug this used to cause when a second call site called Kronos
unconditionally) and for the Stage 0 system-health "kronos" component entry.
Both now report honestly that the model is gone, rather than the pipeline
crashing on import if this file were deleted outright.

If Kronos is ever genuinely worth resurrecting (properly fine-tuned on NSE
data, beating the ensemble in backtest/walkforward.py), re-vendor the repo
and replace this stub — don't try to un-delete weights from git history.
"""
import logging

log = logging.getLogger(__name__)

_kronos_status: dict = {
    "status": "unavailable",
    "detail": "Kronos was dropped in Stage 3 — never ran fine-tuned on NSE data; "
              "replaced by the LightGBM ensemble + quantile regressors "
              "(models/ml/predict.py:predict_with_ensemble). Code/weights deleted.",
    "source": "unavailable",
}


def get_kronos_status() -> dict:
    """Stage 0 status contract — always reports unavailable now. See module docstring."""
    return dict(_kronos_status)


def forecast(df, steps: int | None = None) -> dict:
    """
    Always returns an error result — Kronos is gone. Kept only so
    intelligence/signal_pipeline.py's KRONOS_ENABLED opt-in diagnostic path
    (default off) doesn't hit an ImportError; it already handles this
    result as "diagnostic failed" and skips it, same as any other error.
    """
    return {
        "error": "Kronos removed in Stage 3 — see models/kronos/integration.py docstring",
        "signal": "HOLD",
        "confidence": 0.5,
        "component_status": "unavailable",
        "component_source": "unavailable",
        "component_detail": _kronos_status["detail"],
    }
