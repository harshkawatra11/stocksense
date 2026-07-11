"""
Kronos foundation model integration.
Wraps the upstream Kronos repo (shiyu-coder/Kronos) — KronosTokenizer + Kronos +
KronosPredictor — and exposes a single reasoning-enriched forecast() for the pipeline.

Install once:
    git clone https://github.com/shiyu-coder/Kronos.git models/kronos/kronos_repo
    pip install -r models/kronos/kronos_repo/requirements.txt
"""
import sys
import os
import numpy as np
import pandas as pd
import logging
import torch

from config import settings

log = logging.getLogger(__name__)

KRONOS_DIR = os.path.join(os.path.dirname(__file__), "kronos_repo")
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"
# Local NSE-finetuned model checkpoint (produced by finetune_nse.py). Optional.
# NSE-finetuned predictor checkpoint — produced by finetune_nse.py.
# finetune_nse.py saves to weights/nse_finetuned/{exp_name}/basemodel/best_model/
# Override via env var NSE_KRONOS_WEIGHTS_DIR if you used a custom exp_name.
_WEIGHTS_BASE = os.path.join(os.path.dirname(__file__), "weights", "nse_finetuned")
KRONOS_WEIGHTS_DIR = os.environ.get(
    "NSE_KRONOS_WEIGHTS_DIR",
    # auto-detect: use the first basemodel/best_model found under nse_finetuned/
    next(
        (
            str(p)
            for p in sorted(__import__("pathlib").Path(_WEIGHTS_BASE).glob("*/basemodel/best_model"))
            if p.is_dir()
        ),
        "",  # empty → no finetuned weights, use pretrained
    ),
)

# model_size -> (model HF id, tokenizer HF id, max_context)
_MODEL_SPECS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


def ensure_kronos():
    """
    Verify the Kronos repo is installed and importable. Does NOT auto-clone —
    a silent multi-hundred-MB clone + pip install mid-pipeline is the wrong default.
    """
    if not os.path.isdir(KRONOS_DIR):
        raise RuntimeError(
            f"Kronos repo not found at {KRONOS_DIR}.\nInstall it once with:\n"
            f"  git clone {KRONOS_REPO} \"{KRONOS_DIR}\"\n"
            f"  pip install -r \"{os.path.join(KRONOS_DIR, 'requirements.txt')}\""
        )
    if KRONOS_DIR not in sys.path:
        sys.path.insert(0, KRONOS_DIR)


# Stage 0 truth-layer status of the currently-loaded Kronos model.
# source: "finetuned" | "pretrained" | "mock" | "unavailable"
# status:  "ok"        | "degraded"  | "unavailable" | "unavailable"
_kronos_status: dict = {
    "status": "unavailable", "detail": "model not loaded yet", "source": "unavailable",
}


def get_kronos_status() -> dict:
    """Stage 0 status of the currently-loaded Kronos model, per the shared status contract."""
    return dict(_kronos_status)


def load_kronos_model(model_size: str | None = None):
    """
    Build a KronosPredictor. Prefers the NSE-finetuned checkpoint when present,
    otherwise pulls the pretrained model/tokenizer from HuggingFace.
    Falls back to MockKronosForecaster if Kronos is unavailable — but ALWAYS records
    an explicit status in _kronos_status/get_kronos_status() so callers (combine.py)
    know when they're looking at a degraded/mock component instead of silently
    treating it as a real forecast.
    """
    global _kronos_status
    model_size = model_size or settings.KRONOS_MODEL_SIZE
    model_id, tokenizer_id, max_context = _MODEL_SPECS.get(model_size, _MODEL_SPECS["base"])

    try:
        ensure_kronos()
        from model import Kronos, KronosTokenizer, KronosPredictor

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("CUDA not available — Kronos on CPU (slow). Forecasts will lag.")

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        if os.path.isdir(KRONOS_WEIGHTS_DIR):
            log.info(f"Loading NSE-finetuned Kronos weights from {KRONOS_WEIGHTS_DIR}")
            kmodel = Kronos.from_pretrained(KRONOS_WEIGHTS_DIR)
            _kronos_status = {
                "status": "ok",
                "detail": f"NSE-finetuned weights loaded from {KRONOS_WEIGHTS_DIR}",
                "source": "finetuned",
            }
        else:
            log.info(f"Loading pretrained {model_id}")
            kmodel = Kronos.from_pretrained(model_id)
            _kronos_status = {
                "status": "degraded",
                "detail": f"pretrained weights ({model_id}), not NSE-finetuned — "
                           "trained predominantly on Chinese A-share/crypto data",
                "source": "pretrained",
            }

        predictor = KronosPredictor(kmodel, tokenizer, device=device, max_context=max_context)
        log.info(f"Kronos ({model_size}) ready on {device}, max_context={max_context}")
        return predictor
    except Exception as e:
        log.warning(f"Kronos unavailable ({e}) — using mock forecaster for dev mode")
        _kronos_status = {
            "status": "unavailable",
            "detail": f"mock forecaster — Kronos load failed: {e}",
            "source": "mock",
        }
        return MockKronosForecaster()


class MockKronosForecaster:
    """Dev fallback when Kronos isn't installed yet. Mimics a random-walk forecast."""

    is_mock = True

    def forecast(self, df: pd.DataFrame, steps: int = 5) -> pd.DataFrame:
        last = df["close"].iloc[-1]
        noise = np.random.randn(steps) * 0.008
        closes = last * np.cumprod(1 + noise)
        highs = closes * (1 + abs(np.random.randn(steps)) * 0.005)
        lows = closes * (1 - abs(np.random.randn(steps)) * 0.005)
        opens = np.roll(closes, 1)
        opens[0] = last
        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": [df["volume"].mean()] * steps,
        })


_kronos_model = None


def get_kronos():
    global _kronos_model
    if _kronos_model is None:
        _kronos_model = load_kronos_model()
    return _kronos_model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def prepare_kronos_input(df: pd.DataFrame, lookback: int = 512) -> pd.DataFrame:
    """Last N candles with the OHLCV columns Kronos expects."""
    required = ["open", "high", "low", "close", "volume"]
    df = df[required].copy().tail(lookback)
    return df.dropna()


def _future_timestamps(last_ts: pd.Timestamp, steps: int) -> pd.Series:
    """Next `steps` business days after the last observed timestamp (daily NSE data)."""
    future = pd.bdate_range(start=last_ts + pd.Timedelta(days=1), periods=steps)
    return pd.Series(future)


def _run_predictor(predictor, inp: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Call the real KronosPredictor.predict with constructed timestamps."""
    x_df = inp.reset_index(drop=True)[["open", "high", "low", "close", "volume"]]
    x_timestamp = pd.Series(inp.index)
    y_timestamp = _future_timestamps(pd.Timestamp(inp.index[-1]), steps)
    return predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=steps,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )


def forecast(df: pd.DataFrame, steps: int | None = None) -> dict:
    """
    Forecast next `steps` candles using Kronos. Returns a reasoning-enriched dict.
    steps defaults to config.KRONOS_FORECAST_STEPS.
    """
    steps = steps or settings.KRONOS_FORECAST_STEPS
    model = get_kronos()
    status = get_kronos_status()
    inp = prepare_kronos_input(df)

    if len(inp) < settings.KRONOS_MIN_CANDLES:
        return {
            "error": (
                f"Insufficient data for Kronos forecast "
                f"({len(inp)} candles, need {settings.KRONOS_MIN_CANDLES})"
            ),
            "component_status": status["status"],
            "component_source": status["source"],
            "component_detail": status["detail"],
        }

    try:
        if getattr(model, "is_mock", False):
            forecast_df = model.forecast(inp, steps=steps)
        else:
            forecast_df = _run_predictor(model, inp, steps)
    except Exception as e:
        log.warning(f"Kronos predict failed ({e}) — returning HOLD")
        return {
            "error": f"Kronos predict failed: {e}",
            "component_status": "unavailable",
            "component_source": status["source"],
            "component_detail": f"predict() raised: {e}",
        }

    current_price = float(df["close"].iloc[-1])
    predicted_close = float(forecast_df["close"].iloc[-1])
    predicted_high = float(forecast_df["high"].max())
    predicted_low = float(forecast_df["low"].min())
    move_pct = (predicted_close - current_price) / current_price * 100

    candle_path = " → ".join(f"₹{c:.0f}" for c in forecast_df["close"].values)

    # Sigmoid confidence: maps move magnitude smoothly to (0.5, 0.95).
    k = 0.45
    if move_pct > 1.5:
        signal = "BUY"
        confidence = min(0.5 + 0.45 * (2 * _sigmoid(k * move_pct) - 1), 0.95)
    elif move_pct < -1.5:
        signal = "SELL"
        confidence = min(0.5 + 0.45 * (2 * _sigmoid(k * abs(move_pct)) - 1), 0.95)
    else:
        signal = "HOLD"
        confidence = 0.45

    reasoning = (
        f"Kronos Foundation Model ({signal} @ {confidence*100:.1f}% confidence):\n"
        f"  • Next {steps} candle forecast: {candle_path}\n"
        f"  • Predicted range: ₹{predicted_low:.1f} – ₹{predicted_high:.1f}\n"
        f"  • Expected move: {move_pct:+.2f}% over {steps} sessions\n"
        f"  • Price trajectory: {'upward with momentum' if move_pct > 2 else 'gradual upward' if move_pct > 0 else 'downward pressure'}"
    )

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "predicted_close": round(predicted_close, 2),
        "predicted_high": round(predicted_high, 2),
        "predicted_low": round(predicted_low, 2),
        "move_pct": round(move_pct, 3),
        "candle_forecast": forecast_df.to_dict(orient="records"),
        "reasoning": reasoning,
        # Stage 0 truth-layer status — combine.py checks this to decide whether
        # to blend Kronos in at all (see models/kronos/combine.py).
        "component_status": status["status"],
        "component_source": status["source"],
        "component_detail": status["detail"],
    }
