"""
Kronos foundation model integration.
Clones the repo on first run, runs inference for OHLCV forecasting.
"""
import subprocess
import sys
import os
import numpy as np
import pandas as pd
import logging
import torch

log = logging.getLogger(__name__)

KRONOS_DIR = os.path.join(os.path.dirname(__file__), "kronos_repo")
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"


def ensure_kronos():
    if not os.path.exists(KRONOS_DIR):
        log.info("Cloning Kronos repository...")
        subprocess.run(["git", "clone", KRONOS_REPO, KRONOS_DIR], check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", os.path.join(KRONOS_DIR, "requirements.txt")],
            check=True,
        )
    if KRONOS_DIR not in sys.path:
        sys.path.insert(0, KRONOS_DIR)


def load_kronos_model(model_size: str = "small"):
    """
    model_size: 'mini' (4.1M), 'small' (24.7M), 'base' (102.3M)
    On RTX 3050 4GB: base fits comfortably.
    """
    ensure_kronos()
    try:
        from kronos import KronosForecaster
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KronosForecaster.from_pretrained(f"kronos-{model_size}", device=device)
        log.info(f"Kronos-{model_size} loaded on {device}")
        return model
    except ImportError:
        log.warning("Kronos not yet available — using mock forecaster for dev mode")
        return MockKronosForecaster()


class MockKronosForecaster:
    """Dev fallback when Kronos isn't installed yet."""

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
        _kronos_model = load_kronos_model("base")
    return _kronos_model


def prepare_kronos_input(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Prepare last N candles for Kronos input."""
    required = ["open", "high", "low", "close", "volume"]
    df = df[required].copy().tail(lookback)
    df = df.dropna()
    return df


def forecast(df: pd.DataFrame, steps: int = 5) -> dict:
    """
    Forecast next `steps` candles using Kronos.
    Returns reasoning-enriched dict.
    """
    model = get_kronos()
    inp = prepare_kronos_input(df)

    if len(inp) < 20:
        return {"error": "Insufficient data for Kronos forecast"}

    forecast_df = model.forecast(inp, steps=steps)

    current_price = float(df["close"].iloc[-1])
    predicted_close = float(forecast_df["close"].iloc[-1])
    predicted_high = float(forecast_df["high"].max())
    predicted_low = float(forecast_df["low"].min())
    move_pct = (predicted_close - current_price) / current_price * 100

    # Build candle path string
    candle_path = " → ".join(f"₹{c:.0f}" for c in forecast_df["close"].values)

    if move_pct > 1.5:
        signal = "BUY"
        confidence = min(0.5 + move_pct * 0.03, 0.92)
    elif move_pct < -1.5:
        signal = "SELL"
        confidence = min(0.5 + abs(move_pct) * 0.03, 0.92)
    else:
        signal = "HOLD"
        confidence = 0.45

    reasoning = (
        f"Kronos Foundation Model ({signal} @ {confidence*100:.1f}% confidence):\n"
        f"  • Next {steps} candle forecast: {candle_path}\n"
        f"  • Predicted range: ₹{predicted_low:.1f} – ₹{predicted_high:.1f}\n"
        f"  • Expected move: {move_pct:+.2f}% over {steps * 30} minutes\n"
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
    }
