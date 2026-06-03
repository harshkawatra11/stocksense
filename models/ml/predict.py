"""
LightGBM inference + SHAP-based human-readable reasoning.
"""
import pickle
import shap
import numpy as np
import pandas as pd
import os
import json
import logging
from data.pipeline.feature_engineering import compute_features, get_feature_columns

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")

_model = None
_explainer = None
_feature_cols = None


def load_model():
    global _model, _explainer, _feature_cols
    path = os.path.join(MODEL_DIR, "lgbm_latest.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}. Run models/ml/train.py first.")
    with open(path, "rb") as f:
        _model = pickle.load(f)
    _explainer = shap.TreeExplainer(_model)
    _feature_cols = get_feature_columns()
    log.info("LightGBM model loaded.")


def get_model():
    if _model is None:
        load_model()
    return _model, _explainer, _feature_cols


FEATURE_DESCRIPTIONS = {
    "rsi_14": "RSI(14)",
    "rsi_7": "RSI(7) short-term momentum",
    "rsi_14_slope": "RSI trend (3-day slope)",
    "macd_diff": "MACD histogram",
    "macd_cross": "MACD crossover signal",
    "bb_position": "Bollinger Band position",
    "bb_width": "Bollinger Band squeeze",
    "dist_sma_20": "distance from 20-day SMA",
    "dist_sma_50": "distance from 50-day SMA",
    "dist_sma_200": "distance from 200-day SMA",
    "ema_cross_9_21": "EMA 9/21 crossover",
    "volume_ratio": "volume vs 20-day average",
    "volume_spike": "volume spike (>2x avg)",
    "return_1d": "1-day return",
    "return_5d": "5-day return",
    "return_20d": "20-day return",
    "atr_pct": "volatility (ATR%)",
    "dist_52w_high": "distance from 52-week high",
    "dist_52w_low": "distance from 52-week low",
    "stoch_k": "Stochastic %K",
    "obv_slope": "OBV trend",
    "candle_body": "candle body strength",
    "candle_lower_shadow": "lower wick (buyer support)",
}


def _feature_to_text(feat_name: str, value: float, shap_val: float) -> str:
    direction = "supporting" if shap_val > 0 else "opposing"
    pct = abs(shap_val) * 100

    desc = FEATURE_DESCRIPTIONS.get(feat_name, feat_name.replace("_", " "))

    if feat_name == "rsi_14":
        if value < 30:
            ctx = f"RSI at {value:.0f} — oversold territory"
        elif value > 70:
            ctx = f"RSI at {value:.0f} — overbought territory"
        else:
            ctx = f"RSI at {value:.0f} — neutral zone"
    elif feat_name == "volume_ratio":
        ctx = f"volume {value:.1f}x the 20-day average"
    elif feat_name == "volume_spike" and value == 1:
        ctx = "volume spike detected (>2x average)"
    elif feat_name == "macd_cross" and value == 1:
        ctx = "MACD bullish crossover active"
    elif feat_name == "ema_cross_9_21" and value == 1:
        ctx = "EMA 9 above EMA 21 (bullish alignment)"
    elif feat_name == "dist_sma_200":
        ctx = f"price {abs(value)*100:.1f}% {'above' if value > 0 else 'below'} 200-day SMA"
    elif feat_name == "dist_52w_high":
        ctx = f"price {abs(value)*100:.1f}% below 52-week high"
    elif feat_name == "dist_52w_low":
        ctx = f"price {abs(value)*100:.1f}% above 52-week low"
    elif feat_name == "bb_position":
        if value < 0.2:
            ctx = "price near lower Bollinger Band (oversold)"
        elif value > 0.8:
            ctx = "price near upper Bollinger Band (overbought)"
        else:
            ctx = f"price at {value*100:.0f}% of Bollinger Band width"
    else:
        ctx = f"{desc} = {value:.3f}"

    return f"{ctx} ({direction} signal, weight: {pct:.1f}%)"


def predict_with_reasoning(df: pd.DataFrame, ticker: str) -> dict:
    """
    df: raw OHLCV DataFrame for a single ticker
    Returns: {confidence, signal, reasoning_text, shap_values, features}
    """
    model, explainer, feature_cols = get_model()

    feat_df = compute_features(df, ticker=ticker)
    feat_df = feat_df.dropna(subset=feature_cols)

    if feat_df.empty:
        return {"error": "Not enough data to compute features"}

    latest = feat_df[feature_cols].iloc[[-1]]
    confidence = float(model.predict_proba(latest)[0, 1])
    signal = "BUY" if confidence > 0.55 else ("SELL" if confidence < 0.45 else "HOLD")

    shap_values = explainer.shap_values(latest)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    feature_values = latest.iloc[0].to_dict()
    shap_dict = dict(zip(feature_cols, sv))

    # Top 5 contributors
    top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    reasoning_lines = []
    total_weight = sum(abs(v) for _, v in top_features) + 1e-8
    for feat_name, shap_val in top_features:
        val = feature_values.get(feat_name, 0)
        pct = abs(shap_val) / total_weight * 100
        line = _feature_to_text(feat_name, val, shap_val)
        reasoning_lines.append(line)

    reasoning_text = (
        f"ML Model ({signal} @ {confidence*100:.1f}% confidence):\n"
        + "\n".join(f"  • {line}" for line in reasoning_lines)
    )

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 4),
        "reasoning": reasoning_text,
        "top_features": [
            {"feature": k, "shap": round(v, 4), "value": round(feature_values.get(k, 0), 4)}
            for k, v in top_features
        ],
        "latest_features": {k: round(v, 4) for k, v in feature_values.items()},
    }
