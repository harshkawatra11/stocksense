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
from data.pipeline.feature_engineering import compute_features, get_all_feature_columns
from config import settings

log = logging.getLogger(__name__)

MODEL_SAVE_DIR = os.path.join(str(settings.MODELS_DIR), "ml", "saved")

_model = None
_explainer = None  # module-level singleton — created once
_feature_cols = None
_threshold = 0.5
_model_mtime: float = 0.0

# ------------------------------------------------------------------ #
# Ensemble + quantile models (Stage 3 — replaces Kronos's confidence/    #
# target/stop/ETA role; see models/ml/train.py and models/kronos/       #
# combine.py). Loaded lazily, same hot-reload-on-mtime-change pattern   #
# as the single LightGBM model above.                                   #
# ------------------------------------------------------------------ #
_ensemble_models: list = []          # list of LGBMClassifier, one per seed
_quantile_models: dict = {}          # {"q10": model, "q50": model, "q90": model}
_ensemble_mtime: float = 0.0
_ensemble_status: dict = {
    "status": "unavailable", "detail": "ensemble not loaded yet", "source": "unavailable",
}

# Stage 0 truth-layer status for the LightGBM component.
# status: "ok" | "degraded" | "unavailable"
# source: "loaded" | "stale" | "hot_reload_check_failed" | "unavailable"
STALENESS_THRESHOLD_DAYS = 14
_model_status: dict = {
    "status": "unavailable", "detail": "model not loaded yet", "source": "unavailable",
}


def get_model_status() -> dict:
    """Stage 0 status of the LightGBM component, per the shared status contract.
    Recomputes staleness against the current model's mtime each call (cheap: one
    os.path.getmtime), so a caller polling the health endpoint always sees a fresh
    staleness verdict, not just whatever load_model() saw at load time."""
    if _model_mtime <= 0:
        return dict(_model_status)
    import time as _time
    age_days = (_time.time() - _model_mtime) / 86400.0
    if age_days > STALENESS_THRESHOLD_DAYS:
        return {
            "status": "degraded",
            "detail": f"model is {age_days:.1f} days old (stale threshold {STALENESS_THRESHOLD_DAYS}d) "
                      "— no successful retrain has landed recently",
            "source": "stale",
        }
    # Not stale — report whatever load-time status we have, unless it recorded a
    # hot-reload failure (which stays surfaced until the next successful reload).
    if _model_status.get("source") == "hot_reload_check_failed":
        return dict(_model_status)
    return {"status": "ok", "detail": f"model is {age_days:.1f} days old", "source": "loaded"}


def load_model():
    global _model, _explainer, _feature_cols, _threshold, _model_mtime, _model_status
    path = os.path.join(MODEL_SAVE_DIR, "lgbm_latest.pkl")
    if not os.path.exists(path):
        _model_status = {
            "status": "unavailable",
            "detail": f"no trained model at {path}",
            "source": "unavailable",
        }
        raise FileNotFoundError(f"No trained model at {path}. Run models/ml/train.py first.")
    with open(path, "rb") as f:
        _model = pickle.load(f)

    # Load threshold from sidecar; fall back to 0.5
    threshold_path = os.path.join(MODEL_SAVE_DIR, "lgbm_latest_threshold.json")
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as f:
                data = json.load(f)
            _threshold = float(data.get("threshold", 0.5))
            log.info(f"Loaded classification threshold: {_threshold}")
        except Exception as e:
            log.warning(f"Could not load threshold sidecar: {e}. Using 0.5.")
            _threshold = 0.5
    else:
        log.info("No threshold sidecar found. Using default 0.5.")
        _threshold = 0.5

    # Build explainer once
    _explainer = shap.TreeExplainer(_model)
    _feature_cols = get_all_feature_columns()
    _model_mtime = os.path.getmtime(path)
    _model_status = {"status": "ok", "detail": f"loaded from {path}", "source": "loaded"}
    log.info("LightGBM model loaded.")


def get_model():
    global _model_status
    if _model is None:
        load_model()
    else:
        # Hot-reload after an auto-retrain replaced lgbm_latest.pkl.
        try:
            path = os.path.join(MODEL_SAVE_DIR, "lgbm_latest.pkl")
            if os.path.getmtime(path) > _model_mtime:
                log.info("lgbm_latest.pkl changed on disk — reloading model.")
                load_model()
        except OSError as e:
            # Stage 0: don't swallow this silently — we may now be serving a stale
            # model with no way to tell from the caller's side. Log at warning level
            # and record it so get_model_status() surfaces it until the next
            # successful reload.
            log.warning("Hot-reload staleness check failed (%s) — continuing to serve "
                        "the in-memory model, which may now be stale.", e)
            _model_status = {
                "status": "degraded",
                "detail": f"hot-reload mtime check failed: {e} — serving previously loaded model",
                "source": "hot_reload_check_failed",
            }
    return _model, _explainer, _feature_cols, _threshold


def get_ensemble_status() -> dict:
    """Stage 0-style status for the ensemble+quantile component, same contract
    as get_model_status()."""
    return dict(_ensemble_status)


def load_ensemble() -> bool:
    """
    Load the 3-seed classifier ensemble (models/ml/train.py:train_ensemble_seeds)
    and the q10/q50/q90 quantile regressors (train_quantile_models). Both are
    optional companions to the single lgbm_latest.pkl model — if they're not
    present (e.g. an older train.py run before Stage 3), predict_with_ensemble()
    degrades explicitly rather than crashing the pipeline.
    """
    global _ensemble_models, _quantile_models, _ensemble_mtime, _ensemble_status

    meta_path = os.path.join(MODEL_SAVE_DIR, "lgbm_ensemble_meta.json")
    if not os.path.exists(meta_path):
        _ensemble_status = {
            "status": "unavailable",
            "detail": f"no ensemble metadata at {meta_path} — run models/ml/train.py to produce it",
            "source": "unavailable",
        }
        _ensemble_models = []
        _quantile_models = {}
        return False

    with open(meta_path) as f:
        meta = json.load(f)
    seeds = meta.get("seeds", [])

    loaded_models = []
    for seed in seeds:
        p = os.path.join(MODEL_SAVE_DIR, f"lgbm_seed{seed}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            loaded_models.append(pickle.load(f))

    quantile_bundle_path = os.path.join(MODEL_SAVE_DIR, "lgbm_quantiles.pkl")
    loaded_quantiles = {}
    if os.path.exists(quantile_bundle_path):
        with open(quantile_bundle_path, "rb") as f:
            loaded_quantiles = pickle.load(f)

    _ensemble_models = loaded_models
    _quantile_models = loaded_quantiles
    _ensemble_mtime = os.path.getmtime(meta_path)

    if len(loaded_models) < len(seeds) or not loaded_quantiles:
        _ensemble_status = {
            "status": "degraded",
            "detail": f"partial ensemble: {len(loaded_models)}/{len(seeds)} seed models, "
                      f"quantiles={'ok' if loaded_quantiles else 'missing'}",
            "source": "partial",
        }
    else:
        _ensemble_status = {
            "status": "ok",
            "detail": f"{len(loaded_models)}-seed ensemble + q10/q50/q90 quantiles loaded",
            "source": "loaded",
        }
    log.info("Ensemble load: %s", _ensemble_status["detail"])
    return len(loaded_models) > 0


def get_ensemble():
    global _ensemble_status
    if not _ensemble_models and not _quantile_models:
        load_ensemble()
    else:
        try:
            meta_path = os.path.join(MODEL_SAVE_DIR, "lgbm_ensemble_meta.json")
            if os.path.exists(meta_path) and os.path.getmtime(meta_path) > _ensemble_mtime:
                log.info("Ensemble metadata changed on disk — reloading.")
                load_ensemble()
        except OSError as e:
            log.warning("Ensemble hot-reload check failed (%s) — serving previously loaded ensemble.", e)
            _ensemble_status = {
                "status": "degraded",
                "detail": f"hot-reload mtime check failed: {e} — serving previously loaded ensemble",
                "source": "hot_reload_check_failed",
            }
    return _ensemble_models, _quantile_models


def predict_with_ensemble(df: pd.DataFrame, ticker: str, sector: str | None = None) -> dict:
    """
    Ensemble + quantile inference. Returns:
        {
          "ticker", "signal", "confidence", "agreement", "confidence_std",
          "ml_reasoning", "q10", "q50", "q90", "component_status",
          "component_source", "component_detail",
        }
    confidence = mean predicted probability across the seed ensemble.
    agreement  = 1 - normalized std across seeds (1.0 = seeds fully agree).
    q10/q50/q90 = forward-return quantiles (fractional, e.g. 0.02 = +2%),
    used by models/kronos/combine.py to size stop/target instead of Kronos's
    forecast path. Falls back to the single-model predict_with_reasoning()
    (still fully supported — see its own docstring) when the ensemble/
    quantile models aren't available, so callers always get a usable result.
    """
    _, _, feature_cols, threshold = get_model()  # ensures single model + feature_cols loaded
    ensemble_models, quantile_models = get_ensemble()

    if not ensemble_models:
        # Degrade to the single-model path but keep the same return shape.
        base = predict_with_reasoning(df, ticker, sector=sector)
        if "error" in base:
            return base
        base.update({
            "agreement": None,
            "confidence_std": None,
            "q10": None, "q50": None, "q90": None,
            "component_status": "degraded",
            "component_source": "single_model_fallback",
            "component_detail": get_ensemble_status().get("detail", "ensemble unavailable"),
            "ml_reasoning": base.get("reasoning", ""),
        })
        return base

    feat_df = compute_features(df, ticker=ticker, sector=sector)
    feat_df = feat_df.dropna(subset=feature_cols)
    if feat_df.empty:
        return {"error": "Not enough data to compute features"}

    latest = feat_df[feature_cols].iloc[[-1]]

    probs = [float(m.predict_proba(latest)[0, 1]) for m in ensemble_models]
    mean_conf = float(np.mean(probs))
    std_conf = float(np.std(probs))
    agreement = max(0.0, 1.0 - (std_conf / 0.25))  # 0.25 std ~= max plausible spread -> 0 agreement

    buy_thresh = threshold
    sell_thresh = max(0.5 - (buy_thresh - 0.5), 0.2)
    signal = "BUY" if mean_conf >= buy_thresh else ("SELL" if mean_conf < sell_thresh else "HOLD")

    q10 = q50 = q90 = None
    if quantile_models:
        try:
            q10 = float(quantile_models["q10"].predict(latest)[0])
            q50 = float(quantile_models["q50"].predict(latest)[0])
            q90 = float(quantile_models["q90"].predict(latest)[0])
        except Exception as e:
            log.warning(f"Quantile predict failed for {ticker}: {e}")

    status = get_ensemble_status()
    reasoning = (
        f"Ensemble ({signal} @ {mean_conf*100:.1f}% mean confidence, "
        f"{len(probs)} seeds, agreement={agreement*100:.0f}%):\n"
        f"  • Per-seed probabilities: {', '.join(f'{p:.3f}' for p in probs)}\n"
        + (f"  • Forward {5}d return interval: q10={q10*100:+.1f}% q50={q50*100:+.1f}% q90={q90*100:+.1f}%\n"
           if q10 is not None else "  • Quantile targets unavailable\n")
    )

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(mean_conf, 4),
        "agreement": round(agreement, 4),
        "confidence_std": round(std_conf, 4),
        "threshold": round(buy_thresh, 4),
        "q10": round(q10, 4) if q10 is not None else None,
        "q50": round(q50, 4) if q50 is not None else None,
        "q90": round(q90, 4) if q90 is not None else None,
        "reasoning": reasoning,
        "ml_reasoning": reasoning,
        "component_status": status["status"],
        "component_source": status["source"],
        "component_detail": status["detail"],
    }


FEATURE_DESCRIPTIONS = {
    # RSI
    "rsi_14": "RSI(14)",
    "rsi_7": "RSI(7) short-term momentum",
    "rsi_21": "RSI(21) longer-term momentum",
    "rsi_14_slope": "RSI trend (3-day slope)",
    # MACD
    "macd": "MACD line",
    "macd_signal": "MACD signal line",
    "macd_diff": "MACD histogram",
    "macd_cross": "MACD crossover signal",
    # Bollinger
    "bb_position": "Bollinger Band position",
    "bb_width": "Bollinger Band squeeze",
    # SMAs
    "dist_sma_5": "distance from 5-day SMA",
    "dist_sma_10": "distance from 10-day SMA",
    "dist_sma_20": "distance from 20-day SMA",
    "dist_sma_50": "distance from 50-day SMA",
    "dist_sma_100": "distance from 100-day SMA",
    "dist_sma_200": "distance from 200-day SMA",
    # EMA
    "ema_cross_9_21": "EMA 9/21 crossover",
    # Volatility
    "atr_pct": "volatility (ATR%)",
    # OBV
    "obv_slope": "OBV trend",
    # Volume
    "volume_ratio": "volume vs 20-day average",
    "volume_spike": "volume spike (>2x avg)",
    # Returns
    "return_1d": "1-day return",
    "return_3d": "3-day return",
    "return_5d": "5-day return",
    "return_10d": "10-day return",
    "return_20d": "20-day return",
    "return_60d": "60-day return",
    # Candle
    "candle_body": "candle body strength",
    "candle_upper_shadow": "upper wick (seller pressure)",
    "candle_lower_shadow": "lower wick (buyer support)",
    # 52-week
    "dist_52w_high": "distance from 52-week high",
    "dist_52w_low": "distance from 52-week low",
    # Stochastic
    "stoch_k": "Stochastic %K",
    "stoch_d": "Stochastic %D",
    # Calendar
    "day_of_week": "day of week",
    "month": "month of year",
    "quarter": "quarter of year",
    # Sector
    "sector": "sector encoding",
    # F&O
    "oi_change_pct": "open interest change % (F&O)",
    "pcr": "put-call ratio (F&O)",
    "oi_rising": "OI rising (F&O accumulation signal)",
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
    elif feat_name == "oi_change_pct":
        ctx = f"OI change {value*100:+.1f}% (F&O positioning)"
    elif feat_name == "pcr":
        if value > 1.2:
            ctx = f"PCR {value:.2f} — elevated put activity (bearish hedge)"
        elif value < 0.8:
            ctx = f"PCR {value:.2f} — more calls than puts (bullish bias)"
        else:
            ctx = f"PCR {value:.2f} — neutral positioning"
    elif feat_name == "oi_rising" and value == 1:
        ctx = "OI rising — fresh F&O positions being built"
    else:
        ctx = f"{desc} = {value:.3f}"

    return f"{ctx} ({direction} signal, weight: {pct:.1f}%)"


def predict_with_reasoning(df: pd.DataFrame, ticker: str, sector: str | None = None) -> dict:
    """
    df: raw OHLCV DataFrame for a single ticker
    sector: sector name from the stocks DB table (optional; falls back to static map)
    Returns: {confidence, signal, reasoning_text, shap_values, features}
    """
    model, explainer, feature_cols, threshold = get_model()

    feat_df = compute_features(df, ticker=ticker, sector=sector)
    feat_df = feat_df.dropna(subset=feature_cols)

    if feat_df.empty:
        return {"error": "Not enough data to compute features"}

    latest = feat_df[feature_cols].iloc[[-1]]
    confidence = float(model.predict_proba(latest)[0, 1])

    buy_thresh = threshold
    sell_thresh = max(0.5 - (buy_thresh - 0.5), 0.2)  # symmetric around 0.5
    signal = "BUY" if confidence >= buy_thresh else ("SELL" if confidence < sell_thresh else "HOLD")

    # SHAP — reuse module-level explainer singleton
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
        line = _feature_to_text(feat_name, val, shap_val)
        reasoning_lines.append(line)

    reasoning_text = (
        f"ML Model ({signal} @ {confidence*100:.1f}% confidence | threshold={buy_thresh:.2f}):\n"
        + "\n".join(f"  • {line}" for line in reasoning_lines)
    )

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 4),
        "threshold": round(buy_thresh, 4),
        "reasoning": reasoning_text,
        "top_features": [
            {"feature": k, "shap": round(v, 4), "value": round(feature_values.get(k, 0), 4)}
            for k, v in top_features
        ],
        "latest_features": {k: round(v, 4) for k, v in feature_values.items()},
    }
