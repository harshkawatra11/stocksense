"""
LightGBM training on NSE historical data. Walk-forward validation with TimeSeriesSplit.
Run: python -m models.ml.train
"""
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
import asyncpg
import asyncio
import pickle
import json
import os
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from data.pipeline.feature_engineering import compute_features, get_all_feature_columns
from config import settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_SAVE_DIR = os.path.join(str(settings.MODELS_DIR), "ml", "saved")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 1000,
    "verbose": -1,
    "n_jobs": -1,
}

# ------------------------------------------------------------------ #
# Ensemble + quantile regression (replaces Kronos's target/stop/ETA role
# — see WHAT_TO_DO_NEXT.txt Section 3 / Stage 3 of the build plan).
# ------------------------------------------------------------------ #
# 3-seed classifier ensemble: same LGBM_PARAMS/features/labels, different
# random seeds. Confidence = mean probability across seeds; agreement (1 -
# normalized std) is a secondary signal exposed to combine.py.
ENSEMBLE_SEEDS = [42, 123, 2024]

# Quantile regressors (q10/q50/q90) of the forward N-day return. Horizon
# matches HOLD_DAYS from backtest/walkforward.py's convention (5D) — see
# that module's `HOLD_DAYS = 5` and target_1d_return in feature_engineering.py
# for the 1-day classifier label. The quantile target below is a separate,
# longer horizon (5-day forward return) so q10/q50/q90 can size a
# stop/target band the way Kronos's forecast path used to.
QUANTILE_HORIZON_DAYS = 5
QUANTILE_ALPHAS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
QUANTILE_PARAMS_BASE = {
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 500,
    "verbose": -1,
    "n_jobs": -1,
}


async def load_training_data(conn, tickers: list = None) -> pd.DataFrame:
    query = """
        SELECT time, ticker, open, high, low, close, volume
        FROM ohlcv_daily
        WHERE close IS NOT NULL
    """
    if tickers:
        placeholders = ",".join(f"${i+1}" for i in range(len(tickers)))
        query += f" AND ticker IN ({placeholders})"
        rows = await conn.fetch(query, *tickers)
    else:
        rows = await conn.fetch(query)

    df = pd.DataFrame(rows, columns=["time", "ticker", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df


def _features_for_group(ticker: str, group: pd.DataFrame, feature_cols: list) -> pd.DataFrame | None:
    if len(group) < 300:
        return None
    feat = compute_features(group, ticker=ticker)
    # Quantile regression target: forward QUANTILE_HORIZON_DAYS return.
    # Computed here (not in feature_engineering.py, which is out of this
    # module's scope) from the raw close series, index-aligned to feat.
    feat["target_nd_return"] = (
        group["close"].pct_change(QUANTILE_HORIZON_DAYS).shift(-QUANTILE_HORIZON_DAYS)
    )
    # Memory: keep only what training needs, as float32 (halves footprint on
    # the full 8.6M-row universe), plus the ticker as a category column.
    keep = feat[feature_cols + ["target_buy", "target_nd_return"]].astype("float32")
    keep["ticker"] = ticker
    return keep


def finalize_feature_matrix(all_features: list) -> tuple:
    feature_cols = get_all_feature_columns()
    if not all_features:
        raise ValueError("No valid ticker data for training")

    combined = pd.concat(all_features)
    all_features.clear()
    combined = combined.dropna(subset=feature_cols + ["target_buy"])
    # CRITICAL for honest walk-forward CV: concat order is ticker-blocked, so
    # without this sort TimeSeriesSplit would split across tickers, not time.
    combined = combined.sort_index(kind="stable")
    combined["ticker"] = combined["ticker"].astype("category")

    X = combined[feature_cols]
    y = combined["target_buy"].astype("int8")
    return X, y, combined


async def load_and_build_features(conn, tickers: list = None, batch_size: int = 200) -> tuple:
    """Batched load + feature build: fetches OHLCV per ticker-batch and frees the
    raw rows immediately, instead of holding all ~8.6M rows plus features at once."""
    feature_cols = get_all_feature_columns()
    if tickers is None:
        rows = await conn.fetch("SELECT DISTINCT ticker FROM ohlcv_daily WHERE close IS NOT NULL")
        tickers = sorted(r["ticker"] for r in rows)
    log.info(f"Building features over {len(tickers)} tickers in batches of {batch_size}...")

    all_features = []
    total_rows = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        df = await load_training_data(conn, batch)
        total_rows += len(df)
        for ticker, group in df.groupby("ticker"):
            try:
                keep = _features_for_group(ticker, group, feature_cols)
                if keep is not None:
                    all_features.append(keep)
            except Exception as e:
                log.warning(f"Feature error for {ticker}: {e}")
        del df
        log.info(f"  batch {i//batch_size + 1}/{(len(tickers)+batch_size-1)//batch_size} done "
                 f"({total_rows:,} raw rows so far)")

    log.info(f"Loaded {total_rows:,} raw rows total.")
    return finalize_feature_matrix(all_features)


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """Legacy in-memory path (kept for callers that already hold a raw df)."""
    feature_cols = get_all_feature_columns()
    all_features = []
    for ticker, group in df.groupby("ticker"):
        try:
            keep = _features_for_group(ticker, group, feature_cols)
            if keep is not None:
                all_features.append(keep)
        except Exception as e:
            log.warning(f"Feature error for {ticker}: {e}")
    return finalize_feature_matrix(all_features)


def compute_optimal_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Find threshold maximising F1 on validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_f1 = 0.0
    best_thresh = 0.5
    for t in np.arange(0.3, 0.8, 0.02):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    log.info(f"Optimal threshold: {best_thresh:.2f} (val F1={best_f1:.4f})")
    return float(round(best_thresh, 4))


def train_ensemble_seeds(X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                          scale_pos_weight: float, save_dir: str | None = None, version: str = "") -> list:
    """
    Train ENSEMBLE_SEEDS independent LGBM classifiers on identical data/features,
    differing only by random_state. Used at inference for confidence-by-agreement
    (mean probability + std across seeds) instead of a single point estimate —
    replaces Kronos's fabricated sigmoid-of-move-magnitude confidence (see
    WHAT_TO_DO_NEXT.txt 2.2).

    Saves each seed's model to disk immediately after fitting and frees it from
    memory before starting the next seed (save_dir given) — holding all 3 fitted
    models simultaneously (each ~5M+ rows) alongside the caller's still-live
    feature matrix was OOM-killing the process silently (no Python traceback,
    just a dead process — see logs/retrain_20260712.log). Returns paths, not
    model objects, when save_dir is given.
    """
    results = []
    for seed in ENSEMBLE_SEEDS:
        params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight, "random_state": seed,
                   "bagging_seed": seed, "feature_fraction_seed": seed}
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        log.info(f"Ensemble seed {seed} trained.")
        if save_dir:
            p = os.path.join(save_dir, f"lgbm_seed{seed}.pkl")
            with open(p, "wb") as f:
                pickle.dump(m, f)
            log.info(f"Ensemble seed {seed} saved: {p}")
            results.append(p)
            del m
            gc.collect()
        else:
            results.append(m)
    return results


def train_quantile_models(X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                           save_path: str | None = None) -> dict:
    """
    Train LightGBM quantile regressors (q10/q50/q90) of the forward
    QUANTILE_HORIZON_DAYS return. These replace Kronos's forecast-path role:
    q50 = expected move, q10/q90 = a predictive interval sized into
    stop-loss/target instead of the flat ATR-multiple heuristic
    (see models/kronos/combine.py and intelligence/signal_pipeline.py's
    former target_and_eta()).

    Saves the full {q10,q50,q90} bundle once all three are trained (small
    regressors, unlike the classifier ensemble — memory pressure here is
    minor, but freeing intermediate fits still helps).
    """
    models = {}
    for name, alpha in QUANTILE_ALPHAS.items():
        params = {**QUANTILE_PARAMS_BASE, "objective": "quantile", "alpha": alpha, "metric": "quantile"}
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        models[name] = m
        log.info(f"Quantile model {name} (alpha={alpha}) trained.")
        gc.collect()
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(models, f)
        log.info(f"Quantile models saved: {save_path}")
    return models


async def train(tickers=None):
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    log.info("Loading training data from DB (batched)...")
    X, y, combined = await load_and_build_features(conn, tickers)
    log.info(f"Feature matrix: {X.shape}, class balance: {y.mean():.3f}")

    # Compute scale_pos_weight for class imbalance
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / (n_pos + 1e-9)
    log.info(f"Class imbalance: neg={n_neg}, pos={n_pos}, scale_pos_weight={scale_pos_weight:.2f}")

    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    # TimeSeriesSplit walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_aucs = []
    best_model = None
    best_auc = 0.0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
        )
        val_preds = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        log.info(f"Fold {fold+1}/5 AUC: {fold_auc:.4f}")

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_model = model
            # Capture val set for threshold computation
            best_X_val = X_val
            best_y_val = y_val

    mean_cv_auc = float(np.mean(fold_aucs))
    log.info(f"CV AUC: {mean_cv_auc:.4f} ± {np.std(fold_aucs):.4f}")

    # Compute optimal threshold from best validation fold
    optimal_threshold = compute_optimal_threshold(best_model, best_X_val, best_y_val)

    # Out-of-sample test: last 20%
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Retrain on 80% with best hyperparams for final model
    X_full_train = X.iloc[:split_idx]
    y_full_train = y.iloc[:split_idx]
    val_split = int(len(X_full_train) * 0.9)
    X_ftr, X_fval = X_full_train.iloc[:val_split], X_full_train.iloc[val_split:]
    y_ftr, y_fval = y_full_train.iloc[:val_split], y_full_train.iloc[val_split:]

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(
        X_ftr, y_ftr,
        eval_set=[(X_fval, y_fval)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )

    test_preds = final_model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, test_preds))
    acc = float(accuracy_score(y_test, (test_preds >= optimal_threshold).astype(int)))
    avg_conf = float(np.mean(test_preds))
    n_test = len(y_test)
    n_correct = int(accuracy_score(y_test, (test_preds >= optimal_threshold).astype(int)) * n_test)
    log.info(f"Test AUC: {auc:.4f}  Accuracy: {acc:.4f}")

    # Version + paths
    version = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODEL_SAVE_DIR, f"lgbm_{version}.pkl")
    meta_path = os.path.join(MODEL_SAVE_DIR, f"lgbm_{version}_meta.json")
    threshold_path = os.path.join(MODEL_SAVE_DIR, f"lgbm_{version}_threshold.json")
    latest_path = os.path.join(MODEL_SAVE_DIR, "lgbm_latest.pkl")
    latest_threshold_path = os.path.join(MODEL_SAVE_DIR, "lgbm_latest_threshold.json")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(latest_path, "wb") as f:
        pickle.dump(final_model, f)

    feature_cols = get_all_feature_columns()
    feature_importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
    top_importances = sorted(feature_importance.items(), key=lambda x: -x[1])[:20]

    meta = {
        "version": version,
        "auc": auc,
        "cv_auc": mean_cv_auc,
        "accuracy": acc,
        "optimal_threshold": optimal_threshold,
        "scale_pos_weight": float(scale_pos_weight),
        "n_train": int(len(X_full_train)),
        "n_test": n_test,
        "feature_importance": feature_importance,
        "feature_cols": feature_cols,
        "trained_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Sidecar threshold files
    threshold_data = {"threshold": optimal_threshold, "version": version}
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f)
    with open(latest_threshold_path, "w") as f:
        json.dump(threshold_data, f)

    log.info(f"Model saved: {model_path}")
    log.info(f"Top 10 features: {top_importances[:10]}")

    # ------------------------------------------------------------------ #
    # 3-seed ensemble + quantile regressors (Stage 3 — replaces Kronos)   #
    # ------------------------------------------------------------------ #
    # Free what the classic-model path no longer needs before the memory-
    # heavier ensemble step — holding X/X_test/y_test/best_* alongside 3
    # more full-size LGBM fits is what silently OOM-killed the previous run.
    del X, X_test, y_test, best_model, best_X_val, best_y_val, final_model, test_preds
    gc.collect()

    try:
        ensemble_paths = train_ensemble_seeds(
            X_ftr, y_ftr, X_fval, y_fval, scale_pos_weight,
            save_dir=MODEL_SAVE_DIR, version=version,
        )
        log.info(f"Ensemble saved: {ensemble_paths}")

        # Quantile targets: reuse the same train/val split, but drop rows with
        # no forward-return label (the trailing QUANTILE_HORIZON_DAYS rows).
        # Positional slicing: the time index has duplicate labels across tickers,
        # so .loc[X_ftr.index] would cartesian-explode. X shares combined's row order.
        q_col = "target_nd_return"
        q_train_df = combined.iloc[:val_split].dropna(subset=[q_col])
        q_val_df = combined.iloc[val_split:split_idx].dropna(subset=[q_col])
        if len(q_train_df) > 100 and len(q_val_df) > 20:
            quantile_bundle_path = os.path.join(MODEL_SAVE_DIR, "lgbm_quantiles.pkl")
            train_quantile_models(
                q_train_df[feature_cols], q_train_df[q_col],
                q_val_df[feature_cols], q_val_df[q_col],
                save_path=quantile_bundle_path,
            )
        else:
            log.warning("Not enough labeled rows for quantile training — skipping.")

        ensemble_meta = {
            "version": version,
            "seeds": ENSEMBLE_SEEDS,
            "quantile_horizon_days": QUANTILE_HORIZON_DAYS,
            "quantile_alphas": QUANTILE_ALPHAS,
            "trained_at": datetime.now().isoformat(),
        }
        with open(os.path.join(MODEL_SAVE_DIR, "lgbm_ensemble_meta.json"), "w") as f:
            json.dump(ensemble_meta, f, indent=2)
    except Exception as e:
        log.warning(f"Ensemble/quantile training failed (non-fatal — single-model lgbm_latest.pkl "
                    f"still saved): {e}")

    # Persist to DB
    # Determine period start/end from index
    all_dates = combined.index.sort_values()
    period_start = all_dates[0].date()
    period_end = all_dates[-1].date()

    try:
        await conn.execute("""
            INSERT INTO model_accuracy (model_name, signal_type, timeframe, period_start, period_end,
                total_signals, correct_signals, accuracy, avg_confidence, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())
        """, 'lgbm', 'BUY', 'daily', period_start, period_end, n_test, n_correct, auc, avg_conf)
        log.info("Saved model accuracy to DB")
    except Exception as e:
        log.warning(f"Could not save model_accuracy: {e}")

    try:
        await conn.executemany("""
            INSERT INTO feature_importance (model_version, feature_name, importance_score, recorded_at)
            VALUES ($1,$2,$3,NOW())
        """, [(version, feat, score) for feat, score in top_importances])
        log.info("Saved feature importances to DB")
    except Exception as e:
        log.warning(f"Could not save feature_importance: {e}")

    # Mark retrain completion in the audit trail so the brain's history is complete.
    try:
        from intelligence.activity import log_activity
        await log_activity(
            conn, event_type="RETRAIN",
            note=f"Retrain complete: {version} (AUC {auc:.3f}, threshold {optimal_threshold:.2f})",
            payload={"version": version, "auc": float(auc),
                     "threshold": float(optimal_threshold), "test_rows": int(n_test)},
        )
    except Exception as e:
        log.warning(f"Could not log RETRAIN completion: {e}")

    await conn.close()
    # final_model is freed above (memory pressure) before the ensemble step;
    # nothing in-process consumes this return value today (only the CLI
    # entry point `python -m models.ml.train` calls train(), and discards
    # the result) — return the on-disk path instead of holding it live.
    return latest_path, meta


if __name__ == "__main__":
    asyncio.run(train())
