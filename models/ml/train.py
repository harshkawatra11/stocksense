"""
LightGBM training on NSE historical data. Walk-forward validation with TimeSeriesSplit.
Run: python -m models.ml.train
"""
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


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    all_features = []
    feature_cols = get_all_feature_columns()

    for ticker, group in df.groupby("ticker"):
        if len(group) < 300:
            continue
        try:
            feat = compute_features(group, ticker=ticker)
            feat["ticker"] = ticker
            all_features.append(feat)
        except Exception as e:
            log.warning(f"Feature error for {ticker}: {e}")

    if not all_features:
        raise ValueError("No valid ticker data for training")

    combined = pd.concat(all_features)
    combined = combined.dropna(subset=feature_cols + ["target_buy"])

    X = combined[feature_cols]
    y = combined["target_buy"]
    return X, y, combined


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


async def train(tickers=None):
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    log.info("Loading training data from DB...")
    df = await load_training_data(conn, tickers)

    log.info(f"Building feature matrix from {len(df):,} rows across {df['ticker'].nunique()} stocks...")
    X, y, combined = build_feature_matrix(df)
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
    return final_model, meta


if __name__ == "__main__":
    asyncio.run(train())
