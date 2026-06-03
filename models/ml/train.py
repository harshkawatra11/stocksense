"""
LightGBM training on NSE historical data. Walk-forward validation.
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
from data.pipeline.feature_engineering import compute_features, get_feature_columns
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


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
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
}


async def load_training_data(conn, tickers: list[str] = None) -> pd.DataFrame:
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
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    all_features = []
    feature_cols = get_feature_columns()

    for ticker, group in df.groupby("ticker"):
        if len(group) < 300:
            continue
        try:
            feat = compute_features(group, ticker=ticker)
            feat["ticker"] = ticker
            all_features.append(feat)
        except Exception as e:
            log.warning(f"Feature error for {ticker}: {e}")

    combined = pd.concat(all_features)
    combined = combined.dropna(subset=feature_cols + ["target_buy"])

    X = combined[feature_cols]
    y = combined["target_buy"]
    return X, y


async def train(tickers: list[str] = None):
    conn = await asyncpg.connect(DB_DSN)
    log.info("Loading training data from DB...")
    df = await load_training_data(conn, tickers)
    await conn.close()

    log.info(f"Building feature matrix from {len(df):,} rows across {df['ticker'].nunique()} stocks...")
    X, y = build_feature_matrix(df)
    log.info(f"Feature matrix: {X.shape}, class balance: {y.mean():.3f}")

    # Walk-forward split — last 20% as out-of-sample test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    val_split = int(len(X_train) * 0.9)
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    X_tr = X_train.iloc[:val_split]
    y_tr = y_train.iloc[:val_split]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    test_preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_preds)
    acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
    log.info(f"Test AUC: {auc:.4f}  Accuracy: {acc:.4f}")

    # Save model + metadata
    version = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODEL_DIR, f"lgbm_{version}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"lgbm_{version}_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    feature_importance = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
    meta = {
        "version": version,
        "auc": auc,
        "accuracy": acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": feature_importance,
        "feature_cols": X.columns.tolist(),
        "trained_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Save as latest
    latest_path = os.path.join(MODEL_DIR, "lgbm_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(model, f)

    log.info(f"Model saved: {model_path}")
    log.info(f"Top 10 features: {sorted(feature_importance.items(), key=lambda x: -x[1])[:10]}")
    return model, meta


if __name__ == "__main__":
    asyncio.run(train())
