"""
One-off resume script: finish the 3-seed ensemble + quantile regressors after
the classic single-model retrain already succeeded and was saved (see
logs/retrain_20260712_1447.log — CV AUC 0.6355, test AUC 0.6385, saved as
lgbm_20260712_1553.pkl / lgbm_latest.pkl) but the ensemble step OOM-killed
the process before it could run (train_ensemble_seeds held all 3 fitted
classifiers in memory alongside the full feature matrix — fixed in
models/ml/train.py to save-as-you-go, but re-running the whole of train()
would burn another ~10min data load + ~50min classic retrain for no reason
since that artifact is already good).

This script reproduces train()'s exact data load + 80/20/90 split, then
calls the now-fixed (incremental-save, memory-freeing) train_ensemble_seeds
and train_quantile_models directly. Not meant to be a permanent code path —
delete once the ensemble/quantile artifacts exist; future retrains just use
`python -m models.ml.train`, which now does this correctly in one pass.

Run: python -m models.ml.train_ensemble_resume
"""
import asyncio
import gc
import logging

import asyncpg

from config import settings
from models.ml.train import (
    MODEL_SAVE_DIR,
    load_and_build_features,
    train_ensemble_seeds,
    train_quantile_models,
    QUANTILE_HORIZON_DAYS,
    QUANTILE_ALPHAS,
    ENSEMBLE_SEEDS,
)
from data.pipeline.feature_engineering import get_all_feature_columns
import json
import os
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Must match the version of the already-saved classic model this ensemble
# belongs to, so lgbm_ensemble_meta.json's version lines up with lgbm_latest.
EXISTING_VERSION = "20260712_1553"


async def main():
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    log.info("Loading training data from DB (batched) — same as train()...")
    X, y, combined = await load_and_build_features(conn)
    log.info(f"Feature matrix: {X.shape}, class balance: {y.mean():.3f}")

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / (n_pos + 1e-9)
    log.info(f"scale_pos_weight={scale_pos_weight:.2f}")

    # Reproduce train()'s exact split_idx / val_split derivation.
    split_idx = int(len(X) * 0.8)
    X_full_train = X.iloc[:split_idx]
    val_split = int(len(X_full_train) * 0.9)
    X_ftr, X_fval = X_full_train.iloc[:val_split], X_full_train.iloc[val_split:]
    y_full_train = y.iloc[:split_idx]
    y_ftr, y_fval = y_full_train.iloc[:val_split], y_full_train.iloc[val_split:]

    del X, X_full_train, y_full_train
    gc.collect()

    ensemble_paths = train_ensemble_seeds(
        X_ftr, y_ftr, X_fval, y_fval, scale_pos_weight,
        save_dir=MODEL_SAVE_DIR, version=EXISTING_VERSION,
    )
    log.info(f"Ensemble saved: {ensemble_paths}")

    feature_cols = get_all_feature_columns()
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
        "version": EXISTING_VERSION,
        "seeds": ENSEMBLE_SEEDS,
        "quantile_horizon_days": QUANTILE_HORIZON_DAYS,
        "quantile_alphas": QUANTILE_ALPHAS,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL_SAVE_DIR, "lgbm_ensemble_meta.json"), "w") as f:
        json.dump(ensemble_meta, f, indent=2)
    log.info("lgbm_ensemble_meta.json updated.")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
