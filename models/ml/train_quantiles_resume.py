"""
One-off resume: train ONLY the q10/q50/q90 quantile regressors + refresh
lgbm_ensemble_meta.json, for the F&O-featured retrain of 2026-07-12 (version
20260712_2110) that was killed overnight AFTER the classic model and all 3
ensemble seeds had already saved successfully (see logs/retrain_fo_20260712.log
— seeds saved 21:22/21:31/21:41, process died before the quantile step).

Same split derivation as train.py / train_ensemble_resume.py. Delete once the
quantile bundle exists; future retrains use `python -m models.ml.train`.

Run: python -m models.ml.train_quantiles_resume
"""
import asyncio
import gc
import json
import logging
import os
from datetime import datetime

import asyncpg

from config import settings
from models.ml.train import (
    MODEL_SAVE_DIR,
    load_and_build_features,
    train_quantile_models,
    QUANTILE_HORIZON_DAYS,
    QUANTILE_ALPHAS,
    ENSEMBLE_SEEDS,
)
from data.pipeline.feature_engineering import get_all_feature_columns

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# The already-saved classic model + seeds this quantile bundle belongs to.
EXISTING_VERSION = "20260712_2110"


async def main():
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    log.info("Loading training data (batched, F&O wired) — same as train()...")
    X, y, combined = await load_and_build_features(conn)
    await conn.close()
    log.info(f"Feature matrix: {X.shape}")

    split_idx = int(len(X) * 0.8)
    val_split = int(split_idx * 0.9)
    del X, y
    gc.collect()

    feature_cols = get_all_feature_columns()
    q_col = "target_nd_return"
    q_train_df = combined.iloc[:val_split].dropna(subset=[q_col])
    q_val_df = combined.iloc[val_split:split_idx].dropna(subset=[q_col])
    log.info(f"Quantile rows: train={len(q_train_df):,} val={len(q_val_df):,}")

    quantile_bundle_path = os.path.join(MODEL_SAVE_DIR, "lgbm_quantiles.pkl")
    train_quantile_models(
        q_train_df[feature_cols], q_train_df[q_col],
        q_val_df[feature_cols], q_val_df[q_col],
        save_path=quantile_bundle_path,
    )

    ensemble_meta = {
        "version": EXISTING_VERSION,
        "seeds": ENSEMBLE_SEEDS,
        "quantile_horizon_days": QUANTILE_HORIZON_DAYS,
        "quantile_alphas": QUANTILE_ALPHAS,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL_SAVE_DIR, "lgbm_ensemble_meta.json"), "w") as f:
        json.dump(ensemble_meta, f, indent=2)
    log.info("lgbm_ensemble_meta.json updated to version %s.", EXISTING_VERSION)


if __name__ == "__main__":
    asyncio.run(main())
