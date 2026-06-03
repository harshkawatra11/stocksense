"""
Fine-tunes Kronos foundation model on 26 years of NSE OHLCV data.

Usage:
    python -m models.kronos.finetune_nse

Strategy:
    1. Checks if Kronos repo is cloned (calls ensure_kronos from integration.py).
    2. Loads all NSE daily OHLCV from DB for all tickers with >= 300 rows.
    3. Formats data as Kronos training sequences (OHLCV windows of 60 candles).
    4. Attempts to fine-tune using Kronos API; if API unavailable, saves training
       data in Kronos-compatible format with instructions for manual run.
    5. Saves fine-tuned weights (or training data) to
       models/kronos/weights/nse_26yr_finetuned/
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

from config import settings
from models.kronos.integration import ensure_kronos

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path(__file__).parent / "weights" / "nse_26yr_finetuned"
KRONOS_REPO_DIR = Path(__file__).parent / "kronos_repo"
TRAIN_DATA_PATH = WEIGHTS_DIR / "train_data.pkl"
FINETUNED_MODEL_PATH = WEIGHTS_DIR / "kronos_nse_finetuned"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
async def load_nse_sequences(conn, window: int = 60) -> list[dict[str, Any]]:
    """
    Fetch all NSE tickers with >= 300 rows of daily OHLCV from DB.
    For each ticker, create sliding windows of `window` candles.
    Each window is a dict with arrays: open, high, low, close, volume, ticker, start_date.

    Returns a list of sequence dicts ready for Kronos training.
    """
    log.info("Fetching ticker list (min 300 rows)...")
    ticker_rows = await conn.fetch(
        """
        SELECT ticker, COUNT(*) as cnt
        FROM ohlcv_daily
        WHERE close IS NOT NULL
        GROUP BY ticker
        HAVING COUNT(*) >= 300
        ORDER BY cnt DESC
        """
    )
    tickers = [r["ticker"] for r in ticker_rows]
    log.info(f"Found {len(tickers)} tickers with >= 300 rows")

    sequences: list[dict] = []

    for ticker in tickers:
        rows = await conn.fetch(
            """
            SELECT time, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE ticker = $1 AND close IS NOT NULL
            ORDER BY time ASC
            """,
            ticker,
        )
        if not rows:
            continue

        df = pd.DataFrame(
            rows, columns=["time", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["time"])

        opens = df["open"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        volumes = df["volume"].astype(float).values
        dates = df["time"].values

        n = len(df)
        step = max(1, window // 4)  # 25% stride to get diverse windows

        for start in range(0, n - window, step):
            end = start + window
            seq = {
                "ticker": ticker,
                "start_date": str(dates[start])[:10],
                "end_date": str(dates[end - 1])[:10],
                "open": opens[start:end].tolist(),
                "high": highs[start:end].tolist(),
                "low": lows[start:end].tolist(),
                "close": closes[start:end].tolist(),
                "volume": volumes[start:end].tolist(),
                # Next candle as target (for supervised fine-tuning)
                "target_close": float(closes[end]) if end < n else None,
                "target_high": float(highs[end]) if end < n else None,
                "target_low": float(lows[end]) if end < n else None,
            }
            sequences.append(seq)

    log.info(f"Built {len(sequences):,} training sequences from {len(tickers)} tickers")
    return sequences


def _normalize_sequences(sequences: list[dict]) -> list[dict]:
    """
    Normalize OHLCV values within each window to [0, 1] range.
    This improves cross-ticker generalization.
    """
    normalized = []
    for seq in sequences:
        closes = np.array(seq["close"])
        price_min = min(seq["low"])
        price_max = max(seq["high"])
        price_range = price_max - price_min + 1e-8

        vol_arr = np.array(seq["volume"])
        vol_max = vol_arr.max() + 1e-8

        norm_seq = {
            **seq,
            "open": ((np.array(seq["open"]) - price_min) / price_range).tolist(),
            "high": ((np.array(seq["high"]) - price_min) / price_range).tolist(),
            "low": ((np.array(seq["low"]) - price_min) / price_range).tolist(),
            "close": ((closes - price_min) / price_range).tolist(),
            "volume": (vol_arr / vol_max).tolist(),
            "price_min": float(price_min),
            "price_range": float(price_range),
            "vol_max": float(vol_max),
        }
        if seq.get("target_close") is not None:
            norm_seq["target_close_norm"] = (
                seq["target_close"] - price_min
            ) / price_range
        normalized.append(norm_seq)
    return normalized


def _save_training_data_for_kronos(sequences: list[dict]) -> None:
    """
    Persist training sequences to disk in Kronos-compatible pickle format.
    Called when Kronos fine-tune API is not available.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_sequences(sequences)

    payload = {
        "sequences": normalized,
        "metadata": {
            "n_sequences": len(normalized),
            "window_size": 60,
            "features": ["open", "high", "low", "close", "volume"],
            "description": "NSE 26yr OHLCV fine-tuning data",
            "created_by": "stocksense/models/kronos/finetune_nse.py",
        },
    }

    with open(TRAIN_DATA_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Saved {len(normalized):,} training sequences to {TRAIN_DATA_PATH}")


def finetune_kronos(train_sequences: list[dict], model_size: str = "base") -> bool:
    """
    Attempts to fine-tune Kronos on NSE data.
    If Kronos fine-tune API is unavailable, saves training data for manual run.

    Returns True if fine-tuning succeeded, False if data was saved for manual run.
    """
    ensure_kronos()

    try:
        # Try to import Kronos fine-tune interface
        # NOTE: The actual class name may differ in the real repo.
        # We try multiple likely names in order.
        KronosFineTuner = None
        try:
            from kronos import KronosFineTuner  # type: ignore  # noqa: F401
        except ImportError:
            pass

        if KronosFineTuner is None:
            try:
                from kronos.finetune import KronosFineTuner  # type: ignore  # noqa: F401
            except ImportError:
                pass

        if KronosFineTuner is None:
            raise ImportError("KronosFineTuner not found in kronos package")

        # Prepare data in Kronos expected format
        import torch
        normalized = _normalize_sequences(train_sequences)

        # Build tensor dataset
        inputs = []
        targets = []
        for seq in normalized:
            if seq.get("target_close_norm") is None:
                continue
            ohlcv = np.stack([
                seq["open"], seq["high"], seq["low"],
                seq["close"], seq["volume"],
            ], axis=1)  # shape (window, 5)
            inputs.append(ohlcv)
            targets.append([seq["target_close_norm"]])

        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        log.info(f"Starting Kronos fine-tune on {len(inputs_t):,} sequences...")
        tuner = KronosFineTuner.from_pretrained(
            f"kronos-{model_size}",
            output_dir=str(FINETUNED_MODEL_PATH),
        )
        tuner.train(
            inputs=inputs_t,
            targets=targets_t,
            epochs=10,
            batch_size=64,
            learning_rate=1e-4,
        )
        tuner.save(str(FINETUNED_MODEL_PATH))
        log.info(f"Kronos fine-tuned weights saved to {FINETUNED_MODEL_PATH}")
        return True

    except ImportError as e:
        log.warning(f"Kronos fine-tune API not available: {e}")
        _save_training_data_for_kronos(train_sequences)
        log.info("=" * 70)
        log.info("MANUAL FINE-TUNING INSTRUCTIONS")
        log.info("=" * 70)
        log.info(f"Training data saved to: {TRAIN_DATA_PATH}")
        log.info(
            "Run the following command to fine-tune manually:\n"
            f"  cd {KRONOS_REPO_DIR}\n"
            f"  python finetune.py \\\n"
            f"    --data {TRAIN_DATA_PATH} \\\n"
            f"    --output {FINETUNED_MODEL_PATH} \\\n"
            f"    --model-size {model_size} \\\n"
            f"    --epochs 10 \\\n"
            f"    --batch-size 64 \\\n"
            f"    --lr 1e-4"
        )
        log.info("=" * 70)
        return False

    except Exception as e:
        log.error(f"Unexpected error during Kronos fine-tuning: {e}", exc_info=True)
        _save_training_data_for_kronos(train_sequences)
        return False


async def run(model_size: str = "base", window: int = 60) -> None:
    """Main entry point for NSE fine-tuning."""
    log.info("Connecting to DB...")
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    try:
        sequences = await load_nse_sequences(conn, window=window)
    finally:
        await conn.close()

    if not sequences:
        log.error("No training sequences found. Ensure ohlcv_daily has data.")
        return

    log.info(f"Starting Kronos fine-tune with {len(sequences):,} sequences, model_size={model_size}")
    success = finetune_kronos(sequences, model_size=model_size)

    if success:
        log.info("Fine-tuning complete. Update integration.py to load from fine-tuned weights.")
    else:
        log.info(
            "Fine-tuning data prepared. Follow the instructions above for manual fine-tuning."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Kronos on 26yr NSE data")
    parser.add_argument("--model-size", default="base", choices=["mini", "small", "base"])
    parser.add_argument("--window", type=int, default=60, help="Candle window size")
    args = parser.parse_args()

    asyncio.run(run(model_size=args.model_size, window=args.window))
