"""
Fine-tunes Kronos foundation model on 26 years of NSE OHLCV data
using the real finetune_csv pipeline from the Kronos repo.

Usage:
    python -m models.kronos.finetune_nse [--ticker RELIANCE] [--model-size base]
                                          [--epochs-tokenizer 3] [--epochs-model 2]
                                          [--batch-size 8]

Strategy:
    1. Exports NSE daily OHLCV from DB to a Kronos-compatible CSV file.
       CSV columns: timestamps, open, close, high, low, volume, amount
       (amount = close × volume — trading turnover, required by Kronos tokenizer)
    2. Generates a config.yaml pointing at pretrained weights and the CSV.
    3. Downloads pretrained Kronos weights from HuggingFace if not already present.
    4. Runs finetune_tokenizer.py then finetune_base_model.py from the Kronos repo,
       which implement the real training loop:
         tokenizer.encode → Kronos.forward → head.compute_loss → AdamW
    5. Saves finetuned weights to models/kronos/weights/nse_finetuned/{exp_name}/

Hardware note (RTX 3050 4GB):
    batch_size=8, lookback_window=90, predict_window=10
    Fine-tuning a PRETRAINED model needs few epochs (3 tokenizer + 2 base) —
    more just re-feeds the same candles and overfits. Full ~26yr history per
    ticker is exported (cap 6500 ≈ trading days in 26yr). One overnight run
    (~8-14 hrs on RTX 3050).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pandas as pd
import yaml

from config import settings
from models.kronos.integration import ensure_kronos, KRONOS_DIR, _MODEL_SPECS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
KRONOS_REPO_DIR   = Path(KRONOS_DIR)
WEIGHTS_DIR       = _HERE / "weights"
PRETRAINED_DIR    = WEIGHTS_DIR / "pretrained"
FINETUNED_DIR     = WEIGHTS_DIR / "nse_finetuned"
CSV_DIR           = WEIGHTS_DIR / "csv_exports"

# HuggingFace model IDs for Kronos base weights
_HF_PREDICTOR  = "NeoQuasar/Kronos-base"
_HF_TOKENIZER  = "NeoQuasar/Kronos-Tokenizer-base"


# ---------------------------------------------------------------------------
# Step 1: Export DB → Kronos CSV
# ---------------------------------------------------------------------------
async def export_nse_csv(
    conn,
    ticker: str | None,
    out_path: Path,
    min_rows: int = 300,
) -> int:
    """
    Export NSE daily OHLCV from DB to a Kronos-compatible CSV.

    If ticker is None, exports ALL tickers with >= min_rows rows,
    sorted chronologically and concatenated (one long time series).
    This is the correct approach for fine-tuning — Kronos learns from
    the full distribution of NSE price action, not a single stock.

    CSV columns: timestamps, open, close, high, low, volume, amount
    amount = close * volume (trading turnover — required by Kronos tokenizer)
    """
    if ticker:
        log.info(f"Exporting {ticker} from DB...")
        rows = await conn.fetch(
            """
            SELECT time, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE ticker = $1 AND close IS NOT NULL
              AND open IS NOT NULL AND volume IS NOT NULL
            ORDER BY time ASC
            """,
            ticker,
        )
        if len(rows) < min_rows:
            raise ValueError(
                f"{ticker} has only {len(rows)} rows (need {min_rows}). "
                "Run fetch_historical first."
            )
        tickers_used = [ticker]
    else:
        # Cap per ticker to the natural ceiling of ~26 years of NSE trading days
        # (~250 sessions/yr × 26 ≈ 6500). No NSE equity has more daily candles than
        # this, so the cap is effectively "full history" while bounding RAM.
        # ~6500 × ~700 deep-history tickers × 7 cols ≈ a few hundred MB — fits 16GB.
        max_rows_per_ticker = 6500
        log.info(
            "Bulk export: tickers with >= %d rows, capped at %d rows each (full ~26yr history)...",
            min_rows, max_rows_per_ticker,
        )
        rows = await conn.fetch(
            """
            SELECT ticker, time, open, high, low, close, volume
            FROM (
                SELECT
                    ticker, time, open, high, low, close, volume,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY time DESC) AS rn
                FROM ohlcv_daily
                WHERE close IS NOT NULL AND open IS NOT NULL AND volume IS NOT NULL
            ) ranked
            WHERE rn <= $2
              AND ticker IN (
                  SELECT ticker
                  FROM ohlcv_daily
                  WHERE close IS NOT NULL AND open IS NOT NULL AND volume IS NOT NULL
                  GROUP BY ticker
                  HAVING COUNT(*) >= $1
              )
            ORDER BY ticker ASC, time ASC
            """,
            min_rows,
            max_rows_per_ticker,
        )
        log.info("Bulk export: %d rows fetched across all tickers", len(rows))
        tickers_used = ["all"]

    # IMPORTANT: keep each ticker's candles contiguous and in time order.
    # CustomKlineDataset slices a fixed-length window (lookback+predict+1) as a
    # single OHLCV sequence, so rows MUST be grouped per ticker — interleaving
    # tickers by date would feed the model meaningless cross-stock windows.
    # The single-ticker branch returns 6 columns; the bulk branch returns 7
    # (with a leading `ticker`) — normalize both to the 6 feature columns,
    # preserving the SQL ordering (do NOT re-sort globally by time).
    multi_ticker = ticker is None
    cols = ["ticker", "time", "open", "high", "low", "close", "volume"] if rows and len(rows[0]) == 7 \
        else ["time", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(rows, columns=cols)
    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])
    df["time"] = pd.to_datetime(df["time"])

    # Derive amount = close * volume (turnover). Kronos requires this column;
    # NSE daily data doesn't store it directly but the product is equivalent.
    df["amount"] = df["close"] * df["volume"]

    # Drop rows with any NaN before assigning timestamps so the synthetic
    # sequence below stays gap-free and length-aligned.
    df = df.dropna().reset_index(drop=True)

    if multi_ticker:
        # CRITICAL: Kronos's CustomKlineDataset re-sorts the CSV by `timestamps`
        # on load (df.sort_values('timestamps')). Real NSE daily candles share
        # calendar dates across tickers, so real timestamps would re-interleave
        # every stock by date and destroy per-ticker sequence contiguity —
        # making each training window a mix of ~100 unrelated stocks.
        #
        # We export tickers in contiguous per-ticker blocks (ORDER BY ticker,
        # time) and overwrite timestamps with a synthetic strictly-monotonic
        # minute sequence. This makes the internal sort a no-op and preserves
        # each ticker's candle order. Trade-off: synthetic timestamps drop real
        # calendar features (weekday/month seasonality), which is an acceptable
        # secondary loss — correct price sequences are what matter.
        # freq='min' keeps millions of rows inside the datetime64[ns] range
        # (year 1677-2262); 'B'/'D' would overflow on a multi-million-row corpus.
        ts = pd.date_range("2000-01-01", periods=len(df), freq="min")
        df["timestamps"] = ts.strftime("%Y/%m/%d %H:%M")
    else:
        # Single instrument — real timestamps are safe and preserve calendar features.
        df["timestamps"] = df["time"].dt.strftime("%Y/%m/%d %H:%M")

    # Reorder to match Kronos expected schema exactly:
    # timestamps, open, close, high, low, volume, amount
    df = df[["timestamps", "open", "close", "high", "low", "volume", "amount"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Exported %d rows to %s", len(df), out_path)
    return len(df)


# ---------------------------------------------------------------------------
# Step 2: Download pretrained weights
# ---------------------------------------------------------------------------
def ensure_pretrained_weights(model_size: str = "base") -> tuple[Path, Path]:
    """
    Download pretrained Kronos tokenizer + predictor from HuggingFace if needed.
    Returns (tokenizer_path, predictor_path).
    """
    specs = {
        "base":  ("NeoQuasar/Kronos-base",       "NeoQuasar/Kronos-Tokenizer-base"),
        "mini":  ("NeoQuasar/Kronos-mini",        "NeoQuasar/Kronos-Tokenizer-2k"),
        "small": ("NeoQuasar/Kronos-small",       "NeoQuasar/Kronos-Tokenizer-base"),
    }
    predictor_hf, tokenizer_hf = specs.get(model_size, specs["base"])

    predictor_path  = PRETRAINED_DIR / f"Kronos-{model_size}"
    tokenizer_path  = PRETRAINED_DIR / f"Kronos-Tokenizer-{model_size}"

    for hf_id, local_path in [(tokenizer_hf, tokenizer_path), (predictor_hf, predictor_path)]:
        if local_path.exists() and any(local_path.iterdir()):
            log.info("Weights already present: %s", local_path)
            continue
        log.info("Downloading %s from HuggingFace → %s", hf_id, local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=hf_id, local_dir=str(local_path))
            log.info("Downloaded %s", hf_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {hf_id}: {e}\n"
                f"Install: pip install huggingface_hub\n"
                f"Or manually: huggingface-cli download {hf_id} --local-dir {local_path}"
            ) from e

    return tokenizer_path, predictor_path


# ---------------------------------------------------------------------------
# Step 3: Generate config.yaml
# ---------------------------------------------------------------------------


def generate_config(
    csv_path: Path,
    tokenizer_path: Path,
    predictor_path: Path,
    finetuned_dir: Path,
    exp_name: str,
    *,
    model_size: str = "base",
    lookback_window: int = 90,
    predict_window: int = 10,
    batch_size: int = 8,
    epochs_tokenizer: int = 3,
    epochs_model: int = 2,
) -> Path:
    """
    Write a Kronos config.yaml tuned for RTX 3050 4GB (single GPU, small batch).
    Returns the path to the written config file.
    """
    # max_context comes from the model spec: base/small=512, mini=2048
    max_context = _MODEL_SPECS.get(model_size, _MODEL_SPECS["base"])[2]
    config = {
        "data": {
            "data_path": str(csv_path),
            "lookback_window": lookback_window,
            "predict_window": predict_window,
            "max_context": max_context,
            "clip": 5.0,
            "train_ratio": 0.9,
            "val_ratio": 0.1,
            "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": epochs_tokenizer,
            "basemodel_epochs": epochs_model,
            "batch_size": batch_size,
            "log_interval": 50,
            "num_workers": 2,        # keep low — 16GB RAM
            "seed": 42,
            "tokenizer_learning_rate": 2e-4,
            "predictor_learning_rate": 1e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 4, # effective batch = 8*4 = 32
        },
        "model_paths": {
            "pretrained_tokenizer": str(tokenizer_path),
            "pretrained_predictor": str(predictor_path),
            "exp_name": exp_name,
            "base_path": str(finetuned_dir),
            "base_save_path": "",    # auto-filled by config_loader
            "finetuned_tokenizer": "",
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": "kronos_nse_finetune",
            "description": f"NSE 26yr daily OHLCV fine-tune — {exp_name}",
            "use_comet": False,
            "train_tokenizer": True,
            "train_basemodel": True,
            "skip_existing": True,   # don't restart tokenizer if already done
            "pre_trained_tokenizer": True,
            "pre_trained_predictor": True,
        },
        "device": {
            "use_cuda": True,
            "device_id": 0,
        },
        "distributed": {
            "use_ddp": False,
        },
    }

    finetuned_dir.mkdir(parents=True, exist_ok=True)
    config_path = finetuned_dir / f"config_{exp_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

    log.info("Config written to %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# Step 4: Run the real Kronos fine-tune scripts
# ---------------------------------------------------------------------------
def run_finetune(config_path: Path) -> None:
    """
    Runs the two-phase Kronos fine-tune:
      Phase A — finetune_tokenizer.py  (learns NSE price vocabulary)
      Phase B — finetune_base_model.py (fine-tunes the predictor)

    Both scripts are in KRONOS_REPO_DIR/finetune_csv/ and must be run
    from that directory so their sys.path.append('../') finds model/.
    """
    ensure_kronos()

    finetune_csv_dir = KRONOS_REPO_DIR / "finetune_csv"
    if not finetune_csv_dir.exists():
        raise RuntimeError(
            f"Kronos finetune_csv/ not found at {finetune_csv_dir}.\n"
            f"Re-clone the repo: git clone https://github.com/shiyu-coder/Kronos.git "
            f'"{KRONOS_REPO_DIR}"'
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(KRONOS_REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    for script_name, phase_label in [
        ("finetune_tokenizer.py",   "Phase A — Tokenizer"),
        ("finetune_base_model.py",  "Phase B — Base model"),
    ]:
        script_path = finetune_csv_dir / script_name
        if not script_path.exists():
            raise RuntimeError(
                f"{script_name} not found at {script_path}. "
                "Check the Kronos repo is fully cloned."
            )

        log.info("=" * 60)
        log.info("Starting %s", phase_label)
        log.info("Script: %s", script_path)
        log.info("Config: %s", config_path)
        log.info("=" * 60)

        result = subprocess.run(
            [sys.executable, str(script_path), "--config", str(config_path)],
            cwd=str(finetune_csv_dir),
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"{phase_label} failed with exit code {result.returncode}.\n"
                f"Check logs in the finetuned_dir/logs/ directory."
            )

        log.info("%s complete.", phase_label)

    log.info("=" * 60)
    log.info("Fine-tuning complete.")
    log.info(
        "Finetuned weights are in: %s",
        FINETUNED_DIR,
    )
    log.info(
        "To use them: update KRONOS_WEIGHTS_DIR in integration.py "
        "to point at the basemodel/best_model subdirectory, "
        "or set env var NSE_KRONOS_WEIGHTS_DIR."
    )
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def run(
    ticker: str | None = None,
    model_size: str = settings.KRONOS_MODEL_SIZE,
    epochs_tokenizer: int = 3,
    epochs_model: int = 2,
    batch_size: int = 8,
    lookback_window: int = 90,
    predict_window: int = 10,
) -> None:
    exp_name = f"nse_{ticker or 'all'}_{model_size}"
    csv_path = CSV_DIR / f"{exp_name}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1_000_000:
        log.info("Reusing cached CSV: %s (%.0f MB)", csv_path, csv_path.stat().st_size / 1e6)
        n_rows = lookback_window + predict_window + 10  # known-good, size check is sufficient
    else:
        log.info("Connecting to DB...")
        pool = await asyncpg.create_pool(settings.DATABASE_DSN, min_size=1, max_size=3)
        try:
            async with pool.acquire() as conn:
                n_rows = await export_nse_csv(conn, ticker, csv_path)
        finally:
            await pool.close()

    if n_rows < lookback_window + predict_window + 1:
        log.error(
            "Only %d rows exported — need at least %d. "
            "Run data pipeline first.",
            n_rows, lookback_window + predict_window + 1,
        )
        return

    tokenizer_path, predictor_path = ensure_pretrained_weights(model_size)

    config_path = generate_config(
        csv_path=csv_path,
        tokenizer_path=tokenizer_path,
        predictor_path=predictor_path,
        finetuned_dir=FINETUNED_DIR,
        exp_name=exp_name,
        model_size=model_size,
        lookback_window=lookback_window,
        predict_window=predict_window,
        batch_size=batch_size,
        epochs_tokenizer=epochs_tokenizer,
        epochs_model=epochs_model,
    )

    run_finetune(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Kronos on NSE OHLCV data using the real finetune_csv pipeline"
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Single NSE ticker to fine-tune on (default: all tickers with >=300 rows)"
    )
    parser.add_argument(
        "--model-size", default=settings.KRONOS_MODEL_SIZE, choices=["mini", "small", "base"],
        help=f"Kronos model variant (default: {settings.KRONOS_MODEL_SIZE} from KRONOS_MODEL_SIZE env)"
    )
    parser.add_argument(
        "--epochs-tokenizer", type=int, default=3,
        help="Tokenizer training epochs (default: 3 — fine-tuning a pretrained model "
             "needs few passes; more overfits and repeats data)"
    )
    parser.add_argument(
        "--epochs-model", type=int, default=2,
        help="Base model fine-tune epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size — keep at 8 for RTX 3050 4GB (default: 8)"
    )
    parser.add_argument(
        "--lookback", type=int, default=90,
        help="Lookback window in candles (default: 90)"
    )
    parser.add_argument(
        "--predict", type=int, default=10,
        help="Predict window in candles (default: 10)"
    )
    args = parser.parse_args()

    asyncio.run(run(
        ticker=args.ticker,
        model_size=args.model_size,
        epochs_tokenizer=args.epochs_tokenizer,
        epochs_model=args.epochs_model,
        batch_size=args.batch_size,
        lookback_window=args.lookback,
        predict_window=args.predict,
    ))
