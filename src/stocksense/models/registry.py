"""
Model registry: serialized artifacts on disk + one row per version in
DuckDB (docs/02-data-layer.md). Every entry is reproducible from its
recorded manifest — feature schema version, training window,
hyperparameters, seed — per docs/06-retraining-rigor.md's reproducibility
requirement. The Gate (stocksense.evaluation.gate) is the only writer of
`lifecycle_state`; this module only knows how to serialize/deserialize
and record.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from stocksense.core.config import get_settings
from stocksense.data.store import Store
from stocksense.models.ranker import CrossSectionalRanker, RankerConfig

FEATURE_SCHEMA_VERSION = "phase0-v1"  # bump whenever features.engine's output contract changes


def make_model_id(model_type: str, horizon_bars: int, top_n: int, created_at: datetime) -> str:
    return f"{model_type}_h{horizon_bars}_n{top_n}_{created_at.strftime('%Y%m%dT%H%M%S')}"


def register_candidate(
    ranker: CrossSectionalRanker,
    model_type: str,
    horizon_bars: int,
    top_n: int,
    training_start: str,
    training_end: str,
    metrics: dict,
    store: Store,
) -> str:
    """Serialize the trained model to disk, write its manifest row into
    model_registry as a 'candidate' (not yet promoted), and return the
    model_id. Called after training, before the Gate decides anything.
    """
    settings = get_settings()
    created_at = datetime.now(timezone.utc)
    model_id = make_model_id(model_type, horizon_bars, top_n, created_at)

    model_dir = settings.parquet_dir.parent / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "model.joblib"
    joblib.dump(ranker, artifact_path)

    manifest = {
        "model_id": model_id,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": ranker.feature_names_,
        "hyperparameters": asdict(ranker.config),
        "random_seed": ranker.config.random_state,
        "created_at": created_at.isoformat(),
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    store.insert_model_registry_row(
        {
            "model_id": model_id,
            "model_type": model_type,
            "horizon_bars": horizon_bars,
            "top_n": top_n,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_start": training_start,
            "training_end": training_end,
            "hyperparameters_json": json.dumps(asdict(ranker.config)),
            "random_seed": ranker.config.random_state,
            "created_at": created_at,
            "metrics_json": json.dumps(metrics),
            "gate_decision": None,
            "gate_reason": None,
            "lifecycle_state": "candidate",
            "artifact_path": str(artifact_path),
            "promoted_at": None,
            "rolled_back_at": None,
        }
    )
    return model_id


def load_model(artifact_path: str) -> CrossSectionalRanker:
    return joblib.load(artifact_path)
