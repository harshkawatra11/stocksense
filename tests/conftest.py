from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from stocksense.core.clock import IST, FixedClock
from stocksense.core.config import Paths, Settings, load_settings

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_DATA_SUBDIRS = [
    "parquet",
    "cache",
    "news",
    "panel",
    "features",
    "labels",
    "models",
    "preds",
    "state",
    "reports",
    "logs",
]


_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixed_clock() -> FixedClock:
    return FixedClock(datetime(2026, 11, 12, 10, 0, tzinfo=IST))


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Raw temp data root Path with the D.4 subfolders."""
    root = tmp_path / "data_store"
    for name in _DATA_SUBDIRS:
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture
def tmp_data_root(tmp_data_dir: Path, tmp_path: Path) -> Settings:
    """Settings (real config/, no .env) whose data_root is the temp data dir."""
    s = load_settings(_REPO / "config", tmp_path / "no.env")
    return s.model_copy(update={"paths": Paths(data_root=tmp_data_dir)})
