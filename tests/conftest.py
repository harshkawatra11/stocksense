from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_DATA_SUBDIRS = [
    "parquet", "cache", "news", "panel", "features", "labels", "models", "preds", "state", "reports", "logs",
]


@pytest.fixture
def fixed_clock() -> datetime:
    # Plain tz-aware datetime for now; becomes a core.clock.FixedClock in T1.3.
    return datetime(2026, 11, 12, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    # Returns the temp data root Path with D.4 subfolders; becomes a Settings in T1.2.
    root = tmp_path / "data_store"
    for name in _DATA_SUBDIRS:
        (root / name).mkdir(parents=True)
    return root
