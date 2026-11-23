from __future__ import annotations

import json
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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--write-web-fixtures",
        action="store_true",
        default=False,
        help="API tests write their JSON responses to web/src/test/fixtures/",
    )


@pytest.fixture
def web_fixture(request: pytest.FixtureRequest):
    """Call web_fixture(name, payload) in an API test; writes only when --write-web-fixtures is given."""

    def _write(name: str, payload) -> None:
        if request.config.getoption("--write-web-fixtures"):
            out = _REPO / "web" / "src" / "test" / "fixtures"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    return _write
