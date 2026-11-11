"""Fixture helper: kronos feature files on the synthetic market using the tiny random-weight model."""

from __future__ import annotations

import os

from stocksense.features import kronos_feats as kf
from stocksense.models import kronos_model as km

# sessions before, on and after the truncation cuts (300, 380, 450, 520, 590)
SESSION_INDICES = (290, 300, 330, 380, 400, 450, 480, 520, 560, 590, 600)


def write_test_kronos_files(settings, sessions) -> list:
    """Compute kr_* files (default tag) with STOCKSENSE_KRONOS_TEST_MODEL=1."""
    old = os.environ.get(km.TEST_MODEL_ENV)
    os.environ[km.TEST_MODEL_ENV] = "1"
    try:
        return kf.compute_kronos_range(
            settings, [sessions[i] for i in SESSION_INDICES], kf.default_tag(), km.load_bundle("small")
        )
    finally:
        if old is None:
            os.environ.pop(km.TEST_MODEL_ENV, None)
        else:
            os.environ[km.TEST_MODEL_ENV] = old
