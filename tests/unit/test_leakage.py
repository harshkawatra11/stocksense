"""
Leakage tests — the highest-value test surface in the codebase (per the
build plan). A feature engine that can see the future produces beautiful
backtests and worthless live predictions, and no other test catches it.

Two independent proofs are used:

1. TRUNCATION INVARIANCE: a feature's value on date D, computed from
   history ending on date D, must be identical to its value on date D
   when computed from a longer history that extends past D. If truncating
   the future changes a past feature value, that feature depends on data
   that would not exist yet — the textbook lookahead bug.

2. LABEL PURITY: the label module is the only place forward-looking
   operations are permitted. Feature column names must never include a
   label column, and the feature engine module must not import the label
   module (a structural guarantee, not just a naming convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels


def _synthetic_candles(n_symbols: int = 5, n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV — no network dependency, so this test
    suite runs offline and reproducibly."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for s in range(n_symbols):
        symbol = f"SYN{s}"
        price = 100.0 + s * 10
        for d in dates:
            ret = rng.normal(0.0005, 0.015)
            price *= 1 + ret
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            open_ = price * (1 + rng.normal(0, 0.003))
            vol = abs(rng.normal(1_000_000, 200_000))
            rows.append(
                {
                    "symbol": symbol,
                    "date": d,
                    "open": open_,
                    "high": max(high, open_, price),
                    "low": min(low, open_, price),
                    "close": price,
                    "adj_close": price,
                    "volume": vol,
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_candles() -> pd.DataFrame:
    return _synthetic_candles()


def test_truncation_invariance(synthetic_candles: pd.DataFrame) -> None:
    """Feature values at date D must not change when future rows (after D)
    are removed from the input — the core point-in-time correctness proof
    demanded by docs/03-feature-engineering.md and docs/10-evaluation.md."""
    full = synthetic_candles
    dates = sorted(full["date"].unique())
    cutoff = dates[len(dates) // 2]  # truncate at the halfway point

    feats_full = build_features(full)
    truncated = full[full["date"] <= cutoff]
    feats_truncated = build_features(truncated)

    check_date = dates[len(dates) // 2 - 5]  # well before cutoff, past warm-up window
    fcols = feature_columns(feats_full)

    row_full = feats_full[feats_full["date"] == check_date].sort_values("symbol").reset_index(drop=True)
    row_trunc = feats_truncated[feats_truncated["date"] == check_date].sort_values("symbol").reset_index(drop=True)

    assert len(row_full) == len(row_trunc) > 0

    for col in fcols:
        full_vals = row_full[col].to_numpy(dtype=float)
        trunc_vals = row_trunc[col].to_numpy(dtype=float)
        both_nan = np.isnan(full_vals) & np.isnan(trunc_vals)
        close_enough = np.isclose(full_vals, trunc_vals, equal_nan=False, rtol=1e-9, atol=1e-9)
        ok = both_nan | close_enough
        assert ok.all(), f"leakage detected in feature '{col}': truncating the future changed a past value"


def test_features_never_contain_label_columns(synthetic_candles: pd.DataFrame) -> None:
    """Structural guarantee: no column produced by the feature engine is
    named like a forward-return label, and the two frames stay disjoint
    in their forward-looking columns."""
    feats = build_features(synthetic_candles)
    fcols = set(feature_columns(feats))
    assert not any(c.startswith("fwd_ret_") for c in fcols)

    labeled = add_forward_return_labels(synthetic_candles, horizon_bars=5)
    label_cols = set(labeled.columns) - set(synthetic_candles.columns)
    assert label_cols.isdisjoint(fcols)


def test_feature_engine_module_does_not_import_labels() -> None:
    """The label module is the only place `.shift(-k)` (forward-looking)
    is permitted to exist. Enforced structurally: features.engine must
    have no dependency edge to stocksense.labels."""
    import ast

    import stocksense.features.engine as engine_module

    src = open(engine_module.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(m.startswith("stocksense.labels") for m in imported_modules)

    # Check actual .shift(...) calls for a negative argument (forward shift),
    # not just a textual grep — which would also flag this docstring.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            for arg in node.args:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    pytest.fail("feature engine must never use a forward (negative) shift")
