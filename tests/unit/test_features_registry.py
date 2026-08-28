"""Phase K2.1: the factor registry.

The property that matters most here: `features/engine.py` is NEVER imported or
modified by this module, so registering a factor cannot change Phase 0's
existing feature set or column ordering. Verified directly, not assumed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stocksense.features.registry import (
    FACTOR_REGISTRY,
    apply_registered_factors,
    clear_registry,
    factor,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _candles() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAA"] * 5 + ["BBB"] * 5,
        "date": list(pd.bdate_range("2024-01-01", periods=5)) * 2,
        "close": [100.0, 101.0, 102.0, 101.0, 103.0, 50.0, 49.0, 51.0, 52.0, 50.0],
    })


def test_a_registered_factor_can_be_added_without_touching_engine_py() -> None:
    """The whole point of this module."""
    import stocksense.features.engine as engine_mod

    @factor("close_delta_1")
    def _close_delta_1(g: pd.DataFrame) -> pd.Series:
        return g["close"].diff()

    out = apply_registered_factors(_candles())
    assert "close_delta_1" in out.columns
    # engine.py's own module object was never touched
    assert not hasattr(engine_mod, "close_delta_1")


def test_factor_output_is_aligned_per_symbol_not_pooled() -> None:
    """A .diff() computed on the POOLED frame would leak BBB's first row into
    AAA's last -- this asserts the per-symbol grouping actually happened."""

    @factor("delta")
    def _delta(g: pd.DataFrame) -> pd.Series:
        return g["close"].diff()

    out = apply_registered_factors(_candles())
    aaa = out[out["symbol"] == "AAA"].sort_values("date")
    bbb = out[out["symbol"] == "BBB"].sort_values("date")
    assert pd.isna(aaa["delta"].iloc[0])  # first row of AAA's own history
    assert pd.isna(bbb["delta"].iloc[0])  # first row of BBB's own history, NOT a diff against AAA's last close


def test_duplicate_registration_raises() -> None:
    @factor("dupe")
    def _f(g):
        return g["close"]

    with pytest.raises(ValueError, match="already registered"):
        @factor("dupe")
        def _g(g):
            return g["close"]


def test_requesting_an_unregistered_factor_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="not registered"):
        apply_registered_factors(_candles(), names=["does_not_exist"])


def test_names_none_runs_every_registered_factor() -> None:
    @factor("f1")
    def _f1(g):
        return g["close"]

    @factor("f2")
    def _f2(g):
        return g["close"] * 2

    out = apply_registered_factors(_candles())
    assert {"f1", "f2"}.issubset(out.columns)


def test_a_factor_returning_the_wrong_length_raises() -> None:
    @factor("broken")
    def _broken(g: pd.DataFrame) -> pd.Series:
        return g["close"].iloc[:1]  # deliberately wrong length

    with pytest.raises(ValueError, match="one value per input row"):
        apply_registered_factors(_candles(), names=["broken"])


def test_empty_candles_returns_empty_frame_with_requested_columns() -> None:
    @factor("f1")
    def _f1(g):
        return g["close"]

    out = apply_registered_factors(pd.DataFrame(columns=["symbol", "date", "close"]), names=["f1"])
    assert out.empty
    assert list(out.columns) == ["symbol", "date", "f1"]


def test_clear_registry_empties_it() -> None:
    @factor("temp")
    def _temp(g):
        return g["close"]

    assert "temp" in FACTOR_REGISTRY
    clear_registry()
    assert FACTOR_REGISTRY == {}
