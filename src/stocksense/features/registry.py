"""
Phase K2.1: the factor registry.

WHY THIS EXISTS. `features/engine.py:_per_symbol_features` and `build_features`
hardcode every feature as a direct assignment -- there is no registry, no
plugin seam, no dict of callables. Adding a new factor today means editing that
function by hand. A search loop that generates and tests candidate factors
needs the opposite: a way to add a factor PROGRAMMATICALLY, register it, and
have it show up in `feature_columns()` without touching engine.py at all.

This module is a PURE ADDITION. `features/engine.py` is not modified in any
way, so Phase 0's reproducibility is preserved by construction -- the two
column sets are simply concatenated by whichever caller wants both.

Hard invariant, inherited unchanged from `features/engine.py`'s own docstring:
every registered factor at row (symbol, date) may use only information dated
<= date. A factor function receives one symbol's own history, already sorted
ascending by date -- exactly `features/engine.py:_per_symbol_features`'s own
contract -- so a `.rolling(n)` window is safe and a `.shift(-k)` is a leakage
bug the same way it always has been in this codebase.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

FactorFn = Callable[[pd.DataFrame], pd.Series]

FACTOR_REGISTRY: dict[str, FactorFn] = {}


def factor(name: str):
    """Decorator registering a factor under `name`.

    `fn(group: pd.DataFrame) -> pd.Series`, called once per symbol's own
    history (already sorted ascending by date, same contract as
    `features/engine.py:_per_symbol_features`). The returned Series must be
    index-aligned with `group` -- i.e. `len(fn(group)) == len(group)`.

    Raises `ValueError` on a duplicate name. A silently shadowed factor is a
    correctness bug: a search loop that later cannot tell which definition of
    "factor_x" actually ran is not reproducible.
    """

    def _register(fn: FactorFn) -> FactorFn:
        if name in FACTOR_REGISTRY:
            raise ValueError(
                f"factor {name!r} is already registered -- pick a different name, "
                "or this would silently shadow an earlier registration"
            )
        FACTOR_REGISTRY[name] = fn
        return fn

    return _register


def apply_registered_factors(candles: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Computes every requested registered factor, grouped per symbol exactly
    like `features/engine.py:build_features` does for its own columns.

    `candles` must have `symbol` and `date` columns, sorted or not (this
    function sorts per group before calling each factor, so callers never
    need to pre-sort). `names=None` (the default) runs every registered
    factor. Returns a frame with `symbol`, `date`, plus one column per
    requested factor name.

    Raises `KeyError` naming any requested factor that was never registered
    -- a silent no-op column would be far more confusing than a clear error at
    the point the caller asked for something that does not exist.
    """
    names = names if names is not None else list(FACTOR_REGISTRY)
    missing = [n for n in names if n not in FACTOR_REGISTRY]
    if missing:
        raise KeyError(f"requested factor(s) not registered: {missing}. Registered: {sorted(FACTOR_REGISTRY)}")

    if candles.empty:
        return pd.DataFrame(columns=["symbol", "date", *names])

    out_frames = []
    for symbol, g in candles.groupby("symbol", sort=False):
        g = g.sort_values("date")
        cols = {"symbol": g["symbol"].values, "date": g["date"].values}
        for n in names:
            values = FACTOR_REGISTRY[n](g)
            if len(values) != len(g):
                raise ValueError(
                    f"factor {n!r} returned {len(values)} values for symbol {symbol!r} "
                    f"with {len(g)} rows -- a factor must return one value per input row"
                )
            cols[n] = pd.Series(values).to_numpy()
        out_frames.append(pd.DataFrame(cols))

    return pd.concat(out_frames, ignore_index=True)


def clear_registry() -> None:
    """Test-only escape hatch -- FACTOR_REGISTRY is module-level state, and
    tests that register throwaway factors must not leak them into other
    tests' assertions about what is/isn't registered."""
    FACTOR_REGISTRY.clear()
