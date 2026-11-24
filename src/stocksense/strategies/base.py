"""Strategy registry: one StrategySpec per pre-registered strategy (PLAN-M M5.1).

A per-stock strategy is a function `f -> (signal bool[n], stop float[n])` on a feature dict; cross-sectional
and index strategies carry a `score` or `timing` function. The library modules register their specs on import.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

Kind = Literal["trend", "mr", "breakout", "flow", "event", "xs_factor", "index_timing", "options"]
Source = Literal["marketlab_swing", "marketlab_intraday", "tier", "k151", "flow"]
Family = Literal["m-swing", "m-intra", "m-xs", "m-index", "m-opt"]
LIBRARY_MODULES = (
    "swing",
    "intraday",
    "tiers",
    "k151",
    "k151_app",
    "k151_idx",
    "k151_ann",
    "flowstrats",
    "k151_opt",
)


@dataclass(frozen=True)
class StrategySpec:
    """One strategy: identity, family, timeframe, exits and the function that produces its signals."""

    key: str
    name: str
    source: Source
    tier: Literal["F", "C", "B", "none"]
    family: Family
    timeframe: Literal["1d", "5m", "60m"]
    kind: Kind
    direction: Literal["long", "short", "both"]
    exits: tuple[str, ...]
    default_exit: str
    signal: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]] | None
    score: Callable[..., pl.DataFrame] | None = (
        None  # xs_factor: (settings, start, end) -> (date, sid, score)
    )
    timing: Callable[..., pl.DataFrame] | None = (
        None  # index_timing: (settings, start, end) -> (date, position)
    )
    strength: str = "signal"
    forming: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
    description: str = ""
    reference: str = ""
    status: Literal["ready", "await_data"] = (
        "ready"  # await_data: built and tested, its data is not stored yet
    )


REGISTRY: dict[str, StrategySpec] = {}


def register(spec: StrategySpec) -> StrategySpec:
    """Add a spec; a duplicate key or display name is a programming error."""
    if spec.key in REGISTRY:
        raise KeyError(f"duplicate strategy key {spec.key}")
    if any(s.name == spec.name for s in REGISTRY.values()):
        raise KeyError(f"duplicate strategy name {spec.name}")
    REGISTRY[spec.key] = spec
    return spec


def load_library() -> dict[str, StrategySpec]:
    """Import every library module that exists (each registers its strategies) and return the registry."""
    for mod in LIBRARY_MODULES:
        try:
            importlib.import_module(f"stocksense.strategies.{mod}")
        except ModuleNotFoundError as e:
            if e.name != f"stocksense.strategies.{mod}":
                raise
    return REGISTRY


def by_name_or_key(s: str) -> StrategySpec:
    """The spec with this key or Market Lab display name (case-insensitive on the name)."""
    load_library()
    if s in REGISTRY:
        return REGISTRY[s]
    low = s.lower()
    for spec in REGISTRY.values():
        if spec.name.lower() == low:
            return spec
    raise KeyError(s)


def code_hash() -> str:
    """First 10 hex of the sha1 of the library source text (part of every trial spec)."""
    h = hashlib.sha1()
    here = Path(__file__).parent
    for name in ("indicators", "exits", "costs", "base", *LIBRARY_MODULES):
        p = here / f"{name}.py"
        if p.exists():
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


STRENGTHS: dict[str, Callable[[dict[str, Any], np.ndarray], np.ndarray]] = {
    "signal": lambda f, sig: sig.astype(float),
    "breakout_atr": lambda f, sig: np.where(sig, (f["c"] - f.get("hh20", f["c"])) / f["atr"], 0.0),
    "dev_sigma": lambda f, sig: np.where(sig, np.abs(f["c"] - f["vwap"]) / np.maximum(f["atr"], 1e-12), 0.0),
    "rs63": lambda f, sig: np.where(sig, np.nan_to_num(f.get("rs63", np.zeros(f["n"]))), 0.0),
}
