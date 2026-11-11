"""The feature registry: every registered FeatureSpec, keyed by name (PLAN T4.1)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stocksense.features.base import FeatureSpec

REGISTRY: dict[str, FeatureSpec] = {}

# Feature modules, imported by load_all(). A module that does not exist yet is skipped, so the library
# tasks (T4.3a..k) can land one by one. `seed` holds the framework's first three features.
FEATURE_MODULES: tuple[str, ...] = (
    "seed",
    "price",
    "overnight",
    "risk",
    "volume",
    "delivery",
    "market",
    "calendar_feats",
    "derivs",
    "events",
    "intraday",
)


def register(spec: FeatureSpec, registry: dict[str, FeatureSpec] | None = None) -> None:
    """Add `spec` to a registry (default: the global one); names and output names must be unique."""
    reg = REGISTRY if registry is None else registry
    taken = {o for s in reg.values() for o in s.outputs}
    for o in spec.outputs:
        if o in taken:
            raise ValueError(f"feature output {o!r} already registered")
    if spec.name in reg:
        raise ValueError(f"feature {spec.name!r} already registered")
    reg[spec.name] = spec


def load_all() -> dict[str, FeatureSpec]:
    """Import every feature module (registering its features) and return the global registry."""
    for mod in FEATURE_MODULES:
        full = f"stocksense.features.{mod}"
        try:
            importlib.import_module(full)
        except ModuleNotFoundError as e:
            if e.name != full:
                raise
    return REGISTRY
