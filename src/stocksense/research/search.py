"""
Phase K2.3: the deterministic candidate generator.

DELIBERATELY NO LLM HERE. Per the user's own decision on this build order: a
bounded, reproducible search comes first, so the loop's harness (K2.2), its
objective (K0's ICIR/half-life), and its multiplicity guard (K1's vault +
evaluation/attempts.py) are all proven against something whose behaviour is
fully known in advance. Claude is added as a SECOND generator only after this
one's acceptance gate has passed (see research/harness.py's module docstring)
-- adding a non-deterministic generator before the harness is trusted would
make it impossible to tell whether a surprising result came from a real signal
or from a bug in the plumbing.

THE GRAMMAR. A candidate factor is a short expression tree over a small,
fixed operator set applied to existing feature-engine primitives (the columns
`features/engine.py` already computes -- this module does not invent new raw
data, only new combinations of what's already there). Depth is capped at
`MAX_DEPTH` (3) specifically to keep the search space enumerable and each
candidate's economic story nameable -- an unbounded tree is exactly the
"complexity" failure mode AlphaAgent's regularisers exist to penalise (K4);
here it's avoided structurally instead of penalised after the fact.

FULLY REPRODUCIBLE: `generate_candidates(seed, ...)` with the same seed
always enumerates the same expressions in the same order. No network call, no
model inference, nothing else non-deterministic anywhere in this module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

OPERATORS = ("rank", "zscore", "delta", "ts_mean", "ts_std", "ratio", "neg")
"""The bounded operator set. `rank`/`zscore` are CROSS-SECTIONAL (computed
across all symbols on one date); `delta`/`ts_mean`/`ts_std` are TIME-SERIES
(computed within one symbol's own history, over a window in
`TS_WINDOWS`); `ratio` is binary (needs two child expressions); `neg` is a
sign flip, useful for cheaply doubling the search space with the mirror image
of every unary expression without a second primitive."""

TS_WINDOWS = (5, 10, 20)
"""Windows in bars for the time-series operators -- matching the scale of
existing engine.py windows (5/10/20/50/200) rather than inventing new ones."""

MAX_DEPTH = 3
"""Hard cap on expression nesting. Kept small on purpose -- see module
docstring."""


@dataclass(frozen=True)
class Candidate:
    """One generated factor: a name (stable, derived from its own expression
    so two runs of the same seed produce identically-named candidates), the
    primitive it starts from, and the callable to register."""

    name: str
    primitive: str
    depth: int
    expression: str  # human-readable, e.g. "zscore(ts_mean(ret_5b, 10))"
    fn: "callable"


def _apply_unary(op: str, series_fn) -> tuple[str, "callable"]:
    if op == "rank":
        return "rank", lambda g: series_fn(g).rank(pct=True)
    if op == "zscore":
        def _z(g):
            s = series_fn(g)
            std = s.std()
            return (s - s.mean()) / std if std and np.isfinite(std) and std > 0 else s * 0.0
        return "zscore", _z
    if op == "neg":
        return "neg", lambda g: -series_fn(g)
    raise ValueError(f"{op!r} is not a unary operator")


def _apply_ts(op: str, window: int, series_fn) -> tuple[str, "callable"]:
    if op == "delta":
        return f"delta{window}", lambda g: series_fn(g).diff(window)
    if op == "ts_mean":
        return f"ts_mean{window}", lambda g: series_fn(g).rolling(window, min_periods=max(2, window // 2)).mean()
    if op == "ts_std":
        return f"ts_std{window}", lambda g: series_fn(g).rolling(window, min_periods=max(2, window // 2)).std()
    raise ValueError(f"{op!r} is not a time-series operator")


def generate_candidates(
    seed: int,
    primitives: list[str],
    n: int,
    max_depth: int = MAX_DEPTH,
) -> list[Candidate]:
    """Deterministically enumerates up to `n` candidates built from
    `primitives` (existing engine.py column names, e.g. "ret_5b",
    "rel_volume_20"). Same `seed` -> same candidates, in the same order,
    every time -- the search space is enumerated in a fixed canonical order
    and then deterministically shuffled by `seed`, rather than sampled with a
    stateful RNG whose draw sequence could shift if an earlier call pattern
    changes.

    Depth-1 candidates apply one operator directly to a primitive; deeper
    candidates compose operators up to `max_depth`. `ratio` candidates pair
    two DIFFERENT primitives, since a primitive divided by itself is not a
    factor. The full space is enumerated first, then truncated to `n` -- so
    `n` larger than the space size returns everything, never an error.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    if not primitives:
        raise ValueError("primitives must be non-empty")

    all_candidates: list[Candidate] = []

    def _primitive_fn(p: str):
        return lambda g, _p=p: g[_p]

    # Depth 1: one operator directly on one primitive.
    for p in primitives:
        base = _primitive_fn(p)
        for op in ("rank", "zscore", "neg"):
            label, fn = _apply_unary(op, base)
            all_candidates.append(Candidate(
                name=f"{p}__{label}", primitive=p, depth=1,
                expression=f"{op}({p})", fn=fn,
            ))
        for op in ("delta", "ts_mean", "ts_std"):
            for w in TS_WINDOWS:
                label, fn = _apply_ts(op, w, base)
                all_candidates.append(Candidate(
                    name=f"{p}__{label}", primitive=p, depth=1,
                    expression=f"{op}({p}, {w})", fn=fn,
                ))

    # Depth 2+: compose a unary/ts operator on top of an existing depth-(d-1)
    # candidate, up to max_depth. Composing onto EVERY prior candidate would
    # blow up combinatorially with depth; this composes onto the depth-1 set
    # only, which already gives genuinely different shapes (e.g.
    # zscore(ts_mean(ret_5b, 10))) without an unbounded space.
    depth1 = list(all_candidates)
    current_depth = 1
    frontier = depth1
    while current_depth < max_depth:
        next_frontier = []
        for cand in frontier:
            for op in ("rank", "zscore", "neg"):
                label, fn = _apply_unary(op, lambda g, _c=cand: _c.fn(g))
                next_frontier.append(Candidate(
                    name=f"{cand.name}__{label}", primitive=cand.primitive, depth=cand.depth + 1,
                    expression=f"{op}({cand.expression})", fn=fn,
                ))
        all_candidates.extend(next_frontier)
        frontier = next_frontier
        current_depth += 1

    # Binary ratio candidates: every distinct unordered pair of primitives,
    # depth counted as 1 (a single operator over two leaves).
    for p1, p2 in product(primitives, primitives):
        if p1 >= p2:
            continue  # unordered pair, avoid p/p and duplicate (p1,p2)/(p2,p1)
        f1, f2 = _primitive_fn(p1), _primitive_fn(p2)

        def _ratio(g, _f1=f1, _f2=f2):
            denom = _f2(g)
            return _f1(g) / denom.replace(0, np.nan)

        all_candidates.append(Candidate(
            name=f"{p1}__ratio__{p2}", primitive=p1, depth=1,
            expression=f"ratio({p1}, {p2})", fn=_ratio,
        ))

    # Deterministic order, then deterministic shuffle by seed, then truncate.
    all_candidates.sort(key=lambda c: c.name)
    rng = random.Random(seed)
    order = list(range(len(all_candidates)))
    rng.shuffle(order)
    selected = [all_candidates[i] for i in order[:n]]
    return selected


def register_candidates(candidates: list[Candidate]) -> list[str]:
    """Registers every candidate into features.registry.FACTOR_REGISTRY under
    its own name. Returns the list of registered names, in the order given,
    for direct use as `SweepConfig.extra_factor_names`.

    Deliberately a separate function from `generate_candidates` -- generation
    is pure and side-effect-free (so it can be called repeatedly to inspect a
    seed's output without mutating global state); registration is the one
    place that touches the shared registry, and only ever called once per
    candidate batch by the loop driver.
    """
    from stocksense.features.registry import factor

    names = []
    for c in candidates:
        factor(c.name)(c.fn)
        names.append(c.name)
    return names
