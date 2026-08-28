"""Phase K2.3: the deterministic candidate generator.

No LLM, no network, no model inference anywhere in this module -- the tests
mostly exist to pin down REPRODUCIBILITY (same seed -> same candidates) and
CORRECTNESS of the individual operators, since those are the two properties
the rest of the loop depends on absolutely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.features.registry import FACTOR_REGISTRY, apply_registered_factors, clear_registry
from stocksense.research.search import (
    MAX_DEPTH,
    TS_WINDOWS,
    Candidate,
    generate_candidates,
    register_candidates,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def _candles() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    price = 100.0
    for d in dates:
        rows.append({"symbol": "AAA", "date": d, "ret_5b": price % 7 / 100.0, "rel_volume_20": price % 3})
        price += 1.0
    return pd.DataFrame(rows)


# ---- reproducibility ----


def test_same_seed_produces_identical_candidates() -> None:
    a = generate_candidates(seed=42, primitives=["ret_5b", "rel_volume_20"], n=20)
    b = generate_candidates(seed=42, primitives=["ret_5b", "rel_volume_20"], n=20)
    assert [c.name for c in a] == [c.name for c in b]
    assert [c.expression for c in a] == [c.expression for c in b]


def test_different_seeds_produce_different_orderings() -> None:
    a = generate_candidates(seed=1, primitives=["ret_5b", "rel_volume_20"], n=20)
    b = generate_candidates(seed=2, primitives=["ret_5b", "rel_volume_20"], n=20)
    assert [c.name for c in a] != [c.name for c in b]


def test_n_larger_than_the_space_returns_everything_not_an_error() -> None:
    small = generate_candidates(seed=1, primitives=["ret_5b"], n=10_000)
    exact = generate_candidates(seed=1, primitives=["ret_5b"], n=len(small))
    assert len(small) == len(exact)


# ---- validation ----


def test_rejects_empty_primitives() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_candidates(seed=1, primitives=[], n=5)


def test_rejects_depth_below_one() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        generate_candidates(seed=1, primitives=["ret_5b"], n=5, max_depth=0)


# ---- depth bound ----


def test_no_candidate_exceeds_the_configured_max_depth() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b", "rel_volume_20"], n=10_000, max_depth=MAX_DEPTH)
    assert all(c.depth <= MAX_DEPTH for c in candidates)
    assert any(c.depth == MAX_DEPTH for c in candidates)  # the bound is actually reached, not just respected


# ---- operator correctness, verified by actually computing them ----


def test_rank_candidate_is_bounded_zero_to_one() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b"], n=10_000)
    rank_cand = next(c for c in candidates if c.expression == "rank(ret_5b)")
    names = register_candidates([rank_cand])
    out = apply_registered_factors(_candles(), names=names)
    values = out[names[0]].dropna()
    assert (values >= 0).all() and (values <= 1).all()


def test_zscore_candidate_has_zero_mean() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b"], n=10_000)
    z_cand = next(c for c in candidates if c.expression == "zscore(ret_5b)")
    names = register_candidates([z_cand])
    out = apply_registered_factors(_candles(), names=names)
    # single-symbol frame: zscore is computed over that symbol's own 30-row
    # history via the registry's per-symbol grouping
    assert out[names[0]].mean() == pytest.approx(0.0, abs=1e-9)


def test_neg_candidate_is_the_exact_negation() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b"], n=10_000)
    neg_cand = next(c for c in candidates if c.expression == "neg(ret_5b)")
    names = register_candidates([neg_cand])
    out = apply_registered_factors(_candles(), names=names)
    pd.testing.assert_series_equal(
        out[names[0]].reset_index(drop=True), (-_candles()["ret_5b"]).reset_index(drop=True), check_names=False,
    )


def test_delta_candidate_matches_diff() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b"], n=10_000)
    d_cand = next(c for c in candidates if c.expression == f"delta(ret_5b, {TS_WINDOWS[0]})")
    names = register_candidates([d_cand])
    out = apply_registered_factors(_candles(), names=names)
    expected = _candles()["ret_5b"].diff(TS_WINDOWS[0])
    pd.testing.assert_series_equal(out[names[0]].reset_index(drop=True), expected.reset_index(drop=True), check_names=False)


def test_ratio_candidate_divides_two_distinct_primitives() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b", "rel_volume_20"], n=10_000)
    ratio_cands = [c for c in candidates if "ratio" in c.expression]
    assert ratio_cands  # at least one exists
    for c in ratio_cands:
        assert c.expression.startswith("ratio(")
        p1, p2 = c.expression[len("ratio("):-1].split(", ")
        assert p1 != p2  # never a primitive divided by itself


def test_ratio_handles_zero_denominator_without_crashing() -> None:
    df = pd.DataFrame({"symbol": ["A"] * 3, "date": pd.bdate_range("2024-01-01", periods=3),
                        "p1": [1.0, 2.0, 3.0], "p2": [0.0, 1.0, 0.0]})
    candidates = generate_candidates(seed=1, primitives=["p1", "p2"], n=10_000)
    ratio_cand = next(c for c in candidates if c.expression == "ratio(p1, p2)")
    names = register_candidates([ratio_cand])
    out = apply_registered_factors(df, names=names)
    assert np.isnan(out[names[0]].iloc[0])  # 1/0 -> nan, not inf or a crash
    assert np.isnan(out[names[0]].iloc[2])


# ---- register_candidates / generate_candidates side-effect boundary ----


def test_generate_candidates_does_not_touch_the_registry() -> None:
    generate_candidates(seed=1, primitives=["ret_5b"], n=50)
    assert FACTOR_REGISTRY == {}  # pure -- only register_candidates mutates shared state


def test_register_candidates_returns_names_in_order() -> None:
    candidates = generate_candidates(seed=1, primitives=["ret_5b"], n=5)
    names = register_candidates(candidates)
    assert names == [c.name for c in candidates]
    assert all(n in FACTOR_REGISTRY for n in names)


def test_registering_the_same_seed_batch_twice_raises_on_duplicate() -> None:
    """A search loop that accidentally re-registers a prior batch should fail
    loudly (features/registry.py's own duplicate-name guard), not silently
    keep the old definition."""
    candidates = generate_candidates(seed=1, primitives=["ret_5b"], n=3)
    register_candidates(candidates)
    with pytest.raises(ValueError, match="already registered"):
        register_candidates(candidates)
