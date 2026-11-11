"""The look-ahead guard (PLAN T4.2): every registered feature, computed with as_of=cut, must equal the
full-history computation on all rows with date <= cut, and must return no row after cut.

Design for speed (the registry will grow to ~130 features): the synthetic market and its panel are built
once per session; each feature is computed once on the full context and once per cut.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from stocksense import panel as panel_pkg
from stocksense.data import io
from stocksense.features.base import FeatureContext, FeatureSpec, feature
from stocksense.features.registry import load_all
from tests.fixtures.kronos import write_test_kronos_files
from tests.fixtures.synth import _settings_for, make_market, write_market

CUT_INDICES = (300, 380, 450, 520, 590)  # positions in the 601-session list
TOL = 1e-9

REGISTRY = load_all()


@pytest.fixture(scope="session")
def truncation_env(tmp_path_factory):
    """(settings, sessions, cache): synthetic market on disk, panel/universe/market built once."""
    root = tmp_path_factory.mktemp("truncation_market")
    m = make_market()
    write_market(m, root)
    s = _settings_for(root)
    panel_pkg.build_all(s)
    write_test_kronos_files(s, m.sessions)  # kr_* read stored files (tiny random-weight model, T7.5c)
    return s, m.sessions, {}


def _keys(spec: FeatureSpec) -> list[str]:
    return ["date", "sid"] if spec.kind == "xs" else ["date"]


def _compute(spec: FeatureSpec, settings, as_of: date | None) -> pl.DataFrame:
    res = spec.fn(FeatureContext(settings, as_of=as_of)).collect()
    return res.select(*_keys(spec), *spec.outputs)


def _full(spec: FeatureSpec, settings, cache: dict) -> pl.DataFrame:
    if spec.name not in cache:
        cache[spec.name] = _compute(spec, settings, None)
    return cache[spec.name]


def violations(spec: FeatureSpec, full: pl.DataFrame, trunc: pl.DataFrame, cut: date) -> list[str]:
    """Why `trunc` (computed with as_of=cut) disagrees with `full` on date <= cut ([] when it agrees)."""
    keys = _keys(spec)
    out: list[str] = []
    if not trunc.is_empty() and trunc["date"].max() > cut:
        out.append(f"returns rows after the cut ({trunc['date'].max()} > {cut})")
    a = full.filter(pl.col("date") <= cut).with_columns(pl.lit(True).alias("__in_full"))
    b = trunc.filter(pl.col("date") <= cut).with_columns(pl.lit(True).alias("__in_trunc"))
    j = a.join(b, on=keys, how="full", coalesce=True, suffix="__t")
    n_only_full = j.filter(pl.col("__in_trunc").is_null()).height
    n_only_trunc = j.filter(pl.col("__in_full").is_null()).height
    if n_only_full or n_only_trunc:
        out.append(f"row sets differ ({n_only_full} only in full, {n_only_trunc} only in truncated)")
    both = j.filter(pl.col("__in_full").is_not_null() & pl.col("__in_trunc").is_not_null())
    for c in spec.outputs:
        x, y = pl.col(c), pl.col(f"{c}__t")
        if full.schema[c].is_float():
            same_val = (x == y) | ((x - y).abs() <= TOL) | (x.is_nan() & y.is_nan())
            bad = (x.is_null() != y.is_null()) | (
                x.is_not_null() & y.is_not_null() & ~same_val.fill_null(False)
            )
        else:
            bad = x.ne_missing(y)
        n_bad = both.filter(bad).height
        if n_bad:
            out.append(f"{c}: {n_bad} of {both.height} rows differ between full and truncated")
    return out


# ---- the guard over the whole registry ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_feature_does_not_look_ahead(name, truncation_env):
    s, sessions, cache = truncation_env
    spec = REGISTRY[name]
    full = _full(spec, s, cache)
    assert full.height > 0, f"{name} produced no rows on the synthetic market"
    for idx in CUT_INDICES:
        cut = sessions[idx]
        bad = violations(spec, full, _compute(spec, s, cut), cut)
        assert not bad, f"{name} @ session {idx} ({cut}): " + "; ".join(bad)


def test_registry_is_not_empty():
    assert {"ret_1", "vol_21", "mkt_ret_1"} <= set(REGISTRY)


# ---- meta-tests: the checker itself must catch leaks -------------------------------------------


def _temp_spec(reg: dict, name: str, fn, kind: str = "xs") -> FeatureSpec:
    feature(name, group="meta", kind=kind, label="a deliberately bad feature", registry=reg)(fn)
    return reg[name]


def _check(spec: FeatureSpec, env) -> list[str]:
    s, sessions, _ = env
    full = _compute(spec, s, None)
    found: list[str] = []
    for idx in CUT_INDICES:
        cut = sessions[idx]
        found += violations(spec, full, _compute(spec, s, cut), cut)
    return found


def test_checker_passes_a_clean_feature(truncation_env):
    reg: dict = {}
    spec = _temp_spec(
        reg,
        "clean",
        lambda ctx: ctx.panel().select(
            "date", "sid", pl.col("ret").rolling_mean(5).over("sid").alias("clean")
        ),
    )
    assert _check(spec, truncation_env) == []


def test_checker_catches_next_day_return(truncation_env):
    """The textbook leak: tomorrow's return."""
    reg: dict = {}
    spec = _temp_spec(
        reg,
        "leaky_next_ret",
        lambda ctx: ctx.panel().select(
            "date", "sid", pl.col("ret").shift(-1).over("sid").alias("leaky_next_ret")
        ),
    )
    found = _check(spec, truncation_env)
    assert found and any("leaky_next_ret" in f for f in found)


def test_checker_catches_full_history_normalisation(truncation_env):
    """A subtler leak: demeaning with a statistic over the whole history."""
    reg: dict = {}
    spec = _temp_spec(
        reg,
        "leaky_global_mean",
        lambda ctx: ctx.panel().select(
            "date", "sid", (pl.col("ret") - pl.col("ret").mean().over("sid")).alias("leaky_global_mean")
        ),
    )
    assert _check(spec, truncation_env)


def test_checker_catches_a_feature_that_ignores_as_of(truncation_env):
    """Reading the store directly instead of through the context returns rows after the cut."""
    s, _, _ = truncation_env
    reg: dict = {}
    spec = _temp_spec(
        reg,
        "leaky_bypass",
        lambda ctx: io.read(s, "panel_daily", columns=["date", "sid", "ret"]).select(
            "date", "sid", pl.col("ret").alias("leaky_bypass")
        ),
    )
    found = _check(spec, truncation_env)
    assert any("after the cut" in f for f in found)


def test_checker_catches_a_leaky_market_feature(truncation_env):
    reg: dict = {}
    spec = _temp_spec(
        reg,
        "leaky_mkt",
        lambda ctx: ctx.market().select("date", pl.col("ew500_ret").shift(-1).alias("leaky_mkt")),
        kind="mkt",
    )
    assert _check(spec, truncation_env)
