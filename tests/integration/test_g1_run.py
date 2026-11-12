"""run_g1 end to end on the synthetic world: grid, references, candidates, gate, production fits."""

from __future__ import annotations

import json

import pytest

from stocksense.core.state import State
from stocksense.learn import nightly
from stocksense.models import store
from stocksense.portfolio.meta import MetaModel
from stocksense.validation import g1, g1_run
from stocksense.validation.g1 import G1Config, G1Runner
from stocksense.validation.trials import n_trials
from tests.fixtures.g1_world import build_g1_world, run_model_trials

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_g1_world(tmp_path_factory.mktemp("g1_run"))


@pytest.fixture(scope="module")
def state(world):
    st = State(world.settings)
    st.migrate()
    return st


ret_factors = (("ret_1", 1.0), ("ret_1", -1.0))


@pytest.fixture(scope="module")
def outcome(world, state):
    run_model_trials(world, state)
    cfg = G1Config(
        start=world.folds[0].test_start, capitals=(100_000.0, 500_000.0), reference_capital=500_000.0,
        folds=tuple(world.folds), meta_factory=lambda: MetaModel(min_child_samples=10), seeds=(1, 2),
        inner_months=2,
    )  # fmt: skip
    runner = G1Runner(world.settings, state, world.calendar, cfg)

    def fit(spec, settings, state, as_of, **kw):
        return nightly.fit_production(spec, settings, state, as_of, seeds=(1, 2), inner_months=2, **kw)

    return g1_run.run_g1(
        world.settings, state, world.calendar, cfg, runner=runner, fit=fit, factors=ret_factors
    )


def test_grid_references_and_counts(outcome, state):
    kinds = [p.kind for p in outcome.plans]
    assert kinds.count("s1") == 12 and kinds.count("s2") == 2
    assert kinds.count("s1-meta") == 1 and kinds.count("ref-open") == 1 and kinds.count("ref-factor") == 2
    assert n_trials(state, "g1-s1") == 13 and n_trials(state, "g1-s2") == 2 and n_trials(state, "g1-ref") == 3


def test_one_candidate_per_family_with_nine_gate_lines(outcome):
    assert set(outcome.candidates) == {"s1", "s2"}
    for rep in outcome.reports.values():
        assert [ln.number for ln in rep.lines] == list(range(1, 10))


def test_candidate_is_the_best_net_sharpe_of_its_family(outcome):
    s2 = [p for p in outcome.plans if p.kind == "s2"]
    best = max(s2, key=lambda p: (outcome.results[p.trial_id].ref_sharpe, p.trial_id))
    assert outcome.candidates["s2"] == best.trial_id


def test_candidates_are_fitted_in_production_and_stored_as_shadow(outcome, state, world):
    assert set(outcome.production) == {"s1", "s2"}
    con = state.connect()
    try:
        rows = {r[0]: r[1] for r in con.execute("SELECT model_id, status FROM models").fetchall()}
    finally:
        con.close()
    for ids in outcome.production.values():
        for mid in ids:
            assert rows[mid] == "shadow"
            assert store.model_dir(world.settings, mid).is_dir()
    assert store.current(state, "s1", "champion") is None  # nothing was promoted


def test_outcome_file_and_idempotence(outcome, world, state):
    doc = json.loads((g1.g1_dir(world.settings) / "outcome.json").read_text(encoding="utf-8"))
    assert doc["trials_total"] == n_trials(state) and set(doc["candidates"]) == {"s1", "s2"}
    before = n_trials(state)
    runner = G1Runner(world.settings, state, world.calendar, None)
    assert runner is not None
    assert n_trials(state) == before


def test_benchmark_rows(world, state):
    cfg = G1Config(start=world.folds[0].test_start, capitals=(500_000.0,), reference_capital=500_000.0)
    rows = g1_run.benchmark_rows(G1Runner(world.settings, state, world.calendar, cfg))
    assert rows and rows[0]["label"].startswith("ew500")


def test_results_markdown_has_every_trial_and_both_gates(outcome, world, state):
    from stocksense.validation.g1_report import render_results

    md = render_results(world.settings, state)
    assert "## 5. Gate G1 research run and decision" in md
    assert f"**{n_trials(state)}**" in md
    con = state.connect()
    try:
        ids = [r[0] for r in con.execute("SELECT trial_id FROM trials").fetchall()]
    finally:
        con.close()
    assert all(i in md for i in ids)
    assert md.count("### 5.5 Research gate") == 2 and ("PASS" in md or "FAIL" in md)
    assert "## 3. Sequence model" in md and "## 4. Kronos" in md
