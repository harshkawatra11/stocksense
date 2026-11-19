"""G2-C5: the g2-s1cost grid end to end on the synthetic market (registered first, resumable)."""

from __future__ import annotations

import pytest

from stocksense.core.state import State
from stocksense.validation import g1, g1_run
from stocksense.validation.g1 import G1Config
from stocksense.validation.gate import load_candidate_result
from tests.fixtures.g1_world import build_g1_world, run_model_trials

pytestmark = pytest.mark.integration

CAPITALS = (100_000.0, 500_000.0)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_g1_world(tmp_path_factory.mktemp("s1cost"))


@pytest.fixture(scope="module")
def state(world):
    st = State(world.settings)
    st.migrate()
    return st


@pytest.fixture(scope="module")
def models(world, state):
    return run_model_trials(world, state)


@pytest.fixture(scope="module")
def outcome(world, state, models):
    cfg = G1Config(
        start=world.folds[0].test_start, capitals=CAPITALS, reference_capital=500_000.0,
        folds=tuple(world.folds),
    )  # fmt: skip
    return g1_run.run_s1_cost(
        world.settings, state, world.calendar, cfg, targets=("y_c2o_10",), k_max=10, overlay=False
    )


def test_eight_variants_registered_in_g2_s1cost_and_finished(outcome, state):
    assert [p.variant for p in outcome.plans] == [f"V{i}" for i in range(1, 9)]
    for p in outcome.plans:
        assert p.plan.trial_id.startswith("g2-s1cost-y_c2o_10-")
        assert outcome.results[p.plan.trial_id].data["by_capital"]["500000"]["n_days"] > 0
    assert g1.FAMILY_S1COST == "g2-s1cost"


def test_the_best_net_sharpe_is_stored_as_the_candidate(outcome, world):
    best = max(outcome.results.values(), key=lambda r: (r.ref_sharpe, r.trial_id))
    assert outcome.candidate == best.trial_id
    assert load_candidate_result(outcome.candidate, world.settings).candidate_id == outcome.candidate


def test_a_rerun_skips_finished_trials(outcome, world, state):
    cfg = G1Config(
        start=world.folds[0].test_start, capitals=CAPITALS, reference_capital=500_000.0,
        folds=tuple(world.folds),
    )  # fmt: skip
    again = g1_run.run_s1_cost(
        world.settings, state, world.calendar, cfg, targets=("y_c2o_10",), k_max=10, overlay=False
    )
    assert [p.plan.trial_id for p in again.plans] == [p.plan.trial_id for p in outcome.plans]
    assert again.candidate == outcome.candidate
