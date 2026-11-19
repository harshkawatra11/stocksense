"""fit_production (the production-fit piece of T11.1): no lockbox reads, no label leakage, idempotent."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from stocksense.core.errors import LockboxError
from stocksense.core.state import State
from stocksense.learn import nightly
from stocksense.models import store
from stocksense.models.base import ModelSpec
from tests.fixtures.g1_world import LOCKBOX_START, build_g1_world

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_g1_world(tmp_path_factory.mktemp("nightly"))


@pytest.fixture(scope="module")
def state(world):
    st = State(world.settings)
    st.migrate()
    return st


SPEC = ModelSpec(family="s1-ridge", target="y_c2o_5", universe="u500", entry="close")


def test_cutoff_is_five_sessions_before_as_of(world):
    as_of = world.calendar.prev_session(LOCKBOX_START)
    cut = nightly.production_cutoff(world.calendar, as_of)
    assert world.calendar.shift(cut, 5) == as_of


def test_training_rows_end_before_the_cutoff(world):
    as_of = world.calendar.prev_session(LOCKBOX_START)
    rows, names, kinds, cutoff = nightly.production_training_rows(SPEC, world.settings, as_of, world.calendar)
    assert rows.height > 0 and len(names) == len(kinds)
    end_col = nightly.END_COLUMN["y_c2o_5"]
    assert rows[end_col].max() < cutoff
    assert rows["date"].max() < as_of


def test_lockbox_guard(world, state):
    with pytest.raises(LockboxError):
        nightly.fit_production(
            SPEC, world.settings, state, world.calendar.next_session(LOCKBOX_START), calendar=world.calendar
        )


def test_fit_saves_a_bag_and_is_idempotent(world, state):
    as_of = world.calendar.prev_session(LOCKBOX_START)
    mid = nightly.fit_production(
        SPEC, world.settings, state, as_of, seeds=(1, 2), calendar=world.calendar, inner_months=2
    )
    assert (
        nightly.fit_production(
            SPEC, world.settings, state, as_of, seeds=(1, 2), calendar=world.calendar, inner_months=2
        )
        == mid
    )
    bag = nightly.load_bag(world.settings, mid)
    assert len(bag.members) == 2
    single = store.load_model(world.settings, mid)
    X = np.random.default_rng(0).normal(
        size=(
            30,
            len(
                json.loads((store.model_dir(world.settings, mid) / "feature_manifest.json").read_text())[
                    "features"
                ]
            ),
        )
    )
    p = bag.predict(X)
    assert p.shape == (30,) and np.all(np.abs(p) <= 0.5 + 1e-9) and np.isfinite(p).all()
    assert single is not None
    con = state.connect()
    try:
        n = con.execute("SELECT count(*) FROM models WHERE model_id=?", (mid,)).fetchone()[0]
    finally:
        con.close()
    assert n == 1
    assert pl.Series(p).n_unique() > 1


def test_last_research_session_is_before_the_lockbox(world):
    assert nightly.last_research_session(world.settings, world.calendar) < LOCKBOX_START


def test_fit_production_appends_keys_for_needs_keys_models(world, state):
    from stocksense.models.base import FitInfo, Model

    seen: list[int] = []

    class Probe(Model):
        needs_keys = True
        target_prefix = "r_"

        def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0):
            seen.append(X.shape[1])
            return FitInfo(best_iteration=1)

        def refit_full(self, X, y, w, dates, fit_info, seed):
            seen.append(X.shape[1])

        def predict(self, X):
            return np.zeros(len(X))

        def save(self, dir):
            dir.mkdir(parents=True, exist_ok=True)
            (dir / "weights.bin").write_bytes(b"0")

        def load(self, dir):
            pass

    _, names, _, _ = nightly.production_training_rows(
        SPEC, world.settings, world.calendar.prev_session(LOCKBOX_START), world.calendar
    )
    spec = ModelSpec(family="s1-probekeys", target="y_c2o_5", universe="u500", entry="close")
    nightly.fit_production(
        spec, world.settings, state, world.calendar.prev_session(LOCKBOX_START), seeds=(1,),
        calendar=world.calendar, inner_months=2, model_factory=lambda s: Probe(),
    )  # fmt: skip
    assert seen and all(n == len(names) + 2 for n in seen)
