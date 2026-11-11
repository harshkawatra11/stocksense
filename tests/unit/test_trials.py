from __future__ import annotations

from pathlib import Path

import pytest

from stocksense.core.state import State
from stocksense.validation import trials as trials_mod
from stocksense.validation.trials import (
    finish_trial,
    n_trials,
    register_trial,
    sharpe_variance_across_trials,
)


@pytest.fixture
def state(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    return st


def test_migration_2_creates_table_and_seeds_30_once(state):
    state.migrate()  # idempotent: no second batch of seeds
    con = state.connect()
    assert [r[0] for r in con.execute("SELECT version FROM schema_version ORDER BY version")] == [1, 2, 3]
    rows = con.execute("SELECT family, status, note, metrics_json FROM trials").fetchall()
    assert len(rows) == 30
    assert {r[0] for r in rows} == {"planning-exploration"}
    assert {r[1] for r in rows} == {"done"}
    assert {r[2] for r in rows} == {"signals examined on development data while writing the plan, 2026-11-11"}
    assert {r[3] for r in rows} == {None}
    assert n_trials(state) == 30
    assert n_trials(state, "planning-exploration") == 30
    assert n_trials(state, "ridge") == 0


def test_register_is_idempotent_on_spec(state):
    a = register_trial(state, "ridge", "y_c2o_5", {"alpha": 1.0, "feats": ["a", "b"]})
    b = register_trial(state, "ridge", "y_c2o_5", {"feats": ["a", "b"], "alpha": 1.0})  # key order differs
    assert a == b
    assert n_trials(state) == 31  # 30 seeds + 1
    c = register_trial(state, "ridge", "y_c2o_5", {"alpha": 2.0, "feats": ["a", "b"]})
    d = register_trial(state, "ridge", "y_c2o_10", {"alpha": 1.0, "feats": ["a", "b"]})
    assert len({a, c, d}) == 3
    assert n_trials(state) == 33
    assert n_trials(state, "ridge") == 3
    row = (
        state.connect().execute("SELECT status, family, target FROM trials WHERE trial_id=?", (a,)).fetchone()
    )
    assert row == ("registered", "ridge", "y_c2o_5")


def test_finish_and_sharpe_variance(state):
    ids = [register_trial(state, "ridge", "y", {"i": i}) for i in range(3)]
    for tid, sr in zip(ids, (0.01, 0.03, 0.05), strict=True):
        finish_trial(state, tid, {"sharpe_daily": sr}, f"returns/{tid}.parquet")
    # mean 0.03, deviations -.02, 0, .02 -> sum sq 8e-4, / (3 - 1) = 4e-4; seeds (no metrics) are ignored
    assert sharpe_variance_across_trials(state) == pytest.approx(4e-4)
    assert sharpe_variance_across_trials(state, "ridge") == pytest.approx(4e-4)
    row = (
        state.connect()
        .execute("SELECT status, returns_path FROM trials WHERE trial_id=?", (ids[0],))
        .fetchone()
    )
    assert row == ("done", f"returns/{ids[0]}.parquet")
    with pytest.raises(ValueError):
        finish_trial(state, ids[0], {"sharpe_daily": 9.9}, "x")  # a finished trial is never rewritten
    with pytest.raises(KeyError):
        finish_trial(state, "nope", {}, "x")


def test_variance_nan_with_fewer_than_two(state):
    import math

    assert math.isnan(sharpe_variance_across_trials(state))


def test_module_never_removes_rows():
    src = Path(trials_mod.__file__).read_text(encoding="utf-8").upper()
    assert "DELETE" not in src and "DROP TABLE" not in src
