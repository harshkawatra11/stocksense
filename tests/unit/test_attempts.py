"""Tests for the append-only attempt registry.

The load-bearing property: EVERY evaluated configuration registers, not just
survivors, because that count is the n_trials fed to the Deflated Sharpe.
"""

from __future__ import annotations

import json

import pytest

from stocksense.data.store import Reader, Store
from stocksense.evaluation import attempts


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "hot.duckdb", tmp_path / "parquet")
    yield s
    s.close()


# ------------------------------------------------------------------ config_hash
def test_config_hash_is_deterministic():
    h1 = attempts.config_hash("family_a", {"x": 1, "y": 2})
    h2 = attempts.config_hash("family_a", {"y": 2, "x": 1})  # different key order
    assert h1 == h2


def test_config_hash_differs_for_different_families_or_params():
    base = attempts.config_hash("family_a", {"x": 1})
    assert attempts.config_hash("family_b", {"x": 1}) != base
    assert attempts.config_hash("family_a", {"x": 2}) != base


# ------------------------------------------------------------- register_attempt
def test_register_attempt_returns_an_id_and_is_queryable(store):
    aid = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json='{"a":1}', family="fam"
    )
    assert aid
    row = store.con.execute(
        "SELECT hypothesis_id, family, config_hash FROM evaluation_attempts WHERE attempt_id = ?",
        [aid],
    ).fetchone()
    assert row == ("h1", "fam", "c1")


def test_registration_is_idempotent_on_config_hash(store):
    """A resumed sweep re-offering the same config must get the SAME
    attempt_id back, not a duplicate row -- this is what makes an
    interrupted-and-resumed sweep count each config exactly once."""
    a1 = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    a2 = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    assert a1 == a2
    n = store.con.execute("SELECT count(*) FROM evaluation_attempts").fetchone()[0]
    assert n == 1


def test_same_config_hash_under_a_different_hypothesis_is_a_separate_attempt(store):
    a1 = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    a2 = attempts.register_attempt(
        store, hypothesis_id="h2", config_hash="c1", config_json="{}", family="fam"
    )
    assert a1 != a2


# --------------------------------------------------------------- record_result
def test_record_result_attaches_verdict_and_metrics(store):
    aid = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    attempts.record_result(
        store, aid, verdict="gate_pass", metrics_json='{"sharpe": 1.2}', fail_reason=None
    )
    row = store.con.execute(
        "SELECT verdict, fail_reason, metrics_json FROM evaluation_attempts WHERE attempt_id = ?",
        [aid],
    ).fetchone()
    assert row[0] == "gate_pass"
    assert row[1] is None
    assert json.loads(row[2])["sharpe"] == 1.2


def test_record_result_rejects_an_invalid_verdict(store):
    aid = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    with pytest.raises(ValueError, match="verdict"):
        attempts.record_result(store, aid, verdict="not_a_real_verdict")


def test_record_result_rejects_an_invalid_fail_reason(store):
    aid = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    with pytest.raises(ValueError, match="fail_reason"):
        attempts.record_result(store, aid, verdict="gate_fail", fail_reason="made_up_reason")


# ---------------------------------------------------------- count_attempts / DSR
def test_every_config_registers_not_just_survivors(store):
    """The property that makes n_trials honest: register a batch, mark most
    as screened_out, and the count must still include them all."""
    for i in range(20):
        aid = attempts.register_attempt(
            store, hypothesis_id="h1", config_hash=f"c{i}", config_json="{}", family="fam"
        )
        verdict = "gate_pass" if i == 0 else "screened_out"
        attempts.record_result(store, aid, verdict=verdict)
    store.publish()

    with Reader(store.parquet_root) as r:
        assert attempts.count_attempts(r, "h1") == 20


def test_count_attempts_on_no_data_is_zero(tmp_path):
    with Reader(tmp_path / "empty") as r:
        assert attempts.count_attempts(r) == 0


def test_count_attempts_scopes_to_hypothesis_when_given(store):
    attempts.register_attempt(store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam")
    attempts.register_attempt(store, hypothesis_id="h2", config_hash="c2", config_json="{}", family="fam")
    attempts.register_attempt(store, hypothesis_id="h2", config_hash="c3", config_json="{}", family="fam")
    store.publish()
    with Reader(store.parquet_root) as r:
        assert attempts.count_attempts(r, "h1") == 1
        assert attempts.count_attempts(r, "h2") == 2
        assert attempts.count_attempts(r) == 3


# ------------------------------------------------------------ trial_sharpe_std
def test_trial_sharpe_std_is_measured_not_assumed(store):
    sharpes = [0.5, 1.0, 1.5, -0.2, 2.1]
    for i, sh in enumerate(sharpes):
        aid = attempts.register_attempt(
            store, hypothesis_id="h1", config_hash=f"c{i}", config_json="{}", family="fam"
        )
        attempts.record_result(
            store, aid, verdict="gate_fail", metrics_json=json.dumps({"sharpe": sh})
        )
    store.publish()

    import numpy as np

    with Reader(store.parquet_root) as r:
        got = attempts.trial_sharpe_std(r, "h1")
    assert got == pytest.approx(float(np.std(sharpes, ddof=1)))


def test_trial_sharpe_std_is_nan_with_fewer_than_two_observations(store):
    aid = attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam"
    )
    attempts.record_result(store, aid, verdict="gate_fail", metrics_json='{"sharpe": 1.0}')
    store.publish()
    with Reader(store.parquet_root) as r:
        assert attempts.trial_sharpe_std(r, "h1") != attempts.trial_sharpe_std(r, "h1")  # nan


def test_trial_sharpe_std_ignores_attempts_with_no_metrics_yet(store):
    """A config that has been registered but not yet scored must not poison
    the dispersion calculation."""
    for i, sh in enumerate([1.0, 2.0]):
        aid = attempts.register_attempt(
            store, hypothesis_id="h1", config_hash=f"c{i}", config_json="{}", family="fam"
        )
        attempts.record_result(store, aid, verdict="gate_fail", metrics_json=json.dumps({"sharpe": sh}))
    attempts.register_attempt(
        store, hypothesis_id="h1", config_hash="c_pending", config_json="{}", family="fam"
    )  # no record_result call -- metrics_json stays NULL
    store.publish()

    with Reader(store.parquet_root) as r:
        got = attempts.trial_sharpe_std(r, "h1")
    import numpy as np

    assert got == pytest.approx(float(np.std([1.0, 2.0], ddof=1)))


# ------------------------------------------------------------------ read_attempts
def test_read_attempts_on_empty_store(tmp_path):
    with Reader(tmp_path / "empty") as r:
        df = attempts.read_attempts(r)
    assert df.empty
    assert "attempt_id" in df.columns


def test_read_attempts_returns_all_rows_for_a_hypothesis(store):
    attempts.register_attempt(store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="fam")
    attempts.register_attempt(store, hypothesis_id="h1", config_hash="c2", config_json="{}", family="fam")
    attempts.register_attempt(store, hypothesis_id="h2", config_hash="c3", config_json="{}", family="fam")
    store.publish()
    with Reader(store.parquet_root) as r:
        df = attempts.read_attempts(r, "h1")
    assert len(df) == 2
    assert set(df["config_hash"]) == {"c1", "c2"}
