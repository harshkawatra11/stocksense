from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from stocksense.core.state import State
from stocksense.data import io
from stocksense.models import store
from stocksense.models.base import ModelSpec
from stocksense.models.linear import RidgeModel


@pytest.fixture
def state(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    return st


def _fitted(seed: int = 0, n_feat: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, n_feat)).astype(np.float32)
    y = X[:, 0] * 0.3 + rng.normal(size=200) * 0.1
    m = RidgeModel()
    m.configure([f"f{i}" for i in range(n_feat)], ["xs"] * n_feat)
    info = m.fit(X, y, np.ones(200), np.arange(200) // 20)
    m.refit_full(X, y, np.ones(200), np.arange(200) // 20, info, 0)
    return m, X, info


def _spec(family: str = "s1-ridge", target: str = "y_c2o_5", **params) -> ModelSpec:
    return ModelSpec(family=family, target=target, params=params)


def _save(
    settings, state, family="s1-ridge", target="y_c2o_5", train_end=date(2024, 12, 31), seed=0, **params
):
    m, X, info = _fitted(seed)
    spec = _spec(family, target, **params)
    meta = {
        "feature_names": [f"f{i}" for i in range(4)], "feature_version": 3, "fit_info": info.to_dict(),
        "train_start": "2011-01-03",
    }  # fmt: skip
    return store.save_model(settings, state, m, spec, train_end, meta), m, X


def test_save_load_round_trip_gives_identical_predictions(tmp_data_root, state):
    mid, m, X = _save(tmp_data_root, state)
    loaded = store.load_model(tmp_data_root, mid)
    assert np.array_equal(loaded.predict(X), m.predict(X))
    d = tmp_data_root.paths.data_root / "models" / mid
    for name in ("spec.json", "feature_manifest.json", "fit_info.json", "train_range.json", "git_sha.txt"):
        assert (d / name).is_file(), name
    assert any(p.name.startswith("weights.") for p in d.iterdir())


def test_row_is_a_candidate_with_a_creation_event(tmp_data_root, state):
    mid, _, _ = _save(tmp_data_root, state)
    con = state.connect()
    row = con.execute("SELECT status, validated, family, target, train_end FROM models").fetchone()
    assert row == ("candidate", 0, "s1-ridge", "y_c2o_5", "2024-12-31")
    ev = con.execute("SELECT model_id, event FROM model_events").fetchall()
    assert ev == [(mid, "saved")]
    # the id carries family, target and train end
    assert mid.startswith("s1-ridge-y_c2o_5-") and mid.endswith("-20241231")


def test_set_status_records_events_and_validates_values(tmp_data_root, state):
    mid, _, _ = _save(tmp_data_root, state)
    store.set_status(state, mid, "shadow", "first look")
    store.set_status(state, mid, "retired", "worse than ridge")
    con = state.connect()
    evs = con.execute(
        "SELECT event, detail FROM model_events WHERE model_id=? ORDER BY id", (mid,)
    ).fetchall()
    assert [e[0] for e in evs] == ["saved", "status:shadow", "status:retired"]
    assert evs[1][1] == "first look"
    assert con.execute("SELECT status FROM models").fetchone()[0] == "retired"
    with pytest.raises(ValueError):
        store.set_status(state, mid, "king", "")
    with pytest.raises(KeyError):
        store.set_status(state, "nope", "shadow", "")


def test_one_champion_per_family_prefix(tmp_data_root, state):
    a, _, _ = _save(tmp_data_root, state, family="s1-ridge", seed=1)
    b, _, _ = _save(tmp_data_root, state, family="s1-gbm", seed=2)
    c, _, _ = _save(tmp_data_root, state, family="s2-ridge", seed=3)
    store.set_status(state, a, "champion", "best")
    store.set_status(state, c, "champion", "s2 best")
    assert store.current(state, "s1", "champion") == a
    store.set_status(state, b, "champion", "better")
    assert store.current(state, "s1", "champion") == b
    con = state.connect()
    assert con.execute("SELECT status FROM models WHERE model_id=?", (a,)).fetchone()[0] == "shadow"
    assert store.current(state, "s2", "champion") == c  # other prefix untouched
    assert store.current(state, "s1", "shadow") == a
    assert con.execute("SELECT count(*) FROM models WHERE status='champion'").fetchone()[0] == 2
    demoted = con.execute(
        "SELECT event, detail FROM model_events WHERE model_id=? ORDER BY id", (a,)
    ).fetchall()
    assert demoted[-1][0] == "status:shadow" and b in demoted[-1][1]
    assert store.current(state, "s3", "champion") is None


def test_append_preds_upserts_on_date_and_sid(tmp_data_root, state, fixed_clock):
    mid, _, _ = _save(tmp_data_root, state)
    d1, d2 = date(2025, 1, 6), date(2025, 1, 7)
    f1 = pl.DataFrame(
        {"date": [d1, d1], "sid": [1, 2], "score": [0.1, -0.1]}, schema_overrides={"sid": pl.Int32}
    )
    store.append_preds(tmp_data_root, mid, f1, "live", clock=fixed_clock)
    f2 = pl.DataFrame(
        {"date": [d1, d2], "sid": [2, 1], "score": [0.4, 0.2]}, schema_overrides={"sid": pl.Int32}
    )
    store.append_preds(tmp_data_root, mid, f2, "live", clock=fixed_clock)
    got = io.read(tmp_data_root, "preds", sub=mid).collect().sort("date", "sid")
    assert got.select("date", "sid").rows() == [(d1, 1), (d1, 2), (d2, 1)]
    assert got["score"].to_list() == pytest.approx([0.1, 0.4, 0.2])
    assert set(got["mode"].to_list()) == {"live"}
    with pytest.raises(ValueError):
        store.append_preds(tmp_data_root, mid, f1, "backfill", clock=fixed_clock)
