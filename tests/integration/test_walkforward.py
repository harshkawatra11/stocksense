from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from stocksense.core.state import State
from stocksense.data import io
from stocksense.models.base import Model, ModelSpec, build_model
from stocksense.validation.trials import n_trials
from stocksense.validation.walkforward import backtest_walk_forward, run_walk_forward
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration

SPEC = ModelSpec(family="s2-ridge", target="y_oc", features="all", entry="open")


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("wf_planted"), planted_ic=0.06)


@pytest.fixture(scope="module")
def null_world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("wf_null"), planted_ic=0.0)


def _state(w: WFWorld) -> State:
    st = State(w.settings)
    st.migrate()
    return st


def _run(w: WFWorld, spec: ModelSpec = SPEC, **kw):
    return run_walk_forward(
        spec, w.settings, _state(w), folds=w.folds, calendar=w.calendar, registry=w.registry,
        seeds=(1, 2), inner_months=2, **kw,
    )  # fmt: skip


def test_ridge_recovers_the_planted_ic(planted):
    res = _run(planted)
    print(f"\nplanted 0.06: out-of-fold mean IC {res.ic['mean_ic']:.4f} t {res.ic['ic_tstat']:.1f}")
    assert 0.04 <= res.ic["mean_ic"] <= 0.08
    assert [m["fold"] for m in res.fold_metrics] == [f.name for f in planted.folds]
    assert all(m["n_dates"] > 20 for m in res.fold_metrics)
    assert set(res.preds.columns) == {"date", "sid", "score", "fold"}
    assert res.preds["score"].min() >= -0.5 and res.preds["score"].max() <= 0.5
    # every prediction lies inside its own fold window
    for f in planted.folds:
        d = res.preds.filter(pl.col("fold") == f.name)["date"]
        assert d.min() >= f.test_start and d.max() <= f.test_end


def test_preds_are_written_as_walkforward(planted):
    res = _run(planted)
    stored = io.read(planted.settings, "preds", sub=res.trial_id).collect()
    assert stored.height == res.preds.height
    assert set(stored["mode"].unique()) == {"walkforward"}
    assert stored.select("date", "sid").is_duplicated().sum() == 0


def test_null_signal_has_zero_ic(null_world):
    res = _run(null_world)
    print("\nplanted 0: out-of-fold mean IC", round(res.ic["mean_ic"], 4))
    assert abs(res.ic["mean_ic"]) <= 0.01


def test_second_call_fits_nothing(tmp_path_factory):
    w = build_world(tmp_path_factory.mktemp("wf_resume"), planted_ic=0.06, n=80)
    fits = {"n": 0}

    def counting(spec: ModelSpec) -> Model:
        inner = build_model(spec)
        orig = inner.fit

        def fit(*a, **k):
            fits["n"] += 1
            return orig(*a, **k)

        inner.fit = fit  # type: ignore[method-assign]
        return inner

    first = _run(w, model_factory=counting)
    n1 = fits["n"]
    assert n1 == len(w.folds) * 2  # folds x seeds
    second = _run(w, model_factory=counting)
    assert fits["n"] == n1
    assert second.trial_id == first.trial_id
    assert second.preds.equals(first.preds)
    assert second.ic["mean_ic"] == pytest.approx(first.ic["mean_ic"])


def test_trial_is_registered_before_any_fit(planted):
    st = _state(planted)
    spec = ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t"], entry="open")
    before = n_trials(st, "s2-ridge")
    seen = {}

    def probing(s: ModelSpec) -> Model:
        seen["n"] = n_trials(st, "s2-ridge")
        return build_model(s)

    run_walk_forward(
        spec, planted.settings, st, folds=planted.folds[:1], calendar=planted.calendar, seeds=(1,),
        inner_months=2, model_factory=probing,
    )  # fmt: skip
    assert seen["n"] == before + 1


def test_backtest_records_trial_returns(planted):
    st = _state(planted)
    spec = ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t"], entry="open")
    res = run_walk_forward(
        spec, planted.settings, st, folds=planted.folds, calendar=planted.calendar, seeds=(1,), inner_months=2
    )  # fmt: skip
    bt = backtest_walk_forward(res, spec, planted.settings, st, calendar=planted.calendar, k=10)
    ret = pl.read_parquet(bt.returns_path)
    assert ret.columns == ["date", "ret"] and ret.height == bt.daily.height > 100
    assert {"net", "net_2x", "benchmark"} <= set(bt.daily.columns)
    con = st.connect()
    row = con.execute("SELECT status, returns_path FROM trials WHERE trial_id=?", (res.trial_id,)).fetchone()
    con.close()
    assert row[0] == "done" and row[1] == str(bt.returns_path)
    assert np.isfinite(bt.metrics["sharpe_daily"])
