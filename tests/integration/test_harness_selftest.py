"""Harness self-tests (PLAN T6.8): the validation machinery must pass a planted signal, fail noise, reject a
look-ahead feature before any model sees it, and refuse to open the lockbox twice. Read Phase 7 results only
after these are green."""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST, FixedClock
from stocksense.core.config import Settings
from stocksense.core.errors import LockboxError
from stocksense.core.state import State
from stocksense.features.base import FeatureSpec
from stocksense.features.build import build_features
from stocksense.models.base import FitInfo, Model, ModelSpec
from stocksense.validation.gate import CandidateResult, GateReport, evaluate_research_gate
from stocksense.validation.lockbox import guard, open_lockbox
from stocksense.validation.walkforward import backtest_walk_forward, run_walk_forward
from tests.fixtures.wf_world import WFWorld, build_world, leaky_spec
from tests.unit.test_feature_truncation import _compute, violations

pytestmark = pytest.mark.integration

PLANTED = 0.08


class RandomScores(Model):
    """A 'model' that knows nothing: seeded Gaussian scores."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0):  # type: ignore[no-untyped-def]
        return FitInfo()

    def refit_full(self, X, y, w, dates, fit_info, seed):  # type: ignore[no-untyped-def]
        self._rng = np.random.default_rng(10_000 + seed)

    def predict(self, X):  # type: ignore[no-untyped-def]
        return self._rng.normal(size=len(X))

    def save(self, dir):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def load(self, dir):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class LookaheadError(AssertionError):
    """A feature's value for a date changed when the data after that date was removed."""


def assert_no_lookahead(spec: FeatureSpec, settings: Settings, cuts: list[date]) -> None:
    """The T4.2 truncation checker as a gate: LookaheadError if `spec` changes under any cut."""
    full = _compute(spec, settings, None)
    for cut in cuts:
        bad = violations(spec, full, _compute(spec, settings, cut), cut)
        if bad:
            raise LookaheadError(f"{spec.name} at cut {cut}: {bad}")


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("selftest"), planted_ic=PLANTED)


@pytest.fixture(scope="module")
def state(world) -> State:
    st = State(world.settings)
    st.migrate()
    return st


def _candidate(world, state, spec, factory, label):
    res = run_walk_forward(
        spec, world.settings, state, folds=world.folds, calendar=world.calendar, seeds=(1, 2, 3),
        inner_months=2, model_factory=factory,
    )  # fmt: skip
    bt = backtest_walk_forward(res, spec, world.settings, state, calendar=world.calendar, k=10)
    cand = CandidateResult(
        f"selftest-{label}", spec.family, spec.target, bt.daily,
        ic={"mean_ic": res.ic["mean_ic"], "ic_tstat": res.ic["ic_tstat"]}, trial_ids=(res.trial_id,),
    )  # fmt: skip
    return res, bt, evaluate_research_gate(cand, world.settings, state)


def _show(label: str, report: GateReport) -> None:
    print(f"\n{label}\n{report.table()}")


def test_random_model_fails_the_research_gate(world, state):
    spec = ModelSpec("selftest-random", "y_oc", features="all", entry="open")
    res, bt, report = _candidate(world, state, spec, lambda s: RandomScores(), "random")
    _show("random-score model", report)
    ic = res.ic
    sd = bt.metrics["sharpe_daily"]
    print(f"random model: mean IC {ic['mean_ic']:.4f} t {ic['ic_tstat']:.2f} daily-Sharpe {sd:.4f}")
    assert not report.passed
    by_number = {ln.number: ln for ln in report.lines}
    assert not by_number[1].passed and abs(ic["mean_ic"]) < 0.02
    assert not by_number[2].passed
    assert not by_number[3].passed  # no net Sharpe >= 1 from noise after costs
    assert not by_number[8].passed  # one trial of the family: cannot estimate PBO
    assert sum(not ln.passed for ln in report.lines) >= 5


def test_planted_signal_passes_the_ic_lines(world, state):
    spec = ModelSpec("selftest-ridge", "y_oc", features="all", entry="open")
    res, _, report = _candidate(world, state, spec, None, "ridge")
    _show(f"ridge on planted_ic={PLANTED}", report)
    print(f"planted {PLANTED}: mean IC {res.ic['mean_ic']:.4f} t {res.ic['ic_tstat']:.2f}")
    by_number = {ln.number: ln for ln in report.lines}
    assert by_number[1].passed and by_number[1].value >= 0.02
    assert by_number[2].passed and by_number[2].value >= 3.0


def test_leaky_feature_is_rejected_by_the_truncation_checker_first(world, tmp_path_factory):
    cuts = [world.calendar.shift(world.market.sessions[0], k) for k in (300, 380, 450)]
    # clean features pass the checker
    for name in ("alpha_t", "ret_1", "vol_21", "mkt_ret_1"):
        assert_no_lookahead(world.registry[name], world.settings, cuts)
    leaky = leaky_spec()["leaky_next"]
    with pytest.raises(LookaheadError, match="leaky_next"):
        assert_no_lookahead(leaky, world.settings, cuts)

    # ...and without the checker a model WOULD pass on it: it reads tomorrow's return
    w = build_world(tmp_path_factory.mktemp("selftest_leaky"), planted_ic=PLANTED, n=100)
    build_features(w.settings, min_valid=10, registry={**w.registry, **leaky_spec()})
    st = State(w.settings)
    st.migrate()
    spec = ModelSpec("selftest-leaky-ridge", "y_oc", features=["leaky_next"], entry="open")
    res = run_walk_forward(
        spec, w.settings, st, folds=w.folds, calendar=w.calendar, seeds=(1,), inner_months=2
    )  # fmt: skip
    print(f"\nleaky feature: mean IC {res.ic['mean_ic']:.4f} t {res.ic['ic_tstat']:.1f}")
    cand = CandidateResult(
        "selftest-leaky", spec.family, spec.target,
        backtest_walk_forward(res, spec, w.settings, st, calendar=w.calendar).daily,
        ic={"mean_ic": res.ic["mean_ic"], "ic_tstat": res.ic["ic_tstat"]},
    )  # fmt: skip
    rep = evaluate_research_gate(cand, w.settings, st)
    assert rep.lines[0].passed and rep.lines[1].passed
    assert res.ic["mean_ic"] > 0.3


def test_lockbox_cannot_be_opened_twice(tmp_data_root):
    s = tmp_data_root
    clock = FixedClock(datetime(2026, 11, 12, 10, 0, tzinfo=IST))
    with open_lockbox(s, clock, "cand-1", "final check"):
        pass
    with pytest.raises(LockboxError), open_lockbox(s, clock, "cand-1", "again"):
        pytest.fail("the lockbox opened a second time")
    path = s.paths.data_root / "state" / "lockbox_touches.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [x["candidate_id"] for x in lines] == ["cand-1"]
    with open_lockbox(s, clock, "cand-2", "a different candidate"):
        pass
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    # outside the context the lockbox period is closed
    with pytest.raises(LockboxError):
        guard(pl.DataFrame({"date": [date(2025, 10, 2)]}), "date", "lockbox", s)
