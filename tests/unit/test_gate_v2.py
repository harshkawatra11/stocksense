"""Gate v2 (G2-B1): family-scoped DSR variance and exposure-matched excess benchmark; version selection."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.backtest.metrics import newey_west_tstat, perf_stats
from stocksense.core.state import State
from stocksense.validation.dsr import deflated_sharpe
from stocksense.validation.gate import (
    CandidateResult,
    evaluate_research_gate,
    exposure_matched_benchmark,
    save_candidate_result,
    validate_candidate,
)
from stocksense.validation.trials import (
    finish_trial,
    n_trials,
    register_trial,
    sharpe_variance_across_trials,
)

T = 600


def _dates(n: int = T) -> list[date]:
    out, d = [], date(2020, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DATES = _dates()


@pytest.fixture
def state(tmp_data_root):
    st = State(tmp_data_root)
    st.migrate()
    return st


def _done(state, tmp_path, family: str, i: int, sharpe: float) -> str:
    tid = register_trial(state, family, "y_c2o_10", {"i": i, "f": family})
    path = tmp_path / f"t_{family}_{i}.parquet"
    pl.DataFrame({"date": DATES, "ret": np.zeros(T)}).write_parquet(path)
    finish_trial(state, tid, {"sharpe_daily": sharpe}, path)
    return tid


def _cand(family="f1", trial_ids=(), gross=True, seed=3) -> CandidateResult:
    rng = np.random.default_rng(seed)
    net = rng.normal(0.0008, 0.006, T)
    cols = {
        "date": DATES,
        "net": net,
        "net_2x": net - 0.0002,
        "benchmark": rng.normal(0.0004, 0.01, T),
    }
    if gross:
        cols["gross"] = np.full(T, 0.5)
    return CandidateResult(
        "cand", family, "y_c2o_10", pl.DataFrame(cols), ic={"mean_ic": 0.05, "ic_tstat": 9.0},
        trial_ids=tuple(trial_ids),
    )  # fmt: skip


def _set_created(state, tid: str, iso: str) -> None:
    con = state.connect()
    try:
        con.execute("UPDATE trials SET created_at=? WHERE trial_id=?", (iso, tid))
    finally:
        con.close()


def _val(report, name: str):
    return {ln.name: ln.value for ln in report.lines}[name]


def test_gate_v2_is_loaded_and_scoped(tmp_data_root):
    g2 = tmp_data_root.gate_v2
    assert g2 is not None
    assert g2.version == 2
    assert g2.effective_from.isoformat() == "2026-11-20T00:00:00+05:30"
    assert g2.research.dsr_variance_scope == "family"
    assert g2.research.excess_benchmark == "exposure_matched"
    # every number identical to v1
    assert g2.research.model_dump(exclude={"dsr_variance_scope", "excess_benchmark"}) == (
        tmp_data_root.gate.research.model_dump()
    )
    assert g2.lockbox == tmp_data_root.gate.lockbox and g2.paper == tmp_data_root.gate.paper
    assert g2.promotion == tmp_data_root.gate.promotion and g2.auction == tmp_data_root.gate.auction


def test_family_variance_hand_numbers(state, tmp_path):
    # var([.05,.07,.30], ddof=1) = (0.0081 + 0.0049 + 0.0256) / 2 = 0.0193 (PLAN-G2 B1 quotes 0.019233)
    for i, s in enumerate((0.05, 0.07)):
        _done(state, tmp_path, "f1", i, s)
    _done(state, tmp_path, "f2", 0, 0.30)
    assert sharpe_variance_across_trials(state, family="f1") == pytest.approx(0.0002)
    assert sharpe_variance_across_trials(state) == pytest.approx(0.0193)


def test_v2_dsr_uses_family_variance_and_all_trials_count(state, tmp_path, tmp_data_root):
    ids = [_done(state, tmp_path, "f1", i, s) for i, s in enumerate((0.05, 0.07))]
    _done(state, tmp_path, "f2", 0, 0.30)
    cand = _cand(trial_ids=ids)
    n = n_trials(state)
    rep1 = evaluate_research_gate(cand, tmp_data_root, state, version=1)
    rep2 = evaluate_research_gate(cand, tmp_data_root, state, version=2)
    net = cand.daily["net"].to_numpy()
    st = perf_stats(net, cand.daily["benchmark"].to_numpy(), dates=DATES)
    sr = float(net.mean() / net.std(ddof=1))
    want2 = deflated_sharpe(sr, T, st["skew"], st["kurtosis"], n, 0.0002)
    want1 = deflated_sharpe(sr, T, st["skew"], st["kurtosis"], n, 0.0193)
    assert _val(rep2, "dsr") == pytest.approx(want2)
    assert _val(rep1, "dsr") == pytest.approx(want1)
    assert n == 33  # 30 planning-exploration rows are seeded by State.migrate; N counts all trials
    assert f"N={n} trials" in {ln.name: ln.text for ln in rep2.lines}["dsr"]


def test_v2_single_trial_family_falls_back_to_one_over_t(state, tmp_path, tmp_data_root):
    ids = [_done(state, tmp_path, "f1", 0, 0.05)]
    cand = _cand(trial_ids=ids)
    n = n_trials(state)
    rep = evaluate_research_gate(cand, tmp_data_root, state, version=2)
    net = cand.daily["net"].to_numpy()
    st = perf_stats(net, cand.daily["benchmark"].to_numpy(), dates=DATES)
    sr = float(net.mean() / net.std(ddof=1))
    want = deflated_sharpe(sr, T, st["skew"], st["kurtosis"], n, 1.0 / T)
    assert _val(rep, "dsr") == pytest.approx(want)


def test_exposure_matched_benchmark_hand_numbers():
    b = exposure_matched_benchmark(np.array([0.5, 0.5, 0.5]), np.array([0.02, 0.02, 0.02]))
    assert b == pytest.approx([0.0, 0.01, 0.01])
    net = np.array([0.01, 0.0, 0.02])
    assert net - b == pytest.approx([0.01, -0.01, 0.01])
    # varying gross: the benchmark uses the PREVIOUS day's gross
    b2 = exposure_matched_benchmark(np.array([1.0, 0.0, 0.4]), np.array([0.01, 0.02, 0.05]))
    assert b2 == pytest.approx([0.0, 0.02, 0.0])


def test_v2_excess_tstat_is_exposure_matched(state, tmp_data_root):
    cand = _cand(family="f1")
    rep = evaluate_research_gate(cand, tmp_data_root, state, version=2)
    net = cand.daily["net"].to_numpy()
    bench = cand.daily["benchmark"].to_numpy()
    b = exposure_matched_benchmark(cand.daily["gross"].to_numpy(), bench)
    want = newey_west_tstat(net - b, lags=5)
    assert _val(rep, "excess_tstat") == pytest.approx(want)
    v1 = evaluate_research_gate(cand, tmp_data_root, state, version=1)
    assert _val(v1, "excess_tstat") != pytest.approx(want)


def test_v2_requires_gross_for_non_auction_family(state, tmp_data_root):
    with pytest.raises(ValueError, match="gate v2 needs gross"):
        evaluate_research_gate(_cand(gross=False), tmp_data_root, state, version=2)


def test_v2_auction_family_keeps_zero_benchmark(state, tmp_data_root):
    cand = _cand(family="g1-s2", gross=False)
    rep = evaluate_research_gate(cand, tmp_data_root, state, version=2)
    net = cand.daily["net"].to_numpy()
    assert _val(rep, "excess_tstat") == pytest.approx(newey_west_tstat(net, lags=5))


def test_version_auto_selection_by_trial_creation_time(state, tmp_path, tmp_data_root):
    old = _done(state, tmp_path, "f1", 0, 0.05)
    new = _done(state, tmp_path, "f1", 1, 0.07)
    _set_created(state, old, "2026-11-19T23:59:00+05:30")
    _set_created(state, new, "2026-11-20T00:00:00+05:30")
    assert evaluate_research_gate(_cand(trial_ids=[old, new]), tmp_data_root, state).kind == "research-v1"
    assert evaluate_research_gate(_cand(trial_ids=[new]), tmp_data_root, state).kind == "research-v2"
    # no trial ids at all: nothing proves the candidate is new, so v1
    assert evaluate_research_gate(_cand(trial_ids=[]), tmp_data_root, state).kind == "research-v1"
    # an explicit version wins
    explicit2 = evaluate_research_gate(_cand(trial_ids=[old]), tmp_data_root, state, version=2)
    assert explicit2.kind == "research-v2"
    explicit1 = evaluate_research_gate(_cand(trial_ids=[new]), tmp_data_root, state, version=1)
    assert explicit1.kind == "research-v1"


def test_v1_when_no_gate_v2_in_settings(state, tmp_data_root):
    s = tmp_data_root.model_copy(update={"gate_v2": None})
    rep = evaluate_research_gate(_cand(gross=False, trial_ids=["x"]), s, state)
    assert rep.kind == "research-v1"
    with pytest.raises(ValueError, match="gate v2"):
        evaluate_research_gate(_cand(), s, state, version=2)


def test_table_header_names_version(state, tmp_data_root):
    rep = evaluate_research_gate(_cand(), tmp_data_root, state, version=2)
    assert rep.table().splitlines()[0].startswith("research-v2 gate for cand:")


def test_validate_candidate_passes_gate_version(state, tmp_data_root):
    save_candidate_result(_cand(), tmp_data_root)
    rep = validate_candidate("cand", tmp_data_root, state, gate_version=2)
    assert rep.kind == "research-v2"
