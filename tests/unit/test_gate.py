from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.validation.gate import (
    CandidateResult,
    evaluate_lockbox_gate,
    evaluate_research_gate,
    load_candidate_result,
    save_candidate_result,
    validate_candidate,
    write_report,
)
from stocksense.validation.trials import finish_trial, register_trial

T = 2500
N_FAMILY = 12


def _dates(n: int = T) -> list[date]:
    out, d = [], date(2014, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DATES = _dates()


def _good_returns(seed: int = 5, mean: float = 0.0009, sd: float = 0.006) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, T)


def _candidate(net: np.ndarray, *, net2: np.ndarray | None = None, bench=None, ic=(0.031, 5.2), family="fam"):
    daily = pl.DataFrame(
        {
            "date": DATES,
            "net": net,
            "net_2x": net - 0.0003 if net2 is None else net2,
            "benchmark": np.zeros(T) if bench is None else bench,
        }
    )
    return CandidateResult(
        candidate_id="cand1",
        family=family,
        target="y_c2o_5",
        daily=daily,
        ic={"mean_ic": ic[0], "ic_tstat": ic[1]},
    )


@pytest.fixture
def state(tmp_data_root):
    st = State(tmp_data_root)
    st.migrate()
    return st


def _trial(state, tmp_path, i: int, ret: np.ndarray, sharpe: float | None = None, family="fam") -> None:
    tid = register_trial(state, family, "y_c2o_5", {"i": i})
    path = tmp_path / f"trial{i}.parquet"
    pl.DataFrame({"date": DATES, "ret": ret}).write_parquet(path)
    sd = float(np.std(ret, ddof=1))
    finish_trial(state, tid, {"sharpe_daily": float(ret.mean() / sd) if sharpe is None else sharpe}, path)


def _family(state, tmp_path, best: np.ndarray, n: int = N_FAMILY, seed: int = 100) -> None:
    """`best` is trial 0; the other n-1 trials are pure noise."""
    _trial(state, tmp_path, 0, best)
    rng = np.random.default_rng(seed)
    for i in range(1, n):
        _trial(state, tmp_path, i, rng.normal(0.0, 0.006, T))


def _lines(report) -> dict[int, bool]:
    return {ln.number: ln.passed for ln in report.lines}


def test_baseline_candidate_passes_all_nine_lines(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    assert len(rep.lines) == 9
    assert _lines(rep) == dict.fromkeys(range(1, 10), True), rep.table()
    assert rep.passed
    assert "PASS" in rep.table() and "mean_rank_ic" in rep.table()


def test_line1_mean_rank_ic_fails_below_0_02(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net, ic=(0.015, 5.2)), tmp_data_root, state)
    # 0.015 < 0.02 (gate.toml min_mean_rank_ic)
    assert _lines(rep)[1] is False and _lines(rep)[2] is True and not rep.passed


def test_line2_ic_tstat_fails_below_3(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net, ic=(0.031, 2.9)), tmp_data_root, state)
    assert _lines(rep)[2] is False and _lines(rep)[1] is True


def test_line3_net_sharpe_fails_when_returns_are_thin(state, tmp_path, tmp_data_root):
    # daily mean 0.0002 / sd 0.006 -> annual Sharpe about 0.53 < 1.0
    net = _good_returns(mean=0.0002)
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    assert _lines(rep)[3] is False


def test_line4_sharpe_under_double_slippage_fails(state, tmp_path, tmp_data_root):
    net = _good_returns()  # Sharpe ~2.4
    net2 = net - 0.0010  # doubled slippage eats the whole edge -> Sharpe below 0
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net, net2=net2), tmp_data_root, state)
    assert _lines(rep)[3] is True and _lines(rep)[4] is False


def test_line5_excess_over_benchmark_fails_when_no_better_than_benchmark(state, tmp_path, tmp_data_root):
    net = _good_returns()
    bench = net + np.random.default_rng(9).normal(0.0, 0.004, T)  # same mean, extra noise
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net, bench=bench), tmp_data_root, state)
    assert _lines(rep)[5] is False and _lines(rep)[3] is True


def test_line6_fraction_of_positive_years_fails(state, tmp_path, tmp_data_root):
    years = np.array([d.year for d in DATES])
    bad = {2015, 2017, 2019, 2021}  # 4 losing years of the 10-11 calendar years in the sample
    mean = np.where(np.isin(years, list(bad)), -0.0006, 0.0025)
    net = np.random.default_rng(5).normal(0.0, 0.006, T) + mean
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    assert _lines(rep)[6] is False and _lines(rep)[3] is True


def test_line7_dsr_fails_when_trial_sharpes_are_dispersed(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    # twelve more finished trials with wildly different per-period Sharpes -> large V and SR0
    rng = np.random.default_rng(3)
    for i in range(20, 32):
        _trial(state, tmp_path, i, rng.normal(0.0, 0.006, T), sharpe=0.4 * (1 if i % 2 else -1))
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    assert _lines(rep)[7] is False


def test_line8_pbo_needs_ten_trials(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net, n=6)
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    line = next(ln for ln in rep.lines if ln.number == 8)
    assert line.passed is False
    assert "too few trials to estimate" in line.text


def test_line8_pbo_fails_for_a_pure_noise_family(state, tmp_path, tmp_data_root):
    noise = np.random.default_rng(1).normal(0.0, 0.006, T)
    _family(state, tmp_path, noise, n=N_FAMILY, seed=200)
    rep = evaluate_research_gate(_candidate(_good_returns()), tmp_data_root, state)
    line = next(ln for ln in rep.lines if ln.number == 8)
    # all twelve trials are noise: PBO is about 0.5, above the 0.30 bar
    assert line.passed is False and line.value > 0.30


def test_line9_max_drawdown_fails(state, tmp_path, tmp_data_root):
    net = _good_returns()
    net = net.copy()
    net[1000:1030] = -0.012  # 30 straight -1.2% days: about -30% drawdown
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    line = next(ln for ln in rep.lines if ln.number == 9)
    assert line.passed is False and line.value > 0.25


def test_s2_benchmark_none_means_zero_benchmark(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    cand = _candidate(net)
    cand = CandidateResult(
        cand.candidate_id, cand.family, cand.target, cand.daily.drop("benchmark"), ic=cand.ic
    )
    rep = evaluate_research_gate(cand, tmp_data_root, state)
    assert _lines(rep)[5] is True


def test_ic_computed_from_predictions_when_not_precomputed(state, tmp_path, tmp_data_root):
    rng = np.random.default_rng(4)
    n_dates, n_stocks = 120, 80
    rows_d = np.repeat(np.array(DATES[:n_dates], dtype="datetime64[D]"), n_stocks)
    label = rng.normal(size=n_dates * n_stocks)
    score = 0.2 * label + rng.normal(size=n_dates * n_stocks)
    preds = pl.DataFrame({"date": rows_d, "score": score, "y": label})
    net = _good_returns()
    _family(state, tmp_path, net)
    cand = CandidateResult("cand1", "fam", "y", _candidate(net).daily, ic=None, preds=preds, label_col="y")
    rep = evaluate_research_gate(cand, tmp_data_root, state)
    ic_line = next(ln for ln in rep.lines if ln.number == 1)
    assert 0.1 < ic_line.value < 0.25  # rank corr of score with label ~ 0.2/sqrt(1.04) = 0.19 (pearson)


def test_report_json_roundtrip_and_path(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    rep = evaluate_research_gate(_candidate(net), tmp_data_root, state)
    path = write_report(rep, tmp_data_root)
    assert path == Paths(tmp_data_root).models / "cand1" / "validation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["candidate_id"] == "cand1" and data["passed"] is True
    assert [ln["number"] for ln in data["lines"]] == list(range(1, 10))
    assert data["lines"][0]["name"] == "mean_rank_ic"


def test_lockbox_gate_lines(tmp_data_root):
    good = np.random.default_rng(1).normal(0.0010, 0.006, 150)
    rep = evaluate_lockbox_gate("cand1", good, tmp_data_root)
    assert [ln.passed for ln in rep.lines] == [True, True] and rep.passed
    # negative total return and Sharpe below 0.5
    bad = np.random.default_rng(1).normal(-0.0005, 0.006, 150)
    rep2 = evaluate_lockbox_gate("cand1", bad, tmp_data_root)
    assert [ln.passed for ln in rep2.lines] == [False, False] and not rep2.passed


def test_candidate_result_save_load_roundtrip(tmp_data_root):
    cand = _candidate(_good_returns())
    save_candidate_result(cand, tmp_data_root)
    back = load_candidate_result("cand1", tmp_data_root)
    assert back.family == "fam" and back.ic == cand.ic
    assert back.daily.equals(cand.daily)


def test_lockbox_refused_when_research_gate_fails(state, tmp_path, tmp_data_root):
    net = _good_returns(mean=0.0002)  # fails line 3
    _family(state, tmp_path, net)
    save_candidate_result(_candidate(net), tmp_data_root)
    calls = []
    rep = validate_candidate(
        "cand1", tmp_data_root, state, open_lockbox=True, lockbox_runner=lambda c, s: calls.append(c)
    )
    assert not rep.passed
    assert rep.lockbox is not None and rep.lockbox["status"] == "refused"
    assert calls == []
    touches = Paths(tmp_data_root).state / "lockbox_touches.jsonl"
    assert not touches.exists()
    assert (Paths(tmp_data_root).models / "cand1" / "validation.json").is_file()


def test_lockbox_opens_once_when_research_gate_passes(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    save_candidate_result(_candidate(net), tmp_data_root)

    def runner(cid: str, settings) -> np.ndarray:
        return np.random.default_rng(2).normal(0.001, 0.006, 150)

    rep = validate_candidate("cand1", tmp_data_root, state, open_lockbox=True, lockbox_runner=runner)
    assert rep.passed and rep.lockbox is not None and rep.lockbox["status"] == "passed"
    touches = Paths(tmp_data_root).state / "lockbox_touches.jsonl"
    assert len(touches.read_text(encoding="utf-8").splitlines()) == 1
    from stocksense.core.errors import LockboxError

    with pytest.raises(LockboxError):
        validate_candidate("cand1", tmp_data_root, state, open_lockbox=True, lockbox_runner=runner)


def test_validate_candidate_without_lockbox_only_writes_research_report(state, tmp_path, tmp_data_root):
    net = _good_returns()
    _family(state, tmp_path, net)
    save_candidate_result(_candidate(net), tmp_data_root)
    rep = validate_candidate("cand1", tmp_data_root, state)
    assert rep.passed and rep.lockbox is None
    assert isinstance(Path(Paths(tmp_data_root).models / "cand1" / "validation.json"), Path)
