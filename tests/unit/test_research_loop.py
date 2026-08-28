"""Phase K2.4: the loop driver.

The property that matters most: EVERY candidate registers an attempt,
regardless of whether it survives screening -- that is what makes N in the
Deflated Sharpe Ratio a counted fact. Verified directly by counting
`evaluation_attempts` rows against the batch size, not inferred from the
returned DataFrame alone.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.evaluation.attempts import attempt_count, holdout_id_for
from stocksense.features.registry import clear_registry
from stocksense.research.loop import FAST_DECAY, LOW_IC, SURVIVED, SearchBudget, run_search_iteration


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


@pytest.fixture()
def committed_prereg(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    f = repo / "preregistration_loop_test.md"
    f.write_text("# fixed before any result\n", encoding="utf-8")
    subprocess.run(["git", "add", f.name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "prereg"], cwd=repo, check=True)
    return f


def _candles_with_real_signal(n_symbols: int = 40, n_dates: int = 80, seed: int = 5) -> pd.DataFrame:
    """A raw column ('true_signal') that genuinely predicts forward relative
    return, plus a pure-noise column ('noise_col') that does not -- so a
    correctly-working screen must sort them into different outcomes."""
    rng = np.random.default_rng(seed)
    syms = [f"S{i:03d}" for i in range(n_symbols)]
    dates = pd.bdate_range("2024-01-01", periods=n_dates)

    rows = []
    price = {s: 100.0 for s in syms}
    for d in dates:
        true_signal = rng.normal(size=n_symbols)
        noise_col = rng.normal(size=n_symbols)
        # next period's relative return is driven by true_signal, not noise_col
        fwd_rel = true_signal * 0.02 + rng.normal(scale=0.005, size=n_symbols)
        for i, s in enumerate(syms):
            price[s] *= 1.0 + fwd_rel[i]
            rows.append({
                "symbol": s, "date": d, "adj_close": price[s], "close": price[s],
                "true_signal": float(true_signal[i]), "noise_col": float(noise_col[i]),
            })
    return pd.DataFrame(rows)


def test_every_candidate_registers_an_attempt_regardless_of_outcome(tmp_store, committed_prereg) -> None:
    candles = _candles_with_real_signal()
    budget = SearchBudget(batch_size=6, max_iterations=1, min_icir=100.0)  # impossible bar -> everything fails
    holdout_spec = {"universe": "test", "horizons": [1]}

    results = run_search_iteration(
        tmp_store, candles, feature_cols_base=[], seed=1, primitives=["true_signal", "noise_col"],
        budget=budget, hypothesis_id="h-test", preregistration_path=committed_prereg,
        holdout_spec=holdout_spec, horizons=(1, 3),
    )

    assert len(results) == 6
    holdout_id = holdout_id_for({**holdout_spec, "candidate_name": results.iloc[0]["candidate_name"],
                                  "candidate_expression": results.iloc[0]["expression"]})
    # each candidate has its OWN holdout_id (candidate identity is baked into
    # the spec), so total attempts across all of them equals the batch size
    total_attempts = sum(
        attempt_count(tmp_store, holdout_id_for({**holdout_spec, "candidate_name": r["candidate_name"],
                                                  "candidate_expression": r["expression"]}))
        for _, r in results.iterrows()
    )
    assert total_attempts == 6


def test_impossible_icir_bar_means_nothing_survives(tmp_store, committed_prereg) -> None:
    candles = _candles_with_real_signal()
    budget = SearchBudget(batch_size=4, max_iterations=1, min_icir=1000.0)
    results = run_search_iteration(
        tmp_store, candles, feature_cols_base=[], seed=2, primitives=["true_signal", "noise_col"],
        budget=budget, hypothesis_id="h-test2", preregistration_path=committed_prereg,
        holdout_spec={"universe": "test"}, horizons=(1, 3),
    )
    assert (results["outcome"] != SURVIVED).all()


def test_a_genuinely_predictive_raw_signal_can_survive_screening(tmp_store, committed_prereg) -> None:
    """Not a claim that the SEARCH generator invents good factors -- it tests
    a raw column directly. This confirms the SCREENING MACHINERY itself
    (decay_curve + icir + half_life composition) correctly recognizes real
    signal when hand-generated data contains it, using a lenient bar."""
    candles = _candles_with_real_signal(n_symbols=60, n_dates=120)
    budget = SearchBudget(batch_size=30, max_iterations=1, min_icir=0.3, max_half_life_for_survival=0.5)
    results = run_search_iteration(
        tmp_store, candles, feature_cols_base=[], seed=3, primitives=["true_signal", "noise_col"],
        budget=budget, hypothesis_id="h-test3", preregistration_path=committed_prereg,
        holdout_spec={"universe": "test"}, horizons=(1, 2, 3),
    )
    survivors = results[results["outcome"] == SURVIVED]
    assert not survivors.empty
    # every survivor's underlying primitive should trace back to the real
    # signal, not the noise column, given the bar is set well above noise
    assert set(survivors["primitive"]) <= {"true_signal"}


def test_outcome_is_always_one_of_the_closed_set(tmp_store, committed_prereg) -> None:
    candles = _candles_with_real_signal(n_symbols=20, n_dates=40)
    budget = SearchBudget(batch_size=8, max_iterations=1, min_icir=0.05)
    results = run_search_iteration(
        tmp_store, candles, feature_cols_base=[], seed=4, primitives=["true_signal", "noise_col"],
        budget=budget, hypothesis_id="h-test4", preregistration_path=committed_prereg,
        holdout_spec={"universe": "test"}, horizons=(1, 3),
    )
    assert set(results["outcome"]) <= {LOW_IC, "unstable_ic", FAST_DECAY, SURVIVED}


def test_a_bad_candidate_does_not_abort_the_whole_batch(tmp_store, committed_prereg, monkeypatch) -> None:
    """One candidate raising must not stop the rest of the batch from being
    screened AND registered -- a single flaky expression should not silently
    swallow the whole night's search."""
    from stocksense.research import loop as loop_mod

    candles = _candles_with_real_signal(n_symbols=15, n_dates=30)
    budget = SearchBudget(batch_size=5, max_iterations=1, min_icir=0.01)

    call_count = {"n": 0}
    real_screen = loop_mod.screen_candidate

    def _flaky_screen(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated bad candidate")
        return real_screen(*args, **kwargs)

    monkeypatch.setattr(loop_mod, "screen_candidate", _flaky_screen)

    results = run_search_iteration(
        tmp_store, candles, feature_cols_base=[], seed=5, primitives=["true_signal", "noise_col"],
        budget=budget, hypothesis_id="h-test5", preregistration_path=committed_prereg,
        holdout_spec={"universe": "test"}, horizons=(1, 3),
    )
    assert len(results) == 5  # all 5 candidates present despite one raising
