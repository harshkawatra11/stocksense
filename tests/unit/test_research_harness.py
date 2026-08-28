"""Phase K2.2: the reusable sweep runner.

Full end-to-end numerical correctness (does this harness reproduce a real
gate-passing result) is NOT re-verified here -- that needs 500+ real training
rows per fold and takes several minutes, so it lives in
research/verify_harness_acceptance.py as a manual check against the real
database, per that script's own docstring. These tests exercise the loop
wiring and plumbing that a unit test can cover cheaply: the grid iteration,
which cost point gets gated, the registered-factor merge path, and the empty-
universe / verdict-doc formatting edge cases.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.evaluation.backtest import FoldResult
from stocksense.evaluation.gate import GateVerdict
from stocksense.features.registry import clear_registry, factor
from stocksense.research.harness import SweepConfig, load_and_prepare, run_sweep, write_verdict_doc


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


def _write_minimal_bhavcopy(store, n_symbols: int = 3, n_days: int = 30) -> None:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for i in range(n_symbols):
        symbol = f"SYM{i}"
        price = 100.0 + i * 10
        for d in dates:
            rows.append({
                "symbol": symbol, "series": "EQ", "date": d.date(), "open": price,
                "high": price * 1.01, "low": price * 0.99, "close": price,
                "prev_close": price, "volume": 100000.0, "turnover_inr": price * 100000.0,
                "era": "udiff",
            })
            price *= 1.001
    store.write_bhavcopy_eq(pd.DataFrame(rows))


class _FakeSettings:
    price_source = "bhavcopy"
    use_point_in_time_universe = False
    return_basis = "price"
    random_seed = 42


# ---- load_and_prepare ----


def test_load_and_prepare_returns_none_for_empty_universe(tmp_store) -> None:
    candles, feats, fcols = load_and_prepare(_FakeSettings(), tmp_store, cap_band=None)
    assert candles.empty
    assert feats is None
    assert fcols is None


def test_load_and_prepare_builds_the_standard_feature_set(tmp_store) -> None:
    _write_minimal_bhavcopy(tmp_store)
    candles, feats, fcols = load_and_prepare(_FakeSettings(), tmp_store, cap_band=None)
    assert not feats.empty
    assert "mkt_ret_1b" not in fcols  # excluded everywhere in this codebase, matching every prior sweep
    assert "ret_5b" in fcols  # a real engine.py column, confirming the standard path ran


def test_load_and_prepare_merges_in_registered_factors(tmp_store) -> None:
    """THE property this whole workstream exists for: a searched factor
    reaches a sweep without editing features/engine.py."""

    @factor("my_search_candidate")
    def _candidate(g: pd.DataFrame) -> pd.Series:
        return g["close"].pct_change(3)

    _write_minimal_bhavcopy(tmp_store)
    candles, feats, fcols = load_and_prepare(
        _FakeSettings(), tmp_store, cap_band=None, extra_factor_names=("my_search_candidate",)
    )
    assert "my_search_candidate" in fcols
    assert "my_search_candidate" in feats.columns


def test_load_and_prepare_unknown_factor_raises(tmp_store) -> None:
    _write_minimal_bhavcopy(tmp_store)
    with pytest.raises(KeyError):
        load_and_prepare(_FakeSettings(), tmp_store, cap_band=None, extra_factor_names=("nope",))


# ---- run_sweep grid wiring (train_and_score_fold/simulate_portfolio mocked
# out -- this section tests the LOOP, not model training) ----


def _fold_result(alpha_net: float, fold_id: int = 0, cost_bps: float = 25.0) -> FoldResult:
    return FoldResult(
        fold_id=fold_id, horizon_bars=10, top_n=10, round_trip_cost_bps=cost_bps,
        n_rebalances=12, gross_expectancy=alpha_net + 0.001, net_expectancy=alpha_net,
        benchmark_expectancy=0.0, alpha_gross=alpha_net + 0.001, alpha_net=alpha_net,
        mean_turnover=0.5, information_coefficient=0.05, hit_rate=0.6,
    )


def test_run_sweep_gates_only_at_the_configured_cost_point(tmp_store) -> None:
    _write_minimal_bhavcopy(tmp_store)

    fake_scored = object()
    with patch("stocksense.research.harness.train_and_score_fold", return_value=fake_scored), \
         patch("stocksense.research.harness.make_folds", return_value=[type("F", (), {"fold_id": 0})()]), \
         patch("stocksense.research.harness.simulate_portfolio", side_effect=lambda scored, top_n, round_trip_cost_bps: _fold_result(0.01, cost_bps=round_trip_cost_bps)):
        config = SweepConfig(
            cap_bands=(("full_pit", None),), horizon_grid=(10,), top_n_grid=(10,),
            cost_grid_bps=(10.0, 25.0), gate_cost_bps=25.0,
        )
        result = run_sweep(_FakeSettings(), tmp_store, config)

    # both cost points produce fold rows...
    assert set(result.fold_results["cost_bps"]) == {10.0, 25.0}
    # ...but only the configured gate_cost_bps produces a verdict
    assert len(result.verdicts) == 1
    assert result.verdicts[0]["cost_bps"] == 25.0


def test_run_sweep_produces_one_verdict_per_cap_band_horizon_top_n(tmp_store) -> None:
    _write_minimal_bhavcopy(tmp_store)

    with patch("stocksense.research.harness.train_and_score_fold", return_value=object()), \
         patch("stocksense.research.harness.make_folds", return_value=[type("F", (), {"fold_id": 0})()]), \
         patch("stocksense.research.harness.simulate_portfolio", return_value=_fold_result(0.01)):
        config = SweepConfig(
            cap_bands=(("full_pit", None),), horizon_grid=(10, 20), top_n_grid=(10, 20),
            cost_grid_bps=(25.0,), gate_cost_bps=25.0,
        )
        result = run_sweep(_FakeSettings(), tmp_store, config)

    assert len(result.verdicts) == 4  # 1 cap_band x 2 horizons x 2 top_n


def test_run_sweep_skips_empty_cap_bands_without_crashing(tmp_store) -> None:
    config = SweepConfig(cap_bands=(("full_pit", None),), horizon_grid=(10,), top_n_grid=(10,), cost_grid_bps=(25.0,))
    result = run_sweep(_FakeSettings(), tmp_store, config)  # no bhavcopy data written
    assert result.fold_results.empty
    assert result.verdicts == []


# ---- write_verdict_doc ----


def test_write_verdict_doc_reports_pass_when_any_combination_passes(tmp_path) -> None:
    verdicts = [
        {"cap_band": "full_pit", "horizon_bars": 10, "top_n": 10, "cost_bps": 25.0,
         "verdict": GateVerdict(True, "all criteria passed", {"n_folds": 25, "mean_alpha_net": 0.0167, "hit_rate_pvalue": 0.0001})},
    ]
    path = tmp_path / "verdict_test.md"
    write_verdict_doc(verdicts, path)
    text = path.read_text(encoding="utf-8")
    assert "AT LEAST ONE COMBINATION PASSES" in text
    assert "PASS" in text


def test_write_verdict_doc_reports_fail_when_none_pass(tmp_path) -> None:
    verdicts = [
        {"cap_band": "large", "horizon_bars": 10, "top_n": 10, "cost_bps": 25.0,
         "verdict": GateVerdict(False, "hit rate not significant", {"n_folds": 25, "mean_alpha_net": 0.0006, "hit_rate_pvalue": 0.65})},
    ]
    path = tmp_path / "verdict_test2.md"
    write_verdict_doc(verdicts, path)
    text = path.read_text(encoding="utf-8")
    assert "ALL COMBINATIONS FAIL" in text
    assert "FAIL" in text
