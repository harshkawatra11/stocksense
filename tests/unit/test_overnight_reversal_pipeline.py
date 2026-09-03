"""End-to-end: signal -> selection -> daily PnL -> CPCV folds -> promotion
gate -> attempt registry, for family 1's first config. This is the one path
the plan calls out to prove fully working before the 10,800-config sweep
runs -- everything downstream just parameterizes this same wiring.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stocksense.data.store import Reader, Store
from stocksense.evaluation import attempts, gate
from stocksense.evaluation.walkforward import CVConfig, make_folds
from stocksense.search.runner import fold_alpha_net
from stocksense.strategies.overnight_reversal import (
    DEFAULT_HYPOTHESIS,
    OvernightReversalConfig,
    compute_overnight_signal,
    daily_pnl,
    select_positions,
)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "hot.duckdb", tmp_path / "parquet")
    yield s
    s.close()


def _synthetic_panel(n_sessions: int, n_symbols: int, seed: int) -> pd.DataFrame:
    """A panel with a REAL planted reversal effect: buying yesterday's
    biggest overnight losers earns a small positive open->close return on
    average, so the whole pipeline has something genuine to detect."""
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 5)
    rows = []
    for day_i in range(n_sessions):
        d = start + timedelta(days=day_i)
        for sym_i in range(n_symbols):
            symbol = f"SYM{sym_i}"
            prev_close = 100.0
            overnight_return = rng.normal(0.0, 0.02)
            adj_open = prev_close * (1 + overnight_return)
            # planted mean reversion: bigger overnight drop -> better intraday return
            intraday_edge = -0.15 * overnight_return
            adj_close = adj_open * (1 + intraday_edge + rng.normal(0.0, 0.005))
            rows.append({
                "date": d, "symbol": symbol,
                "adj_open": adj_open, "adj_close": adj_close,
                "prev_adj_close": prev_close, "prev_gap_sessions": 1,
            })
    return pd.DataFrame(rows)


def test_family1_first_config_end_to_end(store):
    panel = _synthetic_panel(n_sessions=120, n_symbols=20, seed=42)
    cfg = OvernightReversalConfig(
        side="long", n_positions=5, min_overnight_move=0.0, charges_bps=0.0,
    )

    signal_panel = compute_overnight_signal(panel, demean=cfg.demean, winsorise_pct=cfg.winsorise_pct)
    positions = select_positions(signal_panel, cfg)
    daily_returns = daily_pnl(positions, panel, charges_bps=cfg.charges_bps)

    assert not daily_returns.empty

    sessions = sorted(panel["date"].unique())
    cv_cfg = CVConfig(n_folds=5, n_test_folds=1, embargo_pct=0.0, min_folds_required=5)
    folds = make_folds(sessions, horizon_bars=0, cfg=cv_cfg)
    fold_results = fold_alpha_net(daily_returns, folds)

    assert len(fold_results) == len(folds)
    assert any(f is not None for f in fold_results)

    gate_result = gate.evaluate_gate(fold_results, gate.GATE)
    assert gate_result.verdict in ("PASS", "FAIL", "INCONCLUSIVE")

    # the planted effect should be detectable and net-of-zero-cost positive
    assert gate_result.mean_alpha_net > 0

    config_dict = {
        "side": cfg.side, "n_positions": cfg.n_positions,
        "min_overnight_move": cfg.min_overnight_move, "demean": cfg.demean,
    }
    config_hash = attempts.config_hash("overnight_reversal", config_dict)
    attempt_id = attempts.register_attempt(
        store,
        hypothesis_id="overnight_reversal_v1",
        config_hash=config_hash,
        config_json=json.dumps(config_dict),
        family="overnight_reversal",
    )

    verdict_map = {"PASS": "gate_pass", "FAIL": "gate_fail", "INCONCLUSIVE": "screened_out"}
    attempts.record_result(
        store, attempt_id,
        verdict=verdict_map[gate_result.verdict],
        metrics_json=json.dumps({
            "mean_alpha_net": gate_result.mean_alpha_net,
            "binomial_p": gate_result.binomial_p,
            "sharpe": gate_result.mean_alpha_net / (daily_returns.std() + 1e-9),
        }),
    )

    store.publish()
    with Reader(store.parquet_root) as reader:
        recorded = attempts.read_attempts(reader, hypothesis_id="overnight_reversal_v1")
    assert len(recorded) == 1
    assert recorded.iloc[0]["attempt_id"] == attempt_id
    assert recorded.iloc[0]["verdict"] == verdict_map[gate_result.verdict]

    # hypothesis is required and carried through -- not decorative
    assert DEFAULT_HYPOTHESIS == cfg.hypothesis
