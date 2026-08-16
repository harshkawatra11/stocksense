"""
Regression test for the audit finding MED-7: simulate_portfolio carried
pristine target weights forward between rebalances instead of weights
drifted by realized returns, silently understating true turnover (and
therefore true cost) on every rebalance after the first.
"""

from __future__ import annotations

import pandas as pd

from stocksense.evaluation.backtest import ScoredFold, simulate_portfolio


def test_weights_drift_with_realized_returns_between_rebalances() -> None:
    """Two rebalance dates, IDENTICAL target both times (A and B tied on
    score). Between the two dates, A gains 100% and B is flat. By the
    second rebalance, the actually-held position is no longer 50/50 —
    it's ~67/33 — so returning to a 50/50 target requires REAL turnover.

    With the pre-fix code (current_weights = target directly), the second
    rebalance's turnover would be exactly 0, because the code compared
    the new target against the undrifted (still-50/50) prior target
    rather than what was actually held. That is the bug this test pins.
    """
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")
    scored = ScoredFold(
        fold_id=0,
        horizon_bars=20,
        n_train_rows=1000,
        rebalance_dates=[d1, d2],
        scores_by_date={
            d1: pd.Series({"A": 1.0, "B": 1.0}),  # tied -> equal weight
            d2: pd.Series({"A": 1.0, "B": 1.0}),  # tied again -> same target as before
        },
        raw_actual_by_date={
            d1: pd.Series({"A": 1.00, "B": 0.00}),  # A doubles, B flat
            d2: pd.Series({"A": 0.00, "B": 0.00}),  # flat second period, isolates the drift effect
        },
        rel_actual_by_date={
            d1: pd.Series({"A": 0.50, "B": -0.50}),
            d2: pd.Series({"A": 0.00, "B": 0.00}),
        },
    )

    result = simulate_portfolio(scored, top_n=2, round_trip_cost_bps=25.0, no_trade_band=0.02)
    assert result is not None
    assert result.n_rebalances == 2

    # Turnover values aren't exposed per-rebalance on FoldResult, but cost
    # is derived from turnover, and net_returns captures the per-period
    # outcome. The second period's net return must be strictly less than
    # its gross (0.0) by the actual rebalancing cost — a bug that made
    # turnover collapse to 0 would make the second net_return exactly 0.
    second_period_net = result.net_returns[1]
    assert second_period_net < 0.0, (
        "second rebalance shows zero cost — weights were not drifted forward, "
        "turnover was computed against the wrong baseline"
    )

    # The drifted weight before the second rebalance should be close to
    # 1.0/1.5 = 0.667 for A, not the undrifted 0.5 — confirm indirectly via
    # the actual mean_turnover recorded in the returned metrics.
    assert result.mean_turnover > 0.05, "mean turnover too small — drift not being tracked"
