"""Market impact: square-root law and Almgren-Chriss optimal execution."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.microstructure.impact import (
    almgren_chriss_trajectory,
    sqrt_impact_bps,
)


def test_sqrt_impact_is_zero_at_zero_participation():
    assert sqrt_impact_bps(participation_rate=0.0, volatility_bps=100.0) == 0.0


def test_sqrt_impact_is_monotone_increasing_in_size():
    small = sqrt_impact_bps(participation_rate=0.01, volatility_bps=100.0)
    large = sqrt_impact_bps(participation_rate=0.25, volatility_bps=100.0)
    assert 0.0 < small < large


def test_sqrt_impact_rejects_negative_participation():
    with pytest.raises(ValueError):
        sqrt_impact_bps(participation_rate=-0.1, volatility_bps=100.0)


def test_sqrt_impact_quadrupling_size_doubles_impact():
    # the "square-root" in the law: impact ~ sqrt(participation).
    base = sqrt_impact_bps(participation_rate=0.01, volatility_bps=100.0)
    quadrupled = sqrt_impact_bps(participation_rate=0.04, volatility_bps=100.0)
    assert quadrupled == pytest.approx(2 * base)


def test_almgren_chriss_starts_full_and_ends_flat():
    trajectory = almgren_chriss_trajectory(
        total_qty=1000.0, n_intervals=10, horizon=1.0,
        risk_aversion=1e-6, volatility=0.02, temporary_impact=0.1,
    )
    assert trajectory[0] == pytest.approx(1000.0)
    assert trajectory[-1] == pytest.approx(0.0, abs=1e-6)
    assert len(trajectory) == 11


def test_almgren_chriss_is_monotonically_decreasing():
    trajectory = almgren_chriss_trajectory(
        total_qty=1000.0, n_intervals=10, horizon=1.0,
        risk_aversion=5.0, volatility=0.02, temporary_impact=0.1,
    )
    assert np.all(np.diff(trajectory) <= 0)


def test_almgren_chriss_zero_risk_aversion_is_linear_twap():
    trajectory = almgren_chriss_trajectory(
        total_qty=1000.0, n_intervals=4, horizon=1.0,
        risk_aversion=0.0, volatility=0.02, temporary_impact=0.1,
    )
    assert trajectory == pytest.approx([1000.0, 750.0, 500.0, 250.0, 0.0])


def test_almgren_chriss_higher_risk_aversion_front_loads_execution():
    # more risk-averse -> liquidate faster early -> less remaining at t=1
    # than a less risk-averse trajectory over the same horizon.
    cautious = almgren_chriss_trajectory(
        total_qty=1000.0, n_intervals=10, horizon=1.0,
        risk_aversion=0.001, volatility=0.02, temporary_impact=0.1,
    )
    urgent = almgren_chriss_trajectory(
        total_qty=1000.0, n_intervals=10, horizon=1.0,
        risk_aversion=10.0, volatility=0.02, temporary_impact=0.1,
    )
    assert urgent[1] < cautious[1]
