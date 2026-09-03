"""Market impact: the square-root law, and Almgren-Chriss optimal execution.

At 87,500 rupees on liquid names impact is small but not zero, and the
previous cost model used a flat slippage constant with no participation
check at all. These give the search a real, size-dependent cost to weigh
against expected alpha before it ever reaches a broker.
"""

from __future__ import annotations

import numpy as np


def sqrt_impact_bps(participation_rate: float, volatility_bps: float, kappa: float = 1.0) -> float:
    """Temporary impact ~ kappa * sigma * sqrt(participation_rate).

    The empirically robust form across venues and asset classes (Almgren
    et al. 2005; Toth et al. 2011): impact grows with the SQUARE ROOT of
    order size relative to volume, not linearly -- quadrupling size only
    doubles the cost. `participation_rate` is order size / expected volume
    over the execution window, so it is dimensionless and comparable
    across names.
    """
    if participation_rate < 0:
        raise ValueError("participation_rate must be non-negative")
    return float(kappa * volatility_bps * np.sqrt(participation_rate))


def almgren_chriss_trajectory(
    total_qty: float,
    n_intervals: int,
    horizon: float,
    risk_aversion: float,
    volatility: float,
    temporary_impact: float,
) -> list[float]:
    """Almgren-Chriss (2001) optimal execution trajectory: holdings remaining
    at each of n_intervals+1 equally-spaced points from 0 to `horizon`.

        kappa = sqrt(risk_aversion * volatility^2 / temporary_impact)
        x_j   = total_qty * sinh(kappa * (horizon - t_j)) / sinh(kappa * horizon)

    `risk_aversion` trades off cost (linger, minimize impact) against risk
    (rush, minimize exposure to price uncertainty). At risk_aversion == 0 the
    trajectory is the risk-neutral limit -- linear, i.e. TWAP -- taken
    explicitly rather than through kappa -> 0, since sinh(kappa*x)/sinh(kappa)
    is only numerically stable away from that limit.
    """
    if total_qty < 0:
        raise ValueError("total_qty must be non-negative")
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")

    t = np.linspace(0.0, horizon, n_intervals + 1)

    if risk_aversion == 0:
        trajectory = total_qty * (1.0 - t / horizon)
        trajectory[-1] = 0.0
        return [float(x) for x in trajectory]

    kappa = np.sqrt(risk_aversion * volatility**2 / temporary_impact)
    trajectory = total_qty * np.sinh(kappa * (horizon - t)) / np.sinh(kappa * horizon)
    trajectory[-1] = 0.0
    return [float(x) for x in trajectory]
