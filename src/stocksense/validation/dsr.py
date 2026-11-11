from __future__ import annotations

import math

from scipy.stats import norm

_EULER_GAMMA = 0.5772156649


def deflated_sharpe(
    sr: float, T: int, skew: float, kurt: float, n_trials: int, var_sr_trials: float
) -> float:
    """Deflated Sharpe ratio as a probability in [0, 1]; `sr` is the per-period (daily) Sharpe, `kurt`
    is non-excess kurtosis (normal = 3), `var_sr_trials` the variance of per-period Sharpes across trials."""
    n = max(n_trials, 2)
    v = var_sr_trials if math.isfinite(var_sr_trials) and var_sr_trials > 0 else 1.0 / T
    sr0 = math.sqrt(v) * (
        (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n) + _EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e))
    )
    denom = 1 - skew * sr + ((kurt - 1) / 4) * sr**2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)))
