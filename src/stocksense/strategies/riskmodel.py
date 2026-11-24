"""Statistical risk model (151 Trading Strategies section 6.3.1, eqs. 425-427) and mean-variance weights
(section 3.18.1, eq. 358). Shared by Appendix A, the stat-arb scores and the dispersion trial.

The model is written from the paper's formulas: Psi = diag(xi^2) + sum_{A<=K} lambda(A) V(A) V(A)^T with
xi^2_i = 1 - sum_A lambda(A) V(A)_i^2, K from the effective rank of the correlation matrix.
"""

from __future__ import annotations

import numpy as np

_EIG_TOL = 1e-12


def stat_risk_model(returns: np.ndarray, k: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xi2 [N], lam [K], V [N,K]) of the correlation matrix of `returns` [N stocks x T periods].

    K is the effective rank round(exp(-sum p ln p)), p = lambda / sum(lambda) over the positive eigenvalues,
    unless `k` is given. Rows with no variance get no loading (xi2 = 1).
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x**2).sum(axis=1))
    z = np.divide(x, sd[:, None], out=np.zeros_like(x), where=sd[:, None] > 0)  # unit-norm rows
    rho = z @ z.T
    w, v = np.linalg.eigh(rho)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    pos = w > _EIG_TOL * max(w[0], 1e-300)
    if k is None:
        p = w[pos] / w[pos].sum()
        k = int(round(float(np.exp(-(p * np.log(p)).sum()))))
    k = int(max(1, min(k, int(pos.sum()), n)))
    lam, vec = w[:k], v[:, :k]
    xi2 = np.maximum(1.0 - (vec**2) @ lam, 1e-8)
    return xi2, lam, vec


def psi_matrix(xi2: np.ndarray, lam: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The model correlation matrix Psi [N,N] (eq. 425)."""
    return np.diag(xi2) + (v * lam) @ v.T


def inv_cov(xi2: np.ndarray, lam: np.ndarray, v: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
    """Inverse of C = diag(sigma) Psi diag(sigma) [N,N] by the Woodbury identity (sigma=None: of Psi)."""
    d_inv = 1.0 / xi2
    vd = v * d_inv[:, None]
    mid = np.linalg.inv(np.diag(1.0 / lam) + v.T @ vd)
    inv_psi = np.diag(d_inv) - vd @ mid @ vd.T
    if sigma is None:
        return inv_psi
    s = 1.0 / np.asarray(sigma, dtype=float)
    return inv_psi * np.outer(s, s)


def _water_fill(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clip to the bounds, then spread the residual sum over names with slack until sum(w) = 0."""
    w = np.clip(w, lower, upper)
    if not (np.isfinite(lower).any() or np.isfinite(upper).any()):
        return w
    for _ in range(100):
        gap = -w.sum()
        if abs(gap) < 1e-13:
            break
        room = np.minimum((upper - w) if gap > 0 else (w - lower), 1e12)
        open_ = room > 1e-15
        if not open_.any():
            break
        share = np.where(open_, room, 0.0)
        step = np.minimum(abs(gap) * share / share.sum(), room)
        w = w + np.sign(gap) * step
    return w


def mv_weights(
    e: np.ndarray,
    inv_c: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dollar_neutral: bool = True,
) -> np.ndarray:
    """Mean-variance weights w = C^-1 E - C^-1 1 (1' C^-1 E)/(1' C^-1 1) (eq. 358, lambda = 1).

    With finite bounds the clipped names are frozen, the rest are re-solved on the sub-block of C^-1 with the
    remaining budget, at most 50 times, then a final water-fill enforces sum(w) = 0 inside the bounds. This
    approximates the paper's bopt.calc.opt; the approximation is recorded in DECISIONS.
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    one = np.ones(n)

    def solve(idx: np.ndarray, budget: float) -> np.ndarray:
        a = inv_c[np.ix_(idx, idx)]
        ce = a @ e[idx]
        if not dollar_neutral:
            return ce
        c1 = a @ one[idx]
        return ce - c1 * (one[idx] @ ce - budget) / (one[idx] @ c1)

    free = np.ones(n, dtype=bool)
    w = np.zeros(n)
    for _ in range(50):
        idx = np.flatnonzero(free)
        if not len(idx):
            break
        budget = -float(w[~free].sum())
        w[idx] = solve(idx, budget)
        over, under = w > upper + 1e-15, w < lower - 1e-15
        bad = free & (over | under)
        if not bad.any():
            break
        w[bad & over] = upper[bad & over]
        w[bad & under] = lower[bad & under]
        free &= ~bad
    return _water_fill(w, lower, upper) if dollar_neutral else np.clip(w, lower, upper)
