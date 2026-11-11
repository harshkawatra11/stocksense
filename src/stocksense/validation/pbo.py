from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from stocksense.validation.splits import cscv_blocks


def _sharpe(s: np.ndarray, q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-period Sharpe (ddof=1) per row from sums, sums of squares and counts; 0 where variance is 0."""
    mean = s / c[:, None]
    var = (q - c[:, None] * mean**2) / (c[:, None] - 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros_like(mean)
    np.divide(mean, sd, out=out, where=sd > 1e-300)
    return out


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> dict[str, Any]:
    """Probability of backtest overfitting by CSCV over daily return columns (one per trial).

    Returns {"pbo": share of logits <= 0, "logits": ndarray of ln(w / (1 - w)) per block split}."""
    r = np.asarray(returns, dtype=float)
    n_trials = r.shape[1]
    if n_trials < 2:
        return {"pbo": float("nan"), "logits": np.array([])}
    blocks = cscv_blocks(r.shape[0], n_blocks)
    sums = np.stack([r[a:b].sum(axis=0) for a, b in blocks])
    sqs = np.stack([(r[a:b] ** 2).sum(axis=0) for a, b in blocks])
    cnt = np.array([b - a for a, b in blocks], dtype=float)
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    is_mask = np.zeros((len(combos), n_blocks))
    is_mask[np.arange(len(combos))[:, None], combos] = 1.0
    oos_mask = 1.0 - is_mask
    sr_is = _sharpe(is_mask @ sums, is_mask @ sqs, is_mask @ cnt)
    sr_oos = _sharpe(oos_mask @ sums, oos_mask @ sqs, oos_mask @ cnt)
    best = np.argmax(sr_is, axis=1)
    rows = np.arange(len(combos))
    chosen = sr_oos[rows, best][:, None]
    # rank among out-of-sample Sharpes, 1 = worst; ties share the average rank
    rank = 1.0 + (sr_oos < chosen).sum(axis=1) + 0.5 * ((sr_oos == chosen).sum(axis=1) - 1.0)
    omega = rank / (n_trials + 1.0)
    logits = np.log(omega / (1.0 - omega))
    return {"pbo": float(np.mean(logits <= 0.0)), "logits": logits}
