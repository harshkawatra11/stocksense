"""Purged, embargoed Combinatorial Cross-Validation (CPCV). PROTECTED.

Two distinct leakage-prevention operations, both required -- dropping either
one leaks (Lopez de Prado, Advances in Financial Machine Learning):

  Purge   -- remove training observations whose LABEL WINDOW overlaps the test
             set. With a 10-day forward label, a training row dated 5 sessions
             before the test block still "knows" 5 sessions of test-period
             returns. purge_bars = horizon_bars of the strategy, applied on
             BOTH sides of every test block (before AND after -- a strategy's
             own features may look backward from a training row too, so the
             conservative choice purges symmetrically).
  Embargo -- additionally drop training observations in the window
             IMMEDIATELY AFTER the test block, killing the serial correlation
             that purging alone leaves. embargo_pct = 0.01 is the AFML
             convention -- on a 4,100-session history that is 41 sessions.

Combinatorial Purged CV, not a single walk-forward: with n_folds=10,
n_test_folds=2 there are C(10,2) = 45 train/test splits. That matters twice
over -- 45 splits are far harder to fool than one, and evaluation.robustness's
PBO needs a MATRIX of paths as its input, which a single walk-forward cannot
supply.

All arithmetic here operates on SESSION INDEX POSITIONS in a sorted, de-
duplicated date sequence, never on calendar-day gaps -- exactly the
session-vs-calendar-day discipline used everywhere else in this codebase,
because NSE trading days are not evenly spaced.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CVConfig:
    n_folds: int = 10  # groups the timeline is cut into
    n_test_folds: int = 2  # groups held out per split
    purge_bars: int | None = None  # None -> horizon_bars of the strategy
    embargo_pct: float = 0.01  # 41 sessions on a 4,100-session history
    min_folds_required: int = 10  # gate refuses to rule on fewer
    session_bounded: bool = True  # a fold boundary may NEVER fall inside a session


@dataclass(frozen=True)
class Fold:
    fold_id: int
    test_group_ids: tuple[int, ...]  # which of the n_folds groups are held out
    train_dates: list[date]
    test_dates: list[date]


def n_cpcv_paths(cfg: CVConfig) -> int:
    """k * C(N, k) / N -- the number of distinct backtest PATHS the C(N,k)
    splits recombine into (Lopez de Prado, AFML ch.12). Verified:
    n_folds=10, n_test_folds=2 -> 9 paths, matching the plan's own example.
    """
    n, k = cfg.n_folds, cfg.n_test_folds
    return k * math.comb(n, k) // n


def _group_boundaries(n_sessions: int, n_folds: int) -> list[tuple[int, int]]:
    """Split session index positions [0, n_sessions) into n_folds contiguous,
    near-equal groups. Returns [(start, end), ...] with end exclusive.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_sessions < n_folds:
        raise ValueError(f"need at least n_folds={n_folds} sessions, got {n_sessions}")
    base, remainder = divmod(n_sessions, n_folds)
    bounds, start = [], 0
    for i in range(n_folds):
        size = base + (1 if i < remainder else 0)  # spread the remainder across the first groups
        bounds.append((start, start + size))
        start += size
    return bounds


def make_folds(dates: list[date], horizon_bars: int, cfg: CVConfig = CVConfig()) -> list[Fold]:
    """The C(n_folds, n_test_folds) purged, embargoed CPCV splits.

    `dates` must be the unique, sorted trading-session calendar for the
    strategy's universe -- one entry per SESSION, never per bar. That is what
    makes `session_bounded` hold by construction: a fold boundary is always a
    boundary between whole sessions, because sessions are the atomic unit
    groups are built from.
    """
    if cfg.n_test_folds < 1 or cfg.n_test_folds >= cfg.n_folds:
        raise ValueError("n_test_folds must be in [1, n_folds)")
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be >= 0")

    sessions = sorted(set(dates))
    if cfg.session_bounded and len(sessions) != len(dates):
        raise ValueError(
            "make_folds requires one row per unique session; duplicate dates were passed "
            "(session_bounded=True forbids sub-session granularity here)"
        )

    n = len(sessions)
    groups = _group_boundaries(n, cfg.n_folds)
    purge = cfg.purge_bars if cfg.purge_bars is not None else horizon_bars
    embargo_n = max(1, round(cfg.embargo_pct * n)) if cfg.embargo_pct > 0 else 0

    folds = []
    for fold_id, test_group_ids in enumerate(
        itertools.combinations(range(cfg.n_folds), cfg.n_test_folds)
    ):
        test_idx: set[int] = set()
        excluded_idx: set[int] = set()
        for gid in test_group_ids:
            s, e = groups[gid]
            test_idx.update(range(s, e))
            excluded_idx.update(range(max(0, s - purge), s))  # purge BEFORE
            excluded_idx.update(range(e, min(n, e + purge + embargo_n)))  # purge+embargo AFTER
        excluded_idx |= test_idx

        train_idx = sorted(set(range(n)) - excluded_idx)
        folds.append(
            Fold(
                fold_id=fold_id,
                test_group_ids=test_group_ids,
                train_dates=[sessions[i] for i in train_idx],
                test_dates=sorted(sessions[i] for i in test_idx),
            )
        )
    return folds
