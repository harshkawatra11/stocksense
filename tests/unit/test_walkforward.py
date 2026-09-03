"""Tests for purged, embargoed Combinatorial Purged CV.

The properties that matter: no training row can see into its test block's
label window (purge), no training row immediately after a test block can
carry its serial correlation into training (embargo), and a fold boundary
never falls inside a session (session_bounded, by construction).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stocksense.evaluation.walkforward import CVConfig, Fold, make_folds, n_cpcv_paths


def _sessions(n: int, start: date = date(2020, 1, 1)) -> list[date]:
    """n consecutive weekday dates -- a stand-in trading calendar."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ n_cpcv_paths
def test_n_cpcv_paths_matches_the_plans_own_worked_example():
    """n_folds=10, n_test_folds=2 -> 9 paths. The exact number cited in the
    plan, from k*C(N,k)/N = 2*45/10."""
    assert n_cpcv_paths(CVConfig(n_folds=10, n_test_folds=2)) == 9


@pytest.mark.parametrize(
    ("n_folds", "n_test_folds", "expected"),
    [(6, 2, 5), (8, 2, 7), (5, 1, 1), (10, 1, 1)],
)
def test_n_cpcv_paths_other_configurations(n_folds, n_test_folds, expected):
    assert n_cpcv_paths(CVConfig(n_folds=n_folds, n_test_folds=n_test_folds)) == expected


# ------------------------------------------------------------------- make_folds
def test_number_of_splits_is_n_choose_k():
    """C(10, 2) = 45 splits, exactly as the plan states."""
    dates = _sessions(500)
    folds = make_folds(dates, horizon_bars=5, cfg=CVConfig(n_folds=10, n_test_folds=2))
    assert len(folds) == 45


def test_every_session_in_every_fold_is_either_train_test_or_purged():
    """No date silently vanishes or duplicates across a single fold's
    train/test partition (purged/embargoed rows are simply excluded from
    both, never double-counted)."""
    dates = _sessions(300)
    folds = make_folds(dates, horizon_bars=3, cfg=CVConfig(n_folds=10, n_test_folds=2))
    for f in folds:
        assert set(f.train_dates).isdisjoint(f.test_dates)
        assert set(f.train_dates) <= set(dates)
        assert set(f.test_dates) <= set(dates)


def test_purge_removes_training_rows_immediately_before_the_test_block():
    """THE purge property: a training row whose forward label window would
    overlap the test set must be excluded. With horizon_bars=5, the 5
    sessions immediately before a test block must be purged from training."""
    dates = _sessions(200)
    cfg = CVConfig(n_folds=10, n_test_folds=1, embargo_pct=0.0)  # isolate purge from embargo
    folds = make_folds(dates, horizon_bars=5, cfg=cfg)

    f = folds[3]  # an interior test group, so both neighbours exist
    test_start_idx = dates.index(min(f.test_dates))
    if test_start_idx >= 5:
        pre_test = [dates[test_start_idx - i] for i in range(1, 6)]
        assert not any(d in f.train_dates for d in pre_test)


def test_purge_removes_training_rows_immediately_after_the_test_block_too():
    """The plan states purge applies on BOTH sides of the test block."""
    dates = _sessions(200)
    cfg = CVConfig(n_folds=10, n_test_folds=1, embargo_pct=0.0)
    folds = make_folds(dates, horizon_bars=5, cfg=cfg)

    f = folds[3]
    test_end_idx = dates.index(max(f.test_dates))
    post_test = [dates[test_end_idx + i] for i in range(1, 6) if test_end_idx + i < len(dates)]
    assert not any(d in f.train_dates for d in post_test)


def test_embargo_extends_the_post_test_exclusion_beyond_purge():
    """Embargo additionally removes sessions immediately after the test block,
    on top of the purge window -- killing serial correlation purging alone
    leaves. embargo_pct=0.01 on 1,000 sessions is 10 sessions."""
    dates = _sessions(1000)
    cfg = CVConfig(n_folds=10, n_test_folds=1, embargo_pct=0.01)
    folds = make_folds(dates, horizon_bars=2, cfg=cfg)

    f = folds[3]
    test_end_idx = dates.index(max(f.test_dates))
    # purge=2 + embargo=10 -> 12 sessions after the block must be excluded
    excluded_after = [
        dates[test_end_idx + i] for i in range(1, 13) if test_end_idx + i < len(dates)
    ]
    assert not any(d in f.train_dates for d in excluded_after)


def test_zero_embargo_means_only_purge_applies_after_the_block():
    dates = _sessions(300)
    cfg = CVConfig(n_folds=10, n_test_folds=1, embargo_pct=0.0)
    folds = make_folds(dates, horizon_bars=3, cfg=cfg)
    f = folds[3]
    test_end_idx = dates.index(max(f.test_dates))
    # The 4th session after the block (beyond the 3-session purge) must be trainable.
    if test_end_idx + 4 < len(dates):
        assert dates[test_end_idx + 4] in f.train_dates


def test_session_bounded_rejects_duplicate_dates():
    """A fold boundary may NEVER fall inside a session. Passing sub-session
    (duplicate-date) granularity would silently violate that, so it is
    rejected outright rather than producing a boundary that splits a session."""
    dates = _sessions(50) + [_sessions(50)[0]]  # one duplicate
    with pytest.raises(ValueError, match="unique session"):
        make_folds(dates, horizon_bars=1, cfg=CVConfig(n_folds=5, n_test_folds=1))


def test_duplicate_dates_allowed_when_session_bounded_is_explicitly_off():
    dates = _sessions(50)
    dup = dates + [dates[0]]
    # Should not raise -- session_bounded=False opts out of the guard.
    make_folds(dup, horizon_bars=1, cfg=CVConfig(n_folds=5, n_test_folds=1, session_bounded=False))


def test_train_and_test_dates_are_always_sorted():
    dates = _sessions(300)
    folds = make_folds(dates, horizon_bars=2, cfg=CVConfig(n_folds=8, n_test_folds=2))
    for f in folds:
        assert f.train_dates == sorted(f.train_dates)
        assert f.test_dates == sorted(f.test_dates)


def test_purge_bars_none_uses_horizon_bars():
    """purge_bars=None means 'use the strategy's own horizon_bars' -- proven
    by comparing against an explicit purge_bars set to the same value."""
    dates = _sessions(200)
    a = make_folds(dates, horizon_bars=4, cfg=CVConfig(n_folds=10, n_test_folds=1, purge_bars=None))
    b = make_folds(dates, horizon_bars=999, cfg=CVConfig(n_folds=10, n_test_folds=1, purge_bars=4))
    assert [f.train_dates for f in a] == [f.train_dates for f in b]


def test_n_test_folds_out_of_range_is_rejected():
    dates = _sessions(100)
    with pytest.raises(ValueError, match="n_test_folds"):
        make_folds(dates, horizon_bars=1, cfg=CVConfig(n_folds=5, n_test_folds=5))
    with pytest.raises(ValueError, match="n_test_folds"):
        make_folds(dates, horizon_bars=1, cfg=CVConfig(n_folds=5, n_test_folds=0))


def test_too_few_sessions_for_the_fold_count_is_rejected():
    dates = _sessions(3)
    with pytest.raises(ValueError, match="n_folds"):
        make_folds(dates, horizon_bars=1, cfg=CVConfig(n_folds=10, n_test_folds=2))


def test_group_sizes_are_near_equal():
    """The remainder from an uneven split must not pile onto one group."""
    dates = _sessions(103)  # 103 / 10 = 10 remainder 3
    folds = make_folds(dates, horizon_bars=0, cfg=CVConfig(n_folds=10, n_test_folds=1, embargo_pct=0))
    # Every single-group test block size is either 10 or 11.
    sizes = {len(f.test_dates) for f in folds}
    assert sizes <= {10, 11}
