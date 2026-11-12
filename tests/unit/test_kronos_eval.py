"""T7.5d evaluation helpers: daily rank IC, paired t, purged expanding walk-forward ridge arms."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.models import kronos_eval as ke


def _panel(n_dates=60, n=40, signal=0.0, seed=0, horizon_days=3):
    rng = np.random.default_rng(seed)
    d0 = date(2020, 1, 1)
    rows = []
    for i in range(n_dates):
        d = d0 + timedelta(days=14 * i)
        base = rng.normal(size=(n, 3))
        kr = rng.normal(size=n)
        y = 0.3 * base[:, 0] + signal * kr + rng.normal(size=n)
        for j in range(n):
            rows.append((d, j, *base[j], kr[j], y[j], d + timedelta(days=horizon_days)))
    return pl.DataFrame(rows, schema=["date", "sid", "b0", "b1", "b2", "kr_a", "y", "end"], orient="row")


def test_daily_rank_ic_perfect_and_negative():
    df = pl.DataFrame({"date": [date(2020, 1, 1)] * 5 + [date(2020, 1, 2)] * 5,
                       "p": [1, 2, 3, 4, 5] * 2, "y": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]})  # fmt: skip
    ic = ke.daily_rank_ic(df, "p", "y").sort("date")
    assert ic["ic"].to_list() == pytest.approx([1.0, -1.0])


def test_paired_t():
    m, t, n = ke.paired_t(np.array([0.1, 0.2, 0.3, 0.2]))
    assert (
        n == 4
        and m == pytest.approx(0.2)
        and t == pytest.approx(0.2 / (np.std([0.1, 0.2, 0.3, 0.2], ddof=1) / 2))
    )


def test_incremental_detects_planted_signal_and_not_noise():
    sig = ke.incremental_walk_forward(
        _panel(signal=0.8), ["b0", "b1", "b2"], ["kr_a"], "y", "end", alpha=10.0
    )
    assert sig["mean_diff"] > 0.05 and sig["t"] > 3 and sig["n_dates"] > 10
    noi = ke.incremental_walk_forward(
        _panel(signal=0.0, seed=5), ["b0", "b1", "b2"], ["kr_a"], "y", "end", alpha=10.0
    )
    assert abs(noi["t"]) < 3


def test_folds_are_expanding_and_purged():
    df = _panel(n_dates=50)
    folds = ke.make_folds(sorted(df["date"].unique().to_list()), train_frac=0.4, n_folds=4)
    assert len(folds) == 4
    dates = sorted(df["date"].unique().to_list())
    assert folds[0][0][-1] < folds[0][1][0] and folds[0][1][0] == dates[int(0.4 * 50)]
    assert sum(len(f[1]) for f in folds) == 50 - int(0.4 * 50)
    assert all(a[1][-1] < b[1][0] for a, b in zip(folds, folds[1:], strict=False))
    # the training pool of a fold is every earlier date; purging by label end date happens in the fit
    res = ke.incremental_walk_forward(df, ["b0"], ["kr_a"], "y", "end", alpha=10.0, return_detail=True)
    assert (res["detail"]["train_end_max"] < res["detail"]["test_start"]).all()
