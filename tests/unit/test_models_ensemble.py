from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.models.ensemble import blend_equal, blend_online, expected_return_map

H = 5  # label horizon in sessions (target y_c2o_5 -> end_5)
TARGET = "y_c2o_5"


def _world(n_dates=320, n=40, seed=0):
    """sessions are consecutive calendar days (fine: only order matters), two models: good and noise."""
    rng = np.random.default_rng(seed)
    d0 = date(2020, 1, 1)
    days = [d0 + timedelta(days=i) for i in range(n_dates + H)]
    rows_l, rows_g, rows_n = [], [], []
    for i in range(n_dates):
        sig = rng.normal(size=n)
        ret = 0.02 * (0.3 * sig + rng.normal(size=n))
        sid = np.arange(n)
        rows_l.append(pl.DataFrame({"date": days[i], "sid": sid, TARGET: ret, "end_5": days[i + H]}))
        rows_g.append(pl.DataFrame({"date": days[i], "sid": sid, "score": sig}))
        rows_n.append(pl.DataFrame({"date": days[i], "sid": sid, "score": rng.normal(size=n)}))
    cast = {"sid": pl.Int32}
    return (
        pl.concat(rows_l).cast(cast),
        pl.concat(rows_g).cast(cast),
        pl.concat(rows_n).cast(cast),
        days,
    )


def test_blend_equal_is_mean_of_ranks_in_range():
    lab, good, noise, _ = _world(n_dates=5)
    b = blend_equal({"good": good, "noise": noise})
    assert b.height == good.height
    assert b["score"].min() >= -0.5 and b["score"].max() <= 0.5
    one = blend_equal([good])
    d0 = good.filter(pl.col("date") == good["date"][0])
    assert one.filter(pl.col("date") == good["date"][0])["score"].sort().to_list() == pytest.approx(
        np.linspace(-0.5, 0.5, d0.height).tolist()
    )


def test_online_weights_sum_to_one_respect_floor_and_favour_good_model():
    lab, good, noise, days = _world()
    blend, w = blend_online({"good": good, "noise": noise}, lab, TARGET, eta=20.0, halflife=60, floor=0.05)
    arr = w.select("good", "noise").to_numpy()
    assert np.allclose(arr.sum(1), 1.0)
    assert (arr >= 0.05 - 1e-12).all()
    assert w["date"].to_list() == days[:320]
    # first date has no usable label: equal weights
    assert arr[0].tolist() == pytest.approx([0.5, 0.5])
    assert arr[250:, 0].min() > 0.8
    assert blend.height == good.height


def test_no_label_ending_at_or_after_t_influences_weights_at_t():
    lab, good, noise, days = _world()
    t = 200
    base = blend_online({"good": good, "noise": noise}, lab, TARGET)[1]
    # corrupt the labels of dates whose horizon reaches t (end >= days[t]): s in (t-H, t) -> end_5 >= days[t]
    bad = lab.with_columns(
        pl.when(pl.col("date") >= days[t - H]).then(-pl.col(TARGET)).otherwise(pl.col(TARGET)).alias(TARGET)
    )
    other = blend_online({"good": good, "noise": noise}, bad, TARGET)[1]
    a = base.filter(pl.col("date") == days[t]).drop("date").to_numpy()
    b = other.filter(pl.col("date") == days[t]).drop("date").to_numpy()
    assert np.array_equal(a, b)
    # sanity: the same corruption does change weights once those labels have ended
    late = days[t + 60]
    assert not np.allclose(
        base.filter(pl.col("date") == late).drop("date").to_numpy(),
        other.filter(pl.col("date") == late).drop("date").to_numpy(),
    )


def test_expected_return_map_slope_floor_and_causality():
    lab, good, noise, days = _world()
    blend = blend_equal({"good": good})
    er = expected_return_map(blend, lab, TARGET, window=504)
    assert set(er.columns) == {"date", "sid", "score", "exp_ret"}
    late = er.filter(pl.col("date") == days[300])
    slope = (late["exp_ret"] / late["score"]).drop_nulls().drop_nans()
    assert slope.n_unique() == 1 and slope[0] > 0
    # no labels known yet -> zero
    assert er.filter(pl.col("date") == days[0])["exp_ret"].abs().max() == 0
    # a perfectly anti-predictive model gets slope floored at 0
    flipped = expected_return_map(blend, lab.with_columns(-pl.col(TARGET)), TARGET)
    assert flipped.filter(pl.col("date") == days[300])["exp_ret"].abs().max() == 0
    # causality: labels ending at or after t do not change b(t)
    t = 250
    bad = lab.with_columns(
        pl.when(pl.col("date") >= days[t - H])
        .then(pl.col(TARGET) * 50)
        .otherwise(pl.col(TARGET))
        .alias(TARGET)
    )
    er3 = expected_return_map(blend, bad, TARGET)
    x = er.filter(pl.col("date") == days[t])["exp_ret"].to_numpy()
    y = er3.filter(pl.col("date") == days[t])["exp_ret"].to_numpy()
    assert np.array_equal(x, y)
