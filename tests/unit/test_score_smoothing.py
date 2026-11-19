"""G2-C2: causal score EMA and the convex (decile, isotonic) expected-return map."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.models.ensemble import (
    _ema_scores,
    expected_return_map_decile,
    smooth_scores,
)
from stocksense.validation.g1 import G1Config, _spec_s1, blend_models

D0 = date(2020, 1, 1)
TARGET = "y_c2o_5"  # end column end_5
LAG = 7  # a label ends 7 calendar days after its date


def dates(n: int) -> list[date]:
    return [D0 + timedelta(days=i) for i in range(n)]


# ---- smoothing --------------------------------------------------------------------------------------


def test_ema_matches_the_hand_numbers():
    f = pl.DataFrame({"date": dates(3), "sid": [1, 1, 1], "score": [-0.5, 0.5, 0.5]})
    out = _ema_scores(f, span=3).sort("date")  # alpha = 2 / (3 + 1) = 0.5
    assert out["score"].to_list() == pytest.approx([-0.5, 0.0, 0.25])


def test_ema_restarts_after_a_gap_of_more_than_five_sessions():
    ds = dates(12)
    keep = [0, 1, 2] + [9, 10, 11]  # rows 2 and 9 are 7 sessions apart: gap 7 > 5
    f = pl.DataFrame({"date": [ds[i] for i in keep], "sid": 1, "score": [0.5] * 3 + [-0.5] * 3})
    other = pl.DataFrame({"date": ds, "sid": 2, "score": [0.0] * 12})  # keeps the session grid full
    out = _ema_scores(pl.concat([f, other]), span=3).filter(pl.col("sid") == 1).sort("date")
    v = out["score"].to_list()
    assert v[3] == pytest.approx(-0.5)  # restarted: no memory of the +0.5 run
    assert v[4] == pytest.approx(-0.5)
    # rows exactly 5 sessions apart (gap 5, not more than 5) do not restart
    keep2 = [0, 1, 2, 7, 8, 9]
    f2 = pl.DataFrame({"date": [ds[i] for i in keep2], "sid": 1, "score": [0.5] * 3 + [-0.5] * 3})
    out2 = _ema_scores(pl.concat([f2, other]), span=3).filter(pl.col("sid") == 1).sort("date")
    assert out2["score"].to_list()[3] == pytest.approx(0.5 * -0.5 + 0.5 * 0.5)


def test_smooth_scores_reranks_to_half_open_range_and_is_causal():
    rng = np.random.default_rng(3)
    ds = dates(40)
    rows = [(d, s, float(rng.uniform(-0.5, 0.5))) for d in ds for s in range(25)]
    f = pl.DataFrame(rows, schema=["date", "sid", "score"], orient="row")
    out = smooth_scores(f, span=5)
    g = out.filter(pl.col("date") == ds[-1])
    assert g["score"].min() == pytest.approx(-0.5) and g["score"].max() == pytest.approx(0.5)
    cut = smooth_scores(f.filter(pl.col("date") <= ds[20]), span=5)
    a = out.filter(pl.col("date") <= ds[20]).sort("date", "sid")
    b = cut.sort("date", "sid")
    assert a["score"].to_list() == pytest.approx(b["score"].to_list())  # later dates change nothing
    # smoothing really does something
    assert out.sort("date", "sid")["score"].to_list() != f.sort("date", "sid")["score"].to_list()


# ---- decile map -------------------------------------------------------------------------------------


def _frames(n_dates: int, n_sid: int = 100, label_of_decile=None):
    """Every date: sid i scores i / n_sid - 0.5 (decile i // 10); label = f(decile)."""
    ds = dates(n_dates)
    label_of_decile = label_of_decile or (lambda j: 0.001 * j)
    rows, lab = [], []
    for d in ds:
        for s in range(n_sid):
            rows.append((d, s, s / n_sid - 0.5))
            lab.append((d, s, label_of_decile(s // (n_sid // 10)), d + timedelta(days=LAG)))
    blend = pl.DataFrame(rows, schema=["date", "sid", "score"], orient="row")
    labels = pl.DataFrame(lab, schema=["date", "sid", TARGET, "end_5"], orient="row")
    return blend, labels


def _decile_of(out: pl.DataFrame, d: date) -> list[float]:
    g = out.filter(pl.col("date") == d).sort("sid")
    return [g["exp_ret"][j * 10] for j in range(10)]


def test_decile_map_recovers_the_planted_slope():
    blend, labels = _frames(60)
    out = expected_return_map_decile(blend, labels, TARGET, shrink_k=0)
    assert _decile_of(out, dates(60)[-1]) == pytest.approx([0.001 * j for j in range(10)])
    # before any label has ended the map is zero (nothing known, nothing traded)
    assert set(out.filter(pl.col("date") == dates(60)[0])["exp_ret"].to_list()) == {0.0}


def test_decile_map_shrinks_toward_the_grand_mean():
    blend, labels = _frames(60)
    t = dates(60)[-1]
    n_known = sum(1 for s in dates(60) if s + timedelta(days=LAG) < t)  # 52 dates, 10 names per decile
    n_j = 10 * n_known
    out = expected_return_map_decile(blend, labels, TARGET, shrink_k=n_j)
    mean = 0.0045
    expect = [0.5 * 0.001 * j + 0.5 * mean for j in range(10)]
    assert _decile_of(out, t) == pytest.approx(expect)


def test_isotonic_repairs_one_inversion():
    # decile 3 pays 0.004 and decile 4 only 0.002: pooled to 0.003 each (equal weights); the rest unchanged
    base = lambda j: {3: 0.004, 4: 0.002}.get(j, 0.001 * j)  # noqa: E731
    blend, labels = _frames(60, label_of_decile=base)
    v = _decile_of(expected_return_map_decile(blend, labels, TARGET, shrink_k=0), dates(60)[-1])
    assert v[3] == pytest.approx(0.003) and v[4] == pytest.approx(0.003)
    assert all(b >= a - 1e-12 for a, b in zip(v, v[1:], strict=False))
    assert v[0] == pytest.approx(0.0) and v[9] == pytest.approx(0.009)


def test_decile_map_never_uses_a_label_that_ends_on_or_after_t():
    blend, labels = _frames(30)
    ds = dates(30)
    t = ds[12]  # known: dates s with s + 7 < t, i.e. s <= 4
    poisoned = labels.with_columns(
        pl.when(pl.col("date").is_between(ds[5], ds[11])).then(1.0).otherwise(pl.col(TARGET)).alias(TARGET)
    )
    a = expected_return_map_decile(blend, labels, TARGET, shrink_k=0).filter(pl.col("date") == t)
    b = expected_return_map_decile(blend, poisoned, TARGET, shrink_k=0).filter(pl.col("date") == t)
    assert a.sort("sid")["exp_ret"].to_list() == pytest.approx(b.sort("sid")["exp_ret"].to_list())
    # control: one session later date 5 is known and the poisoned labels do change the map
    t2 = ds[13]
    a2 = expected_return_map_decile(blend, labels, TARGET, shrink_k=0).filter(pl.col("date") == t2)
    b2 = expected_return_map_decile(blend, poisoned, TARGET, shrink_k=0).filter(pl.col("date") == t2)
    assert a2.sort("sid")["exp_ret"].to_list() != pytest.approx(b2.sort("sid")["exp_ret"].to_list())


# ---- G1 wiring --------------------------------------------------------------------------------------


def test_spec_keys_appear_only_when_non_default():
    cfg = G1Config(capitals=(5e5,), reference_capital=5e5)
    base = _spec_s1(cfg, "y_c2o_10", 10, False, [])
    assert "score_ema" not in base and "er_map" not in base
    from dataclasses import replace

    spec = _spec_s1(replace(cfg, score_ema=5, er_map="decile"), "y_c2o_10", 10, False, [])
    assert spec["score_ema"] == 5 and spec["er_map"] == "decile"
    assert "score_ema" not in _spec_s1(replace(cfg, er_map="decile"), "y_c2o_10", 10, False, [])
    assert "er_map" not in _spec_s1(replace(cfg, score_ema=3), "y_c2o_10", 10, False, [])


def test_blend_models_applies_the_ema_and_decile_options():
    blend, labels = _frames(80)
    rng = np.random.default_rng(0)
    noisy = blend.with_columns((pl.col("score") + pl.Series(rng.normal(0, 0.3, blend.height))).alias("score"))
    frames = {"m": noisy}
    plain = blend_models(frames, labels, TARGET)
    smooth = blend_models(frames, labels, TARGET, score_ema=5, er_map="decile")
    assert plain.frame.height == smooth.frame.height
    assert set(smooth.frame.columns) >= {"date", "sid", "score", "exp_ret", "disagreement"}
    a = plain.frame.sort("date", "sid")["score"].to_numpy()
    b = smooth.frame.sort("date", "sid")["score"].to_numpy()
    assert not np.allclose(a, b)
    last = smooth.frame.filter(pl.col("date") == dates(80)[-1]).sort("score")
    assert last["exp_ret"].to_list() == sorted(last["exp_ret"].to_list())  # convex map is monotone
