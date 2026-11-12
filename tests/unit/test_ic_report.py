from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.validation.ic_report import group_gain_share, oof_report


def _frames(rho: float, n_dates: int = 75, n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = date(2019, 11, 1)
    for i in range(n_dates):
        d = d0 + timedelta(days=i)
        s = rng.normal(size=n)
        y = rho * s + np.sqrt(1 - rho**2) * rng.normal(size=n)
        rows += [(d, k, float(s[k]), float(y[k])) for k in range(n)]
    df = pl.DataFrame(rows, schema=["date", "sid", "score", "y_oc"], orient="row")
    return df.select("date", "sid", "score"), df.select("date", "sid", "y_oc")


def test_report_recovers_planted_ic_and_year_split():
    p, lab = _frames(0.2)
    r = oof_report(p, lab, "y_oc")
    assert 0.12 < r["mean_ic"] < 0.28 and r["ic_tstat"] > 5
    assert set(r["by_year"]) == {2019, 2020}
    assert len(r["deciles"]) == 10 and r["decile_spread"] > 0
    assert r["frac_pos_years"] == 1.0


def test_report_null_signal_small_ic():
    p, lab = _frames(0.0, seed=3)
    assert abs(oof_report(p, lab, "y_oc")["mean_ic"]) < 0.03


def test_group_gain_share_sums_to_one():
    s = group_gain_share({"a": 3.0, "b": 1.0, "c": 4.0}, {"a": "g1", "b": "g1", "c": "g2"})
    assert s == pytest.approx({"g1": 0.5, "g2": 0.5})
    assert group_gain_share({"a": 0.0}, {"a": "g"}) == {"g": 0.0}
