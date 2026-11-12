"""Kronos features (PLAN T7.5c): the seven kr_* features read per-date files computed by
`compute_kronos_day`; a tiny random-weight model (STOCKSENSE_KRONOS_TEST_MODEL=1) keeps this on CPU."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from stocksense import panel as panel_pkg
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features import kronos_feats as kf
from stocksense.features.base import FeatureContext
from stocksense.features.registry import load_all
from stocksense.models import kronos_model as km
from tests.fixtures.synth import _settings_for, make_market, write_market

NAMES = ("kr_ret5_mean", "kr_ret5_std", "kr_up_prob", "kr_oc1_mean", "kr_co1_mean", "kr_range5", "kr_volchg5")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("kronos_feats")
    m = make_market()
    write_market(m, root)
    s = _settings_for(root)
    panel_pkg.build_all(s)
    return s, m.sessions


@pytest.fixture(scope="module")
def bundle():
    return km.build_test_bundle()


def test_features_registered_as_xs_in_group_kronos():
    reg = load_all()
    for n in NAMES:
        assert reg[n].kind == "xs" and reg[n].group == "kronos" and reg[n].window == kf.LOOKBACK


def test_features_from_paths_hand_computed():
    last_close = np.array([100.0])
    # 2 paths, 5 steps, OHLCVA
    p = np.zeros((1, 2, 5, 6))
    p[..., 0] = 100.0  # open
    p[..., 1] = 110.0  # high
    p[..., 2] = 95.0  # low
    p[..., 3] = 100.0  # close
    p[0, 0, 4, 3] = 110.0  # path 0 closes up on step 5
    p[0, 1, 4, 3] = 90.0
    p[0, 0, 0, 0] = 102.0  # open[0] of path 0
    p[0, 0, 0, 3] = 104.04  # close[0] of path 0 -> oc1 = +2%
    p[..., 4] = 300.0
    out = kf.features_from_paths(p, last_close, np.array([100.0]))
    assert out["kr_ret5_mean"][0] == pytest.approx(0.0)
    assert out["kr_ret5_std"][0] == pytest.approx(0.1)
    assert out["kr_up_prob"][0] == pytest.approx(0.5)
    assert out["kr_oc1_mean"][0] == pytest.approx((0.02 + 0.0) / 2)
    assert out["kr_co1_mean"][0] == pytest.approx((0.02 + 0.0) / 2)
    assert out["kr_volchg5"][0] == pytest.approx(3.0)
    assert out["kr_range5"][0] > 0


def test_features_from_paths_nonfinite_becomes_nan():
    p = np.ones((1, 2, 5, 6))
    p[0, 0, 0, 0] = 0.0  # open[0] = 0 -> oc1 infinite
    out = kf.features_from_paths(p, np.array([1.0]), np.array([0.0]))
    assert math.isnan(out["kr_volchg5"][0])  # divide by zero mean volume


def test_compute_day_writes_file_and_is_resumable(env, bundle):
    s, sessions = env
    d = sessions[320]
    done = kf.compute_kronos_range(s, [d], "t1", bundle)
    assert done == [d]
    f = Paths(s).features / "kronos" / "t1" / f"{d.isoformat()}.parquet"
    df = pl.read_parquet(f)
    assert df.columns == ["date", "sid", *NAMES] and df.height > 0
    assert (df["date"] == d).all() and df["sid"].n_unique() == df.height
    assert kf.compute_kronos_range(s, [d], "t1", bundle) == []  # skipped: file exists
    again = kf.compute_kronos_day(s, d, "t1", bundle)
    assert again.equals(df)  # deterministic recompute


def test_early_date_has_no_eligible_series_and_still_writes_a_file(env, bundle):
    s, sessions = env
    d = sessions[100]  # fewer than 256 rows
    kf.compute_kronos_range(s, [d], "t2", bundle)
    df = pl.read_parquet(Paths(s).features / "kronos" / "t2" / f"{d.isoformat()}.parquet")
    assert df.height == 0 and df.columns == ["date", "sid", *NAMES]


def test_day_is_independent_of_later_data(env, bundle):
    """A range run loads the store up to its LAST date; the day it computes for an earlier date must equal
    the single-day computation that only sees data up to that date."""
    s, sessions = env
    d, later = sessions[330], sessions[345]
    kf.compute_kronos_range(s, [d, later], "t3", bundle)
    from_range = pl.read_parquet(Paths(s).features / "kronos" / "t3" / f"{d.isoformat()}.parquet")
    alone = kf.compute_kronos_day(s, d, "t3-alone", bundle)
    assert from_range.sort("sid").equals(alone.sort("sid")) and alone.height > 0


def test_feature_reads_files_null_where_missing_and_respects_as_of(env, bundle, monkeypatch):
    s, sessions = env
    monkeypatch.setenv("STOCKSENSE_KRONOS_TAG", "t4")
    ds = [sessions[300], sessions[340]]
    kf.compute_kronos_range(s, ds, "t4", bundle)
    spec = load_all()["kr_up_prob"]
    full = spec.fn(FeatureContext(s)).collect()
    assert set(full["date"].unique().to_list()) == set(ds)
    cut = spec.fn(FeatureContext(s, as_of=sessions[320])).collect()
    assert set(cut["date"].unique().to_list()) == {sessions[300]}
    assert full.filter(pl.col("date") == ds[0]).sort("sid").equals(cut.sort("sid"))
    # a tag with no files -> empty frame with the right schema, not an error
    monkeypatch.setenv("STOCKSENSE_KRONOS_TAG", "nothing")
    e = spec.fn(FeatureContext(s)).collect()
    assert e.height == 0 and e.columns == ["date", "sid", "kr_up_prob"]
    assert io.read(s, "panel_daily").collect().height > 0


def test_custom_lookback_paths_and_turnover_rank_universe(env, bundle):
    """T7.5d variant: lookback 128, 4 paths, universe = turnover_rank <= max_rank (not u500)."""
    s, sessions = env
    d = sessions[300]
    full = kf.compute_kronos_day(s, d, "v", bundle)
    cfg = kf.KronosConfig(lookback=128, n_paths=4, max_rank=5)
    df = kf.compute_kronos_day(s, d, "v", bundle, config=cfg)
    uni = io.read(s, "panel_universe").filter(pl.col("date") == d)
    top = uni.filter(pl.col("turnover_rank") <= 5).collect()
    assert 0 < df.height <= top.height < full.height
    assert set(df["sid"].to_list()) <= set(top["sid"].to_list())
    assert df.columns == ["date", "sid", *NAMES]
    done = kf.compute_kronos_range(s, [d], "v2", bundle, config=cfg)
    assert done == [d] and kf.compute_kronos_range(s, [d], "v2", bundle, config=cfg) == []
