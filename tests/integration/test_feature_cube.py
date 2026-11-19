from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from stocksense.core.errors import LockboxError
from stocksense.features.build import load_features
from stocksense.features.cube import FeatureCube
from stocksense.models.seqnet import date_to_days
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration

COLS = ["alpha_t", "ret_1", "vol_21"]


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("cube_world"), planted_ic=0.05, n=40)


@pytest.fixture(scope="module")
def cube(world) -> FeatureCube:
    return FeatureCube(world.settings, 1, COLS, "research")


@pytest.fixture(scope="module")
def frame(world) -> pl.DataFrame:
    return load_features(world.settings, "research", columns=COLS).collect().sort("date", "sid")


def test_shape_dtype_and_index(cube, frame):
    assert cube.data.dtype == np.float16
    assert cube.data.shape[0] == frame["date"].n_unique() and cube.data.shape[2] == 3
    assert cube.n_sids_max == frame.group_by("date").len()["len"].max()
    assert np.array_equal(cube.dates, frame["date"].unique().sort().to_numpy())
    assert np.array_equal(cube.days, date_to_days(cube.dates))
    d = frame["date"][100]
    assert (
        cube.date_idx(np.array([d], dtype="datetime64[D]"))[0]
        == np.flatnonzero(cube.dates == np.datetime64(d))[0]
    )


def test_cell_values_match_the_frame(cube, frame):
    for r in (0, 57, 1000, frame.height - 1):
        row = frame.row(r, named=True)
        di = int(cube.date_idx(np.array([row["date"]], dtype="datetime64[D]"))[0])
        x, ok = cube.window(np.array([di]), np.array([row["sid"]]), 1)
        assert ok[0, 0]
        want = np.array([row[c] if row[c] is not None else np.nan for c in COLS], dtype=np.float16)
        assert np.array_equal(x[0, 0], want.astype(np.float32), equal_nan=True)


def test_window_equals_hand_built(cube, frame):
    T = 5
    sid = int(frame["sid"].value_counts().sort("count", descending=True)["sid"][0])
    one = frame.filter(pl.col("sid") == sid).sort("date")
    di = int(cube.date_idx(np.array([one["date"][20]], dtype="datetime64[D]"))[0])
    x, ok = cube.window(np.array([di]), np.array([sid]), T)
    assert x.shape == (1, T, 3) and ok.shape == (1, T)
    for k in range(T):
        day = cube.dates[di - T + 1 + k]
        hit = one.filter(pl.col("date") == day.astype("datetime64[D]").astype(object))
        if hit.height:
            assert ok[0, k]
            want = np.array([hit[c][0] for c in COLS], dtype=np.float64).astype(np.float16).astype(np.float32)
            assert np.array_equal(x[0, k], want, equal_nan=True)
        else:
            assert not ok[0, k] and (x[0, k] == 0).all()


def test_history_before_the_first_date_is_zero_and_masked(cube, frame):
    sid = int(frame["sid"][0])
    x, ok = cube.window(np.array([2]), np.array([sid]), 6)
    assert not ok[0, :3].any() and (x[0, :3] == 0).all()


def test_unknown_sid_is_all_masked(cube):
    x, ok = cube.window(np.array([30]), np.array([987654]), 4)
    assert not ok.any() and (x == 0).all()


def test_per_row_dates_and_scalar_date(cube, frame):
    sids = frame["sid"].unique().sort().to_numpy()[:5]
    xs, oks = cube.window(30, sids, 3)
    xr, okr = cube.window(np.full(5, 30), sids, 3)
    assert np.array_equal(xs, xr, equal_nan=True) and np.array_equal(oks, okr)


def test_lockbox_guard(world):
    with pytest.raises(LockboxError):
        FeatureCube(world.settings, 1, COLS, "lockbox")


def test_start_end_limit(world):
    c = FeatureCube(world.settings, 1, COLS, "research", start=date(2019, 6, 1), end=date(2019, 6, 30))
    assert c.dates.min() >= np.datetime64("2019-06-01") and c.dates.max() <= np.datetime64("2019-06-30")
