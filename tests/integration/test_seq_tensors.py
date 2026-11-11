from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from stocksense.data import io
from stocksense.features import seq
from stocksense.models.seqnet import MemmapSeqProvider, date_to_days
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("seq_world"), planted_ic=0.0, n=40)


def test_tensors_shape_dtype_index_and_provider(world):
    counts = seq.build_sequence_tensors(world.settings, version=1)
    assert counts and sum(counts.values()) > 1000
    root = seq.seq_root(world.settings, 1)
    year = sorted(counts)[-1]
    x = np.load(root / f"X_{year}.npy", mmap_mode="r")
    idx = pl.read_parquet(root / f"index_{year}.parquet")
    assert x.dtype == np.float16 and x.shape == (counts[year], 60, 12)
    assert idx.columns == ["row", "date", "sid"] and idx.height == counts[year]
    assert np.abs(x.astype(np.float32)).max() <= 10.0 and np.isfinite(x.astype(np.float32)).all()
    prov = MemmapSeqProvider(root)
    d = idx["date"].to_numpy()[[0, 5, 9]]
    s = idx["sid"].to_numpy()[[0, 5, 9]]
    got, valid = prov.get(date_to_days(d), s)
    assert valid.all() and np.array_equal(got, np.asarray(x[[0, 5, 9]]))
    _, miss = prov.get(np.array([0]), np.array([1]))
    assert not miss.any()


def test_windows_are_causal(world):
    """Channels of a row do not change when every later session is removed from the inputs."""
    daily = io.read(world.settings, "panel_daily").collect()
    market = io.read(world.settings, "panel_market").collect()
    cut = sorted(daily["date"].unique().to_list())[400]
    full = seq.compute_channels(daily, market, None)
    part = seq.compute_channels(
        daily.filter(pl.col("date") <= cut), market.filter(pl.col("date") <= cut), None
    )
    a = full.filter(pl.col("date") <= cut).sort("sid", "date")
    b = part.sort("sid", "date")
    assert a.height == b.height > 0
    for c in (*seq.CHANNELS, "vol63"):
        x, y = a[c].to_numpy(), b[c].to_numpy()
        assert np.allclose(x, y, equal_nan=True, atol=1e-9), c


def test_wide_span_rule_drops_windows():
    sid = np.zeros(200, dtype=int)
    sx = np.arange(200)
    sx[100:] += 40  # a long suspension at row 100
    v = np.full(200, 0.02)
    rows = seq.window_rows(sid, sx, v)
    assert rows.min() == 60
    assert not np.any((rows >= 100) & (rows < 100 + 59))
    assert date(2020, 1, 1)  # keep the import used
