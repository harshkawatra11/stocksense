from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.data import io
from stocksense.features.base import FeatureContext
from stocksense.features.registry import load_all
from tests.fixtures.feat58 import write_env
from tests.fixtures.handmade import weekdays

REG = load_all()
NAMES = (
    "last_hour_ret",
    "first_hour_ret",
    "first5_ret",
    "vwap_dev",
    "intraday_rv",
    "rv_ratio",
    "vol_last_hour_share",
    "vol_first5_share",
    "open_drive",
)
N = 26
DS = weekdays(N, date(2024, 10, 1))
FULL = N - 1  # session with a complete set of bars
PARTIAL = N - 2  # session missing its 15:25 bar


def day_bars(sym: str, d: date, drop_last: bool = False) -> list[tuple]:
    """75 five-minute bars: open_k = close_{k-1}, close_k = 101 + k, volume 100 (400 for the first bar)."""
    rows = []
    n = 74 if drop_last else 75
    for k in range(n):
        ts = datetime.combine(d, datetime.min.time(), tzinfo=IST) + timedelta(hours=9, minutes=15 + 5 * k)
        c = 101.0 + k
        o = 100.0 if k == 0 else 100.0 + k
        rows.append((sym, ts, "FIVE_MINUTE", o, c + 0.5, o - 0.5, c, 400 if k == 0 else 100))
    return rows


def put_bars(s, rows):
    cols = ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]
    io.upsert(s, "intraday_bars", pl.DataFrame(rows, schema=cols, orient="row"))


def get(s, name, sid=1, as_of=None):
    df = REG[name].fn(FeatureContext(s, as_of)).collect().filter(pl.col("sid") == sid)
    return dict(zip(df["date"].to_list(), df[name].to_list(), strict=True))


@pytest.fixture
def world(tmp_data_root):
    s = tmp_data_root
    closes = [100.0 + (i % 3) for i in range(N)]
    write_env(s, DS, {"AAA": {"sid": 1, "close": closes}, "BBB": {"sid": 2, "close": [50.0] * N}})
    put_bars(s, day_bars("AAA", DS[FULL]) + day_bars("AAA", DS[PARTIAL], drop_last=True))
    return s, closes


def test_point_returns(world):
    s, _ = world
    d = DS[FULL]
    assert get(s, "first5_ret")[d] == pytest.approx(math.log(101 / 100))
    assert get(s, "first_hour_ret")[d] == pytest.approx(math.log(112 / 100))  # bar 10:10 closes at 101 + 11
    assert get(s, "last_hour_ret")[d] == pytest.approx(math.log(175 / 163))  # 15:25 vs 14:25 closes
    assert get(s, "last_hour_ret")[DS[PARTIAL]] is None  # the 15:25 bar is missing
    assert get(s, "first5_ret")[DS[PARTIAL]] == pytest.approx(math.log(101 / 100))


def test_shares_and_vwap(world):
    s, _ = world
    d = DS[FULL]
    assert get(s, "vol_first5_share")[d] == pytest.approx(400 / 7800)
    assert get(s, "vol_last_hour_share")[d] == pytest.approx(1200 / 7800)  # 12 bars from 14:30
    c = 101.0 + np.arange(75)
    o = np.concatenate([[100.0], 100.0 + np.arange(1, 75)])
    tp = ((c + 0.5) + (o - 0.5) + c) / 3
    vol = np.array([400.0] + [100.0] * 74)
    vwap = (tp * vol).sum() / vol.sum()
    assert get(s, "vwap_dev")[d] == pytest.approx(175.0 / vwap - 1.0)
    # incomplete sessions give no session-wide statistics
    assert get(s, "vwap_dev")[DS[PARTIAL]] is None and get(s, "vol_first5_share")[DS[PARTIAL]] is None


def test_realised_vol_and_ratio(world):
    s, closes = world
    d = DS[FULL]
    c = 101.0 + np.arange(75)
    r = np.concatenate([[math.log(101 / 100)], np.log(c[1:] / c[:-1])])
    rv = math.sqrt(float((r**2).sum()))
    assert get(s, "intraday_rv")[d] == pytest.approx(rv)
    la = np.diff(np.log(closes))  # la[k] is the return of row k + 1
    vol21 = la[FULL - 21 : FULL].std(ddof=1)
    assert get(s, "rv_ratio")[d] == pytest.approx(rv / vol21)


def test_open_drive(world):
    s, closes = world
    d = DS[FULL]
    c = np.array(closes)
    tr = np.abs(np.diff(c))  # open = close in the panel, so high = low = close: TR = |close change|
    atr = None
    for i, x in enumerate(tr):  # ewm alpha 1/14, adjust=False, seeded by the first true range (row 1)
        atr = x if atr is None else (1 - 1 / 14) * atr + x / 14
        if i == FULL - 1:
            break
    assert get(s, "open_drive")[d] == pytest.approx(math.log(106 / 100) / (atr / c[FULL]))


def test_null_without_bars_and_without_dataset(world, tmp_data_root):
    s, _ = world
    for name in NAMES:
        b = get(s, name, sid=2)
        assert len(b) == N and all(v is None for v in b.values()), name
        a = get(s, name)
        assert a[DS[3]] is None
    from tests.fixtures.synth import _settings_for

    root = s.paths.data_root.parent / "no_bars"
    root.mkdir()
    s2 = _settings_for(root)
    write_env(s2, DS, {"AAA": {"sid": 1, "close": [100.0] * N}})
    for name in NAMES:
        out = get(s2, name)
        assert len(out) == N and all(v is None for v in out.values()), name
    assert all(REG[n].label for n in NAMES)
