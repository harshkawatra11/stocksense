from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import load_settings
from stocksense.validation.splits import (
    cscv_blocks,
    inner_validation_split,
    sample_weights,
    train_mask,
    walk_forward_folds,
)

SET = load_settings(Path(__file__).resolve().parents[2] / "config", Path("no.env"))


def _weekdays(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


SESS = _weekdays(date(2010, 1, 4), date(2026, 3, 31))
CAL = TradingCalendar.from_dates(SESS)
POS = {d: i for i, d in enumerate(SESS)}


def test_folds_tile_2014_to_research_end():
    folds = walk_forward_folds(SET, CAL)
    # test years 2014..2025 = 12 folds; lockbox_start 2025-10-01 -> research_end 2025-09-30
    assert len(folds) == 12
    assert [f.name for f in folds] == [str(y) for y in range(2014, 2026)]
    assert folds[0].test_start == date(2014, 1, 1)
    assert folds[-1].test_end == date(2025, 9, 30)
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end + timedelta(days=1) == b.test_start  # no gap, no overlap
        assert a.test_end == date(a.test_start.year, 12, 31)
    # train_end = last session before test_start: 2013-12-31 is a Tuesday session
    assert folds[0].train_end == date(2013, 12, 31)


def test_train_mask_hand_example():
    f = next(x for x in walk_forward_folds(SET, CAL) if x.name == "2020")
    # test_start 2020-01-01 (a weekday session here); 5 sessions before: 12-31, 30, 27, 26, 25 -> 2019-12-25
    # row 2019-12-17 with horizon 5 ends 12-24 (< cutoff, kept); row 2019-12-18 ends 12-25 (not < cutoff)
    dates = pl.Series([date(2019, 12, 17), date(2019, 12, 18), date(2010, 12, 31)])
    ends = pl.Series([date(2019, 12, 24), date(2019, 12, 25), date(2011, 1, 6)])
    m = train_mask(ends, f, 5, calendar=CAL, dates=dates, dev_start=date(2011, 1, 3))
    assert m.to_list() == [True, False, False]  # third row is before dev_start


@settings(max_examples=40, deadline=None)
@given(h=st.integers(1, 20), fold_i=st.integers(0, 11), emb=st.integers(1, 10))
def test_no_training_label_reaches_the_embargo(h, fold_i, emb):
    f = walk_forward_folds(SET, CAL)[fold_i]
    dates = [d for d in SESS if date(2011, 1, 3) <= d <= date(2025, 12, 31)]
    ends = [SESS[min(POS[d] + h, len(SESS) - 1)] for d in dates]
    m = train_mask(pl.Series(ends), f, emb, calendar=CAL, dates=pl.Series(dates), dev_start=date(2011, 1, 3))
    s0 = CAL.next_session(f.test_start - timedelta(days=1))
    limit = CAL.shift(s0, -emb)
    kept = [e for e, k in zip(ends, m.to_list(), strict=True) if k]
    assert kept and all(e < limit for e in kept)
    # and every row ending before the limit is kept (nothing dropped needlessly)
    assert len(kept) == sum(1 for e in ends if e < limit)


def test_weights_boundary_and_three_years_back():
    f = next(x for x in walk_forward_folds(SET, CAL) if x.name == "2020")
    s0 = date(2020, 1, 1)
    back = SESS[POS[s0] - 756]
    # 0 sessions back -> 0.5**0 = 1.0 ; 756 sessions back -> 0.5**1 = 0.5 ; 378 back -> 0.5**0.5
    w = sample_weights(pl.Series([s0, back, SESS[POS[s0] - 378]]), f, CAL)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(0.5**0.5)


def test_inner_validation_split():
    # training rows: every session 2018-01-01 .. 2019-12-31; the last 12 months start 2019-01-01
    dates = [d for d in SESS if date(2018, 1, 1) <= d <= date(2019, 12, 31)]
    h = 5
    ends = [SESS[POS[d] + h] for d in dates]
    tr, va = inner_validation_split(pl.Series(dates), pl.Series(ends), 12, 5, calendar=CAL)
    val_start = date(2019, 1, 1)  # a Tuesday session; 5 sessions before: 2018-12-25
    cutoff = date(2018, 12, 25)
    d_arr = np.array(dates)
    assert va.to_list() == [d >= val_start for d in dates]
    assert tr.to_list() == [(d < val_start and e < cutoff) for d, e in zip(dates, ends, strict=True)]
    # last inner-train row: date 2018-12-17 ends 12-24 (< 12-25); 2018-12-18 ends 12-25 (excluded)
    assert max(d_arr[np.array(tr.to_list())]) == date(2018, 12, 17)


def test_cscv_blocks():
    b = cscv_blocks(3000, 16)
    # 3000 / 16 = 187.5 -> first 8 blocks of 188, last 8 of 187; contiguous, covering [0, 3000)
    assert len(b) == 16
    assert b[0] == (0, 188) and b[7] == (1316, 1504) and b[8] == (1504, 1691) and b[-1] == (2813, 3000)
    assert all(b[i][1] == b[i + 1][0] for i in range(15))
    assert [x[1] - x[0] for x in b] == [188] * 8 + [187] * 8
