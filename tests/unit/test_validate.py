from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from stocksense.data import io
from stocksense.data.validate import Issue, validate_dataset
from tests.fixtures.synth import make_market, write_market

MIN_ROWS = 10  # the synthetic market has 40 stocks, not 500+


@pytest.fixture(scope="module")
def market():
    return make_market(n=40, n_sessions=60, seed=3)


def _store(market, root, eq=None):
    write_market(market if eq is None else replace(market, eq=eq), root)


def _codes(issues: list[Issue]) -> list[str]:
    return [i.code for i in issues]


def _vix(settings, dates, closes):
    """Write a small india_vix file bypassing upsert de-duplication (raw pl write)."""
    path = io.write_partition(
        settings, "india_vix", "all", pl.DataFrame({"date": [date(2026, 1, 5)], "close": [1.0]})
    )
    pl.DataFrame({"date": dates, "close": closes}).write_parquet(path)


def test_clean_synthetic_market_has_zero_issues(market, tmp_data_root, tmp_data_dir):
    _store(market, tmp_data_dir)
    assert validate_dataset(tmp_data_root, "bhavcopy_eq", min_session_rows=MIN_ROWS) == []
    assert validate_dataset(tmp_data_root, "bhavcopy_delivery") == []


def test_deleted_session_gives_one_missing_session_before_next(market, tmp_data_root, tmp_data_dir):
    sessions = sorted(market.eq["date"].unique().to_list())
    gone, nxt = sessions[30], sessions[31]
    _store(market, tmp_data_dir, eq=market.eq.filter(pl.col("date") != gone))
    issues = validate_dataset(tmp_data_root, "bhavcopy_eq", min_session_rows=MIN_ROWS)
    assert _codes(issues) == [f"missing_session_before:{nxt}"]
    assert issues[0].date == nxt


def test_high_below_close_is_one_issue(market, tmp_data_root, tmp_data_dir):
    eq = market.eq
    # row 5: high is forced to half the close, so high < close
    bad = eq.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(pl.col("close") * 0.5)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    _store(market, tmp_data_dir, eq=bad)
    issues = validate_dataset(tmp_data_root, "bhavcopy_eq", min_session_rows=MIN_ROWS)
    assert _codes(issues) == ["high_below_max_open_close"]


def test_duplicate_key_is_one_issue(tmp_data_root):
    # 2026-01-05 appears twice
    _vix(tmp_data_root, [date(2026, 1, 5)] * 2 + [date(2026, 1, 6)], [1.0, 2.0, 3.0])
    assert _codes(validate_dataset(tmp_data_root, "india_vix")) == ["duplicate_key"]


def test_null_date_is_one_issue(tmp_data_root):
    _vix(tmp_data_root, [date(2026, 1, 5), None], [1.0, 2.0])
    assert _codes(validate_dataset(tmp_data_root, "india_vix")) == ["null_date"]


def test_small_session_flagged(market, tmp_data_root, tmp_data_dir):
    sessions = sorted(market.eq["date"].unique().to_list())
    d = sessions[40]
    # keep only 20 of ~40 EQ rows on d: far below 85% of the trailing 20-session median
    keep = market.eq.filter(pl.col("date") == d).head(20)
    eq = pl.concat([market.eq.filter(pl.col("date") != d), keep])
    _store(market, tmp_data_dir, eq=eq)
    issues = validate_dataset(tmp_data_root, "bhavcopy_eq", min_session_rows=MIN_ROWS)
    assert any(i.code == "session_row_count_off" and i.date == d for i in issues)


def test_min_session_rows_default_is_500(market, tmp_data_root, tmp_data_dir):
    _store(market, tmp_data_dir)
    issues = validate_dataset(tmp_data_root, "bhavcopy_eq")
    # 40 stocks per session << 500: one session_too_small issue per session
    assert {i.code for i in issues} == {"session_too_small"}
    assert len(issues) == market.eq["date"].n_unique()


def test_unknown_column_flagged(tmp_data_root):
    path = io.write_partition(
        tmp_data_root, "india_vix", "all", pl.DataFrame({"date": [date(2026, 1, 5)], "close": [1.0]})
    )
    pl.DataFrame({"date": [date(2026, 1, 5)], "close": [1.0], "bogus": [1]}).write_parquet(path)
    assert _codes(validate_dataset(tmp_data_root, "india_vix")) == ["unexpected_column:bogus"]


def test_range_filter_restricts_issues(market, tmp_data_root, tmp_data_dir):
    sessions = sorted(market.eq["date"].unique().to_list())
    _store(market, tmp_data_dir, eq=market.eq.filter(pl.col("date") != sessions[30]))
    out = validate_dataset(
        tmp_data_root, "bhavcopy_eq", start=sessions[0], end=sessions[20], min_session_rows=MIN_ROWS
    )
    assert out == []


@pytest.mark.realdata
def test_real_store_has_no_missing_sessions_after_t23():
    from stocksense.core.config import load_settings

    issues = validate_dataset(load_settings(), "bhavcopy_eq")
    found = [i for i in issues if i.code.startswith("missing_session_before:")]
    # PLAN B.3 Finding 2 (19 missing special sessions) was repaired by T2.3's backfill
    assert found == []
