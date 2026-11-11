from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from stocksense.core.clock import SteppingClock
from stocksense.core.errors import DataValidationError
from stocksense.data import io
from stocksense.data.catalog import CATALOG

IST = "Asia/Kolkata"
EQ_COLS = ["symbol", "series", "date", "open", "high", "low", "close", "prev_close", "last_price"]


def _eq_frame(rows: list[tuple], unit: str = "us", with_isin: bool = False, n_trades_int: bool = False):
    """rows: (symbol, series, datetime, close). Old-style: naive timestamp date."""
    df = pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "series": [r[1] for r in rows],
            "date": pl.Series([r[2] for r in rows], dtype=pl.Datetime(unit)),
            "open": [r[3] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[3] for r in rows],
            "prev_close": [r[3] for r in rows],
            "last_price": [r[3] for r in rows],
            "volume": [100] * len(rows),
            "turnover_inr": [1000.0] * len(rows),
            "n_trades": pl.Series([5] * len(rows), dtype=pl.Int64 if n_trades_int else pl.Float64),
            "era": ["udiff"] * len(rows),
        }
    )
    if with_isin:
        df = df.with_columns(isin=pl.lit("INE001A01036"))
    return df


def _put(root: Path, dataset: str, fname: str, df: pl.DataFrame) -> None:
    d = root / "parquet" / dataset
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / fname)


# ---- catalog ----------------------------------------------------------------------------------


def test_catalog_covers_every_d5_dataset():
    expected = {
        # RAW adopted
        "bhavcopy_eq", "bhavcopy_delivery", "corporate_actions", "fo_bhavcopy", "india_vix",
        "intraday_bars", "nse_snapshots",
        # RAW new
        "index_daily", "symbol_changes", "listed_equities", "etf_list", "sectors", "price_bands",
        "holidays", "announcements", "board_meetings", "results_index", "macro_daily", "participant_oi",
        "fii_dii", "deals", "spreads", "preopen", "quotes_1m", "bars_1m", "news_items",
        # DERIVED
        "security_master", "panel_daily", "panel_universe", "panel_market", "adj_events", "features",
        "labels", "preds",
    }  # fmt: skip
    assert set(CATALOG) == expected
    # partition rule names
    assert {s.partition for s in CATALOG.values()} <= {"month", "year", "all", "day"}
    eq = CATALOG["bhavcopy_eq"]
    assert eq.key == ("symbol", "series", "date") and eq.partition == "month" and eq.date_col == "date"
    assert eq.schema["date"] == pl.Date and eq.schema["n_trades"] == pl.Float64
    assert CATALOG["intraday_bars"].schema["ts"] == pl.Datetime("us", IST)


# ---- read -------------------------------------------------------------------------------------


def test_read_mixed_units_and_dtypes(tmp_data_root):
    root = tmp_data_root.paths.data_root
    _put(root, "bhavcopy_eq", "part-2026-08.parquet",
         _eq_frame([("AAA", "EQ", datetime(2026, 8, 3), 10.0)], unit="ms", n_trades_int=True))  # fmt: skip
    _put(root, "bhavcopy_eq", "part-2026-09.parquet",
         _eq_frame([("AAA", "EQ", datetime(2026, 9, 1), 11.0)], unit="us"))  # fmt: skip
    df = io.read(tmp_data_root, "bhavcopy_eq").collect()
    assert df.height == 2
    assert df.schema["date"] == pl.Date and df.schema["n_trades"] == pl.Float64
    assert df.schema["volume"] == pl.Int64
    assert df["date"].to_list() == [date(2026, 8, 3), date(2026, 9, 1)]


def test_missing_new_column_reads_as_null_and_extra_ignored(tmp_data_root):
    root = tmp_data_root.paths.data_root
    _put(root, "bhavcopy_eq", "part-2026-08.parquet", _eq_frame([("AAA", "EQ", datetime(2026, 8, 3), 10.0)]))
    new = _eq_frame([("AAA", "EQ", datetime(2026, 9, 1), 11.0)], with_isin=True).with_columns(junk=pl.lit(1))
    _put(root, "bhavcopy_eq", "part-2026-09.parquet", new)
    df = io.read(tmp_data_root, "bhavcopy_eq").collect()
    assert df["isin"].to_list() == [None, "INE001A01036"]
    assert "junk" not in df.columns
    assert df.schema["name"] == pl.Utf8 and df["name"].null_count() == 2
    assert df.columns == list(CATALOG["bhavcopy_eq"].schema)


def test_read_columns_subset_and_missing_catalog_column(tmp_data_root):
    root = tmp_data_root.paths.data_root
    # a file holding only `date` (as the calendar tests do) is readable with columns=["date"]
    _put(root, "bhavcopy_eq", "part-2026-09.parquet", pl.DataFrame({"date": [datetime(2026, 9, 1)]}))
    df = io.read(tmp_data_root, "bhavcopy_eq", columns=["date"]).collect()
    assert df.columns == ["date"] and df["date"].to_list() == [date(2026, 9, 1)]
    df = io.read(tmp_data_root, "bhavcopy_eq", columns=["date", "symbol"]).collect()
    assert df.columns == ["date", "symbol"] and df["symbol"].to_list() == [None]


def test_read_start_end_and_file_pruning(tmp_data_root):
    root = tmp_data_root.paths.data_root
    rows = [("AAA", "EQ", datetime(2026, 9, d), float(d)) for d in (1, 2, 3, 4)]
    _put(root, "bhavcopy_eq", "part-2026-09.parquet", _eq_frame(rows))
    # an unreadable file outside the range must never be opened
    (root / "parquet" / "bhavcopy_eq" / "part-2020-01.parquet").write_bytes(b"not parquet")
    df = io.read(tmp_data_root, "bhavcopy_eq", start=date(2026, 9, 2), end=date(2026, 9, 3)).collect()
    assert df["date"].to_list() == [date(2026, 9, 2), date(2026, 9, 3)]
    df = io.read(tmp_data_root, "bhavcopy_eq", start=date(2026, 9, 4)).collect()
    assert df["date"].to_list() == [date(2026, 9, 4)]
    assert io.read(tmp_data_root, "bhavcopy_eq", start=date(2027, 1, 1)).collect().height == 0


def test_read_empty_dataset_returns_schema(tmp_data_root):
    lf = io.read(tmp_data_root, "bhavcopy_eq")
    df = lf.collect()
    assert df.height == 0
    assert dict(df.schema) == CATALOG["bhavcopy_eq"].schema
    assert dict(io.read(tmp_data_root, "bhavcopy_eq", columns=["date", "close"]).collect().schema) == {
        "date": pl.Date, "close": pl.Float64,
    }  # fmt: skip


def test_read_intraday_utc_becomes_ist(tmp_data_root):
    root = tmp_data_root.paths.data_root
    ts = pl.Series([datetime(2026, 9, 17, 3, 45), datetime(2026, 9, 17, 3, 50)]).dt.replace_time_zone("UTC")
    raw = pl.DataFrame({"symbol": ["AAA"] * 2, "ts": ts, "interval": ["FIVE_MINUTE"] * 2,
                        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
                        "volume": [1, 2]})  # fmt: skip
    _put(root, "intraday_bars", "part-2026-09.parquet", raw)
    df = io.read(tmp_data_root, "intraday_bars", start=date(2026, 9, 17), end=date(2026, 9, 17)).collect()
    assert df.schema["ts"] == pl.Datetime("us", IST)
    # 03:45 UTC = 09:15 IST
    assert df["ts"][0].hour == 9 and df["ts"][0].minute == 15


def test_read_nested_dataset_by_sub(tmp_data_root):
    root = tmp_data_root.paths.data_root
    d = root / "parquet" / "nse_snapshots" / "gainers"
    d.mkdir(parents=True)
    pl.DataFrame({"symbol": ["A"], "pct": [1.5]}).write_parquet(d / "2026-09-18.parquet")
    pl.DataFrame({"symbol": ["B"], "pct": [2.5]}).write_parquet(d / "2026-09-19.parquet")
    df = io.read(tmp_data_root, "nse_snapshots", sub="gainers", start=date(2026, 9, 19)).collect()
    assert df["symbol"].to_list() == ["B"] and df["pct"].to_list() == [2.5]
    with pytest.raises(DataValidationError):
        io.read(tmp_data_root, "nse_snapshots")  # nested dataset needs `sub`
    with pytest.raises(DataValidationError):
        io.read(tmp_data_root, "no_such_dataset")


def test_list_partitions(tmp_data_root):
    root = tmp_data_root.paths.data_root
    assert io.list_partitions(tmp_data_root, "bhavcopy_eq") == []
    _put(root, "bhavcopy_eq", "part-2026-09.parquet", _eq_frame([("A", "EQ", datetime(2026, 9, 1), 1.0)]))
    _put(root, "bhavcopy_eq", "part-2026-08.parquet", _eq_frame([("A", "EQ", datetime(2026, 8, 1), 1.0)]))
    names = [p.name for p in io.list_partitions(tmp_data_root, "bhavcopy_eq")]
    assert names == ["part-2026-08.parquet", "part-2026-09.parquet"]


# ---- write_partition / upsert -----------------------------------------------------------------


def _vix(d: date, close: float) -> pl.DataFrame:
    return pl.DataFrame({"date": [d], "close": [close]})


def test_write_partition_roundtrip_sorted_no_temp_left(tmp_data_root):
    df = pl.concat([_vix(date(2026, 9, 2), 2.0), _vix(date(2026, 9, 1), 1.0)])
    path = io.write_partition(tmp_data_root, "india_vix", "all", df)
    assert path.name == "part-all.parquet"
    assert [p.name for p in path.parent.iterdir()] == ["part-all.parquet"]
    out = io.read(tmp_data_root, "india_vix").collect()
    assert out["date"].to_list() == [date(2026, 9, 1), date(2026, 9, 2)]  # sorted by key


def test_write_partition_missing_key_column(tmp_data_root):
    with pytest.raises(DataValidationError, match="date"):
        io.write_partition(tmp_data_root, "india_vix", "all", pl.DataFrame({"close": [1.0]}))


def test_write_failure_keeps_old_file_and_removes_temp(tmp_data_root, monkeypatch):
    path = io.write_partition(tmp_data_root, "india_vix", "all", _vix(date(2026, 9, 1), 1.0))
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disk exploded")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="exploded"):
        io.write_partition(tmp_data_root, "india_vix", "all", _vix(date(2026, 9, 2), 2.0))
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["part-all.parquet"]


def test_write_retries_permission_error_five_times(tmp_data_root, monkeypatch):
    calls = []

    def locked(src, dst):
        calls.append(1)
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", locked)
    clock = SteppingClock(datetime(2026, 11, 12, 10, 0, tzinfo=io.IST))
    t0 = clock.now()
    with pytest.raises(PermissionError):
        io.write_partition(tmp_data_root, "india_vix", "all", _vix(date(2026, 9, 1), 1.0), clock=clock)
    # 1 try + 5 retries, 0.5 s apart on the injected clock (no real waiting)
    assert len(calls) == 6
    assert (clock.now() - t0).total_seconds() == 2.5
    assert list(tmp_data_root.paths.data_root.joinpath("parquet", "india_vix").iterdir()) == []


def test_write_retry_then_success(tmp_data_root, monkeypatch):
    real = os.replace
    state = {"n": 0}

    def flaky(src, dst):
        state["n"] += 1
        if state["n"] < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    clock = SteppingClock(datetime(2026, 11, 12, 10, 0, tzinfo=io.IST))
    io.write_partition(tmp_data_root, "india_vix", "all", _vix(date(2026, 9, 1), 1.0), clock=clock)
    assert io.read(tmp_data_root, "india_vix").collect().height == 1


def test_upsert_idempotent_and_replaces_on_key(tmp_data_root):
    new = _eq_frame(
        [
            ("AAA", "EQ", datetime(2026, 9, 1), 10.0),
            ("BBB", "EQ", datetime(2026, 9, 1), 20.0),
            ("AAA", "EQ", datetime(2026, 10, 1), 30.0),
        ]  # fmt: skip
    )
    io.upsert(tmp_data_root, "bhavcopy_eq", new)
    a = io.read(tmp_data_root, "bhavcopy_eq").collect()
    io.upsert(tmp_data_root, "bhavcopy_eq", new)
    b = io.read(tmp_data_root, "bhavcopy_eq").collect()
    assert a.equals(b) and a.height == 3
    # routed to month partitions
    assert [p.name for p in io.list_partitions(tmp_data_root, "bhavcopy_eq")] == [
        "part-2026-09.parquet", "part-2026-10.parquet",
    ]  # fmt: skip
    # same key, new close: replaces
    upd = _eq_frame([("AAA", "EQ", datetime(2026, 9, 1), 99.0)])
    io.upsert(tmp_data_root, "bhavcopy_eq", upd)
    c = io.read(tmp_data_root, "bhavcopy_eq").filter(pl.col("symbol") == "AAA").collect().sort("date")
    assert (
        c["close"].to_list() == [99.0, 30.0] and io.read(tmp_data_root, "bhavcopy_eq").collect().height == 3
    )


def test_upsert_merges_old_partition_without_isin(tmp_data_root):
    root = tmp_data_root.paths.data_root
    _put(root, "bhavcopy_eq", "part-2026-09.parquet",
         _eq_frame([("AAA", "EQ", datetime(2026, 9, 1), 10.0)], unit="ms"))  # fmt: skip
    io.upsert(tmp_data_root, "bhavcopy_eq",
              _eq_frame([("BBB", "EQ", datetime(2026, 9, 2), 20.0)], with_isin=True))  # fmt: skip
    df = io.read(tmp_data_root, "bhavcopy_eq").sort("symbol").collect()
    assert df["symbol"].to_list() == ["AAA", "BBB"]
    assert df["isin"].to_list() == [None, "INE001A01036"]
    assert df.schema["date"] == pl.Date


def test_upsert_null_key_rejected(tmp_data_root):
    bad = _eq_frame([("AAA", "EQ", datetime(2026, 9, 1), 10.0)]).with_columns(symbol=pl.lit(None, pl.Utf8))
    with pytest.raises(DataValidationError, match="symbol"):
        io.upsert(tmp_data_root, "bhavcopy_eq", bad)
    assert io.list_partitions(tmp_data_root, "bhavcopy_eq") == []


def test_upsert_fo_futures_with_null_strike_allowed(tmp_data_root):
    fut = pl.DataFrame(
        {
            "symbol": ["AAA"], "instrument_type": ["STF"], "expiry_date": [date(2026, 9, 29)],
            "strike_price": pl.Series([None], dtype=pl.Float64),
            "option_type": pl.Series([None], dtype=pl.Utf8),
            "date": [date(2026, 9, 1)], "close": [10.0],
        }
    )  # fmt: skip
    io.upsert(tmp_data_root, "fo_bhavcopy", fut)
    io.upsert(tmp_data_root, "fo_bhavcopy", fut)  # null keys compare equal: still one row
    assert io.read(tmp_data_root, "fo_bhavcopy").collect().height == 1


def test_upsert_intraday_partitions_by_ist_date(tmp_data_root):
    ts = [datetime(2026, 9, 30, 23, 0, tzinfo=io.IST), datetime(2026, 10, 1, 9, 15, tzinfo=io.IST)]
    df = pl.DataFrame({"symbol": ["A", "A"], "ts": ts, "interval": ["FIVE_MINUTE"] * 2, "close": [1.0, 2.0]})
    io.upsert(tmp_data_root, "intraday_bars", df)
    assert [p.name for p in io.list_partitions(tmp_data_root, "intraday_bars")] == [
        "part-2026-09.parquet", "part-2026-10.parquet",
    ]  # fmt: skip


def test_write_partition_nested_and_day(tmp_data_root):
    df = pl.DataFrame({"date": [date(2026, 9, 1)], "sid": pl.Series([1], dtype=pl.Int32), "x": [1.5]})
    p = io.write_partition(tmp_data_root, "features", "2026", df, sub="v1")
    assert p == tmp_data_root.paths.data_root / "features" / "v1" / "part-2026.parquet"
    out = io.read(tmp_data_root, "features", sub="v1").collect()
    assert out.columns == ["date", "sid", "x"]  # extra columns kept for keep_extra datasets
    with pytest.raises(DataValidationError):
        io.write_partition(tmp_data_root, "india_vix", "bogus", _vix(date(2026, 9, 1), 1.0))


# ---- real data --------------------------------------------------------------------------------


@pytest.mark.realdata
def test_real_bhavcopy_eq_one_day():
    from stocksense.core.config import load_settings

    s = load_settings()
    df = io.read(s, "bhavcopy_eq", start=date(2026, 9, 29), end=date(2026, 9, 29)).collect()
    assert df.schema["date"] == pl.Date
    # 3,688 is the TOTAL row count for the date (all series); series EQ alone is 2,671
    assert df.height == 3688 and df.filter(pl.col("series") == "EQ").height == 2671
    assert set(df["date"].unique().to_list()) == {date(2026, 9, 29)}


@pytest.mark.realdata
def test_real_corporate_actions_rows():
    from stocksense.core.config import load_settings

    df = io.read(load_settings(), "corporate_actions").collect()
    assert df.height == 35319
    assert df.schema["ex_date"] == pl.Date and df["record_date"].null_count() == df.height


@pytest.mark.realdata
def test_real_intraday_ts_is_ist():
    from stocksense.core.config import load_settings

    df = io.read(load_settings(), "intraday_bars", start=date(2026, 9, 17), end=date(2026, 9, 17)).collect()
    assert df.height > 0
    assert df.schema["ts"] == pl.Datetime("us", IST)
    assert set(df["ts"].dt.date().unique().to_list()) == {date(2026, 9, 17)}
    assert df["ts"].dt.hour().min() == 9
