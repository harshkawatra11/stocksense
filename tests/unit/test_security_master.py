from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from stocksense.data import io
from stocksense.panel.security_master import build_security_master, symbol_sid_map
from tests.fixtures.handmade import eq_rows, weekdays, write_raw
from tests.fixtures.synth import make_market, write_market


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture
def synth_sm(tmp_data_root, market):
    write_market(market, tmp_data_root.paths.data_root)
    return tmp_data_root, build_security_master(tmp_data_root)


def _sid(sm: pl.DataFrame, symbol: str) -> int:
    return sm.filter(pl.col("symbol") == symbol)["sid"].item()


def test_synth_renames(synth_sm):
    s, sm = synth_sm
    # 40 companies, 42 symbols (S07N, S08N are renames of S07, S08).
    assert sm.height == 42
    assert sm["sid"].n_unique() == 40
    # S07: official edge; S08: auto-detected (no symbol_changes row).
    assert _sid(sm, "S07") == _sid(sm, "S07N")
    assert _sid(sm, "S08") == _sid(sm, "S08N")
    csv = (s.paths.data_root / "panel" / "rename_autodetect.csv").read_text(encoding="utf-8")
    assert "S08" in csv and "S08N" in csv
    assert "S07N" not in csv  # official, not auto
    # S19 (new listing at 350) is its own company; S11 (delisted) too.
    assert sm.filter(pl.col("sid") == _sid(sm, "S19")).height == 1
    assert sm.filter(pl.col("sid") == _sid(sm, "S11")).height == 1
    # sids are Int32, numbered from 1
    assert sm["sid"].dtype == pl.Int32 and sm["sid"].min() == 1
    # the file is written and readable through the catalog
    assert io.read(s, "security_master").collect().height == 42


def test_synth_valid_ranges(synth_sm, market):
    _, sm = synth_sm
    row = sm.filter(pl.col("symbol") == "S07").row(0, named=True)
    assert row["first_date"] == market.sessions[0]
    assert row["last_date"] == market.sessions[299]
    assert row["valid_from"] == row["first_date"] and row["valid_to"] == row["last_date"]
    nxt = sm.filter(pl.col("symbol") == "S07N").row(0, named=True)
    assert nxt["valid_from"] == market.sessions[300]
    assert nxt["valid_to"] == date(2099, 12, 31)
    assert nxt["isin_last"] is not None and nxt["name_last"] == "S07 Ltd"


def test_symbol_sid_map(synth_sm):
    s, sm = synth_sm
    m = symbol_sid_map(s)
    assert m.columns == ["symbol", "valid_from", "valid_to", "sid"]
    assert m.height == 42
    assert m.filter(pl.col("symbol") == "S07N")["sid"].item() == _sid(sm, "S07")


def test_unrelated_symbol_not_merged(tmp_data_root):
    d = weekdays(120)
    # X stops after day 39; Y starts on day 40 at a different price (50 vs 100): not a rename.
    x = eq_rows("XXX", d[:40], [100.0] * 40)
    y = eq_rows("YYY", d[40:], [50.0] * 80, prev_closes=[50.0] * 80)
    z = eq_rows("ZZZ", d, [10.0] * 120)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([x, y, z]))
    assert build_security_master(tmp_data_root)["sid"].n_unique() == 3


def test_autodetect_requires_data_end_gap(tmp_data_root):
    # A ends 10 weekdays before the end of data (< 30 calendar days): never auto-merged.
    d = weekdays(40)
    a = eq_rows("AAA", d[:30], [100.0] * 30)
    b = eq_rows("BBB", d[30:], [100.0] * 10, prev_closes=[100.0] * 10)
    z = eq_rows("ZZZ", d, [10.0] * 40)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, z]))
    assert build_security_master(tmp_data_root)["sid"].n_unique() == 3


def test_autodetect_positive(tmp_data_root):
    d = weekdays(120)
    a = eq_rows("AAA", d[:40], [100.0] * 40)
    # B's first row has prev_close = A's last raw close (100.0): a rename.
    b = eq_rows("BBB", d[41:], [100.0] * 79, prev_closes=[100.0] * 79)  # 2 sessions later
    z = eq_rows("ZZZ", d, [10.0] * 120)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, z]))
    sm = build_security_master(tmp_data_root)
    assert sm["sid"].n_unique() == 2
    assert _sid(sm, "AAA") == _sid(sm, "BBB")


def test_autodetect_gap_over_five_sessions_not_merged(tmp_data_root):
    d = weekdays(120)
    a = eq_rows("AAA", d[:40], [100.0] * 40)
    b = eq_rows("BBB", d[46:], [100.0] * 74, prev_closes=[100.0] * 74)  # 6 sessions later
    z = eq_rows("ZZZ", d, [10.0] * 120)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, z]))
    assert build_security_master(tmp_data_root)["sid"].n_unique() == 3


def test_autodetect_ambiguous_not_merged(tmp_data_root):
    d = weekdays(120)
    a = eq_rows("AAA", d[:40], [100.0] * 40)
    b = eq_rows("BBB", d[40:], [100.0] * 80, prev_closes=[100.0] * 80)
    c = eq_rows("CCC", d[40:], [100.0] * 80, prev_closes=[100.0] * 80)  # a second candidate
    z = eq_rows("ZZZ", d, [10.0] * 120)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, c, z]))
    assert build_security_master(tmp_data_root)["sid"].n_unique() == 4


def test_autodetect_excluded_by_fresh_listing_date(tmp_data_root):
    d = weekdays(120)
    a = eq_rows("AAA", d[:40], [100.0] * 40)
    b = eq_rows("BBB", d[40:], [100.0] * 80, prev_closes=[100.0] * 80)
    z = eq_rows("ZZZ", d, [10.0] * 120)
    listed = pl.DataFrame(
        {
            "symbol": ["BBB"],
            "name": ["BBB Ltd"],
            "series": ["EQ"],
            "listing_date": [d[41]],  # within 3 days of B's first row: a genuine new listing
            "isin": [None],
            "face_value": [10.0],
            "as_of": [d[-1]],
        },
        schema_overrides={"isin": pl.Utf8},
    )
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, z]), listed_equities=listed)
    assert build_security_master(tmp_data_root)["sid"].n_unique() == 3


def test_official_edge_outside_window_ignored(tmp_data_root):
    d = weekdays(120)
    a = eq_rows("AAA", d[:40], [100.0] * 40)
    b = eq_rows("BBB", d[60:], [70.0] * 60, prev_closes=[70.0] * 60)
    z = eq_rows("ZZZ", d, [10.0] * 120)
    sc = pl.DataFrame(
        {"company": ["x"], "old_symbol": ["AAA"], "new_symbol": ["BBB"], "change_date": [d[60]]}
    )
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, z]), symbol_changes=sc)
    # AAA's last row is ~28 days before change_date (> 15): the row is ignored.
    assert build_security_master(tmp_data_root)["sid"].n_unique() == 3


def test_official_edge_for_unknown_symbol_ignored(tmp_data_root):
    d = weekdays(30)
    z = eq_rows("ZZZ", d, [10.0] * 30)
    sc = pl.DataFrame(
        {"company": ["mf"], "old_symbol": ["FUND1"], "new_symbol": ["FUND2"], "change_date": [d[5]]}
    )
    write_raw(tmp_data_root, bhavcopy_eq=z, symbol_changes=sc)
    assert build_security_master(tmp_data_root).height == 1


def test_same_symbol_relisted_after_long_gap_is_one_sid(tmp_data_root):
    d = weekdays(400)
    r1 = eq_rows("RRR", d[:50], [100.0] * 50)
    r2 = eq_rows("RRR", d[350:], [20.0] * 50)  # 300 sessions later, a very different price
    z = eq_rows("ZZZ", d, [10.0] * 400)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([r1, r2, z]))
    sm = build_security_master(tmp_data_root)
    assert sm.filter(pl.col("symbol") == "RRR").height == 1
    assert sm["sid"].n_unique() == 2


def test_series_filter(tmp_data_root):
    d = weekdays(30)
    z = eq_rows("ZZZ", d, [10.0] * 30)
    sme = eq_rows("SMS", d, [10.0] * 30, series="SM")  # SME series: not a node
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([z, sme]))
    assert build_security_master(tmp_data_root)["symbol"].to_list() == ["ZZZ"]


def test_stable_numbering(tmp_data_root, market):
    write_market(market, tmp_data_root.paths.data_root)
    first = build_security_master(tmp_data_root)
    again = build_security_master(tmp_data_root)
    assert first.sort("symbol").equals(again.sort("symbol"))
    # 5 new symbols that start before all but the earliest existing ones
    new = pl.concat([eq_rows(f"NEW{i}", market.sessions[:100], [30.0 + i] * 100) for i in range(5)])
    write_raw(tmp_data_root, bhavcopy_eq=new)
    third = build_security_master(tmp_data_root)
    old = dict(zip(first["symbol"], first["sid"], strict=True))
    now = dict(zip(third["symbol"], third["sid"], strict=True))
    assert all(now[k] == v for k, v in old.items())
    # 40 existing sids -> new chains get 41..45
    assert sorted(now[f"NEW{i}"] for i in range(5)) == [41, 42, 43, 44, 45]


def test_existing_sids_merge_keeps_smaller_and_logs(tmp_data_root):
    d = weekdays(120)
    a = eq_rows("AAA", d[:40], [100.0] * 40)
    b = eq_rows("BBB", d[40:], [70.0] * 80, prev_closes=[70.0] * 80)
    z = eq_rows("ZZZ", d, [10.0] * 120)
    write_raw(tmp_data_root, bhavcopy_eq=pl.concat([a, b, z]))
    first = build_security_master(tmp_data_root)
    assert first["sid"].n_unique() == 3
    forced = first.with_columns(
        sid=pl.col("symbol").replace_strict({"AAA": 1, "BBB": 2, "ZZZ": 3}, return_dtype=pl.Int32)
    )
    io.write_partition(tmp_data_root, "security_master", "all", forced)
    sc = pl.DataFrame(
        {"company": ["x"], "old_symbol": ["AAA"], "new_symbol": ["BBB"], "change_date": [d[40]]}
    )
    write_raw(tmp_data_root, symbol_changes=sc)
    sm = build_security_master(tmp_data_root)
    assert _sid(sm, "AAA") == 1 and _sid(sm, "BBB") == 1
    log = (tmp_data_root.paths.data_root / "panel" / "sid_merges.csv").read_text(encoding="utf-8")
    assert "AAA" in log and "BBB" in log
