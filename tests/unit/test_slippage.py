from __future__ import annotations

import json
from datetime import date

import pytest

from stocksense.costs.slippage import SlippageRow, SlippageTable, estimate_open_penalty, slip_frac


@pytest.fixture
def table(tmp_data_root):
    return SlippageTable.load_latest(tmp_data_root)


def test_hand_example(table):
    # rank 120 -> bucket 200 -> 4.0 bps = 0.0004; impact 0.5*0.02*sqrt(1e5/1e8) = 0.000316228;
    # open penalty 3 bps = 0.0003 -> 0.001016228
    s = slip_frac(120, 0.02, 1e5, 1e8, "open", table)
    assert s == pytest.approx(0.001016228, abs=1e-8)
    assert round(s, 6) == 0.001016


def test_spread_buckets(table):
    # rank 30 -> 1.5 bps; rank 400 -> bucket 500 -> 12 bps; rank > 500 -> 20 bps
    assert slip_frac(30, 0.02, 0, 1e8, "now", table) == pytest.approx(0.00015)
    assert slip_frac(400, 0.02, 0, 1e8, "now", table) == pytest.approx(0.0012)
    assert slip_frac(500, 0.02, 0, 1e8, "now", table) == pytest.approx(0.0012)
    assert slip_frac(501, 0.02, 0, 1e8, "now", table) == pytest.approx(0.0020)


def test_zero_order_value_is_spread_plus_penalty(table):
    # rank 30 close: 1.5 bps + 2 bps = 0.00035
    assert slip_frac(30, 0.02, 0.0, 1e8, "close", table) == pytest.approx(0.00035)


def test_monotone_in_order_value(table):
    vals = [slip_frac(100, 0.02, v, 1e8, "open", table) for v in (0, 1e4, 1e5, 1e6, 1e7)]
    assert vals == sorted(vals) and len(set(vals)) == 5


def test_newest_row_not_after_as_of(tmp_data_root):
    # config default row plus two appended rows; rows dated after as_of are ignored
    base = SlippageTable.load_latest(tmp_data_root)
    assert base.rows[0].date == date(2026, 11, 11)
    t = SlippageTable(
        [
            base.rows[0],
            SlippageRow(date(2027, 1, 5), 1.0, 5.0, 4.0, base.rows[0].half_spread_bps_by_rank),
            SlippageRow(date(2027, 3, 1), 2.0, 6.0, 5.0, base.rows[0].half_spread_bps_by_rank),
        ]
    )
    assert t.row_at(date(2027, 2, 1)).impact_k == 1.0
    assert t.row_at(date(2027, 3, 1)).impact_k == 2.0
    assert t.row_at(date(2026, 12, 1)).impact_k == 0.5
    # impact doubles with impact_k: 1.0 * 0.02 * sqrt(1e-3) = 0.000632456; spread 4 bps; open pen 5 bps
    s = slip_frac(120, 0.02, 1e5, 1e8, "open", t, as_of=date(2027, 2, 1))
    assert s == pytest.approx(0.0004 + 0.000632456 + 0.0005, abs=1e-8)


def test_load_latest_reads_appended_rows(tmp_data_root):
    p = tmp_data_root.paths.data_root / "state" / "slippage_table.json"
    p.write_text(
        json.dumps(
            [
                {
                    "date": "2027-01-05",
                    "impact_k": 0.7,
                    "open_penalty_bps": 3.5,
                    "close_penalty_bps": 2.5,
                    "half_spread_bps_by_rank": [[50, 2.0], [500, 15.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    t = SlippageTable.load_latest(tmp_data_root)
    assert len(t.rows) == 2
    assert t.row_at(date(2027, 1, 5)).impact_k == 0.7


def test_scaled_doubles_every_component(table):
    d = table.scaled(2.0)
    one = slip_frac(120, 0.02, 1e5, 1e8, "open", table)
    two = slip_frac(120, 0.02, 1e5, 1e8, "open", d)
    assert two == pytest.approx(2 * one)


def test_estimate_open_penalty_is_deferred(tmp_data_root):
    with pytest.raises(NotImplementedError):
        estimate_open_penalty(tmp_data_root)
