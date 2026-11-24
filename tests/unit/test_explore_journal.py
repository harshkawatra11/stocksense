from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import polars as pl
import pytest

from stocksense.core.state import State
from stocksense.data import io
from stocksense.strategies import journal
from tests.unit.test_explore_scan import sessions

N = 300
DAYS = sessions()[:N]
SIGNAL_I = 280


@pytest.fixture
def state(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    return st


def row(sym="HND", strat="ml_swing_vcp", p_in=100.0, stop=95.0, target=115.0, **kw):
    return {"sym": sym, "strat": strat, "state": "SIGNAL", "p_in": p_in, "stop": stop, "target": target, **kw}


def write_hand_panel(
    settings, high_at: dict[int, float] | None = None, open_at: dict[int, float] | None = None
):
    """One flat stock (open 100, high 101, low 99, close 100) with overrides, plus a rising Nifty."""
    o, h, lo, c = (np.full(N, v) for v in (100.0, 101.0, 99.0, 100.0))
    for i, v in (high_at or {}).items():
        h[i] = v
    for i, v in (open_at or {}).items():
        o[i] = v
    rows = [(1, "HND", d, "EQ", 100, o[i], h[i], lo[i], c[i], 1e6) for i, d in enumerate(DAYS)]
    cols = ["sid", "symbol", "date", "series", "volume", "adj_open", "adj_high", "adj_low", "adj_close"]
    io.upsert(settings, "panel_daily", pl.DataFrame(rows, schema=[*cols, "adj_volume"], orient="row"))
    idx = [("NIFTY 50", d, 1e4 + i, 1e4 + i, 1e4 + i, 1e4 + i) for i, d in enumerate(DAYS)]
    icols = ["index_name", "date", "open", "high", "low", "close"]
    io.upsert(settings, "index_daily", pl.DataFrame(idx, schema=icols, orient="row"))


def test_update_of_an_immutable_field_is_refused(state, fixed_clock):
    assert journal.record_signals(state, [row()], fixed_clock, trade_date=date(2026, 11, 10)) == 1
    con = state.connect()
    for col, val in (
        ("ref_price", 1.0),
        ("stop", 1.0),
        ("target", 1.0),
        ("sym", "X"),
        ("trade_date", "2000-01-01"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(f"UPDATE lab_journal SET {col}=?", (val,))
    con.execute("UPDATE lab_journal SET status='open', reason='ok'")  # status fields may move
    assert con.execute("SELECT status FROM lab_journal").fetchone()[0] == "open"
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE lab_journal SET status='bogus'")


def test_record_signals_is_idempotent_and_skips_other_states(state, fixed_clock):
    rows = [row(), row(sym="OTH"), {**row(sym="FORM"), "state": "FORMING"}]
    d = date(2026, 11, 10)
    assert journal.record_signals(state, rows, fixed_clock, trade_date=d) == 2
    assert journal.record_signals(state, rows, fixed_clock, trade_date=d) == 0
    assert journal.record_signals(state, rows, fixed_clock, trade_date=date(2026, 11, 11)) == 2  # new day
    con = state.connect()
    assert con.execute("SELECT count(*) FROM lab_journal").fetchone()[0] == 4
    r = con.execute("SELECT status, created_ts, side FROM lab_journal LIMIT 1").fetchone()
    assert r == ("planned", "2026-11-12T10:00:00+05:30", 1)


def test_trade_date_comes_from_t_in_when_not_given(state, fixed_clock):
    t_in = int(np.datetime64("2026-11-11", "s").astype(np.int64))  # nominal next day of the 10th
    journal.record_signals(state, [{**row(), "t_in": t_in}], fixed_clock)
    assert state.connect().execute("SELECT trade_date FROM lab_journal").fetchone()[0] == "2026-11-10"


def test_status_goes_planned_open_closed_as_bars_complete(state, tmp_data_root, fixed_clock):
    write_hand_panel(tmp_data_root, high_at={283: 116.0})  # 3R target (115) reached on day 283
    journal.record_signals(state, [row()], fixed_clock, trade_date=DAYS[SIGNAL_I])

    def status():
        sql = "SELECT status, closed_ts, close_price, reason FROM lab_journal"
        return state.connect().execute(sql).fetchone()

    none = {"opened": 0, "closed": 0, "expired": 0}
    assert journal.update_marks(state, tmp_data_root, DAYS[SIGNAL_I]) == none
    assert status()[0] == "planned"  # the entry session has not completed yet
    assert journal.update_marks(state, tmp_data_root, DAYS[SIGNAL_I + 1]) == {**none, "opened": 1}
    assert status()[0] == "open"
    assert journal.update_marks(state, tmp_data_root, DAYS[SIGNAL_I + 2])["closed"] == 0
    assert journal.update_marks(state, tmp_data_root, DAYS[283]) == {**none, "closed": 1}
    assert status() == ("closed", str(DAYS[283]), 115.0, "target")
    assert journal.update_marks(state, tmp_data_root, DAYS[290]) == none
    ref = state.connect().execute("SELECT ref_price, stop, target FROM lab_journal").fetchone()
    assert ref == (100.0, 95.0, 115.0)  # the record never changed


def test_a_gap_under_the_stop_expires_the_plan(state, tmp_data_root, fixed_clock):
    write_hand_panel(tmp_data_root, open_at={SIGNAL_I + 1: 94.0})  # opens under the 95 stop: never entered
    journal.record_signals(state, [row()], fixed_clock, trade_date=DAYS[SIGNAL_I])
    assert journal.update_marks(state, tmp_data_root, DAYS[SIGNAL_I + 1])["expired"] == 1
    status, reason = state.connect().execute("SELECT status, reason FROM lab_journal").fetchone()
    assert status == "expired" and reason.startswith("skipped")


def test_unregistered_strategy_uses_the_fallback_exit(state, tmp_data_root, fixed_clock):
    write_hand_panel(tmp_data_root)
    journal.record_signals(state, [row(strat="picks:trial-1")], fixed_clock, trade_date=DAYS[SIGNAL_I])
    assert journal.update_marks(state, tmp_data_root, DAYS[SIGNAL_I + 1])["opened"] == 1


def test_read_returns_recent_rows_newest_first(state, tmp_data_root, fixed_clock):
    journal.record_signals(state, [row()], fixed_clock, trade_date=date(2026, 11, 10))
    journal.record_signals(state, [row(sym="OLD")], fixed_clock, trade_date=date(2026, 9, 1))
    out = journal.read(tmp_data_root, days=30, clock=fixed_clock)
    assert [t["sym"] for t in out["trades"]] == ["HND"] and out["paper"] is True
    assert {"id", "trade_date", "strat", "sym", "ref_price", "status"} <= set(out["trades"][0])
    assert len(journal.read(tmp_data_root, days=90, clock=fixed_clock)["trades"]) == 2


def test_read_without_the_table_is_empty(tmp_data_root, fixed_clock):
    assert journal.read(tmp_data_root, clock=fixed_clock) == {"trades": [], "paper": True}
