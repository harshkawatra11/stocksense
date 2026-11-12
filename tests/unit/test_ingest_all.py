from __future__ import annotations

from datetime import date, datetime

import polars as pl
from typer.testing import CliRunner

from stocksense.cli import app
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.core.state import State
from stocksense.data.ingest_all import catch_up, default_steps, exit_code
from stocksense.data.ingest_common import IngestContext, IngestResult

NOW = datetime(2026, 11, 12, 19, 0, tzinfo=IST)


def _ctx(settings):
    clock = FixedClock(NOW)
    state = State(settings, clock)
    state.migrate()
    return IngestContext(settings, clock, TradingCalendar.from_dates([date(2026, 11, 11)]), None, state)  # type: ignore[arg-type]


def test_order_is_respected_and_a_failure_does_not_stop_the_rest(tmp_data_root):
    seen = []

    def ok(name):
        def f(ctx):
            seen.append(name)
            return [IngestResult(name, date(2026, 11, 12), 1, "ok")]

        return f

    def boom(ctx):
        seen.append("b")
        raise RuntimeError("down")

    steps = [("a", ok("a")), ("b", boom), ("c", ok("c"))]
    res = catch_up(_ctx(tmp_data_root), steps)
    assert seen == ["a", "b", "c"]
    assert [(r.dataset, r.status) for r in res] == [("a", "ok"), ("b", "failed"), ("c", "ok")]


def test_default_step_order_matches_the_card():
    names = [n for n, _ in default_steps()]
    assert names == [
        "reference", "eq", "fo", "index", "corporate_actions", "announcements",
        "board_results", "macro_misc", "intraday",
    ]  # fmt: skip


def test_exit_code_logic():
    def table(eq, dl, ca):
        return pl.DataFrame(
            {
                "dataset": ["bhavcopy_eq", "bhavcopy_delivery", "corporate_actions", "fo_bhavcopy"],
                "fresh": [eq, dl, ca, False],
            }
        )

    assert exit_code(table(True, True, True)) == 0  # fo is not required
    assert exit_code(table(True, False, True)) == 1
    assert exit_code(table(True, True, None)) == 1
    assert exit_code(pl.DataFrame({"dataset": ["bhavcopy_eq"], "fresh": [True]})) == 1  # required one absent


def test_cli_catch_up_on_an_empty_store_exits_1(tmp_data_root):
    res = CliRunner().invoke(app, ["ingest", "--all", "--catch-up"], obj=tmp_data_root)
    assert res.exit_code == 1 and "no data" in res.output
