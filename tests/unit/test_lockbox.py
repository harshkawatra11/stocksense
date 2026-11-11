from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl
import pytest

from stocksense.core.errors import LockboxError
from stocksense.core.paths import Paths
from stocksense.validation.lockbox import guard, open_lockbox, research_end


def _frame() -> pl.DataFrame:
    # lockbox_start = 2025-10-01 -> research_end = 2025-09-30
    return pl.DataFrame(
        {
            "date": [date(2025, 9, 29), date(2025, 9, 30), date(2025, 10, 1), date(2025, 10, 2)],
            "x": [1, 2, 3, 4],
        }
    )


def test_research_end(tmp_data_root):
    assert research_end(tmp_data_root) == date(2025, 9, 30)


def test_research_drops_lockbox_rows(tmp_data_root):
    out = guard(_frame(), "date", "research", tmp_data_root)
    # rows dated 2025-10-01 and 2025-10-02 are dropped; 09-29 and 09-30 stay
    assert out["x"].to_list() == [1, 2]
    assert out["date"].max() < tmp_data_root.research.lockbox_start


def test_research_handles_datetime_and_lazy(tmp_data_root):
    f = _frame().with_columns(pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=15))
    assert guard(f, "date", "research", tmp_data_root)["x"].to_list() == [1, 2]
    lazy = guard(_frame().lazy(), "date", "research", tmp_data_root)
    assert isinstance(lazy, pl.LazyFrame) and lazy.collect()["x"].to_list() == [1, 2]


def test_production_keeps_everything(tmp_data_root):
    assert guard(_frame(), "date", "production", tmp_data_root).height == 4


def test_unknown_purpose(tmp_data_root):
    with pytest.raises(ValueError):
        guard(_frame(), "date", "peek", tmp_data_root)


def test_lockbox_outside_context_raises(tmp_data_root):
    with pytest.raises(LockboxError):
        guard(_frame(), "date", "lockbox", tmp_data_root)


def test_lockbox_inside_context_returns_all_and_closes(tmp_data_root, fixed_clock):
    with open_lockbox(tmp_data_root, fixed_clock, "cand-1", "final check"):
        assert guard(_frame(), "date", "lockbox", tmp_data_root).height == 4
    with pytest.raises(LockboxError):
        guard(_frame(), "date", "lockbox", tmp_data_root)


def test_one_line_per_open_and_second_open_raises(tmp_data_root, fixed_clock):
    path = Paths(tmp_data_root).state / "lockbox_touches.jsonl"
    with open_lockbox(tmp_data_root, fixed_clock, "cand-1", "r1"):
        pass
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(LockboxError), open_lockbox(tmp_data_root, fixed_clock, "cand-1", "again"):
        pytest.fail("must not enter")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1  # a refused open writes nothing
    with open_lockbox(tmp_data_root, fixed_clock, "cand-2", "r2"):
        pass
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["candidate_id"] for x in lines] == ["cand-1", "cand-2"]
    assert set(lines[0]) == {"ts", "candidate_id", "reason", "git_sha"}
    # fixed clock: 2026-11-12 10:00 IST
    assert lines[0]["ts"] == datetime(2026, 11, 12, 10, 0, tzinfo=fixed_clock.now().tzinfo).isoformat()
    assert lines[0]["reason"] == "r1"


def test_failure_inside_still_counts_as_a_touch(tmp_data_root, fixed_clock):
    with pytest.raises(RuntimeError), open_lockbox(tmp_data_root, fixed_clock, "cand-9", "boom"):
        raise RuntimeError("x")
    with pytest.raises(LockboxError), open_lockbox(tmp_data_root, fixed_clock, "cand-9", "retry"):
        pytest.fail("must not enter")
