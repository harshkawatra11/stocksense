from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path

import pytest

from stocksense.core.clock import IST, FixedClock
from stocksense.data.ingest_common import IngestResult
from stocksense.marketlab.feeds import CatchUp, make_ingest_step, make_panel_step, make_scan_step
from stocksense.marketlab.store import LabStore

DAY = date(2026, 11, 12)  # a Thursday


def at(h: int, m: int = 0) -> FixedClock:
    return FixedClock(datetime(2026, 11, 12, h, m, tzinfo=IST))


def names(store):
    return [(s["name"], s["state"]) for s in store.catchup["steps"]]


def test_step_order_and_states(tmp_data_root):
    store, calls = LabStore(), []

    def mk(name, detail="ok"):
        return name, lambda: (calls.append(name), detail)[1]

    steps = [mk("ingest"), mk("panel", "deferred: low memory"), mk("scan", "skipped: exists"), mk("flow_eod")]
    CatchUp(tmp_data_root, at(20), store, steps).run()
    assert calls == ["ingest", "panel", "scan", "flow_eod"]
    assert names(store) == [
        ("ingest", "done"),
        ("panel", "deferred"),
        ("scan", "skipped"),
        ("flow_eod", "done"),
    ]
    assert store.catchup["state"] == "done"
    assert store.snapshot_status()["catchup"]["steps"][1]["detail"] == "deferred: low memory"


def test_a_failing_step_does_not_stop_the_next(tmp_data_root):
    store, calls = LabStore(), []

    def boom():
        raise RuntimeError("nse down")

    steps = [("ingest", boom), ("panel", lambda: (calls.append("panel"), "x")[1])]
    CatchUp(tmp_data_root, at(20), store, steps).run()
    assert calls == ["panel"]
    assert names(store) == [("ingest", "failed"), ("panel", "done")]
    assert store.catchup["steps"][0]["detail"].startswith("RuntimeError: nse down")


def test_deferred_market_hours_step_is_rechecked_with_a_stop_event(tmp_data_root):
    store, n = LabStore(), {"i": 0}

    def panel():
        n["i"] += 1
        return "deferred: market hours" if n["i"] < 3 else "built"

    cu = CatchUp(tmp_data_root, at(11), store, [("panel", panel)], retry_s=0.01)
    cu.run(threading.Event())
    assert n["i"] == 3 and names(store) == [("panel", "done")]
    stop = threading.Event()
    stop.set()
    n["i"] = 0
    CatchUp(tmp_data_root, at(11), LabStore(), [("panel", panel)], retry_s=0.01).run(stop)
    assert n["i"] == 1  # a set stop event ends the wait loop


def _panel(clock, *, fresh=False, ram=8.0, built=None):
    built = built if built is not None else []
    step = make_panel_step(
        None, clock, is_fresh=lambda: fresh, build=lambda: built.append(1), free_ram_gb=lambda: ram
    )
    return step, built


def test_panel_deferred_in_market_hours_and_on_low_memory(tmp_data_root):
    step, built = _panel(at(11))
    assert step() == "deferred: market hours" and built == []
    step, built = _panel(at(15, 44))
    assert step() == "deferred: market hours"
    step, built = _panel(at(20), ram=2.0)
    assert step() == "deferred: low memory" and built == []
    step, built = _panel(at(20), ram=3.0)
    assert step() == "built" and built == [1]
    step, built = _panel(at(8, 59))
    assert step() == "built"
    step, built = _panel(at(11), fresh=True)
    assert step() == "fresh" and built == []


def test_ingest_step_runs_catch_up_only_when_behind():
    ran = []

    def run():
        ran.append(1)
        return [
            IngestResult("a", DAY, 5, "ok"),
            IngestResult("b", DAY, 0, "failed"),
            IngestResult("c", DAY, 1, "ok"),
        ]

    assert make_ingest_step(None, at(20), is_fresh=lambda: True, run=run)() == "fresh" and ran == []
    assert make_ingest_step(None, at(20), is_fresh=lambda: False, run=run)() == "2 ok, 1 failed"


def test_scan_step_skips_when_the_scan_exists(tmp_data_root):
    seen = []
    exists = {"v": True}
    step = make_scan_step(
        tmp_data_root,
        at(20),
        asof=lambda: DAY,
        exists=lambda d: exists["v"],
        run=lambda d: seen.append(d) or Path("x.parquet"),
    )
    assert step().startswith("skipped") and seen == []
    exists["v"] = False
    assert "x.parquet" in step() and seen == [DAY]


def test_run_lab_bat_has_no_scheduler_and_starts_serve():
    bat = (Path(__file__).resolve().parents[2] / "run-lab.bat").read_text(encoding="utf-8")
    low = bat.lower()
    assert "schtasks" not in low and "startup" not in low
    assert "-m stocksense serve" in bat and "http://127.0.0.1:8765" in bat and "npm --prefix web ci" in bat


@pytest.mark.parametrize("flag", [True, False])
def test_app_catchup_flag_default(tmp_data_root, tmp_path, flag):
    from stocksense.api.app import create_app

    app = create_app(tmp_data_root, start_feeds=False, clock=at(20), web_out=tmp_path / "n")
    assert app.state.catchup_enabled is True
    app.state.catchup_enabled = flag
    assert app.state.catchup_enabled is flag
