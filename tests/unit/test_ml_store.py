from __future__ import annotations

import json
from datetime import datetime, timedelta

from stocksense.core.clock import IST, SteppingClock
from stocksense.core.config import Paths as ConfigPaths
from stocksense.core.paths import Paths
from stocksense.marketlab.store import LabStore, TTLCache

T0 = datetime(2026, 11, 12, 10, 0, tzinfo=IST)


def test_ok_and_fail_counts_and_status_is_json():
    st = LabStore()
    st.set_interval("tape", 4)
    st.ok("tape", T0)
    st.fail("tape", T0, RuntimeError("boom"))
    st.fail("tape", T0, RuntimeError("boom"))
    snap = st.snapshot_status()
    assert snap["feeds"]["tape"] == {
        "interval_s": 4.0,
        "last_ok": T0.isoformat(),
        "errors": 2,
        "last_err": "RuntimeError: boom",
    }
    assert snap["ready"] is False
    st.tape["x"] = 1
    assert st.snapshot_status()["ready"] is True
    json.dumps(snap)


def test_ttl_cache_computes_once_within_ttl_then_again():
    clock = SteppingClock(T0)
    cache = TTLCache(clock)
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    assert cache.get("k", 60, fn) == 1 and cache.get("k", 60, fn) == 1
    clock.advance(timedelta(seconds=61))
    assert cache.get("k", 60, fn) == 2 and len(calls) == 2
    assert cache.get("other", 60, fn) == 3


def test_paths_lab(tmp_path):
    class S:
        paths = ConfigPaths(data_root=tmp_path)

    assert Paths(S()).lab == tmp_path / "lab"  # type: ignore[arg-type]
