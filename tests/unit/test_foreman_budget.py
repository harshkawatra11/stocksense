from __future__ import annotations

import pytest

from stocksense.data.store import Store
from stocksense.foreman.budget import KillSwitch, check_budget


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def test_check_budget_within_limit_on_fresh_day(tmp_store: Store) -> None:
    status = check_budget(tmp_store, max_invocations=10)
    assert status.within_budget is True
    assert status.invocations_used == 0


def test_check_budget_blocks_once_cap_reached(tmp_store: Store) -> None:
    from datetime import date

    tmp_store.increment_budget(date.today(), invocations=10)
    status = check_budget(tmp_store, max_invocations=10)
    assert status.within_budget is False
    assert "cap" in status.reason


def test_kill_switch_starts_unstopped() -> None:
    ks = KillSwitch()
    assert ks.stopped is False


def test_kill_switch_stop_is_observable() -> None:
    ks = KillSwitch()
    ks.stop()
    assert ks.stopped is True
