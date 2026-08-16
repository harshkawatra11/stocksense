"""Runner tests: resume-after-failure, idempotency across runs, and
job_runs written even on hard failure — the properties the plan's
verification section names explicitly for Phase 0."""

from __future__ import annotations

import pytest

from stocksense.data.store import Store
from stocksense.harness.graph import Graph, Node
from stocksense.harness.runner import run_graph


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def test_all_nodes_completed_and_context_populated(tmp_store: Store) -> None:
    g = Graph(
        [
            Node("a", fn=lambda ctx: {"value": 1}),
            Node("b", fn=lambda ctx: {"value": ctx["a"]["value"] + 1}, depends_on=("a",)),
        ]
    )
    result = run_graph(g, tmp_store)
    assert result.all_succeeded
    assert result.context["b"]["value"] == 2

    runs = tmp_store.read_job_runs()
    assert len(runs) == 2
    assert set(runs["status"]) == {"completed"}


def test_job_runs_written_on_hard_failure(tmp_store: Store) -> None:
    def boom(ctx):
        raise RuntimeError("simulated failure")

    g = Graph([Node("a", fn=lambda ctx: {}), Node("b", fn=boom, depends_on=("a",))])
    result = run_graph(g, tmp_store)
    assert not result.all_succeeded
    assert "b" in result.failed_nodes()

    runs = tmp_store.read_job_runs()
    statuses = dict(zip(runs["job_name"], runs["status"]))
    assert statuses["a"] == "completed"
    assert statuses["b"] == "failed"


def test_downstream_of_failure_is_skipped_not_silently_dropped(tmp_store: Store) -> None:
    def boom(ctx):
        raise RuntimeError("fail")

    g = Graph(
        [
            Node("a", fn=lambda ctx: {}),
            Node("b", fn=boom, depends_on=("a",)),
            Node("c", fn=lambda ctx: {}, depends_on=("b",)),
        ]
    )
    result = run_graph(g, tmp_store, stop_on_failure=True)
    outcome_by_name = {o.name: o.status for o in result.outcomes}
    assert outcome_by_name["a"] == "completed"
    assert outcome_by_name["b"] == "failed"
    assert outcome_by_name["c"] == "skipped"


def test_idempotency_key_skips_on_second_invocation(tmp_store: Store) -> None:
    calls = []

    def track(ctx):
        calls.append(1)
        return {}

    g = Graph([Node("ingest", fn=track, idempotency_key="ingest:2024-01-15")])
    run_graph(g, tmp_store)
    run_graph(g, tmp_store)  # second run, same idempotency key

    assert len(calls) == 1  # not re-executed
    runs = tmp_store.read_job_runs()
    assert len(runs) == 1  # only one job_runs row, not two


def test_resume_only_reexecutes_failed_node_onward(tmp_store: Store) -> None:
    """Simulates: first run, 'a' succeeds (idempotency-keyed), 'b' fails.
    A second run of the same graph should skip 'a' (already completed for
    its key) and retry 'b'."""
    a_calls = []
    b_calls = []
    fail_b = [True]

    def node_a(ctx):
        a_calls.append(1)
        return {}

    def node_b(ctx):
        b_calls.append(1)
        if fail_b[0]:
            raise RuntimeError("transient failure")
        return {}

    g = Graph(
        [
            Node("a", fn=node_a, depends_on=(), idempotency_key="a:run1"),
            Node("b", fn=node_b, depends_on=("a",)),
        ]
    )
    run_graph(g, tmp_store)  # a succeeds, b fails
    assert len(a_calls) == 1
    assert len(b_calls) == 1

    fail_b[0] = False
    run_graph(g, tmp_store)  # a skipped (idempotent), b retried and succeeds
    assert len(a_calls) == 1  # NOT re-executed
    assert len(b_calls) == 2  # retried
