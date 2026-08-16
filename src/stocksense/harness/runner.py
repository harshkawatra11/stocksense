"""
Graph runner (docs/11-agent-harness.md). Executes a Graph in topological
order, writing one job_runs row per node — including on hard failure,
which is how "did last night actually run?" gets answered from the
database rather than from log scrollback (docs/08-operations.md).

Two guarantees this module exists to provide:
1. **Resume, not restart.** If node C fails after A and B succeeded, a
   re-run of the same graph re-executes only C onward — it does not
   redo A and B's work.
2. **Idempotency across runs.** A node with an idempotency_key that has
   already succeeded for that key (e.g. "ingest:2024-01-15") is skipped
   even on a fresh graph invocation, not just within one run.
"""

from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from stocksense.harness.graph import Graph

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class NodeOutcome:
    name: str
    status: str  # 'completed' | 'failed' | 'skipped'
    run_id: str
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    outcomes: list[NodeOutcome]
    context: dict

    @property
    def all_succeeded(self) -> bool:
        return all(o.status in ("completed", "skipped") for o in self.outcomes)

    def failed_nodes(self) -> list[str]:
        return [o.name for o in self.outcomes if o.status == "failed"]


def _already_succeeded(store, idempotency_key: str) -> bool:
    rows = store.con.execute(
        "SELECT 1 FROM job_runs WHERE job_name = ? AND status = 'completed' LIMIT 1",
        [idempotency_key],
    ).fetchall()
    return len(rows) > 0


def run_graph(graph: Graph, store, context: dict | None = None,
              stop_on_failure: bool = True) -> RunResult:
    """Execute every node in topological order. `context` accumulates
    each node's returned dict, keyed by node name, so downstream nodes
    can read upstream outputs (`context[dep_name]`).

    A node whose idempotency_key already has a 'completed' job_runs row
    is skipped without being invoked — its prior output is not
    reconstructed into context, since job_runs stores status, not
    payloads; nodes needing their output persisted across runs must do
    so themselves (e.g. by writing to the store).
    """
    order = graph.topological_order()
    ctx = dict(context or {})
    outcomes: list[NodeOutcome] = []
    failed_names: set[str] = set()

    for name in order:
        node = graph[name]

        if any(dep in failed_names for dep in node.depends_on):
            outcomes.append(NodeOutcome(name, "skipped", run_id="", error="upstream dependency failed"))
            failed_names.add(name)
            continue

        if node.idempotency_key and _already_succeeded(store, node.idempotency_key):
            outcomes.append(NodeOutcome(name, "skipped", run_id=""))
            continue

        run_id = str(uuid.uuid4())
        job_name = node.idempotency_key or name
        started_at = datetime.now(timezone.utc)
        store.start_job_run(run_id, job_name, started_at)

        try:
            result = node.fn(ctx) or {}
            ctx[name] = result
            store.finish_job_run(run_id, "completed", datetime.now(timezone.utc), json.dumps({"node": name}))
            outcomes.append(NodeOutcome(name, "completed", run_id=run_id))
        except Exception as e:  # noqa: BLE001 — must always write job_runs, even on unexpected failure
            tb = traceback.format_exc()
            store.finish_job_run(
                run_id, "failed", datetime.now(timezone.utc),
                json.dumps({"node": name, "error": str(e), "traceback": tb[-2000:]}),
            )
            log.error("harness_node_failed", node=name, error=str(e))
            outcomes.append(NodeOutcome(name, "failed", run_id=run_id, error=str(e)))
            failed_names.add(name)
            if stop_on_failure:
                # mark all remaining nodes as skipped rather than silently
                # omitting them, so RunResult accounts for every node
                remaining = order[order.index(name) + 1:]
                for r in remaining:
                    outcomes.append(NodeOutcome(r, "skipped", run_id="", error="halted after upstream failure"))
                break

    return RunResult(outcomes=outcomes, context=ctx)
