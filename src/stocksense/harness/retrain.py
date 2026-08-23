"""
Phase G4: the weekly retrain-through-gate-through-registry loop, as a
harness.Graph -- the same reusable Graph/Node/Runner infrastructure
Phase G2's reconcile loop (harness/loops.py) uses, keyed by ISO week
rather than calendar date so re-running mid-week is a no-op
(docs/05-nightly-pipeline.md's idempotency requirement, at weekly
rather than daily granularity).
"""

from __future__ import annotations

from datetime import date

from stocksense.harness.graph import Graph, Node
from stocksense.models.train_candidate import train_candidate_core


def build_weekly_retrain_graph(
    store, horizon: int, top_n: int, cost_bps: float = 25.0,
    turnover_rank_band: tuple[float, float] | None = None, settings=None,
) -> Graph:
    """A single-node graph -- train_candidate_core already IS
    train -> gate -> register in one call, so there is no natural
    sub-step boundary the way grade/record had for the reconcile graph.
    Kept as a Graph rather than a bare function call anyway, so it gets
    the harness's job_runs heartbeat, resumability, and idempotency for
    free, and composes alongside the reconcile graph under one
    scheduler (the remaining piece of Phase G4/G5's scheduling wiring).

    `turnover_rank_band`: passed straight through to train_candidate_core
    -- Phase G1 found the edge concentrated in mid/small caps, not large
    (research/verdict_bhavcopy_rerun.md); this is the parameter that
    lets the weekly loop actually retrain on whichever cap band was
    chosen to run live, rather than only ever the unrestricted universe.

    `settings`: must be threaded through explicitly (Phase H1's --cap-
    band bug, found live: get_settings() constructs a fresh Settings()
    from the process environment on every call, with no caching, so a
    caller-mutated settings.price_source override has no effect unless
    passed all the way down -- train_candidate_core already accepts it,
    this was the missing link between this graph and that function).
    """
    iso_year, iso_week, _ = date.today().isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"

    def _train(ctx: dict) -> dict:
        result = train_candidate_core(horizon, top_n, cost_bps, store, settings=settings, turnover_rank_band=turnover_rank_band)
        return {
            "model_id": result.model_id,
            "lifecycle_state": result.lifecycle_state,
            "n_fold_results": result.n_fold_results,
            "gate_passed": None if result.verdict is None else result.verdict.passed,
            "gate_reason": None if result.verdict is None else result.verdict.reason,
        }

    return Graph(
        [
            Node(
                "train_candidate",
                fn=_train,
                idempotency_key=f"weekly_retrain:{horizon}:{top_n}:{turnover_rank_band}:{week_key}",
            ),
        ]
    )
