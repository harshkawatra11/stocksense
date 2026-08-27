"""
Phase J2.c: the paper trading graph, a SIBLING to harness.loops.build_
reconcile_graph -- deliberately not merged into it. The prediction
ledger is the research record of last resort; a bug in the paper book
must never be able to stop it writing. Idempotency-keyed by calendar
date, same property every other nightly step in this codebase has.
"""

from __future__ import annotations

from datetime import date

from stocksense.harness.graph import Graph, Node
from stocksense.paper.account import get_account
from stocksense.paper.engine import run_pending_rebalances


def build_paper_graph(store, account_id: str) -> Graph:
    today_key = date.today().isoformat()

    def _run(ctx: dict) -> dict:
        account = get_account(store, account_id)
        if account.status != "active":
            return {"ran": False, "reason": f"account status is {account.status!r}, not active"}
        runs = run_pending_rebalances(store, account)
        return {
            "ran": True,
            "n_rebalances_processed": len(runs),
            "rebalance_dates": [r.rebalance_date for r in runs],
        }

    return Graph([Node("run_pending_rebalances", fn=_run, idempotency_key=f"paper_run:{account_id}:{today_key}")])
