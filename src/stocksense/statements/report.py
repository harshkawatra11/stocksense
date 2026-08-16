"""
The Kundli report — assembles diagnostics + counterfactuals into a fact
sheet and narrates it via the agent bridge (docs/12-statement-forensics.md).

Structure, matching the plan:
1. Verdict line — one sentence, the single biggest problem
2. The houses — a scored panel across behavior dimensions
3. Doshas found — severity-ranked, each with the number that proves it
4. Remedies — specific, quantified prescriptions
5. Counterfactual table — what each fix would have been worth
6. What you do well — genuinely, not as a softener

Python assembles the fact sheet (1-2, 5) deterministically; only the
narrative framing (verdict wording, remedy phrasing, strengths) goes
through Claude, and it is handed the exact numbers rather than asked to
compute anything — the compute/narrate split from docs/11.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from stocksense.agent.claude_cli import AgentRequest, invoke
from stocksense.statements.counterfactuals import Counterfactual, run_all as run_all_counterfactuals
from stocksense.statements.diagnostics import Diagnostic, run_all as run_all_diagnostics

_SEVERITY_ORDER = {"critical": 0, "high": 1, "notable": 2, "ok": 3}


def build_fact_sheet(positions: pd.DataFrame) -> dict:
    diagnostics = run_all_diagnostics(positions)
    counterfactuals = run_all_counterfactuals(positions)

    n = len(positions)
    total_net = float(positions["net_pnl"].sum()) if n else 0.0
    total_gross = float(positions["gross_pnl"].sum()) if n else 0.0
    total_charges = float(positions["charges"].sum()) if n else 0.0
    win_rate = float((positions["net_pnl"] > 0).mean()) if n else None

    ranked_diagnostics = sorted(diagnostics, key=lambda d: _SEVERITY_ORDER[d.severity])
    findings = [
        {"name": d.name, "value": d.value, "unit": d.unit, "severity": d.severity, "detail": d.detail}
        for d in ranked_diagnostics if d.severity != "ok"
    ]
    strengths = [
        {"name": d.name, "value": d.value, "unit": d.unit}
        for d in ranked_diagnostics if d.severity == "ok" and d.value is not None
    ]

    return {
        "n_positions": n,
        "total_net_pnl": total_net,
        "total_gross_pnl": total_gross,
        "total_charges": total_charges,
        "win_rate": win_rate,
        "findings": findings,
        "strengths": strengths,
        "counterfactuals": [asdict(cf) for cf in counterfactuals],
        "insufficient_sample": n < 30,
    }


_REPORT_PROMPT = """You are writing a "Kundli" report: a direct, specific
diagnostic profile of a trader's real behavior, built entirely from the
computed fact sheet below. Structure your response as:

1. VERDICT — one sentence naming the single biggest problem, using its
   exact number from the fact sheet.
2. DOSHAS FOUND — for each finding (severity != ok), one short paragraph
   naming the problem, citing its number, and explaining the consequence
   in plain language. Rank by severity (critical first).
3. REMEDIES — one specific, quantified, actionable prescription per
   dosha (e.g. "cap position size at ₹X" using an actual number from the
   fact sheet, not a generic suggestion).
4. COUNTERFACTUAL TABLE — narrate what each "what if" scenario would
   have been worth, using the exact delta_pnl figures. State plainly
   these are historical arithmetic, not predictions.
5. WHAT YOU DO WELL — genuinely, using the 'strengths' list; skip this
   section if strengths is empty rather than inventing praise.

If n_positions is small (insufficient_sample=true), open with an
explicit caveat that findings are low-confidence given the sample size.
Be direct and specific. Never state a number that isn't in the fact
sheet below."""


def generate_kundli(positions: pd.DataFrame, store=None, period_label: str = "all") -> dict:
    """Returns {'fact_sheet': dict, 'narrative': str, 'run_id': str}.
    Persists diagnostics/counterfactuals to the store when provided."""
    run_id = str(uuid.uuid4())
    fact_sheet = build_fact_sheet(positions)

    if store is not None:
        _persist(store, run_id, positions)

    result = invoke(
        AgentRequest(prompt=_REPORT_PROMPT, facts=fact_sheet, skill="statement-forensics"),
        store=store, job_run_id=run_id,
    )

    return {"run_id": run_id, "fact_sheet": fact_sheet, "narrative": result.output_text, "agent_status": result.status}


def _persist(store, run_id: str, positions: pd.DataFrame) -> None:
    diagnostics = run_all_diagnostics(positions)
    counterfactuals = run_all_counterfactuals(positions)
    as_of = datetime.now(timezone.utc).date()

    diag_df = pd.DataFrame(
        [
            {
                "run_id": run_id, "as_of": as_of, "metric_name": d.name, "metric_value": d.value,
                "metric_unit": d.unit, "severity": d.severity, "cohort": d.cohort,
                "detail_json": json.dumps(d.detail, default=str),
            }
            for d in diagnostics
        ]
    )
    if not diag_df.empty:
        store.write_diagnostics(diag_df)

    cf_df = pd.DataFrame(
        [
            {
                "run_id": run_id, "scenario_name": cf.scenario_name, "actual_pnl": cf.actual_pnl,
                "scenario_pnl": cf.scenario_pnl, "delta_pnl": cf.delta_pnl,
                "n_trades_affected": cf.n_trades_affected, "detail_json": json.dumps(cf.detail, default=str),
            }
            for cf in counterfactuals
        ]
    )
    if not cf_df.empty:
        store.write_counterfactuals(cf_df)
