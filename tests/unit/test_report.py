"""Kundli fact sheet assembly and persistence — the deterministic half
of report.py. generate_kundli's narrative step (agent invocation) is
exercised via a monkeypatched invoke() so this suite doesn't depend on
having the Claude CLI installed."""

from __future__ import annotations

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.statements.report import build_fact_sheet, generate_kundli


def _position(symbol, entry, exit_, gross_pnl, charges=0.0):
    return {
        "symbol": symbol, "segment": "equity_delivery", "open_date": "2024-01-01", "open_time": "09:20",
        "close_date": "2024-01-01", "close_time": "15:00", "quantity": 1, "entry_price": entry,
        "exit_price": exit_, "gross_pnl": gross_pnl, "charges": charges, "net_pnl": gross_pnl - charges,
        "holding_seconds": 3600, "is_intraday": True, "mae": None, "mfe": None,
    }


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _sample_positions() -> pd.DataFrame:
    rows = [_position("AAA", 100, 90, -50, charges=60) for _ in range(20)]  # heavy cost drag + losses
    rows += [_position("BBB", 100, 105, 5, charges=1) for _ in range(5)]
    return pd.DataFrame(rows)


def test_build_fact_sheet_structure() -> None:
    fs = build_fact_sheet(_sample_positions())
    assert fs["n_positions"] == 25
    assert "findings" in fs and isinstance(fs["findings"], list)
    assert "counterfactuals" in fs and len(fs["counterfactuals"]) == 7
    assert fs["insufficient_sample"] is True  # 25 positions < MIN_POSITIONS_FOR_SIGNIFICANCE (30)


def test_findings_ranked_by_severity() -> None:
    fs = build_fact_sheet(_sample_positions())
    severities = [f["severity"] for f in fs["findings"]]
    order = {"critical": 0, "high": 1, "notable": 2}
    assert severities == sorted(severities, key=lambda s: order.get(s, 3))


def test_insufficient_sample_flag_on_thin_data() -> None:
    fs = build_fact_sheet(pd.DataFrame([_position("AAA", 100, 110, 10)]))
    assert fs["insufficient_sample"] is True


def test_generate_kundli_persists_and_narrates(tmp_store: Store, monkeypatch) -> None:
    captured = {}

    def fake_invoke(req, store=None, job_run_id=None):
        captured["facts"] = req.facts
        from stocksense.agent.claude_cli import AgentResult
        from datetime import datetime, timezone
        return AgentResult(
            agent_run_id="test", output_text="VERDICT: your cost drag is high.",
            status="ok", error=None, started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("stocksense.statements.report.invoke", fake_invoke)

    result = generate_kundli(_sample_positions(), store=tmp_store)
    assert result["narrative"] == "VERDICT: your cost drag is high."
    assert captured["facts"]["n_positions"] == 25

    diagnostics = tmp_store.read_diagnostics(result["run_id"])
    assert len(diagnostics) == 13  # all doshas persisted, ok and non-ok

    cfs = tmp_store.read_counterfactuals(result["run_id"])
    assert len(cfs) == 7
