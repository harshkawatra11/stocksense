"""Query agent tests: ask() must return citations tied to what was
actually retrieved, and must degrade gracefully to an honest 'nothing
found' when the corpus has no relevant content -- never inventing an
answer with no grounding."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stocksense.agent.claude_cli import AgentResult
from stocksense.data.store import Store
from stocksense.rag.agent import ask
from stocksense.rag.index import index_document, rebuild_fts_index


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def test_ask_returns_no_results_message_on_empty_corpus(tmp_store) -> None:
    result = ask("why did I lose money in March?", tmp_store)
    assert result["n_chunks_retrieved"] == 0
    assert "Nothing relevant" in result["answer"]
    assert result["citations"] == []


def test_ask_retrieves_and_narrates_with_citations(tmp_store, monkeypatch) -> None:
    index_document(tmp_store, "research", "verdict.md", "Phase 0 Verdict", "The gate passed with p=0.0085 for h=20.")
    rebuild_fts_index(tmp_store)

    captured = {}

    def fake_invoke(req, store=None, job_run_id=None):
        captured["facts"] = req.facts
        return AgentResult(
            agent_run_id="t", output_text="The gate passed with p=0.0085 [1].", status="ok", error=None,
            started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("stocksense.rag.agent.invoke", fake_invoke)
    result = ask("did the gate pass for h=20?", tmp_store)

    assert result["n_chunks_retrieved"] == 1
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_ref"] == "verdict.md"
    assert "[1]" in result["answer"]
    assert "retrieved_passages" in captured["facts"]
    assert captured["facts"]["retrieved_passages"][0]["source_ref"] == "verdict.md"


def test_ask_facts_contain_only_retrieved_content_not_arbitrary_knowledge(tmp_store, monkeypatch) -> None:
    index_document(tmp_store, "docs", "cost.md", "Cost Model", "intraday round trip costs 8.3 basis points")
    rebuild_fts_index(tmp_store)

    captured = {}

    def fake_invoke(req, store=None, job_run_id=None):
        captured["facts"] = req.facts
        return AgentResult(agent_run_id="t", output_text="answer", status="ok", error=None,
                            started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc))

    monkeypatch.setattr("stocksense.rag.agent.invoke", fake_invoke)
    ask("what does intraday cost?", tmp_store)

    passage_content = [p["content"] for p in captured["facts"]["retrieved_passages"]]
    assert any("8.3 basis points" in c for c in passage_content)
