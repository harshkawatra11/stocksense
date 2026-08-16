"""RAG indexing tests: chunking, idempotent re-indexing, and FTS
retrieval end-to-end on a real DuckDB store (fts extension verified
working locally). Vector search is exercised in test_rag_retrieve.py
with a stubbed embedder, since this environment has no embedding model
pulled -- degraded-mode behavior (embed_text returning None) is
asserted directly, not skipped."""

from __future__ import annotations

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.rag.index import chunk_text, content_hash, index_document, rebuild_fts_index
from stocksense.rag.retrieve import fts_search


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def test_chunk_text_splits_on_paragraph_boundaries() -> None:
    content = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunk_text(content, chunk_size=100, overlap=10)
    assert len(chunks) == 1  # small enough to fit in one chunk
    assert "First paragraph" in chunks[0]


def test_chunk_text_splits_long_content_into_multiple_chunks() -> None:
    paragraphs = [f"Paragraph {i} with some words in it to pad length." for i in range(50)]
    content = "\n\n".join(paragraphs)
    chunks = chunk_text(content, chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_chunk_text_handles_oversized_single_paragraph() -> None:
    huge_paragraph = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(huge_paragraph, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_text_empty_content_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_content_hash_deterministic() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_index_document_writes_document_and_chunks(tmp_store) -> None:
    result = index_document(tmp_store, "research", "phase0_verdict.md", "Phase 0 Verdict", "The gate passed with p=0.0085.")
    assert result["reindexed"] is True
    assert result["n_chunks"] == 1

    docs = tmp_store.con.execute("SELECT * FROM rag_documents").fetchdf()
    assert len(docs) == 1
    chunks = tmp_store.con.execute("SELECT * FROM rag_chunks").fetchdf()
    assert len(chunks) == 1


def test_index_document_is_idempotent_on_unchanged_content(tmp_store) -> None:
    index_document(tmp_store, "research", "verdict.md", "Verdict", "unchanged content")
    result2 = index_document(tmp_store, "research", "verdict.md", "Verdict", "unchanged content")
    assert result2["reindexed"] is False

    docs = tmp_store.con.execute("SELECT * FROM rag_documents").fetchdf()
    assert len(docs) == 1  # not duplicated


def test_index_document_reindexes_and_replaces_chunks_on_content_change(tmp_store) -> None:
    index_document(tmp_store, "research", "verdict.md", "Verdict", "original content here")
    result2 = index_document(tmp_store, "research", "verdict.md", "Verdict", "completely different new content")
    assert result2["reindexed"] is True

    docs = tmp_store.con.execute("SELECT * FROM rag_documents").fetchdf()
    assert len(docs) == 1  # still one doc row, not two
    assert "completely different" in docs.iloc[0]["content"]


def test_embeddings_absent_still_indexes_via_fts_degraded_mode(tmp_store) -> None:
    """The load-bearing property: with no embedding model available in
    this environment, index_document must still succeed and the chunk
    must still be findable by FTS -- the exact 'FTS-only is a genuinely
    supported degraded mode' claim."""
    index_document(tmp_store, "docs", "cost_model.md", "Cost Model", "the intraday cost drag on this portfolio is high")
    rebuild_fts_index(tmp_store)

    results = fts_search(tmp_store, "cost drag", top_k=5)
    assert len(results) == 1
    assert "cost drag" in results[0].content


def test_embed_chunks_false_skips_embed_call_entirely(tmp_store, monkeypatch) -> None:
    """embed_chunks=False must not call embed_text at all -- the
    performance property: a caller who already knows (via
    embeddings_available(), checked once) that no embedding model is
    available should be able to skip N redundant ~2s network round
    trips across N chunks, not just get None back from each of them."""
    calls = []
    monkeypatch.setattr("stocksense.rag.embed.embed_text", lambda text: calls.append(text) or [0.1] * 768)

    index_document(tmp_store, "docs", "a.md", "A", "some content here", embed_chunks=False)
    assert calls == []  # never called

    chunks = tmp_store.con.execute("SELECT embedding FROM rag_chunks").fetchdf()
    assert pd.isna(chunks.iloc[0]["embedding"])


def test_embed_chunks_true_calls_embedder_and_stores_result(tmp_store, monkeypatch) -> None:
    monkeypatch.setattr("stocksense.rag.embed.embed_text", lambda text: [0.1] * 768)

    index_document(tmp_store, "docs", "a.md", "A", "some content here", embed_chunks=True)
    chunks = tmp_store.con.execute("SELECT embedding FROM rag_chunks").fetchdf()
    assert chunks.iloc[0]["embedding"] is not None


def test_fts_search_finds_relevant_document_over_irrelevant_ones(tmp_store) -> None:
    index_document(tmp_store, "docs", "a.md", "A", "the survivorship bias break-even shock rate is 6.4 percent annually")
    index_document(tmp_store, "docs", "b.md", "B", "the kundli report covers behavioral diagnostics and counterfactuals")
    rebuild_fts_index(tmp_store)

    results = fts_search(tmp_store, "survivorship bias shock rate", top_k=5)
    assert len(results) >= 1
    assert results[0].source_ref == "a.md"
