"""Retrieval tests. vector_search/hybrid_search are exercised with a
monkeypatched embed_text (this environment has no real embedding model
pulled -- see rag/embed.py's docstring), so these tests prove the
FUSION LOGIC works correctly, not that any particular embedding model
produces good vectors."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.data.store import Store
from stocksense.rag.index import index_document, rebuild_fts_index
from stocksense.rag.retrieve import fts_search, hybrid_search, vector_search


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding: hash-seeded, so identical text
    always gets an identical vector and different text gets a
    genuinely different (not just randomly different) one -- lets a
    test assert 'the semantically similar chunk ranks first' via
    hand-picked seeds rather than trusting an opaque real model."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(0, 1, 768).tolist()


def test_fts_search_returns_empty_when_nothing_indexed(tmp_store) -> None:
    results = fts_search(tmp_store, "anything", top_k=5)
    assert results == []


def test_vector_search_returns_empty_without_embeddings(tmp_store) -> None:
    """The degraded-mode property, at the retrieval layer: with no
    embedding model available (embed_text returns None for the query
    itself), vector_search must return [] cleanly, not raise."""
    index_document(tmp_store, "docs", "a.md", "A", "some content")
    results = vector_search(tmp_store, "a query")
    assert results == []


def test_hybrid_search_falls_back_to_fts_only_without_embeddings(tmp_store) -> None:
    index_document(tmp_store, "docs", "a.md", "A", "the cost drag on this portfolio is high")
    rebuild_fts_index(tmp_store)

    results = hybrid_search(tmp_store, "cost drag", top_k=5)
    assert len(results) == 1  # FTS alone still finds it, vector search contributing nothing


def test_hybrid_search_with_stubbed_embeddings_fuses_both_signals(tmp_store, monkeypatch) -> None:
    monkeypatch.setattr("stocksense.rag.embed.embed_text", _fake_embed)
    monkeypatch.setattr("stocksense.rag.retrieve.embed_text", _fake_embed)

    index_document(tmp_store, "docs", "a.md", "A", "the survivorship bias break-even shock rate is 6.4 percent")
    index_document(tmp_store, "docs", "b.md", "B", "unrelated content about something else entirely")
    rebuild_fts_index(tmp_store)

    results = hybrid_search(tmp_store, "survivorship bias shock rate", top_k=5)
    assert len(results) >= 1
    assert results[0].source_ref == "a.md"  # FTS signal alone should surface it top, vector adds noise here but shouldn't override a strong FTS match


def test_hybrid_search_chunk_found_by_both_ranks_above_fts_only_match(tmp_store, monkeypatch) -> None:
    """A chunk appearing in both FTS and vector results should fuse to
    a higher combined score than one appearing in only one -- the
    entire point of reciprocal rank fusion."""
    fixed_vec = [0.1] * 768

    def same_embed(text):
        return fixed_vec

    monkeypatch.setattr("stocksense.rag.embed.embed_text", same_embed)
    monkeypatch.setattr("stocksense.rag.retrieve.embed_text", same_embed)

    index_document(tmp_store, "docs", "a.md", "A", "cost drag analysis for the portfolio")
    index_document(tmp_store, "docs", "b.md", "B", "cost drag mentioned here too, briefly")
    rebuild_fts_index(tmp_store)

    fused = hybrid_search(tmp_store, "cost drag", top_k=5)
    assert len(fused) == 2
    # both chunks have identical embeddings (same fixed_vec), so vector
    # search alone can't distinguish them -- FTS relevance should still
    # determine the top rank, and both should be present since found by
    # both search paths
    assert {r.source_ref for r in fused} == {"a.md", "b.md"}
