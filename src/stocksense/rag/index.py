"""
RAG corpus indexing (docs/14-rag.md). Chunking + content-hashed,
idempotent writes into rag_documents/rag_chunks. Embeddings are
attempted but never required — a chunk with no embedding still gets
indexed and is still findable by full-text search, which is what makes
FTS-only degraded mode (no Ollama running, or no embedding model pulled)
a genuinely supported path rather than a documented aspiration.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from stocksense.rag import embed as embed_module

DEFAULT_CHUNK_SIZE = 512  # words, not tokens -- a simple, dependency-free proxy
DEFAULT_OVERLAP = 64


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(content: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Paragraph-aware chunking: splits on blank lines first (so a
    chunk boundary tends to land between logical sections, not mid-
    sentence), then packs paragraphs into ~chunk_size-word windows with
    overlap, falling back to a hard word-count split for any single
    paragraph longer than chunk_size on its own."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = para.split()
        if len(para_words) > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
            for i in range(0, len(para_words), chunk_size - overlap):
                chunks.append(" ".join(para_words[i:i + chunk_size]))
            continue

        if current_words + len(para_words) > chunk_size and current:
            chunks.append("\n\n".join(current))
            # start the next chunk with an overlap tail of the previous one
            overlap_words = "\n\n".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_words = len(overlap_words)

        current.append(para)
        current_words += len(para_words)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def index_document(store, source_type: str, source_ref: str, title: str, content: str,
                    metadata: dict | None = None, embed_chunks: bool = True) -> dict:
    """Idempotent: if `content`'s hash matches an already-indexed
    document with the same source_ref, this is a no-op (returns
    reindexed=False). Otherwise removes old chunks for that source_ref
    (if any) and writes fresh ones -- so re-indexing after a source file
    changes never leaves stale chunks alongside new ones.

    `embed_chunks=False` skips the embedding call entirely for every
    chunk in this call. Pass this when a caller has already confirmed
    (once, via rag.embed.embeddings_available()) that no embedding
    model is available -- each embed_text call costs a real ~2s network
    round trip even to determine "unavailable," and a multi-document
    indexing run making that call once per chunk turns a few seconds of
    work into minutes for no benefit, since the answer doesn't change
    between chunks within one indexing session."""
    import json

    h = content_hash(content)
    existing = store.con.execute(
        "SELECT doc_id, content_hash FROM rag_documents WHERE source_ref = ?", [source_ref]
    ).fetchdf()

    if not existing.empty and existing.iloc[0]["content_hash"] == h:
        return {"reindexed": False, "doc_id": existing.iloc[0]["doc_id"], "n_chunks": 0}

    doc_id = existing.iloc[0]["doc_id"] if not existing.empty else str(uuid.uuid4())
    if not existing.empty:
        store.con.execute("DELETE FROM rag_chunks WHERE doc_id = ?", [doc_id])
        store.con.execute("DELETE FROM rag_documents WHERE doc_id = ?", [doc_id])

    store.con.execute(
        "INSERT INTO rag_documents (doc_id, source_type, source_ref, title, content, content_hash, indexed_at, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [doc_id, source_type, source_ref, title, content, h, datetime.now(timezone.utc), json.dumps(metadata or {})],
    )

    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        # call through the module (not a name imported at module-load
        # time) so tests can monkeypatch stocksense.rag.embed.embed_text
        # and have it take effect here -- an `from ... import embed_text`
        # binds the function object once at import time and a later
        # monkeypatch of the source module has no effect on that binding.
        embedding = embed_module.embed_text(chunk) if embed_chunks else None
        chunk_id = str(uuid.uuid4())
        if embedding is not None:
            store.con.execute(
                "INSERT INTO rag_chunks (chunk_id, doc_id, chunk_index, content, token_count, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                [chunk_id, doc_id, i, chunk, len(chunk.split()), embedding],
            )
        else:
            store.con.execute(
                "INSERT INTO rag_chunks (chunk_id, doc_id, chunk_index, content, token_count) VALUES (?, ?, ?, ?, ?)",
                [chunk_id, doc_id, i, chunk, len(chunk.split())],
            )

    return {"reindexed": True, "doc_id": doc_id, "n_chunks": len(chunks)}


def rebuild_fts_index(store) -> None:
    """DuckDB's fts index is a snapshot, not incrementally maintained --
    rebuild it after a batch of index_document calls, not after each one
    (rebuilding per-document would be needless repeated work for a
    multi-document indexing run)."""
    store.con.execute("INSTALL fts")
    store.con.execute("LOAD fts")
    store.con.execute("PRAGMA create_fts_index('rag_chunks', 'chunk_id', 'content', overwrite=1)")
