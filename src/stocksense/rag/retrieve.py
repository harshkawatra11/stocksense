"""
Hybrid retrieval (docs/14-rag.md): BM25 full-text search (DuckDB `fts`,
always available) fused with vector search (DuckDB `vss`, only when
embeddings exist for a chunk) via reciprocal rank fusion. FTS-only is
the floor, not a fallback bolted on afterward — vector search only ever
ADDS to what FTS already found, so a query never returns fewer or worse
results because Ollama happened to be down.
"""

from __future__ import annotations

from dataclasses import dataclass

from stocksense.rag.embed import embed_text

RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    content: str
    source_type: str
    source_ref: str
    title: str
    score: float


def fts_search(store, query: str, top_k: int = 10) -> list[RetrievedChunk]:
    store.con.execute("INSTALL fts")
    store.con.execute("LOAD fts")
    try:
        rows = store.con.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.content, d.source_type, d.source_ref, d.title,
                   fts_main_rag_chunks.match_bm25(c.chunk_id, ?) AS score
            FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.doc_id
            WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT ?
            """,
            [query, top_k],
        ).fetchall()
    except Exception:
        return []  # no FTS index built yet (e.g. nothing indexed) -- empty results, not a crash
    return [RetrievedChunk(r[0], r[1], r[2], r[3], r[4], r[5], float(r[6])) for r in rows]


def vector_search(store, query: str, top_k: int = 10) -> list[RetrievedChunk]:
    query_embedding = embed_text(query)
    if query_embedding is None:
        return []  # degraded mode: no embeddings available for this query either

    store.con.execute("INSTALL vss")
    store.con.execute("LOAD vss")
    rows = store.con.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.content, d.source_type, d.source_ref, d.title,
               array_distance(c.embedding, ?::FLOAT[768]) AS distance
        FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.doc_id
        WHERE c.embedding IS NOT NULL
        ORDER BY distance ASC LIMIT ?
        """,
        [query_embedding, top_k],
    ).fetchall()
    # convert distance (lower=better) to a score (higher=better) for RRF consistency
    return [RetrievedChunk(r[0], r[1], r[2], r[3], r[4], r[5], -float(r[6])) for r in rows]


def hybrid_search(store, query: str, top_k: int = 10) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: each result's fused score is
    sum(1 / (RRF_K + rank)) across whichever result lists it appeared
    in. A chunk found by both FTS and vector search ranks higher than
    one found by only one -- but a chunk found ONLY by FTS still ranks,
    since vector_search legitimately returns [] with no embeddings
    configured."""
    fts_results = fts_search(store, query, top_k=top_k * 2)
    vec_results = vector_search(store, query, top_k=top_k * 2)

    fused: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}
    for rank, chunk in enumerate(fts_results):
        fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup[chunk.chunk_id] = chunk
    for rank, chunk in enumerate(vec_results):
        fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup.setdefault(chunk.chunk_id, chunk)

    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_lookup[cid].chunk_id, chunk_lookup[cid].doc_id, chunk_lookup[cid].content,
            chunk_lookup[cid].source_type, chunk_lookup[cid].source_ref, chunk_lookup[cid].title,
            score=fused[cid],
        )
        for cid in ranked_ids
    ]
