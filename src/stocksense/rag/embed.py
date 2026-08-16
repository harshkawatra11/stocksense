"""
Optional embedding layer via local Ollama. Deliberately fails soft:
`embed_text` returns None (not raises) whenever Ollama isn't running or
the configured embedding model isn't pulled, and every caller in this
package treats None as "no vector for this chunk" rather than an error —
FTS-only is a genuinely supported degraded mode (docs/14-rag.md), not
just a documented aspiration, precisely because nothing downstream
assumes embeddings exist.

Verified 2026-08-16: this environment has Ollama installed and running
but only `qwen2.5:3b` pulled (a chat model, not embedding-capable) — no
embedding model is available today. embed_text will correctly return
None in that state; pulling `nomic-embed-text` (~274MB) later requires
no code change here, only `ollama pull nomic-embed-text`.
"""

from __future__ import annotations

import structlog
import requests

log = structlog.get_logger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # matches rag_chunks.embedding's FLOAT[768] column
_TIMEOUT_S = 10


def embed_text(text: str) -> list[float] | None:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=_TIMEOUT_S,
        )
        if resp.status_code != 200:
            return None
        embedding = resp.json().get("embedding")
        if not embedding or len(embedding) != EMBED_DIM:
            log.warning("embed_dim_mismatch", expected=EMBED_DIM, got=len(embedding) if embedding else 0)
            return None
        return embedding
    except requests.exceptions.RequestException:
        return None  # Ollama not running, model not pulled, or network error -- all treated the same: no embedding


def embeddings_available() -> bool:
    """Cheap availability probe (a real embed call, not just a health
    check, since the failure mode we actually care about -- model not
    pulled -- only shows up on a real request)."""
    return embed_text("availability probe") is not None
