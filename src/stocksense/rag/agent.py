"""
RAG query agent (docs/14-rag.md): `stocksense ask "<question>"`.
retrieve -> build a fact sheet of retrieved chunks -> Claude narrates an
answer -> every claim in the answer must trace to a retrieved chunk,
same compute/narrate discipline as everywhere else in this system
(agent/claude_cli.py's numeric tripwire doesn't apply directly here
since chunks are prose, not numbers, but the citation requirement is the
prose equivalent: no claim without a source).
"""

from __future__ import annotations

from stocksense.agent.claude_cli import AgentRequest, invoke
from stocksense.rag.retrieve import RetrievedChunk, hybrid_search

_ASK_PROMPT = """Answer the question below using ONLY the retrieved
passages provided as facts. Every claim in your answer must be
attributable to a specific passage -- cite it inline as [1], [2] etc.
matching the passage numbers below. If the retrieved passages don't
contain enough information to answer, say so plainly rather than
filling the gap with general knowledge about markets or trading --
this system's entire value is that its answers trace to StockSense's
own computed data and research, not to what an LLM generally believes
about finance.

Question: {question}

Retrieved passages:
{passages}
"""


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c.source_type}/{c.source_ref}, \"{c.title}\")\n{c.content}\n")
    return "\n".join(lines)


def ask(question: str, store, top_k: int = 5) -> dict:
    chunks = hybrid_search(store, question, top_k=top_k)
    if not chunks:
        return {
            "answer": "Nothing relevant found in the indexed corpus for this question.",
            "citations": [], "n_chunks_retrieved": 0,
        }

    facts = {
        "question": question,
        "retrieved_passages": [
            {"index": i + 1, "source_type": c.source_type, "source_ref": c.source_ref, "title": c.title, "content": c.content}
            for i, c in enumerate(chunks)
        ],
    }
    prompt = _ASK_PROMPT.format(question=question, passages=_format_passages(chunks))
    result = invoke(AgentRequest(prompt=prompt, facts=facts, skill="claude-report-writing"), store=store)

    return {
        "answer": result.output_text,
        "citations": [{"index": i + 1, "source_ref": c.source_ref, "title": c.title} for i, c in enumerate(chunks)],
        "n_chunks_retrieved": len(chunks),
        "agent_status": result.status,
    }
