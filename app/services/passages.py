"""The one way chunks are turned into numbered passages for an LLM prompt.

The answer model (`[n] (source: x)` + text, cited as [n]) and the graders (CRAG, Self-RAG) use
the same numbering, so a grader's "passage 3" is the answer's "[3]". Phase 6 replaces this
format with spotlighting in one place.
"""

from app.models import RetrievedChunk


def format_passages(
    chunks: list[RetrievedChunk], max_chars: int | None = None, with_sources: bool = False
) -> str:
    """`[1] text`, `[2] text`, ... separated by blank lines.

    max_chars cuts each chunk's text (None = whole chunk); with_sources adds `(source: x)`.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text if max_chars is None else chunk.text[:max_chars]
        header = f"[{i}] (source: {chunk.source})\n" if with_sources else f"[{i}] "
        blocks.append(header + text)
    return "\n\n".join(blocks)
