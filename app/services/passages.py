"""The one way chunks are turned into numbered passages for an LLM prompt.

The answer model (cites [n]) and the graders (CRAG, Self-RAG, grounding) use the same
numbering, so a grader's "document 3" is the answer's "[3]". Since Phase 6 every passage is
spotlighted (`<<DOCUMENT id=n ...>> ... <</DOCUMENT>>`, security/spotlighting.py).
"""

from app.models import RetrievedChunk
from app.security.spotlighting import build_spotlighted_context


def format_passages(
    chunks: list[RetrievedChunk], max_chars: int | None = None, with_sources: bool = False
) -> str:
    """Spotlighted documents, separated by blank lines.

    max_chars cuts each chunk's text (None = whole chunk); with_sources adds `source=x`.
    """
    return build_spotlighted_context(chunks, max_chars=max_chars, with_sources=with_sources)
