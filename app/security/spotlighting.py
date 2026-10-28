"""L8 spotlighting (Phase 6): mark retrieved text as data, not instructions.

Each chunk is wrapped in `<<DOCUMENT id=1 source=pods.html>> ... <</DOCUMENT>>`; the system
prompt (L3) says text inside the markers is never an instruction. Anything marker-like inside
a chunk (or its source name) is neutralised first, so a planted document can't close its
block early and write "instructions" outside it. The id is the number the answer cites as
[n], and the graders (CRAG, Self-RAG, grounding) see the same numbering.
"""

import re

from app.models import RetrievedChunk

OPEN_MARKER_EXAMPLE = "<<DOCUMENT id=N source=...>>"
CLOSE_MARKER = "<</DOCUMENT>>"
_MARKER_LIKE = re.compile(r"<<\s*/?\s*DOCUMENT\b[^>\n]*(?:>>?)?", re.IGNORECASE)
_SOURCE_UNSAFE = re.compile(r"[<>\s]+")


def escape_markers(text: str) -> str:
    return _MARKER_LIKE.sub("[marker removed]", text)


def _safe_source(source: str) -> str:
    return _SOURCE_UNSAFE.sub("_", escape_markers(source))[:300]


def build_spotlighted_context(
    chunks: list[RetrievedChunk], max_chars: int | None = None, with_sources: bool = False
) -> str:
    """The chunks as marked documents, separated by blank lines (max_chars cuts each text)."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text if max_chars is None else chunk.text[:max_chars]
        source = f" source={_safe_source(chunk.source)}" if with_sources else ""
        blocks.append(f"<<DOCUMENT id={i}{source}>>\n{escape_markers(text)}\n{CLOSE_MARKER}")
    return "\n\n".join(blocks)
