"""format_passages: since Phase 6 every passage is spotlighted (L8)."""

from app.models import RetrievedChunk
from app.services.passages import format_passages

CHUNKS = [
    RetrievedChunk(text="Pods are the smallest units.", source="pods.html", score=0.9),
    RetrievedChunk(text="A Service exposes Pods.", source="service.html", score=0.8),
]


def test_numbered_with_sources_for_the_answer_model():
    assert format_passages(CHUNKS, with_sources=True) == (
        "<<DOCUMENT id=1 source=pods.html>>\nPods are the smallest units.\n<</DOCUMENT>>\n\n"
        "<<DOCUMENT id=2 source=service.html>>\nA Service exposes Pods.\n<</DOCUMENT>>"
    )


def test_plain_and_cut_for_graders():
    assert format_passages(CHUNKS, max_chars=4) == (
        "<<DOCUMENT id=1>>\nPods\n<</DOCUMENT>>\n\n<<DOCUMENT id=2>>\nA Se\n<</DOCUMENT>>"
    )


def test_whole_chunks_by_default():
    assert "Pods are the smallest units." in format_passages(CHUNKS)
    assert "A Service exposes Pods." in format_passages(CHUNKS)


def test_no_chunks():
    assert format_passages([]) == ""
