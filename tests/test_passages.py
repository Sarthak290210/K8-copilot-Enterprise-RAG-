from app.models import RetrievedChunk
from app.services.passages import format_passages

CHUNKS = [
    RetrievedChunk(text="Pods are the smallest units.", source="pods.html", score=0.9),
    RetrievedChunk(text="A Service exposes Pods.", source="service.html", score=0.8),
]


def test_numbered_with_sources_for_the_answer_model():
    assert format_passages(CHUNKS, with_sources=True) == (
        "[1] (source: pods.html)\nPods are the smallest units.\n\n"
        "[2] (source: service.html)\nA Service exposes Pods."
    )


def test_plain_and_cut_for_graders():
    assert format_passages(CHUNKS, max_chars=4) == "[1] Pods\n\n[2] A Se"


def test_whole_chunks_by_default():
    assert (
        format_passages(CHUNKS) == "[1] Pods are the smallest units.\n\n[2] A Service exposes Pods."
    )


def test_no_chunks():
    assert format_passages([]) == ""
