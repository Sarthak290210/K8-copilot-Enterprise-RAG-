import pytest
from pydantic import ValidationError

from app.models import ChatResponse, QueryRequest, RetrievedChunk, RetrievedChunkPreview


def test_preview_collapses_whitespace_and_truncates():
    chunk = RetrievedChunk(
        text="Pods   are\n\nsmall " + "x" * 500, source="pods.html", score=0.81234
    )
    preview = RetrievedChunkPreview.from_chunk(chunk)
    assert preview.snippet.startswith("Pods are small ")
    assert len(preview.snippet) == 200
    assert preview.score == 0.8123


def test_query_request_limits():
    assert QueryRequest(question="What is a Pod?").flags is None
    with pytest.raises(ValidationError):
        QueryRequest(question="")
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 2001)


def test_chat_response_defaults():
    response = ChatResponse(answer="hi")
    assert response.sources == [] and response.confidence is None
    assert response.metadata.intent == "rag"
