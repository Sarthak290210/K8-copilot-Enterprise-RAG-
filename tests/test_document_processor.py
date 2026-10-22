"""Docling conversion + chunking on tiny fixtures (models load from the local cache)."""

import pytest

from app.services.document_processor import DocumentProcessor


@pytest.fixture(scope="module")
def processor():
    return DocumentProcessor()


def test_txt_file_gives_chunks_with_source(processor, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(
        "Pods are the smallest deployable units in Kubernetes.\n\n"
        "A Pod groups one or more containers with shared storage and network.",
        encoding="utf-8",
    )
    chunks = processor.process_document(path)
    assert len(chunks) >= 1
    assert all(c["source"] == "notes.txt" for c in chunks)  # original name, not the .md stream
    assert "smallest deployable units" in chunks[0]["text"]
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_html_headings_are_kept(processor, tmp_path):
    path = tmp_path / "pods.html"
    path.write_text(
        "<html><body><h1>Pods</h1><p>A Pod is a group of containers.</p>"
        "<h2>Lifecycle</h2><p>Pods move through phases such as Pending and Running.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    chunks = processor.process_document(path, source="custom-name.html")
    assert chunks and all(c["source"] == "custom-name.html" for c in chunks)
    joined = " ".join(c["text"] for c in chunks)
    assert "Pending" in joined
    assert any(c["metadata"]["headings"] for c in chunks)
    assert all("" not in c["metadata"]["headings"] for c in chunks)


def test_unsupported_extension(processor, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.process_document(path)
