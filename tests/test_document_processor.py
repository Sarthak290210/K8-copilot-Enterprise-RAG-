"""Docling conversion + chunking on tiny fixtures (models load from the local cache)."""

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.text import SectionHeaderItem, TextItem, TitleItem

from app.services.document_processor import (
    MAX_HEADING_CHARS,
    DocumentProcessor,
    demote_long_headings,
)


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


def test_demote_long_headings_keeps_short_ones():
    doc = DoclingDocument(name="t")
    doc.add_title(text="RBAC Good Practices")
    doc.add_heading(text="Least privilege", level=1)
    long_text = "Ideally, minimal RBAC rights should be assigned to users. " * 5
    doc.add_heading(text=long_text, level=1)
    doc.add_title(text="x" * (MAX_HEADING_CHARS + 1))
    assert demote_long_headings(doc) == 2
    kinds = [type(item) for item in doc.texts]
    assert kinds == [TitleItem, SectionHeaderItem, TextItem, TextItem]
    assert doc.texts[2].text == long_text and doc.texts[2].self_ref == "#/texts/2"
    assert demote_long_headings(doc) == 0  # idempotent


def test_long_heading_text_reaches_the_chunks(processor, tmp_path):
    """Two headers in a row: the chunker kept only the last as the heading, so a paragraph
    mislabelled as a header vanished (rbac-good-practices.pdf lost pages 1-2 this way)."""
    paragraph = "Least privilege: grant only the RBAC rights a workload needs. " * 5
    path = tmp_path / "rbac.html"
    path.write_text(
        f"<html><body><h2>{paragraph}</h2><h2>Escalation risks</h2>"
        "<p>The escalate verb lets users create bindings beyond their rights.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    chunks = processor.process_document(path)
    joined = " ".join(c["text"] for c in chunks)
    assert "grant only the RBAC rights" in joined
    assert "escalate verb" in joined
    assert all(paragraph.strip() not in c["metadata"]["headings"] for c in chunks)


def test_unsupported_extension(processor, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.process_document(path)
