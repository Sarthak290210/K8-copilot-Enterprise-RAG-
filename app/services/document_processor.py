"""Any supported file -> clean, token-bounded chunks (Docling + HybridChunker)."""

from io import BytesIO
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.io import DocumentStream
from transformers.models.auto.tokenization_auto import AutoTokenizer

from app.config import settings

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".html", ".htm", ".md", ".txt"})
# bge-base reads at most 512 tokens; chunks are sized to fit exactly.
MAX_CHUNK_TOKENS = 512


class DocumentProcessor:
    def __init__(self, table_structure: bool = True) -> None:
        # The corpus PDFs are born-digital, so OCR is off (it's the slowest step and adds noise).
        # Table structure recognition (~half of PDF time) can be turned off for bulk noise docs.
        pdf_options = PdfPipelineOptions(do_ocr=False, do_table_structure=table_structure)
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        )
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(settings.embedding_model),
            max_tokens=MAX_CHUNK_TOKENS,
        )
        # Splits on document structure (headings, tables, lists), merges small neighbours,
        # and never exceeds the embedding model's token limit.
        self.chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    def process_document(
        self, file_path: Path, source: str | None = None, max_pages: int | None = None
    ) -> list[dict[str, Any]]:
        """Convert and chunk one file -> [{"text", "source", "metadata"}].

        `source` defaults to the file name (what answers cite and evals match on).
        `max_pages` keeps only the first N pages of a PDF (bulk noise ingestion).
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        source = source or file_path.name

        if ext == ".txt":
            # Docling has no plain-text reader; plain text is valid Markdown (paragraphs).
            stream = DocumentStream(
                name=f"{file_path.stem}.md", stream=BytesIO(file_path.read_bytes())
            )
            document = self.converter.convert(stream).document
        elif max_pages and ext == ".pdf":
            document = self.converter.convert(file_path, page_range=(1, max_pages)).document
        else:
            document = self.converter.convert(file_path).document

        chunks = []
        for index, chunk in enumerate(self.chunker.chunk(document)):
            # contextualize() prefixes the chunk with its heading path, which helps retrieval.
            text = self.chunker.contextualize(chunk).strip()
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "metadata": {
                        "chunk_index": index,
                        "headings": [h for h in (getattr(chunk.meta, "headings", None) or []) if h],
                        "page": _first_page(chunk),
                    },
                }
            )
        return chunks


def _first_page(chunk: Any) -> int | None:
    """Page number of the chunk's first item (PDFs only; None for html/docx/txt)."""
    for item in getattr(chunk.meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            if getattr(prov, "page_no", None):
                return int(prov.page_no)
    return None
