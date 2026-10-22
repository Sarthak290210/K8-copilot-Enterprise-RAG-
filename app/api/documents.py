import asyncio
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from app.middleware.auth import User, get_current_user, require_admin
from app.services import embedding_service, vector_store
from app.services.document_processor import SUPPORTED_EXTENSIONS, DocumentProcessor

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class UploadResult(BaseModel):
    source: str
    chunks_indexed: int


class DocumentList(BaseModel):
    sources: list[str]


@lru_cache
def get_processor() -> DocumentProcessor:
    """Docling models load once (several seconds), on the first upload."""
    return DocumentProcessor()


@router.post("/upload", response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile, _: Annotated[User, Depends(require_admin)]
) -> UploadResult:
    # Keep only the base name: a client-supplied path must never choose where we write.
    source = Path(file.filename or "").name
    if Path(source).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File is larger than 20 MB")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / source
        path.write_bytes(data)
        try:
            # Docling is CPU-heavy: run it off the event loop.
            chunks = await asyncio.to_thread(get_processor().process_document, path, source)
        except Exception as exc:
            logger.warning("Could not process upload {}: {}", source, type(exc).__name__)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "Could not read this document"
            ) from None
    if not chunks:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No text found in document")

    embeddings = await embedding_service.embed_texts([c["text"] for c in chunks])
    # Replace any previous version, so a shorter new version leaves no stale chunks behind.
    await vector_store.delete_source(source)
    indexed = await vector_store.upsert_chunks(chunks, embeddings)
    logger.info("Indexed upload {} ({} chunks)", source, indexed)
    return UploadResult(source=source, chunks_indexed=indexed)


@router.get("", response_model=DocumentList)
async def list_documents(_: Annotated[User, Depends(get_current_user)]) -> DocumentList:
    return DocumentList(sources=await vector_store.list_sources())
