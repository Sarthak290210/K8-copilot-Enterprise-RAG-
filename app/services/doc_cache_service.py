"""Document dedup (Phase 5): never ingest the same file twice.

The SHA-256 of a file's bytes is its identity. After an ingest we store
  docs/<hash>.json              {source, chunks, max_pages, ingested_at}
  sources/<sha256(source)>.json {hash}   <- the content now indexed under that file name
A file counts as "already ingested" only if its hash has a record AND that record's source still
points at this hash: when a source is replaced by new content, its old hash stops counting (its
chunks were deleted from Qdrant), even though the old record stays in storage.

Sync on purpose (file hashing and local/S3 I/O): the API calls it via asyncio.to_thread.
Callers that decide "skip" use the async `find_indexed`, which also asks Qdrant whether the
chunks are still there.
Fail-soft: a storage error is logged and treated as "not ingested", so ingest still works.
"""

import asyncio
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from app.storage.storage_backend import StorageBackend, get_storage_backend


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_file_hash(file_path: str | Path, chunk_size: int = 8192) -> str:
    """Streams the file in blocks, so big PDFs are never loaded into memory at once."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


class DocCacheService:
    def __init__(self, backend: StorageBackend | None = None) -> None:
        self.backend = backend or get_storage_backend()

    @staticmethod
    def _metadata_key(content_hash: str) -> str:
        return f"docs/{content_hash}.json"

    @staticmethod
    def _source_key(source: str) -> str:
        return f"sources/{hashlib.sha256(source.encode()).hexdigest()}.json"

    def _read_json(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self.backend.read_bytes(key)
            data = json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("Doc cache read failed ({})", type(exc).__name__)
            return None
        return data if isinstance(data, dict) else None

    def exists(self, content_hash: str) -> bool:
        """Has a file with these bytes ever been ingested? (See find_ingested for "is it still
        indexed".)"""
        return self.get_metadata(content_hash) is not None

    def get_metadata(self, content_hash: str) -> dict[str, Any] | None:
        return self._read_json(self._metadata_key(content_hash))

    def source_hash(self, source: str) -> str | None:
        """The hash of the content currently indexed under `source`, if recorded."""
        record = self._read_json(self._source_key(source))
        value = record.get("hash") if record else None
        return value if isinstance(value, str) else None

    def set_metadata(self, content_hash: str, metadata: dict[str, Any]) -> bool:
        """Record an ingest. `metadata` needs `source`; `ingested_at` is added if missing."""
        record = {"ingested_at": dt.datetime.now(dt.UTC).isoformat(), **metadata}
        try:
            self.backend.save_bytes(
                self._metadata_key(content_hash), json.dumps(record).encode("utf-8")
            )
            self.backend.save_bytes(
                self._source_key(record["source"]),
                json.dumps({"hash": content_hash, "source": record["source"]}).encode("utf-8"),
            )
        except Exception as exc:
            logger.warning("Doc cache write failed ({})", type(exc).__name__)
            return False
        return True

    def find_ingested(
        self, content_hash: str, max_pages: int | None = None
    ) -> dict[str, Any] | None:
        """The ingest record if these bytes are still indexed (under any file name) with the
        same page cap; None means "ingest it"."""
        metadata = self.get_metadata(content_hash)
        if metadata is None or metadata.get("max_pages") != max_pages:
            return None
        if self.source_hash(str(metadata.get("source"))) != content_hash:
            return None  # that source was re-ingested with other content since
        return metadata

    async def find_indexed(
        self, content_hash: str, max_pages: int | None = None
    ) -> dict[str, Any] | None:
        """find_ingested, plus a check that Qdrant still holds that source's chunks (as many
        as were recorded). The records outlive the index (`./cache` survives
        `docker compose down -v`; a failed upsert after delete_source), so they alone can't
        say "skip". A Qdrant error also means "ingest it" (re-ingesting replaces, never
        duplicates)."""
        from app.services import vector_store

        metadata = await asyncio.to_thread(self.find_ingested, content_hash, max_pages)
        if metadata is None:
            return None
        try:
            indexed = await vector_store.count_source(str(metadata.get("source")))
        except Exception as exc:
            logger.warning("Could not count indexed chunks ({})", type(exc).__name__)
            return None
        if indexed == 0 or indexed != metadata.get("chunks", indexed):
            logger.info("{} is recorded but not (fully) indexed; re-ingesting", metadata["source"])
            return None
        return metadata
