"""One small interface for "store these bytes under this key", so the doc-dedup cache works the
same on a local folder (dev) and on S3 (deployed). Switch with STORAGE_BACKEND=local|s3."""

from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import settings


class StorageBackend(ABC):
    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def save_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes | None:
        """The stored bytes, or None if the key doesn't exist."""

    @abstractmethod
    def url_for(self, key: str) -> str: ...


@lru_cache
def get_storage_backend() -> StorageBackend:
    """The configured backend (one per process). Imports are lazy: boto3 only loads for s3."""
    if settings.storage_backend == "local":
        from app.storage.local_storage import LocalStorage

        return LocalStorage(settings.local_storage_dir)
    if settings.storage_backend == "s3":
        from app.storage.s3_storage import S3Storage

        return S3Storage()
    raise ValueError(f"Unknown STORAGE_BACKEND: {settings.storage_backend!r}")
