"""Storage in a local folder (default `./cache`, gitignored)."""

from pathlib import Path

from app.storage.storage_backend import StorageBackend


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str | Path = "./cache") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """base_dir / key, refusing anything that would land outside base_dir (`..`, absolute
        paths)."""
        path = (self.base_dir / key).resolve()
        if not path.is_relative_to(self.base_dir) or path == self.base_dir:
            raise ValueError(f"Invalid storage key: {key!r}")
        return path

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def save_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: a crash never leaves a half-written file

    def read_bytes(self, key: str) -> bytes | None:
        path = self._resolve(key)
        return path.read_bytes() if path.is_file() else None

    def url_for(self, key: str) -> str:
        return self._resolve(key).as_uri()
