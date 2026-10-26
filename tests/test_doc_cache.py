import hashlib

from app.services.doc_cache_service import (
    DocCacheService,
    compute_content_hash,
    compute_file_hash,
)
from app.storage.local_storage import LocalStorage


def make_cache(tmp_path):
    return DocCacheService(LocalStorage(tmp_path / "store"))


def test_same_bytes_same_hash(tmp_path):
    path = tmp_path / "a.pdf"
    data = b"%PDF" + bytes(range(256)) * 100  # > one 8 KB block
    path.write_bytes(data)
    assert compute_file_hash(path, chunk_size=1000) == compute_content_hash(data)
    assert compute_content_hash(data) == hashlib.sha256(data).hexdigest()
    assert compute_content_hash(b"a") != compute_content_hash(b"b")


def test_ingest_record_round_trip(tmp_path):
    cache = make_cache(tmp_path)
    h = compute_content_hash(b"pods")
    assert not cache.exists(h) and cache.find_ingested(h) is None
    assert cache.set_metadata(h, {"source": "pods.txt", "chunks": 3, "max_pages": None})
    record = cache.find_ingested(h)
    assert record is not None and record["chunks"] == 3 and "ingested_at" in record
    assert cache.source_hash("pods.txt") == h


def test_second_ingest_of_same_file_is_skipped_but_not_with_another_page_cap(tmp_path):
    cache = make_cache(tmp_path)
    h = compute_content_hash(b"manual")
    cache.set_metadata(h, {"source": "manual.pdf", "chunks": 40, "max_pages": 30})
    assert cache.find_ingested(h, max_pages=30) is not None
    assert cache.find_ingested(h, max_pages=None) is None  # whole file wasn't indexed


def test_replaced_source_no_longer_counts_old_content(tmp_path):
    cache = make_cache(tmp_path)
    v1, v2 = compute_content_hash(b"v1"), compute_content_hash(b"v2")
    cache.set_metadata(v1, {"source": "pods.txt", "chunks": 1, "max_pages": None})
    cache.set_metadata(v2, {"source": "pods.txt", "chunks": 1, "max_pages": None})
    assert cache.exists(v1)  # the record stays...
    assert cache.find_ingested(v1) is None  # ...but its chunks were replaced by v2
    assert cache.find_ingested(v2) is not None


def test_storage_errors_are_soft(tmp_path):
    class Broken(LocalStorage):
        def read_bytes(self, key):
            raise OSError("disk gone")

        def save_bytes(self, key, data):
            raise OSError("disk gone")

    cache = DocCacheService(Broken(tmp_path / "store"))
    assert cache.find_ingested("abc") is None  # -> ingest it
    assert cache.set_metadata("abc", {"source": "x"}) is False
