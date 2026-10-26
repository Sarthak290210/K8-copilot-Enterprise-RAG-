import pytest

from app.storage.local_storage import LocalStorage


def test_save_read_exists(tmp_path):
    storage = LocalStorage(tmp_path / "store")
    assert not storage.exists("docs/a.json") and storage.read_bytes("docs/a.json") is None
    storage.save_bytes("docs/a.json", b"{}")  # creates parent folders
    assert storage.exists("docs/a.json") and storage.read_bytes("docs/a.json") == b"{}"
    assert storage.url_for("docs/a.json").startswith("file://")


@pytest.mark.parametrize("key", ["../escape.txt", "docs/../../escape.txt", "", "."])
def test_path_traversal_blocked(tmp_path, key):
    storage = LocalStorage(tmp_path / "store")
    with pytest.raises(ValueError, match="Invalid storage key"):
        storage.save_bytes(key, b"x")
    assert not (tmp_path / "escape.txt").exists()


def test_absolute_path_blocked(tmp_path):
    storage = LocalStorage(tmp_path / "store")
    with pytest.raises(ValueError):
        storage.read_bytes(str(tmp_path / "other.txt"))
