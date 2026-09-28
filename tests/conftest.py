"""Shared fixtures: every test that touches storage gets its own tmp_path-based
STORAGE_DIR/DB_PATH so tests never read or write the real storage/factory.db."""
import pytest

from core.config import settings
from storage import db


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "storage" / "factory.db")
    monkeypatch.setattr(settings, "MEDIA_MUSIC_DIR", tmp_path / "storage" / "music")
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings.MEDIA_MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db(settings.DB_PATH)
    return settings
