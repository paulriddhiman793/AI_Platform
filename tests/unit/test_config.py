"""Unit tests for config module."""
import pytest
from api.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "ai_platform"


def test_settings_cors_origins():
    """Test CORS origins parsing."""
    settings = Settings(frontend_origins="http://a.com,http://b.com")
    assert settings.cors_origins == ["http://a.com", "http://b.com"]


def test_settings_env_override(monkeypatch):
    """Test environment variable override."""
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000


def test_settings_validation():
    """Test settings validation."""
    with pytest.raises(ValueError):
        Settings(port=0)  # Invalid port
    
    with pytest.raises(ValueError):
        Settings(port=70000)  # Invalid port
    
    with pytest.raises(ValueError):
        Settings(groq_max_tokens=100)  # Below minimum


def test_settings_expand_path(tmp_path):
    """Test path expansion."""
    test_dir = tmp_path / "test_projects"
    settings = Settings(platform_storage_root=str(test_dir))
    assert settings.storage_path == test_dir.resolve()
    assert settings.storage_path.exists()