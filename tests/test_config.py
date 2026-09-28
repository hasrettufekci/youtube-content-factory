import pytest

import core.config as config_module
from core.config import settings, require


def test_settings_load_without_any_api_keys():
    # Import-time already happened (conftest imports core.config); this just
    # asserts that accessing settings never raised even with no keys set.
    assert settings.VIDEO_WIDTH == 1920
    assert settings.VIDEO_HEIGHT == 1080
    assert settings.STORAGE_DIR.exists()


def test_claude_timeout_seconds_defaults_to_240_on_the_loaded_singleton():
    # Same convention as test_settings_load_without_any_api_keys above:
    # asserts against the already-loaded settings singleton, trusting the
    # test environment has no CLAUDE_TIMEOUT_SECONDS override (no .env is
    # committed to this repo, and none of the fixtures set one).
    assert settings.CLAUDE_TIMEOUT_SECONDS == 240.0


def test_float_env_helper_returns_default_when_var_unset(monkeypatch):
    # Deliberately tests the real `_float()` helper Settings.CLAUDE_TIMEOUT_
    # SECONDS is built from (not a reimplementation of it) - and deliberately
    # does NOT importlib.reload(core.config), for the exact reason explained
    # in tests/test_scene_planner.py's test_scene_bounds_respect_env_override:
    # that would swap the module-level `settings` singleton for a brand-new
    # object, silently orphaning every other already-imported module's own
    # `from core.config import settings` reference. Calling the pure helper
    # function directly avoids that entirely.
    monkeypatch.delenv("CLAUDE_TIMEOUT_SECONDS", raising=False)
    assert config_module._float("CLAUDE_TIMEOUT_SECONDS", 240.0) == 240.0


def test_float_env_helper_parses_override_when_var_set(monkeypatch):
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "90")
    assert config_module._float("CLAUDE_TIMEOUT_SECONDS", 240.0) == 90.0
    # Non-integer values are valid too (it's a float, not an int).
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "97.5")
    assert config_module._float("CLAUDE_TIMEOUT_SECONDS", 240.0) == 97.5


def test_require_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(settings, "FAL_KEY", "")
    with pytest.raises(RuntimeError, match="FAL_KEY"):
        require("FAL_KEY")


def test_require_returns_value_when_present(monkeypatch):
    monkeypatch.setattr(settings, "FAL_KEY", "sk-test-123")
    assert require("FAL_KEY") == "sk-test-123"
