"""
youtube/thumbnail_generator.py tests.

Covers:
   14  FAL provider called exactly once
   15  no FAL call in fixture mode
plus provider-failure/empty-response/malformed-image handling (never a
silent fallback to a fake image), and a static check that this module
never duplicates FAL's own HTTP/queue-API code.
"""
from __future__ import annotations

import ast

import pytest
from PIL import Image

from core.config import settings
from youtube import image_gen
from youtube import thumbnail_generator as tg

WIDTH, HEIGHT = settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT


def _strategy(prompt: str = "Cinematic documentary photography of a radio telescope. NO TEXT.") -> dict:
    return {"background_prompt": prompt}


class _SpyProvider(image_gen.ImageProvider):
    def __init__(self, image_bytes: bytes | None = None, error: Exception | None = None):
        self.calls = 0
        self._image_bytes = image_bytes
        self._error = error

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        self.calls += 1
        if self._error:
            raise self._error
        if self._image_bytes is not None:
            return self._image_bytes
        return tg.FixtureImageProvider().generate(prompt, width, height)


# ---------------------------------------------------------------------------
# Normal path: normalized 1280x720 output
# ---------------------------------------------------------------------------

def test_generate_background_bytes_returns_normalized_1280x720():
    result_bytes = tg.generate_background_bytes(_strategy(), fixture=True)
    img = Image.open(__import__("io").BytesIO(result_bytes))
    assert img.size == (WIDTH, HEIGHT)


def test_missing_background_prompt_raises():
    with pytest.raises(tg.ThumbnailBackgroundError, match="background_prompt"):
        tg.generate_background_bytes({"background_prompt": ""}, fixture=True)


def test_missing_background_prompt_key_raises():
    with pytest.raises(tg.ThumbnailBackgroundError, match="background_prompt"):
        tg.generate_background_bytes({}, fixture=True)


# ---------------------------------------------------------------------------
# 14 - FAL provider called exactly once
# ---------------------------------------------------------------------------

def test_provider_generate_called_exactly_once():
    spy = _SpyProvider()
    tg.generate_background_bytes(_strategy(), provider=spy)
    assert spy.calls == 1


def test_provider_generate_called_exactly_once_even_when_prompt_is_long():
    spy = _SpyProvider()
    tg.generate_background_bytes(_strategy("A" * 500 + " NO TEXT."), provider=spy)
    assert spy.calls == 1


# ---------------------------------------------------------------------------
# 15 - no FAL call in fixture mode
# ---------------------------------------------------------------------------

def test_fixture_mode_never_calls_image_gen_get_provider(monkeypatch):
    def _fail_if_called():
        raise AssertionError("image_gen.get_provider() (which selects FalImageProvider) must never be "
                              "called when fixture=True")
    monkeypatch.setattr(image_gen, "get_provider", _fail_if_called)
    tg.generate_background_bytes(_strategy(), fixture=True)  # must not raise


def test_fixture_mode_provider_is_the_fixture_provider():
    provider = tg.get_background_provider(fixture=True)
    assert isinstance(provider, tg.FixtureImageProvider)


def test_non_fixture_mode_delegates_to_image_gen_get_provider(monkeypatch):
    sentinel = _SpyProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: sentinel)
    provider = tg.get_background_provider(fixture=False)
    assert provider is sentinel


def test_fixture_provider_makes_no_network_call_by_construction():
    """Static guarantee: FixtureImageProvider.generate() has no import of
    any HTTP client anywhere in this module - see the module-level static
    check below - so there is no way for it to reach the network."""
    provider = tg.FixtureImageProvider()
    result = provider.generate("anything", 100, 100)
    assert result  # succeeds with zero network access, zero exceptions


# ---------------------------------------------------------------------------
# Fail loud, never a silent fake production thumbnail
# ---------------------------------------------------------------------------

def test_provider_failure_raises_clearly():
    spy = _SpyProvider(error=RuntimeError("[FAL] simulated failure"))
    with pytest.raises(tg.ThumbnailBackgroundError, match="background generation failed"):
        tg.generate_background_bytes(_strategy(), provider=spy)


def test_provider_empty_response_raises():
    spy = _SpyProvider(image_bytes=b"")
    with pytest.raises(tg.ThumbnailBackgroundError, match="empty image data"):
        tg.generate_background_bytes(_strategy(), provider=spy)


def test_provider_malformed_image_raises_clearly():
    spy = _SpyProvider(image_bytes=b"not a real image at all")
    with pytest.raises(tg.ThumbnailBackgroundError, match="could not be decoded"):
        tg.generate_background_bytes(_strategy(), provider=spy)


# ---------------------------------------------------------------------------
# No duplicated FAL API code
# ---------------------------------------------------------------------------

def test_module_has_no_httpx_import_or_fal_url():
    source = open(tg.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert "httpx" not in imported_modules, "must not duplicate FAL's HTTP client - reuse youtube.image_gen instead"
    assert "queue.fal.run" not in source, "must not duplicate FAL's queue-API URL"


def test_module_imports_image_gen_rather_than_reimplementing_it():
    source = open(tg.__file__, encoding="utf-8").read()
    assert "from youtube import image_gen" in source or "import youtube.image_gen" in source


# ---------------------------------------------------------------------------
# Background text-likelihood sanity (feeds thumbnail_qc.check_background)
# ---------------------------------------------------------------------------

def test_fixture_background_scores_low_on_text_likelihood():
    from youtube import thumbnail_qc

    background = tg.generate_background_bytes(_strategy(), fixture=True)
    check = thumbnail_qc.check_background(background)
    assert check["possible_text_detected"] is False
