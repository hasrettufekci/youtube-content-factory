"""
FAL Background Generation (Thumbnail System V1 - stage 2 of 4:
    STRATEGIST -> FAL BACKGROUND GENERATION -> TEXT OVERLAY -> QC).

Reuses youtube/image_gen.py's EXISTING ImageProvider abstraction and
get_provider() selection (FalImageProvider for real generation,
MockImageProvider for IMAGE_PROVIDER=mock) rather than duplicating any FAL
HTTP/queue-API code - this module contains no httpx import and no FAL URL
of its own; see get_background_provider() below and
tests/test_thumbnail_generator.py's static "no duplicated FAL code" check.

This module's whole job: send the strategist's background_prompt to the
configured provider, and normalize whatever comes back to EXACTLY
settings.THUMBNAIL_WIDTH x settings.THUMBNAIL_HEIGHT via
youtube/image_normalize.py's EXISTING normalize_image_bytes_to_video_size()
- the same one-source-of-truth padding/letterboxing helper the video
pipeline already uses for its 1920x1080 contract (see
youtube/final_visual_render_demo.py), just called with a 1280x720 target
instead. Never stretched, never cropped, never duplicated here.

FAIL LOUD, NEVER FAKE: if the provider raises (a real FAL error, timeout,
bad response), this raises ThumbnailBackgroundError - it NEVER falls back
to a silently-generated mock/placeholder image standing in for a real
production thumbnail. The ONLY way a non-FAL image is ever used is the
explicit, opt-in `fixture=True` path below (see FixtureImageProvider),
which callers (tests, --fixture demo runs) choose deliberately.

FIXTURE MODE: for tests and safe/offline demo runs, get_background_provider(
fixture=True) returns FixtureImageProvider - a small, fully deterministic
(hashlib-based, not Python's per-process-salted built-in hash()) offline
stand-in implementing the SAME ImageProvider interface youtube/image_gen.py
already defines. It is deliberately NOT youtube.image_gen.MockImageProvider:
that provider draws the prompt text onto the image (useful for image_gen's
own scene-preview purpose), which would defeat this pipeline's own "no
readable text in the generated background" contract even in test/fixture
mode - FixtureImageProvider draws a solid color field with a plain border
and nothing else. Fixture mode makes NO network call of any kind.
"""
from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageDraw

from core.config import settings
from core.logging_setup import get_logger
from youtube import image_gen
from youtube import image_normalize as inorm

logger = get_logger("thumbnail_generator")

WIDTH = settings.THUMBNAIL_WIDTH
HEIGHT = settings.THUMBNAIL_HEIGHT


class ThumbnailBackgroundError(RuntimeError):
    """Raised for a missing/empty background_prompt, a provider failure
    (network/API error, empty response), or a provider response that
    can't be decoded/normalized as an image. Never raised as a disguised
    fallback to a fake image - see the module docstring."""


class FixtureImageProvider(image_gen.ImageProvider):
    """Deterministic, fully offline background provider for tests and
    `--fixture` demo runs - see the module docstring's FIXTURE MODE
    section. Draws NO text of any kind (unlike
    youtube.image_gen.MockImageProvider), so it honors this pipeline's
    "no readable text in the background" contract even in test mode."""

    _PALETTE = (
        (24, 28, 38), (36, 22, 46), (14, 30, 30), (42, 24, 18),
        (20, 34, 20), (30, 20, 34), (16, 28, 40), (38, 30, 16),
    )

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        color = self._PALETTE[int(digest[:8], 16) % len(self._PALETTE)]

        img = Image.new("RGB", (width, height), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width - 1, height - 1], outline=(90, 90, 90), width=4)
        # A simple deterministic radial-ish vignette so QC's text-likelihood
        # heuristic sees something photo-like rather than a perfectly flat
        # field, without drawing anything resembling text/typography.
        inset = min(width, height) // 6
        draw.ellipse([inset, inset, width - inset, height - inset],
                     outline=(min(color[0] + 40, 255), min(color[1] + 40, 255), min(color[2] + 40, 255)), width=3)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def get_background_provider(fixture: bool = False) -> image_gen.ImageProvider:
    """fixture=True -> FixtureImageProvider (never touches
    youtube.image_gen.get_provider()/FAL at all - see
    tests/test_thumbnail_generator.py's "no FAL call in fixture mode"
    test). fixture=False (default) -> youtube.image_gen.get_provider()
    UNCHANGED (FalImageProvider when IMAGE_PROVIDER=fal, image_gen's own
    MockImageProvider when IMAGE_PROVIDER=mock) - this module never
    reimplements that selection logic."""
    if fixture:
        return FixtureImageProvider()
    return image_gen.get_provider()


def generate_background_bytes(
    strategy: dict, provider: image_gen.ImageProvider | None = None, fixture: bool = False,
) -> bytes:
    """Makes exactly ONE provider.generate() call (see
    tests/test_thumbnail_generator.py's call-count test) and returns
    PNG bytes normalized to exactly WIDTH x HEIGHT. `provider`, when
    given, is used as-is (test injection / explicit control); otherwise
    resolved via get_background_provider(fixture=fixture)."""
    prompt = (strategy or {}).get("background_prompt")
    if not prompt or not str(prompt).strip():
        raise ThumbnailBackgroundError("strategy has no background_prompt to send to the image provider")

    resolved_provider = provider if provider is not None else get_background_provider(fixture=fixture)
    provider_label = type(resolved_provider).__name__
    logger.info(
        "[THUMBNAIL_GENERATOR] requesting background: provider=%s size=%dx%d "
        "(exactly one generation call - fixture=%s)",
        provider_label, WIDTH, HEIGHT, fixture,
    )

    try:
        raw_bytes = resolved_provider.generate(prompt, WIDTH, HEIGHT)
    except Exception as e:
        raise ThumbnailBackgroundError(f"background generation failed (provider={provider_label}): {e}") from e

    if not raw_bytes:
        raise ThumbnailBackgroundError(f"provider {provider_label} returned empty image data")

    try:
        normalized_bytes = inorm.normalize_image_bytes_to_video_size(raw_bytes, WIDTH, HEIGHT)
    except inorm.ImageNormalizationError as e:
        raise ThumbnailBackgroundError(
            f"provider {provider_label} returned an image that could not be decoded/normalized: {e}"
        ) from e

    logger.info("[THUMBNAIL_GENERATOR] background ready: provider=%s %dx%d", provider_label, WIDTH, HEIGHT)
    return normalized_bytes
