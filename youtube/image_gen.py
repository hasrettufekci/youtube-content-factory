"""
Scene image generation (spec section 16).

Provider-abstracted (like tts.py) so swapping the image backend later is a
one-line config change (IMAGE_PROVIDER=fal|mock), same reasoning as the TTS
abstraction the spec asks for in section 17.

- FalImageProvider: real fal.ai queue API (flux/dev by default), 1920x1080.
- MockImageProvider: no network - renders a labeled placeholder locally with
  Pillow so the whole pipeline (storage, ffmpeg assembly, QC) is runnable
  and testable with zero API keys/cost. Never used unless IMAGE_PROVIDER=mock.

Images are written straight to their final location (scene_001.png, ...)
inside the video's project folder and, per spec section 25/26, generation
is idempotent/resumable: an existing scene_NNN.png is left alone and not
regenerated.

Standalone test:
    python -m youtube.image_gen --video-id 1
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import httpx

from core.config import settings, require
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import overlay_renderer
from youtube import programmatic_graphic_renderer

logger = get_logger("image_gen")

WIDTH = settings.VIDEO_WIDTH
HEIGHT = settings.VIDEO_HEIGHT


class ImageProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, width: int, height: int) -> bytes:
        """Return PNG/JPEG bytes for the given prompt."""


def _call_with_retries(description: str, fn: Callable[[], httpx.Response]) -> httpx.Response:
    """Run one httpx call (`fn`, a zero-arg callable that performs the
    request and calls .raise_for_status() itself) with retry-with-backoff
    on transient failures - HTTP 429/5xx, timeouts, connection errors -
    mirroring the pattern already used in youtube/llm_client.py's
    `_post_with_retries` and youtube/tts.py's OpenAITTSProvider. A
    non-retryable 4xx (bad prompt, bad API key, ...) fails immediately
    instead of burning through retries on an error that will never
    resolve itself."""
    last_error: Exception | None = None
    for attempt in range(1, settings.FAL_MAX_RETRIES + 1):
        try:
            return fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 or e.response.status_code >= 500:
                last_error = e
            else:
                raise RuntimeError(
                    f"[FAL] {description} failed: {e.response.status_code} {e.response.text[:300]}"
                ) from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_error = e

        if attempt < settings.FAL_MAX_RETRIES:
            wait = min(2 ** attempt, 15)
            logger.warning(
                "[FAL] %s attempt %d/%d failed (%s), retrying in %ss",
                description, attempt, settings.FAL_MAX_RETRIES, last_error, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"[FAL] {description} gave up after {settings.FAL_MAX_RETRIES} attempts: {last_error}")


class FalImageProvider(ImageProvider):
    def generate(self, prompt: str, width: int, height: int) -> bytes:
        fal_key = require("FAL_KEY")
        fal_url = f"https://queue.fal.run/{settings.FAL_IMAGE_MODEL}"
        headers = {"Authorization": f"Key {fal_key}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "image_size": {"width": width, "height": height}, "num_images": 1}

        with httpx.Client(timeout=120) as client:
            def _submit() -> httpx.Response:
                resp = client.post(fal_url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp

            submit_data = _call_with_retries("submit", _submit).json()

            status_url = submit_data["status_url"]
            result_url = submit_data["response_url"]

            for _ in range(90):
                def _poll() -> httpx.Response:
                    resp = client.get(status_url, headers=headers)
                    resp.raise_for_status()
                    return resp

                status_resp = _call_with_retries("status poll", _poll)
                status = status_resp.json().get("status")
                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    raise RuntimeError(f"[FAL] image generation failed: {status_resp.text}")
                time.sleep(2)
            else:
                raise RuntimeError("[FAL] image generation timed out")

            def _fetch_result() -> httpx.Response:
                resp = client.get(result_url, headers=headers)
                resp.raise_for_status()
                return resp

            result_data = _call_with_retries("result fetch", _fetch_result).json()

        images = result_data.get("images", [])
        if not images:
            raise RuntimeError(f"[FAL] no images in result: {result_data}")

        def _download() -> httpx.Response:
            resp = httpx.get(images[0]["url"], timeout=60)
            resp.raise_for_status()
            return resp

        image_resp = _call_with_retries("image download", _download)
        if not image_resp.content:
            raise RuntimeError("[FAL] downloaded image is empty")
        return image_resp.content


class MockImageProvider(ImageProvider):
    """Offline placeholder image generator - no network, no API key. Lets the
    pipeline, ffmpeg assembly and quality checks be exercised end-to-end
    without paying for or requiring fal.ai."""

    _PALETTE = [
        (18, 24, 38), (30, 20, 45), (12, 32, 32), (40, 22, 18),
        (20, 30, 20), (28, 18, 32), (14, 26, 36), (34, 28, 14),
    ]

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont

        color = self._PALETTE[abs(hash(prompt)) % len(self._PALETTE)]
        img = Image.new("RGB", (width, height), color)
        draw = ImageDraw.Draw(img)

        # simple vignette-ish frame so scenes are visually distinguishable in a preview
        draw.rectangle([0, 0, width - 1, height - 1], outline=(90, 90, 90), width=4)

        try:
            font = ImageFont.load_default(size=36)
        except TypeError:
            font = ImageFont.load_default()

        wrapped = textwrap.fill(prompt, width=60)
        draw.multiline_text((60, 60), wrapped, fill=(230, 230, 230), font=font, spacing=10)
        draw.text((60, height - 80), "[MOCK IMAGE - set IMAGE_PROVIDER=fal for real generation]",
                   fill=(160, 160, 160), font=font)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def get_provider() -> ImageProvider:
    if settings.IMAGE_PROVIDER == "mock":
        return MockImageProvider()
    return FalImageProvider()


def _is_valid_existing_image(dest: Path) -> bool:
    """Whether an existing scene_NNN.png counts as "already generated" for
    resume purposes (spec Round 2 section 6/12: a failed scene must be
    retried without re-generating the scenes that already succeeded - but
    a zero-byte or truncated file left behind by, say, a process killed
    mid-write during an earlier interrupted run must NOT be mistaken for a
    successful generation, or a corrupt/blank frame would be silently
    baked into the final video). A non-empty file is as far as we can
    check without an image-decoding dependency; combined with the atomic
    write below (write-to-temp then rename), a file only ever reaches
    `dest` after being fully written, so "exists and non-empty" is a
    reliable completeness signal here."""
    try:
        return dest.stat().st_size > 0
    except OSError:
        return False


def _write_image_atomic(dest: Path, image_bytes: bytes) -> None:
    """Atomic write: write to a sibling temp file, then rename into place.
    `Path.replace` is an atomic rename on both POSIX and Windows, so a
    crash/interrupt mid-write can never leave a corrupt file sitting at the
    final scene_NNN.png path - it's either the old temp leftovers (ignored,
    not scene_NNN.png) or the complete new file, never something in
    between. Extracted verbatim from generate_scene_images()'s previous
    inline body so every strategy branch below (a real FAL image, a mock
    image, or a programmatic graphic) goes through the exact same
    resume-safe write path - Visual Generator V2 integration must not
    weaken this."""
    tmp_dest = dest.with_name(dest.name + ".tmp")
    tmp_dest.write_bytes(image_bytes)
    tmp_dest.replace(dest)


def _generate_image_bytes_for_scene(scene: dict, provider: ImageProvider) -> bytes:
    """Strategy-aware decision layer sitting on top of ImageProvider
    (Visual Generator V2 integration). Dispatches on
    scene.get("visual_strategy") - Scene Planner V2's DETERMINISTICALLY
    resolved field (resolve_visual_strategy() in scene_planner.py; the LLM
    never decides visual_strategy directly). A v1-shaped scene (no
    visual_strategy key at all) defaults to "ai_generated", so v1 scenes
    flow through this same function unchanged - full backward
    compatibility, not a separate code path.

    Invariants that must never be weakened here:
      - "programmatic_graphic" NEVER calls provider.generate() - Phase 2's
        build_visual_prompt() deliberately leaves visual_prompt=None for
        this strategy, so passing that through to an AI image provider
        would either crash on a None prompt or silently produce a
        meaningless picture. A local, deterministic Pillow chart/graphic
        (youtube/programmatic_graphic_renderer.py - Phase 4) is rendered
        from scene["data_points"]/scene["exact_text"] instead - no AI
        model, no FAL, no API credits spent on this strategy at all.
      - every other branch sends ONLY scene.get("visual_prompt") to the
        provider - never exact_text/data_points/text_render. Those are
        overlay/accuracy metadata for the overlay renderer (see
        youtube/overlay_renderer.py) to draw as literal, pixel-accurate
        text on top of the AI image; burning them into the AI generation
        prompt itself risks the model mangling a code or number it has no
        reason to render faithfully (e.g. "6EQUJ5", "1420.726 MHz", a
        stated RA/Dec value).
      - once the base image comes back from the provider, scene["overlay_required"]
        (never the visual_strategy string directly - see below) is the
        single gate for whether overlay_renderer.render_scene_overlay_bytes()
        runs on it before this function returns. Today overlay_required is
        only ever True for strategy=="ai_generated_plus_overlay" (see
        scene_planner.py's build_scene_prompt()), but gating on the field
        instead of the strategy name means every branch below - including
        a v1 scene (no visual_strategy key, overlay_required absent/False)
        - is already correct without a strategy-specific special case.

    The scene_type-based fallback-transparency logging below is a SECOND,
    orthogonal axis - not a routing decision, and it never changes what
    gets sent to the provider. There is no real research/archival source
    system yet, so scene_type=="research_evidence" and
    scene_type=="document_recreation" scenes currently fall through to the
    same AI generation as everything else; the logging exists purely so
    that fallback is visible instead of silent. (scene_type=="document_recreation"
    is checked here, not the literal "archival_source" - resolve_visual_strategy()'s
    rule 0 unconditionally normalizes archival_source to document_recreation,
    so a stored/validated scene can never actually carry the literal value
    "archival_source"; document_recreation is what a real archival-source
    scene shows in practice.)
    """
    strategy = scene.get("visual_strategy") or "ai_generated"
    scene_type = scene.get("scene_type")
    scene_id = scene.get("scene_id")

    logger.info("[IMAGE][scene=%03d][strategy=%s]", scene_id, strategy)

    if strategy == "programmatic_graphic":
        png_bytes, renderer_name = programmatic_graphic_renderer.render_programmatic_graphic(
            scene, width=WIDTH, height=HEIGHT,
        )
        logger.info(
            "[IMAGE][PROGRAMMATIC] scene=%03d renderer=%s", scene_id, renderer_name,
        )
        return png_bytes

    prompt = scene.get("visual_prompt")
    if not prompt:
        raise RuntimeError(
            f"[IMAGE] scene {scene_id}: strategy={strategy!r} but visual_prompt is empty/None - "
            f"refusing to call the image provider without a prompt"
        )

    if scene_type == "research_evidence":
        logger.info(
            "[IMAGE][RESEARCH_EVIDENCE][FALLBACK=AI] scene=%03d - no real research/archival source "
            "system yet, falling back to AI generation", scene_id,
        )
    elif scene_type == "document_recreation":
        logger.info(
            "[IMAGE][ARCHIVAL][FALLBACK=AI] scene=%03d - no real research/archival source system "
            "yet, falling back to AI generation", scene_id,
        )

    image_bytes = provider.generate(prompt, WIDTH, HEIGHT)

    if scene.get("overlay_required"):
        overlaid = overlay_renderer.render_scene_overlay_bytes(image_bytes, scene)
        if overlaid is image_bytes:
            # overlay_required was True but the scene carried no
            # exact_text/data_points/hypothesis_badge to actually draw
            # (missing-metadata edge case) - render_scene_overlay_bytes()
            # already no-ops safely in this case; log it as such rather
            # than falsely claiming an overlay was rendered.
            logger.warning(
                "[IMAGE][OVERLAY] scene=%03d overlay_required=True but no exact_text/data_points/"
                "hypothesis_badge found - base AI image returned unchanged", scene_id,
            )
        else:
            image_bytes = overlaid
            logger.info(
                "[IMAGE][OVERLAY] scene=%03d rendered exact_text=%d data_points=%d",
                scene_id, len(scene.get("exact_text") or []), len(scene.get("data_points") or []),
            )

    return image_bytes


def generate_scene_images(paths: ps.ProjectPaths, scenes: list[dict]) -> list:
    provider = get_provider()
    logger.info("[FAL] generating %d scene images (provider=%s)", len(scenes), settings.IMAGE_PROVIDER)

    result_paths = []
    for scene in scenes:
        dest = paths.scene_image(scene["scene_id"])
        if dest.exists():
            if _is_valid_existing_image(dest):
                logger.info("[FAL] scene %d already generated, skipping", scene["scene_id"])
                result_paths.append(dest)
                continue
            logger.warning(
                "[FAL] scene %d has an empty/corrupt existing image file, regenerating", scene["scene_id"]
            )

        image_bytes = _generate_image_bytes_for_scene(scene, provider)
        if not image_bytes:
            raise RuntimeError(f"[FAL] scene {scene['scene_id']}: provider returned empty image data")

        _write_image_atomic(dest, image_bytes)
        result_paths.append(dest)

    return result_paths


def run_and_save(paths: ps.ProjectPaths) -> list:
    # The precondition check (missing scenes.json) lives inside the try
    # block below, not before it: every failure here must go through the
    # same save_failure() path as a real generation error, or the video is
    # left stuck with no FAILED state recorded and no diagnostic trail.
    run_id = db.start_run(paths.video_id, "images")
    try:
        scenes = ps.load_json(paths.scenes_json)
        if not scenes:
            raise RuntimeError(f"[FAL] no scenes.json found at {paths.scenes_json} - run scene_planner.py first")

        result = generate_scene_images(paths, scenes)
        ps.save_state(paths, PipelineState.IMAGES_GENERATED, extra={"image_count": len(result)})
        db.finish_run(run_id, "success")
        return result
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "images", str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate scene images from scenes.json (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        result = run_and_save(paths)
        print(f"OK - {len(result)} scene images saved under {paths.root}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
