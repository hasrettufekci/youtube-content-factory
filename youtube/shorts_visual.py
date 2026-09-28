"""
Shorts Engine V1 - visual strategy: which of the Long's existing scene
images to reuse for a given Short, and how to turn a 16:9 image into a
9:16 (1080x1920 by default - settings.SHORTS_WIDTH/HEIGHT) one.

REUSE, NOT REGENERATION (spec, non-negotiable): reuse the Long's existing
scene images whenever possible; only generate a NEW vertical FAL image
when a scene's own composition genuinely cannot be safely center-cropped
to 9:16 without losing meaningful content. Never blindly stretch (no
distortion).

SCENE SELECTION: a Short's narration is a condensation of exactly ONE
Long script section (see youtube/shorts_planner.py). scenes.json has no
explicit field linking a scene back to its originating section (scene
narration is a plain word-count-based chunk of the Long's documentary
narration - youtube/scene_planner.py), so this module maps a section's
character span within the documentary narration onto the scene(s) whose
own narration chunks overlap that same span (a deterministic, proportional
approximation - never an LLM call, never a second research/classification
pass).

CROP SUITABILITY HEURISTIC: built from scene_planner.py's REAL VISUAL_TYPES
enum (["cinematic", "wide", "closeup", "archival", "map", "portrait",
"abstract"]), not a fabricated one:
    - {"map", "wide"}: BAD - these compositions are deliberately meant to
      fill a wide 16:9 frame edge to edge; a narrow center crop would very
      likely cut off the meaningful content (a map's labeled region, a
      wide establishing shot's subject) -> generate a vertical replacement.
    - {"cinematic", "closeup", "archival", "portrait", "abstract"}: GOOD -
      single-subject/centered compositions crop safely -> vertical crop.
"""
from __future__ import annotations

from pathlib import Path

from core.config import settings
from core.logging_setup import get_logger
from storage import project_store as ps
from youtube import image_gen

logger = get_logger("shorts_visual")

# scene_planner.py's real VISUAL_TYPES split into "safe to center-crop to
# 9:16" vs "needs a dedicated vertical regeneration" - see module docstring.
CROP_SAFE_VISUAL_TYPES = {"cinematic", "closeup", "archival", "portrait", "abstract"}
CROP_UNSAFE_VISUAL_TYPES = {"map", "wide"}


def _non_cta_sections_in_order(script: dict) -> list[dict]:
    return [s for s in script.get("sections", []) if s.get("type") != "cta"]


def _documentary_narration(script: dict) -> str:
    """The exact text scene_planner.py's plan_scenes_v2() chunks into
    scenes (everything except the "cta" section, which becomes its own
    separate, dedicated final scene - see scene_planner.py's
    _split_documentary_and_cta()/_build_cta_scene())."""
    return "\n\n".join(s["narration"].strip() for s in _non_cta_sections_in_order(script))


def _section_word_span(script: dict, section_type: str) -> tuple[int, int]:
    """(start_word_index, end_word_index) of `section_type`'s own narration
    within _documentary_narration(script)'s word sequence - computed
    directly from section order/word counts (never a substring search,
    which could misfire on repeated words) since documentary narration is
    exactly "\\n\\n".join(section narrations in order)."""
    words_before = 0
    for s in _non_cta_sections_in_order(script):
        section_words = len(s["narration"].split())
        if s.get("type") == section_type:
            return words_before, words_before + section_words
        words_before += section_words
    raise ValueError(f"[SHORTS_VISUAL] section {section_type!r} not found in script")


def select_scenes_for_section(scenes: list[dict], script: dict, section_type: str) -> list[dict]:
    """Non-CTA scenes (schema_version==2 scenes never carry is_cta=True
    except the dedicated CTA scene - legacy v1 scenes have no is_cta field
    at all, so `not s.get("is_cta")` correctly keeps them too) whose own
    word-range within the documentary narration overlaps `section_type`'s
    word-range. Falls back to the single middle scene if, for any reason
    (an unusually shaped script), nothing overlaps - this must never raise
    or return an empty list, since a Short's visual production has to
    proceed with SOME hero image."""
    candidate_scenes = [s for s in scenes if not s.get("is_cta")]
    if not candidate_scenes:
        candidate_scenes = list(scenes)
    if not candidate_scenes:
        return []

    start, end = _section_word_span(script, section_type)

    selected = []
    cursor = 0
    for scene in candidate_scenes:
        narration = scene.get("narration_en") or scene.get("narration") or ""
        scene_word_count = max(len(narration.split()), 1)
        scene_start, scene_end = cursor, cursor + scene_word_count
        if scene_start < end and scene_end > start:  # overlap test
            selected.append(scene)
        cursor = scene_end

    if not selected:
        selected = [candidate_scenes[len(candidate_scenes) // 2]]
    return selected


def classify_crop_suitability(scene: dict) -> str:
    """"good" (safe to center-crop) or "bad" (needs vertical regeneration) -
    see module docstring. Unknown/missing visual_type defaults to "good"
    (the majority-safe case) rather than triggering an unnecessary FAL
    regeneration."""
    visual_type = scene.get("visual_type", "cinematic")
    return "bad" if visual_type in CROP_UNSAFE_VISUAL_TYPES else "good"


def crop_to_vertical(source_path: Path, dest_path: Path, width: int | None = None, height: int | None = None) -> Path:
    """Deterministic, no-distortion vertical crop: scale the source image
    so its height exactly matches the target height (preserving aspect
    ratio - never a non-uniform stretch), then center-crop the resulting
    (now-wider-than-target) width down to the target width. Mirrors
    youtube/image_normalize.py's own determinism/no-distortion conventions,
    using crop instead of pad (the correct transform for "narrow a 16:9
    frame into 9:16" rather than "fit a slightly-short frame").

    Source images in this codebase are always 16:9 (settings.VIDEO_WIDTH x
    settings.VIDEO_HEIGHT), so scaling height up to a 9:16 target always
    leaves the width wider than the target (croppable) - a same-edge-
    extension pad fallback is included regardless, purely as a safety net
    against a future non-16:9 source, never exercised by any current
    caller."""
    from PIL import Image

    width = width or settings.SHORTS_WIDTH
    height = height or settings.SHORTS_HEIGHT

    with Image.open(source_path) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size
        scale = height / src_h
        new_w = max(round(src_w * scale), 1)
        resized = img.resize((new_w, height), Image.LANCZOS)

        if new_w >= width:
            left = (new_w - width) // 2
            result = resized.crop((left, 0, left + width, height))
        else:
            # Safety-net pad path (not expected for 16:9 sources - see
            # docstring): center the scaled image and extend its own edge
            # columns to fill the remaining width, never a flat synthetic bar.
            result = Image.new("RGB", (width, height))
            left_pad = (width - new_w) // 2
            result.paste(resized, (left_pad, 0))
            if left_pad > 0:
                left_col = resized.crop((0, 0, 1, height)).resize((left_pad, height))
                result.paste(left_col, (0, 0))
            right_pad = width - new_w - left_pad
            if right_pad > 0:
                right_col = resized.crop((new_w - 1, 0, new_w, height)).resize((right_pad, height))
                result.paste(right_col, (left_pad + new_w, 0))

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(dest_path, format="PNG")
    return dest_path


def build_vertical_visual(paths: ps.ProjectPaths, scene: dict, dest_path: Path,
                           width: int | None = None, height: int | None = None) -> Path:
    """One scene -> one vertical (9:16) image at `dest_path`, reusing the
    Long's own existing scene image whenever the crop-suitability heuristic
    says it's safe, generating a dedicated vertical image via the SAME
    image_gen.ImageProvider abstraction (already width/height-parameterized
    - zero changes needed to image_gen.py) only when it isn't."""
    width = width or settings.SHORTS_WIDTH
    height = height or settings.SHORTS_HEIGHT
    source_path = paths.scene_image(scene["scene_id"])

    if classify_crop_suitability(scene) == "good" and source_path.exists():
        logger.info("[SHORTS_VISUAL] scene_id=%d visual_type=%s -> vertical crop (reused)",
                    scene.get("scene_id"), scene.get("visual_type"))
        return crop_to_vertical(source_path, dest_path, width=width, height=height)

    logger.info("[SHORTS_VISUAL] scene_id=%d visual_type=%s -> generating dedicated vertical image (FAL)",
                scene.get("scene_id"), scene.get("visual_type"))
    prompt = scene.get("visual_prompt") or scene.get("narration_en") or scene.get("narration") or ""
    provider = image_gen.get_provider()
    image_bytes = provider.generate(prompt, width, height)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(image_bytes)
    return dest_path


def build_visuals_for_short(paths: ps.ProjectPaths, scenes: list[dict], script: dict,
                             section_type: str, short_number: int,
                             width: int | None = None, height: int | None = None) -> list[Path]:
    """Selects the Long scene(s) for this Short's section, builds one
    vertical image per selected scene (short_NN_visual_0.png, _1.png, ...
    under paths.short_dir()), and returns their paths in order - ready to
    pass straight into assemble.assemble_video(scene_image_paths=...)."""
    selected = select_scenes_for_section(scenes, script, section_type)
    out_paths = []
    for i, scene in enumerate(selected):
        dest = paths.short_dir() / f"short_{short_number:02d}_visual_{i:02d}.png"
        build_vertical_visual(paths, scene, dest, width=width, height=height)
        out_paths.append(dest)
    return out_paths
