"""
Thumbnail generation (spec section 21).

1280x720. A strong single hero image (reuses the already-generated
scene_001.png by default - free, and it's the image tied to the video's
opening hook - or can generate a dedicated one via the image provider),
darkened/contrast-boosted, with a short (2-5 word) high-contrast text
overlay generated from the script's title/hook.

Standalone test:
    python -m youtube.thumbnail --video-id 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import settings
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import llm_client
from youtube.localizer import LANGUAGE_NAMES

logger = get_logger("thumbnail")

WIDTH = settings.THUMBNAIL_WIDTH
HEIGHT = settings.THUMBNAIL_HEIGHT

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

SYSTEM_PROMPT = """You write short YouTube thumbnail text overlays for a factual documentary
channel. Given a video title and hook, return ONLY a JSON object:
{"text": "2 TO 5 WORDS, punchy, curiosity-driving, ALL CAPS, no punctuation at the end"}
The text must be something the video can actually back up - no exaggeration beyond what the
title/hook already say."""

# The thumbnail image itself is shared/language-neutral (spec Round 3
# section 17) - only the short overlay text is ever language-specific, and
# only if a future caller asks for a non-English one (the pipeline itself
# always calls this with the default "en" today: one shared thumbnail, not
# a duplicate per language). Reuses youtube.localizer's name table so
# there's a single place mapping a language code to a prompt-ready name.


def _system_prompt(language: str) -> str:
    if language == "en":
        return SYSTEM_PROMPT
    name = LANGUAGE_NAMES.get(language, language)
    return SYSTEM_PROMPT + f"\n\nWrite the text in {name} ({language}), not English."


def generate_thumbnail_text(title: str, hook: str, language: str = "en") -> str:
    try:
        data, _ = llm_client.chat_json(_system_prompt(language), f"Title: {title}\nHook: {hook}", max_tokens=200)
        text = str(data.get("text", "")).strip()
        if text:
            return text[:40]
    except Exception as e:
        logger.warning("[THUMBNAIL] LLM text generation failed (%s), falling back to title", e)
    words = title.upper().split()
    return " ".join(words[:5])


def _load_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def compose_thumbnail(hero_image_path: Path, text: str, dest: Path,
                       width: int | None = None, height: int | None = None) -> Path:
    """`width`/`height` (Shorts Engine V1, additive): override the module's
    WIDTH/HEIGHT (1280x720) target - e.g. width=1080/height=1920 for a
    Short's vertical thumbnail - reusing this exact same cover-crop/
    darken/contrast/scrim/font-fit/outlined-text treatment for both shapes.
    None (every pre-existing caller) is 100% unchanged behavior."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    width = width or WIDTH
    height = height or HEIGHT

    img = Image.open(hero_image_path).convert("RGB")

    # Cover-crop to the target aspect ratio
    target_ratio = width / height
    w, h = img.size
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((width, height), Image.LANCZOS)

    # High-contrast look: darken + boost contrast/saturation so text pops
    img = ImageEnhance.Brightness(img).enhance(0.75)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Color(img).enhance(1.15)

    # Bottom gradient scrim so text stays legible over any background
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scrim_top = int(height * 0.55)
    for y in range(scrim_top, height):
        alpha = int(180 * (y - scrim_top) / (height - scrim_top))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    font_size = 92
    font = _load_font(font_size)
    while font.getbbox(text)[2] > width * 0.92 and font_size > 36:
        font_size -= 4
        font = _load_font(font_size)

    bbox = font.getbbox(text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - text_w) / 2
    y = height - text_h - 60

    # Outline for legibility against any image
    outline_range = 4
    for dx in range(-outline_range, outline_range + 1, 2):
        for dy in range(-outline_range, outline_range + 1, 2):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 214, 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG")
    return dest


def run_and_save(paths: ps.ProjectPaths, hero_scene_id: int = 1, language: str = "en") -> Path:
    # Precondition checks (missing script.json / no scene images) are
    # inside the try block, not before it: every failure here - including
    # these - must go through the same save_failure() path as everything
    # else, or the video is left stuck in whatever state it was already in
    # (never recorded FAILED) with an error a resume would hit again and
    # again with no diagnostic trail in state.json.
    run_id = db.start_run(paths.video_id, "thumbnail")
    try:
        script = ps.load_json(paths.script_json)
        if not script:
            raise RuntimeError(f"[THUMBNAIL] no script.json found at {paths.script_json}")

        hero = paths.scene_image(hero_scene_id)
        if not hero.exists():
            scenes = paths.scene_images()
            if not scenes:
                raise RuntimeError(f"[THUMBNAIL] no scene images found under {paths.root} - run image_gen.py first")
            hero = scenes[0]

        text = generate_thumbnail_text(script["title"], script.get("hook", ""), language=language)
        compose_thumbnail(hero, text, paths.thumbnail_png)
        ps.save_state(paths, PipelineState.THUMBNAIL_READY, extra={"thumbnail_text": text})
        db.finish_run(run_id, "success")
        return paths.thumbnail_png
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "thumbnail", str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the video thumbnail (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        result = run_and_save(paths)
        print(f"OK - thumbnail saved to {result}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
