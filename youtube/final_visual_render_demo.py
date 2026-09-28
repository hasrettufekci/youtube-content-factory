"""
YOUTUBE_CONTENT_FACTORY - SAFE FINAL VISUAL RENDER PASS.

Takes an EXISTING, already-generated production project (its scenes.json
and its already-generated scene_NNN.png files) and produces a NEW,
complete set of final scene images using the CURRENT rendering system -
Scene Planner V2's visual_strategy/overlay_required fields, the
deterministic Pillow overlay renderer (youtube/overlay_renderer.py), and
the programmatic graphic renderer (youtube/programmatic_graphic_renderer.py).

This is a pure LOCAL RENDERING PASS over data that already exists on disk.
It never calls Claude, FAL, or OpenAI, never re-runs research or script
generation, never regenerates TTS/video, and never writes anything under
the source project's own directory (storage/2026/09/<slug>/) - only under
a brand-new output directory this module creates itself.

Per-scene rule (mirrors image_gen.py's own real dispatch precedence in
_generate_image_bytes_for_scene() exactly - programmatic_graphic is
checked FIRST and unconditionally, then overlay_required, then a plain
copy; see _decide_action()):

  A) visual_strategy == "programmatic_graphic":
     render a complete new PNG with programmatic_graphic_renderer -
     the scene's EXISTING scene_NNN.png (if any - it may be an old
     diagnostic placeholder from before that renderer existed) is never
     read or reused; the programmatic renderer builds the whole image
     from scene metadata alone, exactly like image_gen.py does live.

  B) overlay_required == true (and not A):
     take the scene's EXISTING scene_NNN.png as the base image - never
     regenerated - normalize it to exactly 1920x1080 (see below), then run
     it through the deterministic overlay renderer, exactly like
     image_gen.py does live after a real provider.generate() call, except
     the "provider" step is skipped entirely because the base image
     already exists on disk.

  C) neither A nor B:
     normalize the EXISTING scene_NNN.png to exactly 1920x1080 (see
     below) and write it out - a true byte-for-byte copy when the source
     is already 1920x1080 (the common case), or a padded copy when it
     isn't.

NORMALIZATION: the real production project's existing scene images turned
out to be 1920x1072, not 1920x1080 - 8 pixels short of the video frame
size this pass (and the assembler) require. Every base image (for both
the "overlay" and "copy" actions above) is passed through
youtube/image_normalize.py's normalize_image_bytes_to_video_size() BEFORE
anything else happens to it - this is the ONE place that padding logic
lives (never duplicated here, in overlay_renderer.py, or in image_gen.py -
see that module's docstring). It never stretches/distorts the image and
never crops real content: a 1920x1072 source is kept at its native
resolution with 4px of edge-extended padding added top and bottom (8px
total); an already-1920x1080 source passes through completely unchanged
(a true no-op - byte-identical output). Scene 026 (and any other
programmatic_graphic scene) never goes through this at all - the
programmatic renderer always produces a complete 1920x1080 image directly
from scene metadata, with no existing base image involved.

Usage (against the real production project - e.g. on the Windows machine,
once this file is copied over):

    python -m youtube.final_visual_render_demo --slug the-wow-signal-6
    python -m youtube.final_visual_render_demo --project-dir "D:\\...\\storage\\2026\\09\\the-wow-signal-6"

Output goes to debug_audit/final_visual_render_v2/<slug>/ by default -
NEVER to the source project directory. scenes.json, research.json,
script_en.json, narration_en.mp3, and video.mp4 are never read for
anything other than scenes.json (read-only), and none of them are ever
written to.

VALIDATION: render_project() processes every declared scene (it does not
stop at the first problem, so one bad scene doesn't hide problems with the
other 32) but collects every failure - a missing source image, an output
that can't be reopened, an output that isn't 1920x1080, or a renderer
exception - into summary.failures. summary.ok is False if there is even
one failure, and main() then prints every failure and exits non-zero. This
means the command FAILS LOUDLY on any problem (never silently skips a
scene and calls the run a success) while still producing one complete,
readable report of everything that's wrong in a single run, rather than
stopping at the first bad scene out of 33.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from core.config import settings
from core.logging_setup import get_logger
from youtube import image_normalize as inorm
from youtube import overlay_renderer as ov
from youtube import programmatic_graphic_renderer as pg

logger = get_logger("final_visual_render_demo")

EXPECTED_WIDTH = settings.VIDEO_WIDTH
EXPECTED_HEIGHT = settings.VIDEO_HEIGHT


class SceneRenderError(RuntimeError):
    """One scene's failure - carries scene_id and a human-readable reason.
    Raised by render_scene(), caught and collected (never silently
    swallowed) by render_project()."""

    def __init__(self, scene_id: int | None, reason: str):
        self.scene_id = scene_id
        self.reason = reason
        super().__init__(f"scene {scene_id!r}: {reason}")


@dataclass
class RenderSummary:
    total: int = 0
    copied: int = 0
    overlay: int = 0
    programmatic: int = 0
    failures: list[tuple[int, str]] = field(default_factory=list)
    scene_actions: dict[int, str] = field(default_factory=dict)
    scene_renderers: dict[int, str] = field(default_factory=dict)  # scene_id -> renderer name (programmatic only)

    @property
    def ok(self) -> bool:
        return not self.failures


def _load_scenes_json(project_dir: Path) -> dict[int, dict]:
    scenes_path = project_dir / "scenes.json"
    if not scenes_path.exists():
        raise FileNotFoundError(f"no scenes.json found at {scenes_path}")
    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    return {s["scene_id"]: s for s in scenes if "scene_id" in s}


def _find_project_dir(slug: str) -> Path | None:
    """Read-only search under storage/<year>/<month>/<slug> - never writes
    or modifies anything it finds."""
    storage_root = settings.STORAGE_DIR
    if not storage_root.exists():
        return None
    matches = sorted(storage_root.glob(f"*/*/{slug}"))
    return matches[0] if matches else None


def _decide_action(scene: dict) -> str:
    """Purely structural three-way decision - mirrors image_gen.py's own
    real dispatch precedence in _generate_image_bytes_for_scene() exactly:
    "programmatic_graphic" is checked FIRST and unconditionally (a
    programmatic scene never also gets overlaid - it renders its own
    complete graphic), then overlay_required, then a plain copy. Never
    branches on scene_id, topic, or slug - works identically for any
    project."""
    if scene.get("visual_strategy") == "programmatic_graphic":
        return "programmatic"
    if scene.get("overlay_required"):
        return "overlay"
    return "copy"


def _validate_output_dimensions(path: Path, scene_id: int) -> None:
    """Opens `path` and confirms it decodes to a valid image of exactly
    (EXPECTED_WIDTH, EXPECTED_HEIGHT) - raises SceneRenderError otherwise.
    Applied to every scene's output, whichever action produced it -
    including "copy", so a wrongly-sized EXISTING source image fails the
    pass loudly instead of being silently propagated into the final set."""
    try:
        with Image.open(path) as img:
            img.load()
            size = img.size
    except Exception as e:
        raise SceneRenderError(scene_id, f"output PNG at {path} could not be opened: {e}") from e
    if size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise SceneRenderError(
            scene_id, f"output at {path} is {size[0]}x{size[1]}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}",
        )


def _write_bytes_atomic(dest_path: Path, data: bytes) -> None:
    """Atomic write: write to a sibling temp file, then rename into place
    - `Path.replace` is an atomic rename on both POSIX and Windows, so a
    crash/interrupt mid-write can never leave a corrupt file at dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(dest_path)


def render_scene(scene: dict, src_image_path: Path, dest_path: Path) -> tuple[str, str | None]:
    """Renders ONE scene from its EXISTING base image at src_image_path to
    dest_path, per the three-way action in _decide_action() - the base
    image is never regenerated, never re-prompted, and no provider/API of
    any kind is ever called here. Returns (action, renderer_name_or_None)
    - renderer_name is only set for action=="programmatic". Raises
    SceneRenderError on any failure: a missing source image, an
    unreadable/wrong-size output, or an underlying renderer exception -
    never silently skips or leaves a partial/incorrect file at dest_path.

    For "overlay" and "copy", the source image is first passed through
    youtube/image_normalize.py's normalize_image_bytes_to_video_size() -
    the ONE place padding/letterbox logic lives (see that module's
    docstring for why: the real production project's existing images are
    1920x1072, not 1920x1080). "programmatic" never touches the existing
    base image at all - the programmatic renderer always produces a
    complete 1920x1080 image directly from scene metadata."""
    scene_id = scene.get("scene_id")
    action = _decide_action(scene)

    if action != "programmatic" and not src_image_path.exists():
        raise SceneRenderError(scene_id, f"expected source image missing: {src_image_path}")

    renderer_name: str | None = None
    try:
        if action == "programmatic":
            _out_path, renderer_name = pg.render_programmatic_graphic_to_file(
                scene, dest_path, width=EXPECTED_WIDTH, height=EXPECTED_HEIGHT,
            )
        elif action == "overlay":
            src_bytes = src_image_path.read_bytes()
            normalized_bytes = inorm.normalize_image_bytes_to_video_size(
                src_bytes, EXPECTED_WIDTH, EXPECTED_HEIGHT,
            )
            overlaid_bytes = ov.render_scene_overlay_bytes(normalized_bytes, scene)
            _write_bytes_atomic(dest_path, overlaid_bytes)
        else:  # copy
            src_bytes = src_image_path.read_bytes()
            normalized_bytes = inorm.normalize_image_bytes_to_video_size(
                src_bytes, EXPECTED_WIDTH, EXPECTED_HEIGHT,
            )
            _write_bytes_atomic(dest_path, normalized_bytes)
    except SceneRenderError:
        raise
    except inorm.ImageNormalizationError as e:
        raise SceneRenderError(scene_id, f"source image could not be normalized: {e}") from e
    except Exception as e:
        raise SceneRenderError(scene_id, f"{action} renderer raised: {e!r}") from e

    _validate_output_dimensions(dest_path, scene_id)
    return action, renderer_name


def render_project(project_dir: Path, out_dir: Path) -> RenderSummary:
    """Renders every scene declared in project_dir/scenes.json using ONLY
    files already on disk under project_dir (scenes.json + existing
    scene_NNN.png files) - never calls Claude, FAL, OpenAI, or any
    provider (see render_scene/_decide_action, and the module's own "no
    network import" guarantee - this file imports no httpx/requests/
    ImageProvider anywhere). Writes exclusively under out_dir; project_dir
    - the real production folder - is only ever READ, never written to or
    deleted from.

    Processes every scene (never stops at the first failure, so one bad
    scene doesn't hide problems with the other 32 in a single run), but
    the whole pass is considered failed - summary.ok is False - if ANY
    scene failed. The caller (main()) must treat that as a hard failure,
    never a silent partial success - see the module docstring's
    VALIDATION section."""
    scenes_by_id = _load_scenes_json(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = RenderSummary(total=len(scenes_by_id))
    for scene_id in sorted(scenes_by_id):
        scene = scenes_by_id[scene_id]
        src_path = project_dir / f"scene_{scene_id:03d}.png"
        dest_path = out_dir / f"scene_{scene_id:03d}.png"
        try:
            action, renderer_name = render_scene(scene, src_path, dest_path)
        except SceneRenderError as e:
            logger.error("[FINAL_RENDER] scene=%03d action=FAILED reason=%s", scene_id, e.reason)
            summary.failures.append((scene_id, e.reason))
            continue

        summary.scene_actions[scene_id] = action
        if action == "copy":
            summary.copied += 1
            logger.info("[FINAL_RENDER] scene=%03d action=copy", scene_id)
        elif action == "overlay":
            summary.overlay += 1
            logger.info("[FINAL_RENDER] scene=%03d action=overlay", scene_id)
        else:  # programmatic
            summary.programmatic += 1
            summary.scene_renderers[scene_id] = renderer_name
            logger.info("[FINAL_RENDER] scene=%03d action=programmatic renderer=%s", scene_id, renderer_name)

    return summary


def _build_contact_sheet(rendered_paths: list[Path], out_path: Path, columns: int = 6) -> Path:
    thumbs = []
    thumb_w = 320
    for path in rendered_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * ratio)))
        thumbs.append((path.stem, thumb))

    if not thumbs:
        raise ValueError("no rendered scenes to build a contact sheet from")

    thumb_h = thumbs[0][1].height
    label_h = 24
    rows = (len(thumbs) + columns - 1) // columns
    pad = 10
    sheet_w = columns * thumb_w + (columns + 1) * pad
    sheet_h = rows * (thumb_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 22))
    draw = ImageDraw.Draw(sheet)
    label_font = ov.resolve_font(16, bold=True)

    for i, (name, thumb) in enumerate(thumbs):
        col, row = i % columns, i // columns
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb_h + 3), name, font=label_font, fill=(210, 210, 210))

    sheet.save(out_path, format="PNG")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SAFE FINAL VISUAL RENDER PASS: re-render every scene of an existing project from its "
                     "already-generated scene images and scenes.json, using the current rendering system "
                     "(copy / overlay / programmatic per scene). Read-only against the source project - "
                     "never calls Claude/FAL/OpenAI, never regenerates prompts/TTS/research/video.",
    )
    parser.add_argument("--slug", help="Project slug under storage/<year>/<month>/<slug> (e.g. the-wow-signal-6)")
    parser.add_argument("--project-dir", help="Explicit project directory (overrides --slug)")
    parser.add_argument("--out-dir", default=None,
                         help="Output directory (default: debug_audit/final_visual_render_v2/<slug>) - "
                              "never the source project directory")
    args = parser.parse_args(argv)

    if args.project_dir:
        project_dir = Path(args.project_dir)
    elif args.slug:
        project_dir = _find_project_dir(args.slug)
        if project_dir is None:
            print(f"FAILED: no project found for slug={args.slug!r} under {settings.STORAGE_DIR}", file=sys.stderr)
            return 1
    else:
        print("FAILED: one of --slug or --project-dir is required", file=sys.stderr)
        return 1

    if not (project_dir / "scenes.json").exists():
        print(f"FAILED: no scenes.json found at {project_dir}", file=sys.stderr)
        return 1

    slug_name = args.slug or project_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("debug_audit") / "final_visual_render_v2" / slug_name

    logger.info("[FINAL_RENDER] project=%s (read-only) -> %s", project_dir, out_dir)
    summary = render_project(project_dir, out_dir)

    rendered_paths = [out_dir / f"scene_{sid:03d}.png" for sid in sorted(summary.scene_actions)]
    contact_sheet_path = None
    if rendered_paths:
        contact_sheet_path = _build_contact_sheet(rendered_paths, out_dir / "contact_sheet.png")

    print(
        f"total={summary.total} copied={summary.copied} overlay={summary.overlay} "
        f"programmatic={summary.programmatic} failures={len(summary.failures)}"
    )
    if contact_sheet_path:
        print(f"Contact sheet: {contact_sheet_path}")

    if not summary.ok:
        print("FAILED - one or more scenes could not be rendered:", file=sys.stderr)
        for scene_id, reason in summary.failures:
            print(f"  scene {scene_id:03d}: {reason}", file=sys.stderr)
        return 1

    print(f"OK - rendered {summary.total} scene(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
