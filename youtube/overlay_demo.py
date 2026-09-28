"""
Scene Planner V2 - Phase 3 overlay renderer: local visual audit/demo tool.

Standalone, read-only, offline. Does NOT call Claude, does NOT call FAL,
does NOT touch storage/ - it only ever READS an existing project's
scene_NNN.png + scenes.json and writes rendered results to a brand-new
directory under debug_audit/overlay_v2/<slug>/, so the real production
artifacts (e.g. storage/2026/09/the-wow-signal-6/scene_001.png) are never
at risk of being overwritten.

Usage (against a real project already on disk - e.g. on the Windows
production machine, once this file is copied over):

    python -m youtube.overlay_demo --slug the-wow-signal-6
    python -m youtube.overlay_demo --project-dir "D:\\...\\storage\\2026\\09\\the-wow-signal-6"

Usage without a real project available (e.g. in this sandbox, which has no
FAL key and has never generated the real Wow! Signal V2 production run) -
builds a synthetic-but-realistic fixture using the EXACT exact_text/
data_points values documented for the real production scenes 001, 005,
006, 008, 017, 019, 024, 032, so the renderer's actual output can still be
visually inspected end to end without spending any API credits:

    python -m youtube.overlay_demo --synthetic

By default, renders scenes 001, 005, 006, 008, 017, 019, 024, 032 (the
representative set named in the overlay renderer task) - pass --scenes to
override, e.g. --scenes 1 2 3.

Also builds one contact-sheet PNG combining every rendered scene into a
single grid image, for a fast visual review of the whole set at once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from core.config import settings
from core.logging_setup import get_logger
from youtube import overlay_renderer

logger = get_logger("overlay_demo")

DEFAULT_SCENE_IDS = [1, 5, 6, 8, 17, 19, 24, 32]

# The EXACT exact_text/data_points values documented for the real
# production run (video_id=7, slug the-wow-signal-6, 33 scenes) - used only
# to build a synthetic fixture when no real project is available locally,
# so this demo still exercises the real renderer against the real factual
# strings it must preserve character-for-character. Every value below is
# copied verbatim from that documentation, never altered/reworded.
_REAL_PRODUCTION_EXACT_TEXT = {
    1: ["Wow", "August 15, 1977", "72 seconds"],
    5: ["6EQUJ5", "Wow!"],
    6: ["1420 megahertz"],
    8: ["30 times background noise", "constellation Sagittarius"],
    17: ["August 15, 1977", "266P/Christensen"],
    19: ["February-May 2020", "Arecibo Observatory, Puerto Rico"],
    24: ["72 seconds", "1977"],
    32: ["72 seconds", "Wow"],
}

# Which of each scene's values are treated as a "measurement"/"coordinate"
# style data point (-> data_label render style) vs. a bare exact_text
# quote/name/date (-> headline/caption style) - a reasonable, generic
# classification by shape (contains digits + a unit-like word), not a
# per-scene/topic-specific rule.
_MEASUREMENT_HINT_WORDS = ("megahertz", "hertz", "seconds", "times")


def _infer_data_type(value: str) -> str:
    lowered = value.lower()
    if any(word in lowered for word in _MEASUREMENT_HINT_WORDS):
        return "measurement"
    return "exact_text"


def _build_synthetic_scene(scene_id: int, values: list[str]) -> dict:
    data_points = [{"data_type": _infer_data_type(v), "value": v, "exact_render": True} for v in values]
    return {
        "scene_id": scene_id,
        "schema_version": 2,
        "visual_strategy": "ai_generated_plus_overlay",
        "overlay_required": True,
        "exact_text": values,
        "data_points": data_points,
        "text_render": {"items": values, "hypothesis_badge": scene_id == 8},  # scene 8 doubles as a hypothesis-badge demo
        "claim": f"Synthetic audit scene {scene_id} (see _REAL_PRODUCTION_EXACT_TEXT)",
    }


def _synthetic_base_image_bytes(scene_id: int, width: int, height: int) -> bytes:
    """A deterministic, documentary-toned gradient + soft focal shape -
    stands in for a real FAL-generated frame so the overlay's placement
    relative to a "subject" can still be judged visually."""
    import io

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    seed = (scene_id * 37) % 255
    for y in range(height):
        t = y / height
        r = int(8 + 26 * t + (seed % 20))
        g = int(12 + 22 * t)
        b = int(22 + 44 * t + (seed % 15))
        draw.line([(0, y), (width, y)], fill=(min(r, 255), min(g, 255), min(b, 255)))
    cx = width // 2 + ((seed % 7) - 3) * 40
    cy = int(height * 0.42)
    radius = int(height * 0.19)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(58, 68, 88))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_synthetic_fixture(out_dir: Path, scene_ids: list[int]) -> tuple[Path, dict]:
    """Writes a synthetic project (scene_NNN.png + scenes.json) under
    out_dir/_synthetic_source/ - a READ-only-afterward staging area this
    script itself creates and owns (never storage/) - and returns
    (project_dir, {scene_id: scene}). Only ever includes the scene ids the
    real production documentation actually gives values for."""
    project_dir = out_dir / "_synthetic_source"
    project_dir.mkdir(parents=True, exist_ok=True)

    scenes_by_id = {}
    for scene_id in scene_ids:
        values = _REAL_PRODUCTION_EXACT_TEXT.get(scene_id)
        if not values:
            continue
        scene = _build_synthetic_scene(scene_id, values)
        scenes_by_id[scene_id] = scene
        image_bytes = _synthetic_base_image_bytes(scene_id, settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT)
        (project_dir / f"scene_{scene_id:03d}.png").write_bytes(image_bytes)

    (project_dir / "scenes.json").write_text(
        json.dumps(list(scenes_by_id.values()), indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return project_dir, scenes_by_id


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


def _build_contact_sheet(rendered_paths: list[Path], out_path: Path, columns: int = 4) -> Path:
    thumbs = []
    thumb_w = 480
    for path in rendered_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * ratio)))
        thumbs.append((path.stem, thumb))

    if not thumbs:
        raise ValueError("no rendered scenes to build a contact sheet from")

    thumb_h = thumbs[0][1].height
    label_h = 28
    rows = (len(thumbs) + columns - 1) // columns
    pad = 12
    sheet_w = columns * thumb_w + (columns + 1) * pad
    sheet_h = rows * (thumb_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 22))
    draw = ImageDraw.Draw(sheet)
    label_font = overlay_renderer.resolve_font(18, bold=True)

    for i, (name, thumb) in enumerate(thumbs):
        col, row = i % columns, i // columns
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb_h + 4), name, font=label_font, fill=(210, 210, 210))

    sheet.save(out_path, format="PNG")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the overlay system against an existing project's scenes for visual audit "
                     "(offline, read-only - never touches storage/, never calls FAL/Claude).",
    )
    parser.add_argument("--slug", help="Project slug under storage/<year>/<month>/<slug> (e.g. the-wow-signal-6)")
    parser.add_argument("--project-dir", help="Explicit project directory (overrides --slug)")
    parser.add_argument("--synthetic", action="store_true",
                         help="Build a synthetic fixture from the documented real production exact_text "
                              "values instead of reading a real project (used automatically as a fallback "
                              "if neither --slug nor --project-dir resolves to a real project).")
    parser.add_argument("--scenes", type=int, nargs="+", default=DEFAULT_SCENE_IDS,
                         help=f"Scene ids to render (default: {DEFAULT_SCENE_IDS})")
    parser.add_argument("--out-dir", default=str(Path("debug_audit") / "overlay_v2"),
                         help="Output directory (default: debug_audit/overlay_v2) - never storage/")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_dir: Path | None = None
    if args.project_dir:
        project_dir = Path(args.project_dir)
    elif args.slug:
        project_dir = _find_project_dir(args.slug)
        if project_dir is None:
            logger.warning("[OVERLAY_DEMO] no project found for slug=%r under %s", args.slug, settings.STORAGE_DIR)

    scenes_by_id: dict[int, dict]
    if project_dir is not None and (project_dir / "scenes.json").exists():
        logger.info("[OVERLAY_DEMO] using real project at %s (read-only)", project_dir)
        scenes_by_id = _load_scenes_json(project_dir)
    else:
        if not args.synthetic and (args.slug or args.project_dir):
            logger.warning("[OVERLAY_DEMO] falling back to a synthetic fixture (no real project found)")
        logger.info(
            "[OVERLAY_DEMO] building a synthetic fixture from the documented real production exact_text "
            "values for scenes %s", sorted(_REAL_PRODUCTION_EXACT_TEXT),
        )
        project_dir, scenes_by_id = _build_synthetic_fixture(out_dir, args.scenes)

    rendered_paths: list[Path] = []
    for scene_id in args.scenes:
        scene = scenes_by_id.get(scene_id)
        src_image = project_dir / f"scene_{scene_id:03d}.png"
        if scene is None or not src_image.exists():
            logger.warning("[OVERLAY_DEMO] scene %03d not available (scene=%s, image exists=%s) - skipping",
                            scene_id, scene is not None, src_image.exists())
            continue

        dest = out_dir / f"scene_{scene_id:03d}.png"
        overlay_renderer.render_scene_overlay(src_image, scene, dest)
        logger.info(
            "[IMAGE][OVERLAY] scene=%03d rendered exact_text=%d data_points=%d -> %s",
            scene_id, len(scene.get("exact_text") or []), len(scene.get("data_points") or []), dest,
        )
        rendered_paths.append(dest)

    if not rendered_paths:
        print("FAILED: no scenes were rendered - nothing available at the given slug/project-dir/scene ids",
              file=sys.stderr)
        return 1

    contact_sheet_path = _build_contact_sheet(rendered_paths, out_dir / "contact_sheet.png")

    print(f"OK - rendered {len(rendered_paths)} scene(s) under {out_dir}")
    print(f"Contact sheet: {contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
