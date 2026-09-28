"""
Scene Planner V2 - Phase 4 programmatic graphic renderer: local visual
audit/demo tool (mirrors youtube/overlay_demo.py's structure/contract for
the Phase 3 overlay renderer).

Standalone, read-only, offline. Does NOT call Claude, does NOT call FAL,
does NOT touch storage/ - it only ever READS an existing project's
scenes.json and writes rendered results to a brand-new directory under
debug_audit/programmatic_graphics_v2/<slug>/, so real production artifacts
are never at risk.

Usage (against the real production project - e.g. on the Windows machine,
once this file is copied over):

    python -m youtube.programmatic_graphic_demo --slug the-wow-signal-6
    python -m youtube.programmatic_graphic_demo --project-dir "D:\\...\\storage\\2026\\09\\the-wow-signal-6"

Usage without the real project directory reachable (this sandbox has no
copy of storage/2026/09/the-wow-signal-6 on disk) - renders scene 026
from its REAL production metadata, exactly as provided by the user from
their actual scenes.json (see _KNOWN_SCENE_026 below - this is the real
payload, not a synthetic stand-in):

    python -m youtube.programmatic_graphic_demo --offline-fixture

On the Windows machine, --slug/--project-dir always take priority and
read the live scenes.json directly - --offline-fixture is only ever the
fallback used here because this sandbox cannot reach that file, and it
renders the same real scene 026 object either way.

Also builds one contact-sheet PNG combining every rendered scene into a
single grid image when more than one programmatic_graphic scene is found,
for a fast visual review of the whole set at once - identical convention
to overlay_demo.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from core.config import settings
from core.logging_setup import get_logger
from youtube import overlay_renderer, programmatic_graphic_renderer as pg

logger = get_logger("programmatic_graphic_demo")

DEFAULT_SCENE_IDS = [26]

# Scene 026's REAL production metadata, exactly as provided by the user
# from storage/2026/09/the-wow-signal-6/scenes.json - used as the offline
# fallback only because this sandbox cannot read that file directly. NOT
# invented/synthetic - identical to the object used in
# tests/test_programmatic_graphic_renderer.py's _REAL_SCENE_026.
_KNOWN_SCENE_026 = {
    "scene_id": 26,
    "scene_type": "scientific_graphic",
    "visual_strategy": "programmatic_graphic",
    "exact_text": [],
    "data_points": [
        {
            "data_type": "curve",
            "value": "celestial beam-crossing intensity profile",
            "exact_render": False,
        },
    ],
    "text_render": {
        "items": [],
        "hypothesis_badge": False,
    },
    "visual_relevance_reason": (
        "Plots the combined characteristics—strength, frequency specificity, bandwidth, "
        "and celestial intensity profile—as overlaid evidence showing all anomalies present simultaneously."
    ),
    "visual_prompt": None,
    "narration": None,
}


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


def _programmatic_scene_ids(scenes_by_id: dict[int, dict]) -> list[int]:
    """Every scene actually classified visual_strategy=="programmatic_graphic"
    in this project, in scene_id order - never hardcoded to scene 026, so
    this demo naturally covers whatever a real project actually contains."""
    return sorted(sid for sid, scene in scenes_by_id.items() if scene.get("visual_strategy") == "programmatic_graphic")


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
        description="Render the programmatic graphic system against an existing project's "
                     "programmatic_graphic scenes for visual audit (offline, read-only - "
                     "never touches storage/, never calls FAL/Claude, never spends API credits).",
    )
    parser.add_argument("--slug", help="Project slug under storage/<year>/<month>/<slug> (e.g. the-wow-signal-6)")
    parser.add_argument("--project-dir", help="Explicit project directory (overrides --slug)")
    parser.add_argument("--offline-fixture", action="store_true",
                         help="Render scene 026 from its known REAL metadata (see module docstring's "
                              "_KNOWN_SCENE_026 - provided directly by the user from the real scenes.json, not "
                              "invented) instead of reading a live project (used automatically as a fallback if "
                              "neither --slug nor --project-dir resolves to a real project with a "
                              "programmatic_graphic scene).")
    parser.add_argument("--scenes", type=int, nargs="+", default=None,
                         help="Explicit scene ids to render (default: every visual_strategy=='programmatic_graphic' "
                              "scene actually found in the project; falls back to scene 26 for --offline-fixture)")
    parser.add_argument("--out-dir", default=str(Path("debug_audit") / "programmatic_graphics_v2"),
                         help="Output directory (default: debug_audit/programmatic_graphics_v2) - never storage/")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_dir: Path | None = None
    if args.project_dir:
        project_dir = Path(args.project_dir)
    elif args.slug:
        project_dir = _find_project_dir(args.slug)
        if project_dir is None:
            logger.warning("[PROGRAMMATIC_DEMO] no project found for slug=%r under %s", args.slug, settings.STORAGE_DIR)

    scenes_by_id: dict[int, dict]
    using_offline_fixture = False
    if project_dir is not None and (project_dir / "scenes.json").exists():
        logger.info("[PROGRAMMATIC_DEMO] using real project at %s (read-only)", project_dir)
        scenes_by_id = _load_scenes_json(project_dir)
        scene_ids = args.scenes if args.scenes else _programmatic_scene_ids(scenes_by_id)
        if not scene_ids:
            logger.warning(
                "[PROGRAMMATIC_DEMO] real project at %s has no visual_strategy=='programmatic_graphic' scenes - "
                "falling back to the known scene 026 metadata", project_dir,
            )
            using_offline_fixture = True
    else:
        if not args.offline_fixture and (args.slug or args.project_dir):
            logger.warning(
                "[PROGRAMMATIC_DEMO] live project not reachable here - falling back to scene 026's known "
                "real metadata (see module docstring)",
            )
        using_offline_fixture = True

    if using_offline_fixture:
        logger.info(
            "[PROGRAMMATIC_DEMO] rendering scene 026 from its known REAL metadata (provided directly by the "
            "user from the actual scenes.json - see module docstring's _KNOWN_SCENE_026; the live project file "
            "was not reachable from this environment)",
        )
        scenes_by_id = {26: _KNOWN_SCENE_026}
        scene_ids = args.scenes if args.scenes else [26]

    rendered_paths: list[Path] = []
    for scene_id in scene_ids:
        scene = scenes_by_id.get(scene_id)
        if scene is None:
            logger.warning("[PROGRAMMATIC_DEMO] scene %03d not available in this project - skipping", scene_id)
            continue
        if scene.get("visual_strategy") != "programmatic_graphic":
            logger.warning(
                "[PROGRAMMATIC_DEMO] scene %03d has visual_strategy=%r, not 'programmatic_graphic' - "
                "skipping (this demo is for programmatic_graphic scenes only)",
                scene_id, scene.get("visual_strategy"),
            )
            continue

        dest = out_dir / f"scene_{scene_id:03d}.png"
        _out_path, renderer_name = pg.render_programmatic_graphic_to_file(scene, dest)
        logger.info(
            "[IMAGE][PROGRAMMATIC] scene=%03d renderer=%s -> %s", scene_id, renderer_name, dest,
        )
        print(f"scene {scene_id:03d}: renderer={renderer_name} -> {dest}")
        rendered_paths.append(dest)

    if not rendered_paths:
        print("FAILED: no programmatic_graphic scenes were rendered - nothing available at the given "
              "slug/project-dir/scene ids", file=sys.stderr)
        return 1

    print(f"OK - rendered {len(rendered_paths)} scene(s) under {out_dir}")

    if len(rendered_paths) > 1:
        contact_sheet_path = _build_contact_sheet(rendered_paths, out_dir / "contact_sheet.png")
        print(f"Contact sheet: {contact_sheet_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
