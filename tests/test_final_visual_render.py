"""
YOUTUBE_CONTENT_FACTORY - SAFE FINAL VISUAL RENDER PASS
(youtube/final_visual_render_demo.py).

Deliberately never calls Claude, FAL, or OpenAI - every fixture project
here is hand-built directly on disk (scenes.json + real Pillow-generated
scene_NNN.png base images), and render_project()/render_scene() are
exercised directly against these fixture directories (no --slug/storage
lookup needed for most tests - see test_main_cli_uses_slug_lookup for the
one test that does exercise that path via the isolated_storage fixture).

Covers exactly the 12 categories requested:
    1  output directory creation
    2  33 scenes discovered
    3  copy behavior
    4  overlay behavior
    5  programmatic behavior
    6  scene 026 routes to conceptual_curve
    7  no provider/API calls
    8  all outputs are 1920x1080
    9  missing input scene fails clearly
   10  deterministic output
   11  storage is never modified
   12  full existing suite remains green (run separately, not a unit test
       here - see the task's delivery report)
"""
from __future__ import annotations

import ast
import io
import json

import pytest
from PIL import Image

from youtube import final_visual_render_demo as fvr
from youtube import overlay_renderer as ov
from youtube import programmatic_graphic_renderer as pg

WIDTH, HEIGHT = 1920, 1080

# Scene 026's REAL production metadata (identical to the object already
# verified in tests/test_programmatic_graphic_renderer.py's _REAL_SCENE_026).
_REAL_SCENE_026 = {
    "scene_id": 26,
    "scene_type": "scientific_graphic",
    "visual_strategy": "programmatic_graphic",
    "exact_text": [],
    "data_points": [
        {"data_type": "curve", "value": "celestial beam-crossing intensity profile", "exact_render": False},
    ],
    "text_render": {"items": [], "hypothesis_badge": False},
    "visual_relevance_reason": (
        "Plots the combined characteristics—strength, frequency specificity, bandwidth, "
        "and celestial intensity profile—as overlaid evidence showing all anomalies present simultaneously."
    ),
    "visual_prompt": None,
    "narration": None,
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_base_image_bytes(seed: int, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    color = (10 + seed % 200, 20, 30)
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _copy_scene(scene_id: int) -> dict:
    return {"scene_id": scene_id, "visual_strategy": "ai_generated", "overlay_required": False}


def _overlay_scene(scene_id: int) -> dict:
    return {
        "scene_id": scene_id,
        "visual_strategy": "ai_generated_plus_overlay",
        "overlay_required": True,
        "exact_text": ["6EQUJ5", "Wow!"],
        "data_points": [{"data_type": "exact_text", "value": "6EQUJ5", "exact_render": True}],
        "text_render": {"items": ["6EQUJ5", "Wow!"], "hypothesis_badge": False},
    }


def _build_fixture_project(tmp_path, scenes: list[dict], skip_image_for: set[int] | None = None):
    """Builds a project directory on disk (scenes.json + scene_NNN.png
    base images for every scene, unless its scene_id is in
    skip_image_for) and returns the project directory Path. Never touches
    settings.STORAGE_DIR - these are plain, isolated tmp_path fixtures."""
    skip_image_for = skip_image_for or set()
    project_dir = tmp_path / "fake_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    for scene in scenes:
        sid = scene["scene_id"]
        if sid in skip_image_for:
            continue
        (project_dir / f"scene_{sid:03d}.png").write_bytes(_make_base_image_bytes(sid))
    return project_dir


# ---------------------------------------------------------------------------
# 1 - output directory creation
# ---------------------------------------------------------------------------

def test_output_directory_is_created(tmp_path):
    project_dir = _build_fixture_project(tmp_path, [_copy_scene(1)])
    out_dir = tmp_path / "brand_new_output_dir" / "nested"
    assert not out_dir.exists()

    summary = fvr.render_project(project_dir, out_dir)

    assert out_dir.exists()
    assert summary.ok
    assert (out_dir / "scene_001.png").exists()


# ---------------------------------------------------------------------------
# 2 - 33 scenes discovered
# ---------------------------------------------------------------------------

def test_33_scenes_discovered(tmp_path):
    scenes = [_copy_scene(i) for i in range(1, 34)]
    project_dir = _build_fixture_project(tmp_path, scenes)
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.total == 33
    assert summary.ok
    assert summary.copied == 33
    assert len(summary.scene_actions) == 33


# ---------------------------------------------------------------------------
# 3 - copy behavior
# ---------------------------------------------------------------------------

def test_copy_action_is_byte_identical_to_source(tmp_path):
    scene = _copy_scene(7)
    project_dir = _build_fixture_project(tmp_path, [scene])
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.scene_actions[7] == "copy"
    assert summary.copied == 1
    src_bytes = (project_dir / "scene_007.png").read_bytes()
    dest_bytes = (out_dir / "scene_007.png").read_bytes()
    assert src_bytes == dest_bytes


def test_copy_action_selected_for_v1_style_scene_with_no_overlay_or_strategy(tmp_path):
    scene = {"scene_id": 9}  # v1-shaped: no visual_strategy/overlay_required keys at all
    project_dir = _build_fixture_project(tmp_path, [scene])
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.scene_actions[9] == "copy"
    assert summary.ok


# ---------------------------------------------------------------------------
# 4 - overlay behavior
# ---------------------------------------------------------------------------

def test_overlay_action_matches_overlay_renderer_output_exactly(tmp_path):
    scene = _overlay_scene(5)
    project_dir = _build_fixture_project(tmp_path, [scene])
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.scene_actions[5] == "overlay"
    assert summary.overlay == 1
    src_bytes = (project_dir / "scene_005.png").read_bytes()
    expected_bytes = ov.render_scene_overlay_bytes(src_bytes, scene)
    dest_bytes = (out_dir / "scene_005.png").read_bytes()
    assert dest_bytes == expected_bytes
    # and it must actually differ from the untouched base image - proof
    # the overlay was really drawn, not just copied
    assert dest_bytes != src_bytes


def test_overlay_action_not_selected_when_overlay_required_is_false(tmp_path):
    scene = {"scene_id": 3, "visual_strategy": "ai_generated_plus_overlay", "overlay_required": False}
    project_dir = _build_fixture_project(tmp_path, [scene])
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.scene_actions[3] == "copy"


# ---------------------------------------------------------------------------
# 5 - programmatic behavior
# ---------------------------------------------------------------------------

def test_programmatic_action_matches_renderer_output_exactly_and_ignores_base_image(tmp_path):
    scene = {
        "scene_id": 12, "visual_strategy": "programmatic_graphic", "scene_type": "scientific_graphic",
        "data_points": [{"data_type": "timeline", "value": "January 1, 2000", "exact_render": True},
                         {"data_type": "timeline", "value": "January 1, 2020", "exact_render": True}],
        "exact_text": ["January 1, 2000", "January 1, 2020"],
        "text_render": {"items": ["January 1, 2000", "January 1, 2020"], "hypothesis_badge": False},
    }
    project_dir = _build_fixture_project(tmp_path, [scene])
    # corrupt the base image on disk - the programmatic renderer must
    # never read it, so this must not affect the output at all
    (project_dir / "scene_012.png").write_bytes(b"not a real png at all")
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert summary.scene_actions[12] == "programmatic"
    expected_bytes, expected_renderer = pg.render_programmatic_graphic(scene, width=WIDTH, height=HEIGHT)
    dest_bytes = (out_dir / "scene_012.png").read_bytes()
    assert dest_bytes == expected_bytes
    assert summary.scene_renderers[12] == expected_renderer == "timeline"


# ---------------------------------------------------------------------------
# 6 - scene 026 routes to conceptual_curve
# ---------------------------------------------------------------------------

def test_scene_026_real_metadata_routes_to_conceptual_curve(tmp_path):
    project_dir = _build_fixture_project(tmp_path, [_REAL_SCENE_026])
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert summary.scene_actions[26] == "programmatic"
    assert summary.scene_renderers[26] == "conceptual_curve"
    img = Image.open(out_dir / "scene_026.png")
    assert img.size == (WIDTH, HEIGHT)


# ---------------------------------------------------------------------------
# 7 - no provider/API calls
# ---------------------------------------------------------------------------

def test_module_source_has_no_network_or_provider_imports():
    source = open(fvr.__file__).read()
    tree = ast.parse(source)
    banned_modules = {"httpx", "requests", "urllib", "urllib2", "socket", "aiohttp"}
    imported_modules = set()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
            imported_names.update(alias.name for alias in node.names)
    assert not (imported_modules & banned_modules), f"unexpected network import(s): {imported_modules & banned_modules}"
    assert "image_gen" not in imported_modules, "must never import image_gen.py (the provider dispatch layer)"
    assert "ImageProvider" not in imported_names


def test_render_project_never_calls_a_provider(tmp_path, monkeypatch):
    """Belt-and-suspenders functional check: even if something tried to
    reach image_gen.get_provider()/ImageProvider.generate() during this
    pass, it would show up as an AttributeError/ImportError here, since
    this module has no reference to either at all (see the static check
    above) - this test just confirms a full render_project() run over a
    mix of all three actions completes with zero errors and without ever
    importing youtube.image_gen as a side effect."""
    import sys
    scenes = [_copy_scene(1), _overlay_scene(2), _REAL_SCENE_026]
    project_dir = _build_fixture_project(tmp_path, scenes)
    out_dir = tmp_path / "out"
    sys.modules.pop("youtube.image_gen", None)

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert "youtube.image_gen" not in sys.modules


# ---------------------------------------------------------------------------
# 8 - all outputs are 1920x1080
# ---------------------------------------------------------------------------

def test_all_output_actions_produce_1920x1080(tmp_path):
    scenes = [_copy_scene(1), _overlay_scene(2), _REAL_SCENE_026]
    project_dir = _build_fixture_project(tmp_path, scenes)
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    for scene_id in (1, 2, 26):
        img = Image.open(out_dir / f"scene_{scene_id:03d}.png")
        assert img.size == (WIDTH, HEIGHT)


def test_differently_sized_source_gets_normalized_not_failed(tmp_path):
    """A source image whose aspect ratio doesn't match the target (e.g. an
    800x600 legacy/misc source) must be letterboxed to 1920x1080 via
    youtube.image_normalize, never failed and never distorted - this is the
    explicit "non-target aspect ratios: preserve aspect ratio, use
    deterministic letterbox/padding, never distort" requirement. A merely
    wrongly-sized-but-decodable image is not the "malformed" case - see
    test_malformed_source_image_fails_the_copy_action_loudly below for that."""
    scene = _copy_scene(4)
    project_dir = _build_fixture_project(tmp_path, [scene])
    (project_dir / "scene_004.png").write_bytes(_make_base_image_bytes(4, width=800, height=600))
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert summary.scene_actions[4] == "copy"
    img = Image.open(out_dir / "scene_004.png")
    assert img.size == (WIDTH, HEIGHT)


def test_malformed_source_image_fails_the_copy_action_loudly(tmp_path):
    scene = _copy_scene(4)
    project_dir = _build_fixture_project(tmp_path, [scene])
    # overwrite with bytes that are not a decodable image at all - a real
    # corrupted/truncated source, distinct from a merely wrongly-sized one
    (project_dir / "scene_004.png").write_bytes(b"not a real png at all")
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert not summary.ok
    failed_ids = [sid for sid, _reason in summary.failures]
    assert 4 in failed_ids


# ---------------------------------------------------------------------------
# 9 - missing input scene fails clearly
# ---------------------------------------------------------------------------

def test_missing_source_image_fails_clearly_not_silently(tmp_path):
    scenes = [_copy_scene(1), _overlay_scene(2)]
    project_dir = _build_fixture_project(tmp_path, scenes, skip_image_for={2})
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert not summary.ok
    assert not (out_dir / "scene_002.png").exists()
    failed_ids = {sid for sid, _reason in summary.failures}
    assert failed_ids == {2}
    reason = dict(summary.failures)[2]
    assert "missing" in reason.lower()
    # scene 1 (unaffected) must still have been processed - a missing
    # scene must not silently abort/skip the REST of the batch either
    assert summary.scene_actions.get(1) == "copy"
    assert (out_dir / "scene_001.png").exists()


def test_main_cli_exits_nonzero_and_reports_failure_for_missing_scene(tmp_path, capsys):
    scenes = [_copy_scene(1), _overlay_scene(2)]
    project_dir = _build_fixture_project(tmp_path, scenes, skip_image_for={2})
    out_dir = tmp_path / "out"

    exit_code = fvr.main(["--project-dir", str(project_dir), "--out-dir", str(out_dir)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "scene 002" in captured.err or "002" in captured.err


# ---------------------------------------------------------------------------
# 10 - deterministic output
# ---------------------------------------------------------------------------

def test_render_project_is_deterministic_across_runs(tmp_path):
    scenes = [_copy_scene(1), _overlay_scene(2), _REAL_SCENE_026]
    project_dir = _build_fixture_project(tmp_path, scenes)
    out_dir_a = tmp_path / "out_a"
    out_dir_b = tmp_path / "out_b"

    summary_a = fvr.render_project(project_dir, out_dir_a)
    summary_b = fvr.render_project(project_dir, out_dir_b)

    assert summary_a.ok and summary_b.ok
    for scene_id in (1, 2, 26):
        bytes_a = (out_dir_a / f"scene_{scene_id:03d}.png").read_bytes()
        bytes_b = (out_dir_b / f"scene_{scene_id:03d}.png").read_bytes()
        assert bytes_a == bytes_b


# ---------------------------------------------------------------------------
# 11 - storage (the source project) is never modified
# ---------------------------------------------------------------------------

def test_source_project_directory_is_never_modified(tmp_path):
    scenes = [_copy_scene(1), _overlay_scene(2), _REAL_SCENE_026]
    project_dir = _build_fixture_project(tmp_path, scenes)

    before = {
        p.name: (p.stat().st_size, p.read_bytes())
        for p in sorted(project_dir.iterdir())
    }
    before_names = set(before)

    out_dir = tmp_path / "out"
    summary = fvr.render_project(project_dir, out_dir)

    after = {p.name: (p.stat().st_size, p.read_bytes()) for p in sorted(project_dir.iterdir())}
    assert summary.ok, summary.failures
    assert set(after) == before_names, "no files may be added to or removed from the source project directory"
    for name in before_names:
        assert after[name] == before[name], f"{name} content changed - source project must never be modified"


def test_main_cli_uses_slug_lookup_and_writes_only_under_debug_audit(isolated_storage, tmp_path):
    """Exercises the --slug lookup path (the one that mirrors the real
    Windows invocation: python -m youtube.final_visual_render_demo --slug
    the-wow-signal-6) via the isolated_storage fixture's own tmp
    STORAGE_DIR, and confirms output lands under debug_audit/, never
    inside the storage project directory."""
    project_dir = isolated_storage.STORAGE_DIR / "2026" / "09" / "the-wow-signal-6-test"
    project_dir.mkdir(parents=True, exist_ok=True)
    scenes = [_copy_scene(1), _overlay_scene(2)]
    (project_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    for scene in scenes:
        (project_dir / f"scene_{scene['scene_id']:03d}.png").write_bytes(_make_base_image_bytes(scene["scene_id"]))

    out_dir = tmp_path / "debug_audit" / "final_visual_render_v2" / "the-wow-signal-6-test"
    exit_code = fvr.main([
        "--slug", "the-wow-signal-6-test",
        "--out-dir", str(out_dir),
    ])

    assert exit_code == 0
    assert (out_dir / "scene_001.png").exists()
    assert (out_dir / "scene_002.png").exists()
    assert (out_dir / "contact_sheet.png").exists()
    # nothing was written inside the storage project directory itself
    assert set(p.name for p in project_dir.iterdir()) == {"scenes.json", "scene_001.png", "scene_002.png"}


# ---------------------------------------------------------------------------
# 12 - normalization integration: the real 1920x1072 production bug
#
# The real generated AI scene images in storage/2026/09/the-wow-signal-6/
# are 1920x1072, 8px short of the 1920x1080 this pipeline requires. These
# tests replicate that exact shape (never 1920x1080) for the copy and
# overlay actions, proving youtube.image_normalize is wired in BEFORE both,
# and that the programmatic path (scene 026) never needed it at all.
# ---------------------------------------------------------------------------

def test_overlay_action_receives_a_normalized_1920x1080_base(tmp_path):
    """Category 7: the overlay renderer must draw onto an already-normalized
    1920x1080 canvas, not the raw 1920x1072 source - proven two ways: the
    final output is exactly 1920x1080, and the overlay's own bytes-for-bytes
    output (computed by normalizing first, exactly as the pipeline should)
    matches the pipeline's actual output exactly."""
    scene = _overlay_scene(5)
    project_dir = _build_fixture_project(tmp_path, [scene])
    (project_dir / "scene_005.png").write_bytes(_make_base_image_bytes(5, width=1920, height=1072))
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert summary.scene_actions[5] == "overlay"
    img = Image.open(out_dir / "scene_005.png")
    assert img.size == (WIDTH, HEIGHT)

    from youtube import image_normalize as inorm
    src_bytes = _make_base_image_bytes(5, width=1920, height=1072)
    normalized_bytes = inorm.normalize_image_bytes_to_video_size(src_bytes, WIDTH, HEIGHT)
    expected_bytes = ov.render_scene_overlay_bytes(normalized_bytes, scene)
    dest_bytes = (out_dir / "scene_005.png").read_bytes()
    assert dest_bytes == expected_bytes


def test_programmatic_scene_026_unaffected_by_1920x1072_base_image(tmp_path):
    """Category 8: scene 026 (programmatic) must still come out exactly
    1920x1080 even when its on-disk base image is the buggy 1920x1072 size
    - the programmatic renderer never reads the base image at all, so this
    proves the real-world bug can never leak into it."""
    project_dir = _build_fixture_project(tmp_path, [_REAL_SCENE_026])
    (project_dir / "scene_026.png").write_bytes(_make_base_image_bytes(26, width=1920, height=1072))
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert summary.scene_actions[26] == "programmatic"
    assert summary.scene_renderers[26] == "conceptual_curve"
    img = Image.open(out_dir / "scene_026.png")
    assert img.size == (WIDTH, HEIGHT)


def test_realistic_project_with_1920x1072_sources_renders_all_scenes_at_1920x1080(tmp_path):
    """Category 9: the closest sandbox-achievable proxy for "final render of
    the real fixture produces all expected dimensions" - a fixture project
    mirroring the real dispatch mix (copy scenes, an overlay scene, and the
    real scene 026 payload) with every non-programmatic base image built at
    the exact real-world buggy size (1920x1072), asserting the whole batch
    succeeds and every single output is exactly 1920x1080."""
    scenes = [_copy_scene(1), _copy_scene(2), _overlay_scene(3), _REAL_SCENE_026]
    project_dir = tmp_path / "realistic_project"
    project_dir.mkdir()
    (project_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    for scene in scenes:
        sid = scene["scene_id"]
        if sid == 26:
            continue  # programmatic - no base image needed/used
        (project_dir / f"scene_{sid:03d}.png").write_bytes(
            _make_base_image_bytes(sid, width=1920, height=1072)
        )
    out_dir = tmp_path / "out"

    summary = fvr.render_project(project_dir, out_dir)

    assert summary.ok, summary.failures
    assert summary.total == 4
    for scene in scenes:
        sid = scene["scene_id"]
        img = Image.open(out_dir / f"scene_{sid:03d}.png")
        assert img.size == (WIDTH, HEIGHT), f"scene {sid} is {img.size}, expected {(WIDTH, HEIGHT)}"
