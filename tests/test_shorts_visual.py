"""
youtube/shorts_visual.py tests - scene selection + crop-vs-regenerate
strategy.

Covers spec scenarios:
    7  9:16 output configuration
    8  existing Long image reuse
    9  vertical crop fallback
   10  FAL vertical regeneration only when necessary
"""
from __future__ import annotations

from PIL import Image

from core.config import settings
from youtube import image_gen, shorts_visual


def _script():
    return {
        "sections": [
            {"type": "hook", "narration": "A strange signal was heard one night by a lonely radio telescope."},
            {"type": "introduction", "narration": "The observatory kept careful records of every reading."},
            {"type": "story_development", "narration": "Investigators pored over the printout for weeks."},
            {"type": "evidence", "narration": "Nothing in the archives matched the pattern they found."},
            {"type": "mystery", "narration": "To this day nobody has explained where the signal came from."},
            {"type": "reveal", "narration": "The most surprising detail was that it never repeated."},
            {"type": "conclusion", "narration": "The mystery endures as one of astronomy's greatest puzzles."},
            {"type": "cta", "narration": "Subscribe for the full story."},
        ],
    }


def _scenes():
    doc_words = " ".join(s["narration"] for s in _script()["sections"] if s["type"] != "cta").split()
    chunk = len(doc_words) // 4
    types = ["cinematic", "map", "closeup", "wide"]
    scenes = []
    for i in range(4):
        words = doc_words[i * chunk:(i + 1) * chunk] or doc_words[-1:]
        scenes.append({
            "scene_id": i + 1, "schema_version": 2, "duration_hint": 4.0,
            "narration_en": " ".join(words), "visual_type": types[i],
            "visual_prompt": f"documentary shot {i + 1}",
        })
    return scenes


# ---------------------------------------------------------------------------
# scene selection (proportional word-overlap mapping)
# ---------------------------------------------------------------------------

def test_select_scenes_for_section_returns_overlapping_scenes():
    scenes = _scenes()
    selected = shorts_visual.select_scenes_for_section(scenes, _script(), "hook")
    assert selected  # never empty
    assert selected[0]["scene_id"] == scenes[0]["scene_id"]  # the hook is at the very start


def test_select_scenes_for_section_never_returns_empty_even_on_odd_input():
    scenes = [{"scene_id": 1, "narration_en": "", "visual_type": "cinematic"}]
    selected = shorts_visual.select_scenes_for_section(scenes, _script(), "reveal")
    assert selected  # falls back to the middle scene rather than an empty list


# ---------------------------------------------------------------------------
# 7/9/10 - crop-suitability heuristic, built from scene_planner's real
# VISUAL_TYPES
# ---------------------------------------------------------------------------

def test_classify_crop_suitability_map_and_wide_are_bad():
    assert shorts_visual.classify_crop_suitability({"visual_type": "map"}) == "bad"
    assert shorts_visual.classify_crop_suitability({"visual_type": "wide"}) == "bad"


def test_classify_crop_suitability_others_are_good():
    for vt in ("cinematic", "closeup", "archival", "portrait", "abstract"):
        assert shorts_visual.classify_crop_suitability({"visual_type": vt}) == "good"


def test_classify_crop_suitability_unknown_defaults_to_good():
    assert shorts_visual.classify_crop_suitability({}) == "good"


# ---------------------------------------------------------------------------
# 7/9 - vertical crop transform: no distortion, exact target size
# ---------------------------------------------------------------------------

def test_crop_to_vertical_produces_exact_target_size_no_distortion(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT), (10, 20, 30)).save(source)
    dest = tmp_path / "vertical.png"

    shorts_visual.crop_to_vertical(source, dest, width=1080, height=1920)

    with Image.open(dest) as out:
        assert out.size == (1080, 1920)


def test_crop_to_vertical_center_crop_preserves_center_pixel_color(tmp_path):
    """A uniformly-colored source, once scaled+cropped, must still be that
    same uniform color everywhere - proof the transform never distorts
    (stretches unevenly) or introduces artifacts."""
    source = tmp_path / "source.png"
    Image.new("RGB", (1920, 1080), (200, 50, 10)).save(source)
    dest = tmp_path / "vertical.png"
    shorts_visual.crop_to_vertical(source, dest, width=1080, height=1920)
    with Image.open(dest) as out:
        assert out.getpixel((540, 960)) == (200, 50, 10)


# ---------------------------------------------------------------------------
# 8/10 - build_vertical_visual(): reuse via crop for "good" types, FAL/
# provider regeneration only for "bad" types
# ---------------------------------------------------------------------------

def test_build_vertical_visual_reuses_existing_image_for_good_visual_type(tmp_path, monkeypatch, isolated_storage):
    from storage import project_store as ps

    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    paths, _ = ps.create_project("Reuse Test Topic")
    Image.new("RGB", (settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT), (5, 5, 5)).save(paths.scene_image(1))

    called = {"n": 0}
    real_generate = image_gen.MockImageProvider.generate

    def spy_generate(self, prompt, width, height):
        called["n"] += 1
        return real_generate(self, prompt, width, height)

    monkeypatch.setattr(image_gen.MockImageProvider, "generate", spy_generate)

    dest = tmp_path / "short_visual.png"
    scene = {"scene_id": 1, "visual_type": "cinematic", "visual_prompt": "a cinematic shot"}
    shorts_visual.build_vertical_visual(paths, scene, dest, width=1080, height=1920)

    assert dest.exists()
    with Image.open(dest) as out:
        assert out.size == (1080, 1920)
    assert called["n"] == 0  # never called the image provider - pure crop reuse


def test_build_vertical_visual_generates_dedicated_image_for_bad_visual_type(tmp_path, monkeypatch, isolated_storage):
    from storage import project_store as ps

    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    paths, _ = ps.create_project("Regen Test Topic")
    # Deliberately do NOT create scene_image(1) - a "map"/"wide" scene must
    # regenerate regardless of whether the original even exists.

    called = {"n": 0}
    real_generate = image_gen.MockImageProvider.generate

    def spy_generate(self, prompt, width, height):
        called["n"] += 1
        assert (width, height) == (1080, 1920)
        return real_generate(self, prompt, width, height)

    monkeypatch.setattr(image_gen.MockImageProvider, "generate", spy_generate)

    dest = tmp_path / "short_visual.png"
    scene = {"scene_id": 1, "visual_type": "map", "visual_prompt": "a labeled star map"}
    shorts_visual.build_vertical_visual(paths, scene, dest, width=1080, height=1920)

    assert dest.exists()
    assert called["n"] == 1


def test_build_visuals_for_short_returns_one_path_per_selected_scene(isolated_storage, monkeypatch):
    from storage import project_store as ps

    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    paths, _ = ps.create_project("Multi Scene Topic")
    scenes = _scenes()
    for s in scenes:
        Image.new("RGB", (settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT), (1, 2, 3)).save(paths.scene_image(s["scene_id"]))

    out_paths = shorts_visual.build_visuals_for_short(paths, scenes, _script(), "hook", short_number=1)
    assert len(out_paths) >= 1
    assert all(p.exists() for p in out_paths)
