"""
Thumbnail generation (spec Round 2 section 10):

    - must be 1280x720 regardless of the source hero image's aspect ratio
    - production (LLM overlay-text call) must not silently emit no text or
      crash the stage if the LLM call fails - it must fall back to a real,
      title-derived overlay
    - correctly linked to this video's own script/metadata (title/hook),
      never someone else's
"""
from PIL import Image

import pytest

from core.state import PipelineState
from storage import db, project_store as ps
from youtube import thumbnail as th


def test_thumbnail_is_always_1280x720_regardless_of_source_aspect_ratio(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Wide Source Image Topic")
    ps.save_json(paths.script_json, {"title": "The Mystery", "hook": "a strange tale"})
    # a very wide (21:9-ish) source image, unlike the 16:9 target
    Image.new("RGB", (2560, 900), (10, 20, 30)).save(paths.scene_image(1), format="PNG")

    monkeypatch.setattr(th.llm_client, "chat_json", lambda *a, **k: ({"text": "STRANGE TALE"}, []))

    result = th.run_and_save(paths)

    with Image.open(result) as im:
        assert im.size == (th.WIDTH, th.HEIGHT)


def test_thumbnail_is_always_1280x720_for_a_tall_source_image(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Tall Source Image Topic")
    ps.save_json(paths.script_json, {"title": "The Mystery", "hook": "a strange tale"})
    Image.new("RGB", (900, 1600), (10, 20, 30)).save(paths.scene_image(1), format="PNG")

    monkeypatch.setattr(th.llm_client, "chat_json", lambda *a, **k: ({"text": "STRANGE TALE"}, []))

    result = th.run_and_save(paths)

    with Image.open(result) as im:
        assert im.size == (th.WIDTH, th.HEIGHT)


def test_llm_failure_falls_back_to_title_text_instead_of_crashing(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("LLM Overlay Failure Topic")
    ps.save_json(paths.script_json, {"title": "The Dyatlov Pass Incident", "hook": "nine hikers, one mystery"})
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(paths.scene_image(1), format="PNG")

    def _boom(*a, **k):
        raise RuntimeError("simulated Claude API outage")

    monkeypatch.setattr(th.llm_client, "chat_json", _boom)

    result = th.run_and_save(paths)

    # did not crash, and a real (non-empty) fallback overlay was recorded
    assert result.exists()
    data = ps.load_json(paths.state_json)
    assert data["thumbnail_text"]  # non-empty
    assert data["thumbnail_text"] != ""


def test_generated_overlay_text_is_derived_from_this_videos_own_title(isolated_storage, monkeypatch):
    """Guards against a mixed-up hero image / title from a different
    video_id ever leaking into this one's thumbnail (spec: "correctly
    link to video/topic metadata")."""
    paths, _ = ps.create_project("This Videos Own Unique Title Topic")
    ps.save_json(paths.script_json, {"title": "This Videos Own Unique Title", "hook": "a hook"})
    Image.new("RGB", (1920, 1080), (5, 5, 5)).save(paths.scene_image(1), format="PNG")

    captured = {}

    def _fake_chat_json(system, user, **kwargs):
        captured["user"] = user
        return {"text": "UNIQUE TITLE"}, []

    monkeypatch.setattr(th.llm_client, "chat_json", _fake_chat_json)

    th.run_and_save(paths)

    assert "This Videos Own Unique Title" in captured["user"]


def test_no_scene_images_raises_clear_error_not_a_crash(isolated_storage):
    paths, _ = ps.create_project("No Scene Images At All Topic")
    ps.save_json(paths.script_json, {"title": "T", "hook": "h"})
    # no scene_image(1) or anything else written

    with pytest.raises(RuntimeError, match="no scene images found"):
        th.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED


def test_falls_back_to_first_available_scene_when_hero_scene_id_missing(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Hero Scene Missing Falls Back Topic")
    ps.save_json(paths.script_json, {"title": "T", "hook": "h"})
    # scene 1 (the default hero) is missing, but scene 2 exists
    Image.new("RGB", (1920, 1080), (0, 0, 0)).save(paths.scene_image(2), format="PNG")

    monkeypatch.setattr(th.llm_client, "chat_json", lambda *a, **k: ({"text": "TEXT"}, []))

    result = th.run_and_save(paths, hero_scene_id=1)

    assert result.exists()
