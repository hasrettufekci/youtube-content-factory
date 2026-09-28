"""
Small-scale but REAL integration test: exercises the actual research -> script
-> scenes -> images -> narration -> video -> thumbnail -> metadata chain with
real ffmpeg/Pillow/wave-module audio, no mocked file I/O. Only the Claude API calls
are mocked (llm_client.chat_json/chat), since this suite must run with zero
API keys/cost/network - see tests/test_pipeline_integration.py's use of
IMAGE_PROVIDER=mock / TTS_PROVIDER=mock for the same reason on the media side.

Deliberately small (3 scenes, ~20s narration) to keep the automated suite
fast; a full 25-40 scene / 8-10 minute production run was additionally run
by hand once against this same code path (see the project's delivery
report) to validate at real spec scale.
"""
import shutil

import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import researcher, script_generator, scene_planner, image_gen, tts, assemble, thumbnail, metadata, quality_check

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

FAKE_RESEARCH = {
    "topic": "Test Mystery Topic",
    "summary": "A short test summary of the topic for pipeline testing purposes.",
    "facts": ["Fact one about the topic.", "Fact two about the topic."],
    "timeline": [{"date": "1900", "event": "Something happened."}],
    "sources": [{"title": "Test Source", "url": "https://example.test/article", "publisher": "Test Publisher", "published_at": "1999", "relevance": "primary account"}],
    "uncertainties": ["It is unclear exactly why this happened."],
    "search_notes": ["Searched for background information."],
}

_SECTION_TYPES = script_generator.REQUIRED_SECTION_TYPES


def _fake_script_sections(words_per_section=6):
    return [
        {"type": t, "title": t.replace("_", " ").title(), "narration": f"This is the {t.replace('_', ' ')} part of the test script right now."}
        for t in _SECTION_TYPES
    ]


@pytest.fixture
def small_scene_planner(monkeypatch):
    """Scene planner targets a handful of scenes instead of 25-40, and the
    script's word-count gate is relaxed to match, so the ffmpeg assembly step
    in this test finishes in seconds instead of minutes while still exercising
    every real code path (just at a smaller scale than a real 8-10min video)."""
    monkeypatch.setattr(scene_planner, "MIN_SCENES", 3)
    monkeypatch.setattr(scene_planner, "MAX_SCENES", 4)
    monkeypatch.setattr(script_generator, "TARGET_WORDS_MIN", 40)
    monkeypatch.setattr(script_generator, "TARGET_WORDS_MAX", 200)


def test_full_pipeline_mechanics_with_mocked_llm_and_mocked_media(isolated_storage, small_scene_planner, monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")

    # --- research (Claude call mocked) ---
    def fake_research_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        return dict(FAKE_RESEARCH), []
    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_research_chat_json)

    paths, created = ps.create_project("Test Mystery Topic", category="Mysteries")
    assert created
    researcher.run_and_save(paths, "Test Mystery Topic", category="Mysteries")
    assert paths.research_json.exists()
    assert ps.load_state(paths) == PipelineState.RESEARCHED

    # --- script (Claude call mocked) ---
    def fake_script_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "The Test Mystery: What Really Happened", "sections": _fake_script_sections()}, []
    monkeypatch.setattr(script_generator.llm_client, "chat_json", fake_script_chat_json)

    script = script_generator.run_and_save(paths)
    assert paths.script_json.exists()
    assert ps.load_state(paths) == PipelineState.SCRIPT_GENERATED
    assert script["word_count"] > 0

    # --- scenes (Claude call mocked for visual prompts) ---
    def fake_scenes_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        import re
        chunk_count = len(re.findall(r"(?m)^\d+\. ", user))
        return [
            {"scene_id": i + 1, "visual_prompt": f"A cinematic test scene number {i + 1}.", "visual_type": "cinematic"}
            for i in range(chunk_count)
        ], []
    monkeypatch.setattr(scene_planner.llm_client, "chat_json", fake_scenes_chat_json)

    scenes = scene_planner.run_and_save(paths)
    assert paths.scenes_json.exists()
    assert 3 <= len(scenes) <= 4
    assert ps.load_state(paths) == PipelineState.SCENES_PLANNED

    # --- images (mock provider - real Pillow-rendered PNGs, no network) ---
    image_gen.run_and_save(paths)
    assert len(paths.scene_images()) == len(scenes)
    assert ps.load_state(paths) == PipelineState.IMAGES_GENERATED
    from PIL import Image
    with Image.open(paths.scene_image(1)) as im:
        assert im.size == (settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT)

    # --- narration (mock provider - pure Python `wave` module, no network, no external binary) ---
    tts.run_and_save(paths)
    assert paths.narration_mp3.exists()
    assert paths.narration_mp3.stat().st_size > 0
    assert ps.load_state(paths) == PipelineState.AUDIO_GENERATED

    # --- video assembly (REAL ffmpeg Ken Burns + crossfade + mux) ---
    assemble.run_and_save(paths)
    assert paths.video_mp4.exists()
    assert ps.load_state(paths) == PipelineState.VIDEO_ASSEMBLED

    probe = quality_check._ffprobe_json(paths.video_mp4)
    video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    assert video_streams and video_streams[0]["width"] == settings.VIDEO_WIDTH
    assert video_streams[0]["height"] == settings.VIDEO_HEIGHT
    assert audio_streams, "assembled video has no audio track"

    # --- thumbnail (Claude call mocked for the short text overlay) ---
    def fake_thumb_chat_json(system, user, max_tokens=200, model=None, use_web_search=False):
        return {"text": "TEST MYSTERY REVEALED"}, []
    monkeypatch.setattr(thumbnail.llm_client, "chat_json", fake_thumb_chat_json)

    thumbnail.run_and_save(paths)
    assert paths.thumbnail_png.exists()
    with Image.open(paths.thumbnail_png) as im:
        assert im.size == (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT)
    assert ps.load_state(paths) == PipelineState.THUMBNAIL_READY

    # --- metadata (deterministic, no LLM call) ---
    metadata.run_and_save(paths, category="Mysteries")
    assert paths.metadata_json.exists()
    meta = ps.load_json(paths.metadata_json)
    assert meta["title"]
    assert meta["source_count"] == 1

    # --- quality check: everything except the "is this really 8-10 minutes"
    # gate should pass, since this test deliberately uses a 3-scene/~15s
    # narration instead of a full documentary to keep the suite fast.
    report = quality_check.run_quality_check(paths)
    for name in ("video_file_exists", "resolution_1920x1080", "audio_present",
                 "audio_duration_matches_video", "scene_image_count_matches_plan",
                 "thumbnail_exists", "metadata_exists", "research_sources_present"):
        assert report.checks[name]["passed"], f"{name}: {report.checks[name]['detail']}"
    assert report.checks["duration_in_documentary_range"]["passed"] is False  # expected at this test's small scale
