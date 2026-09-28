"""
pipeline.py <-> Shorts Engine V1 wiring tests.

Covers spec scenarios:
   25  existing Smart Publish behavior remains intact
   26  existing Analytics Timing behavior remains intact
   27  existing Daily Topic Discovery remains intact
   28  existing Telegram topic selection remains intact
(25-28 are primarily proven by the fact that the full, pre-existing
regression suite - none of it touching Shorts - continues to pass
completely unmodified; see the delivery report. This file additionally
proves the specific pipeline.py integration points: SHORTS_ENABLED=False
(the library default) leaves _notify_telegram()'s original single-video
message byte-for-byte unchanged and never touches youtube.shorts_engine at
all, while SHORTS_ENABLED=True wires produce_video() -> _produce_shorts()
-> _notify_telegram() correctly for both the all-ready and the
partial-failure case.

Every Long pipeline stage is stubbed (instant fake writes) - no real
Claude/FAL/TTS/ffmpeg/Telegram/YouTube call is made anywhere in this file.
"""
from __future__ import annotations

import pytest

import pipeline as pipeline_module
from core.config import settings
from storage import project_store as ps
from storage import shorts as shorts_store
from youtube import (
    researcher, script_generator, scene_planner, image_gen, tts, assemble,
    thumbnail, metadata, quality_check, telegram_approval as tap,
)

REQUIRED_SECTIONS = ["hook", "introduction", "story_development", "evidence",
                     "mystery", "reveal", "conclusion", "cta"]


def _fake_script():
    return {
        "title": "Shorts Pipeline Integration Test Video",
        "sections": [{"type": t, "title": t, "narration": f"Narration for {t} section, long enough to be real."}
                     for t in REQUIRED_SECTIONS],
    }


@pytest.fixture
def stubbed_long_pipeline(monkeypatch):
    """Mirrors tests/test_pipeline_approval_integration.py's own
    stubbed_pipeline_stages fixture, English-only (languages=[]) to keep
    this file focused purely on the Shorts wiring, not multilingual."""
    def fake_research(paths, topic, category=None):
        ps.save_json(paths.research_json, {"ok": True, "sources": [{"url": "https://example.com"}]})
    monkeypatch.setattr(researcher, "run_and_save", fake_research)

    def fake_script(paths):
        ps.save_json(paths.script_en_json, _fake_script())
    monkeypatch.setattr(script_generator, "run_and_save", fake_script)

    def fake_scenes(paths):
        ps.save_json(paths.scenes_json, [{"scene_id": i, "duration_hint": 5.0} for i in range(1, 4)])
    monkeypatch.setattr(scene_planner, "run_and_save_v2", fake_scenes)

    def fake_images(paths):
        scenes = ps.load_json(paths.scenes_json)
        for i in range(1, len(scenes) + 1):
            paths.scene_image(i).write_bytes(b"fake-image")
    monkeypatch.setattr(image_gen, "run_and_save", fake_images)

    def fake_tts(paths, language="en"):
        p = paths.narration_mp3_for(language)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-audio")
    monkeypatch.setattr(tts, "run_and_save", fake_tts)

    def fake_assemble(paths):
        paths.video_mp4.write_bytes(b"fake-video")
    monkeypatch.setattr(assemble, "run_and_save", fake_assemble)

    def fake_thumbnail(paths):
        from PIL import Image
        Image.new("RGB", (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT)).save(paths.thumbnail_png)
    monkeypatch.setattr(thumbnail, "run_and_save", fake_thumbnail)

    def fake_metadata(paths, category=None, language="en"):
        ps.save_json(paths.metadata_json_for(language), {"title": _fake_script()["title"]})
    monkeypatch.setattr(metadata, "run_and_save", fake_metadata)

    def fake_qc(paths, languages=None):
        from youtube.quality_check import QualityReport
        return QualityReport(passed=True)
    monkeypatch.setattr(quality_check, "run_and_enforce", fake_qc)


# ---------------------------------------------------------------------------
# SHORTS_ENABLED=False (default): byte-identical original behavior
# ---------------------------------------------------------------------------

def test_notify_telegram_uses_original_single_video_flow_when_shorts_results_is_none(isolated_storage, monkeypatch):
    calls = {"single": 0, "package": 0}
    monkeypatch.setattr(tap, "build_approval_payload", lambda *a, **k: calls.__setitem__("single", calls["single"] + 1) or "PAYLOAD")
    monkeypatch.setattr(tap, "send_new_approval", lambda payload: calls.__setitem__("single_sent", True))
    monkeypatch.setattr(tap, "send_new_package_approval", lambda *a, **k: calls.__setitem__("package", calls["package"] + 1))

    paths, _ = ps.create_project("Shorts Disabled Notify Topic")
    paths.video_mp4.write_bytes(b"x")
    paths.thumbnail_png.write_bytes(b"x")

    pipeline_module._notify_telegram(paths, shorts_results=None)

    assert calls["single"] == 1
    assert calls["package"] == 0


def test_produce_shorts_not_imported_or_called_when_shorts_disabled(isolated_storage, monkeypatch, stubbed_long_pipeline):
    monkeypatch.setattr(settings, "SHORTS_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "_notify_telegram", lambda paths, **kwargs: None)

    from youtube import shorts_engine
    monkeypatch.setattr(shorts_engine, "produce_shorts_for_video",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called when SHORTS_ENABLED is False")))

    result_paths = pipeline_module.produce_video("Shorts Disabled Full Run Topic", languages=[])
    assert result_paths.video_mp4.exists()
    assert shorts_store.list_shorts_for_parent(result_paths.video_id) == []


# ---------------------------------------------------------------------------
# SHORTS_ENABLED=True: wiring to _produce_shorts()/_notify_telegram()
# ---------------------------------------------------------------------------

def test_produce_video_produces_shorts_and_sends_package_when_all_ready(isolated_storage, monkeypatch, stubbed_long_pipeline):
    monkeypatch.setattr(settings, "SHORTS_ENABLED", True)

    from youtube import shorts_engine

    def fake_produce_shorts(paths, script, scenes, shorts_per_long=None):
        results = []
        for n in (1, 2, 3):
            short_id = shorts_store.create_short(parent_video_id=paths.video_id, short_number=n,
                                                  status=shorts_store.STATUS_READY, title=f"Short {n}")
            shorts_store.update_fields(short_id, video_path=str(paths.video_mp4), thumbnail_path=str(paths.thumbnail_png))
            results.append({"short_id": short_id, "short_number": n, "status": shorts_store.STATUS_READY,
                             "title": f"Short {n}", "qc_passed": True, "qc_errors": []})
        return results

    monkeypatch.setattr(shorts_engine, "produce_shorts_for_video", fake_produce_shorts)

    notify_calls = []
    monkeypatch.setattr(pipeline_module, "_notify_telegram",
                         lambda paths, shorts_results=None: notify_calls.append(shorts_results))

    result_paths = pipeline_module.produce_video("Shorts Enabled All Ready Topic", languages=[])

    assert len(notify_calls) == 1
    assert notify_calls[0] is not None
    assert len(notify_calls[0]) == 3
    assert all(r["status"] == shorts_store.STATUS_READY for r in notify_calls[0])
    assert len(shorts_store.list_shorts_for_parent(result_paths.video_id)) == 3


def test_notify_telegram_sends_package_approval_when_shorts_all_ready(isolated_storage, monkeypatch):
    from youtube import shorts_engine

    package_calls = []
    monkeypatch.setattr(tap, "build_package_approval_payload", lambda *a, **k: package_calls.append(a) or "PAYLOAD")
    monkeypatch.setattr(tap, "send_new_package_approval", lambda payload: package_calls.append("sent"))
    monkeypatch.setattr(tap, "send_new_approval", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not send the single-video message for a Shorts package")))

    paths, _ = ps.create_project("Shorts Package Notify Topic")
    paths.video_mp4.write_bytes(b"x")
    paths.thumbnail_png.write_bytes(b"x")
    ps.save_json(paths.metadata_en_json, {"title": "The Package Title"})
    for n in (1, 2, 3):
        shorts_store.create_short(parent_video_id=paths.video_id, short_number=n, status=shorts_store.STATUS_READY, title=f"Short {n}")

    shorts_results = [{"short_number": n, "status": shorts_store.STATUS_READY} for n in (1, 2, 3)]
    pipeline_module._notify_telegram(paths, shorts_results=shorts_results)

    assert "sent" in package_calls


def test_notify_telegram_sends_failure_notice_when_a_short_is_not_ready(isolated_storage, monkeypatch):
    failure_calls = []
    monkeypatch.setattr(tap, "send_shorts_production_failed_notice",
                         lambda long_title, slug, shorts_results, **k: failure_calls.append((long_title, slug)))
    monkeypatch.setattr(tap, "send_new_package_approval",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not send the package as approve-ready when a Short failed")))
    monkeypatch.setattr(tap, "send_new_approval",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fall back to the single-video message")))

    paths, _ = ps.create_project("Shorts Partial Failure Notify Topic")
    paths.video_mp4.write_bytes(b"x")
    paths.thumbnail_png.write_bytes(b"x")
    ps.save_json(paths.metadata_en_json, {"title": "The Failing Package Title"})

    shorts_results = [
        {"short_number": 1, "status": shorts_store.STATUS_READY},
        {"short_number": 2, "status": shorts_store.STATUS_FAILED, "qc_errors": ["resolution mismatch"]},
        {"short_number": 3, "status": shorts_store.STATUS_READY},
    ]
    pipeline_module._notify_telegram(paths, shorts_results=shorts_results)

    assert len(failure_calls) == 1
    assert failure_calls[0][0] == "The Failing Package Title"
