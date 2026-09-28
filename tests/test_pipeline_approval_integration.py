"""
pipeline.py <-> Telegram Approval V1 architecture integration tests.

Confirms the production pipeline's own real assembler/thumbnail output
(storage/<slug>/video.mp4, storage/<slug>/thumbnail.png - the actual
production source of truth) is what reaches Telegram Approval V1, NOT
youtube/final_video_assembly_demo.py's narrow, the-wow-signal-6-specific
33-scene debug_audit/ remediation output (that module is untouched and
stays scoped to its own one-off job). Also confirms the legacy
telegram_bot.py video_id-based notification is no longer triggered from
produce_video() (so a produced video only ever gets ONE approval message),
that an English-only run works without ever mutating
settings.LANGUAGE_LOCALIZATIONS globally, and that the approval state
machine (upload only after APPROVE, never after REJECT) behaves the same
way for a production-pipeline-sourced approval as it already does for a
manually-sent demo one (see tests/test_telegram_approval.py for the full
state-machine suite this deliberately does not duplicate).

Every pipeline stage function is monkeypatched to instantly write a tiny
fake artifact - no real Claude/FAL/TTS/ffmpeg/Telegram/YouTube call is
made anywhere in this file.
"""
from __future__ import annotations

import pytest

import pipeline as pipeline_module
import telegram_bot as tb
from core.config import settings
from storage import project_store as ps
from storage import telegram_approvals as ta
from youtube import (
    researcher, script_generator, scene_planner, image_gen, tts, assemble,
    thumbnail, metadata, quality_check, telegram_approval as tap,
)

FAKE_TITLE = "E2E Architecture Test Video"


def _write_bytes(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def stubbed_pipeline_stages(monkeypatch):
    """Every _stage_plan() stage becomes an instant fake write - mirrors
    tests/test_pipeline_resume.py's own _record_and_write pattern, kept
    local here so this file has no cross-file fixture coupling."""
    calls: list[str] = []

    def fake_research(paths, topic, category=None):
        calls.append("research")
        ps.save_json(paths.research_json, {"ok": True})
    monkeypatch.setattr(researcher, "run_and_save", fake_research)

    def fake_script(paths):
        calls.append("script_en")
        ps.save_json(paths.script_en_json, {"ok": True, "sections": []})
    monkeypatch.setattr(script_generator, "run_and_save", fake_script)

    def fake_localizer_run_and_save(paths, language="tr"):
        calls.append(f"localization_{language}")
        ps.save_json(paths.script_localized_json(language), {"ok": True, "sections": []})
    from youtube import localizer
    monkeypatch.setattr(localizer, "run_and_save", fake_localizer_run_and_save)

    def fake_scenes(paths):
        calls.append("scenes")
        ps.save_json(paths.scenes_json, [{"scene_id": i} for i in range(1, 4)])
    monkeypatch.setattr(scene_planner, "run_and_save_v2", fake_scenes)

    def fake_scene_sync(paths, language="tr"):
        calls.append(f"scene_sync_{language}")
    monkeypatch.setattr(scene_planner, "sync_localized_narrations", fake_scene_sync)

    def fake_images(paths):
        calls.append("images")
        scenes = ps.load_json(paths.scenes_json)
        for i in range(1, len(scenes) + 1):
            _write_bytes(paths.scene_image(i))
    monkeypatch.setattr(image_gen, "run_and_save", fake_images)

    def fake_tts(paths, language="en"):
        calls.append(f"narration_{language}")
        _write_bytes(paths.narration_mp3_for(language))
    monkeypatch.setattr(tts, "run_and_save", fake_tts)

    def fake_assemble(paths):
        calls.append("video_assembly")
        _write_bytes(paths.video_mp4, content=b"FAKE_VIDEO_BYTES" * 4)
    monkeypatch.setattr(assemble, "run_and_save", fake_assemble)

    def fake_thumbnail(paths):
        calls.append("thumbnail")
        _write_bytes(paths.thumbnail_png, content=b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    monkeypatch.setattr(thumbnail, "run_and_save", fake_thumbnail)

    def fake_metadata(paths, category=None, language="en"):
        calls.append(f"metadata_{language}")
        ps.save_json(paths.metadata_json_for(language), {
            "title": FAKE_TITLE, "description": "d", "tags": ["a"], "category_id": "27",
            "language": language, "privacy_status": "private", "source_count": 1,
        })
    monkeypatch.setattr(metadata, "run_and_save", fake_metadata)

    monkeypatch.setattr(
        quality_check, "run_and_enforce",
        lambda paths, languages=None: quality_check.QualityReport(passed=True, checks={"ok": {"passed": True, "detail": ""}}),
    )
    return calls


@pytest.fixture
def telegram_configured(monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tb, "_probe_duration_seconds", lambda path: 137.0)


@pytest.fixture
def fake_telegram_send(monkeypatch):
    sent = {"photos": [], "videos": []}

    def fake_photo(image_path, caption, keyboard, chat_id=None):
        message_id = 5000 + len(sent["photos"])
        sent["photos"].append({"image_path": image_path, "caption": caption, "keyboard": keyboard,
                                "chat_id": chat_id, "message_id": message_id})
        return message_id

    def fake_video(video_path, caption="", chat_id=None):
        sent["videos"].append({"video_path": video_path})
        return None

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_photo)
    monkeypatch.setattr(tap, "send_video_preview", fake_video)
    return sent


class _FakeYouTube:
    """Stand-in for youtube.uploader - mirrors tests/test_telegram_approval.py's
    own _FakeYouTube exactly, kept local to avoid cross-file coupling."""

    def __init__(self):
        self.video_calls = []
        self.thumbnail_calls = []

    def upload_video(self, video_path, metadata, channel_key=None):
        self.video_calls.append((video_path, dict(metadata), channel_key))
        return {"video_id": "prodVideoId789", "url": "https://www.youtube.com/watch?v=prodVideoId789",
                "channel_key": channel_key or "default"}

    def upload_thumbnail(self, video_id, thumbnail_path, channel_key=None):
        self.thumbnail_calls.append((video_id, thumbnail_path, channel_key))


@pytest.fixture
def fake_youtube(monkeypatch):
    from youtube import uploader
    fake = _FakeYouTube()
    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", fake.upload_thumbnail)
    return fake


def _produce(topic="Architecture Test Topic", **kwargs):
    return pipeline_module.produce_video(topic, languages=kwargs.pop("languages", []), **kwargs)


# ---------------------------------------------------------------------------
# 1/2/5/6 - production assembler/thumbnail paths reach Approval V1 and are
# persisted reliably on the approval's own DB row.
# ---------------------------------------------------------------------------

def test_production_video_and_thumbnail_paths_are_persisted_on_the_approval_row(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send,
):
    paths = _produce()

    row = ta.get_latest_approval_by_slug(paths.slug)
    assert row is not None
    assert row["status"] == tap.STATUS_READY
    # The REAL production assembler/thumbnail output - never
    # youtube/final_video_assembly_demo.py's debug_audit/final_video_v2/
    # convention, which this pipeline-sourced approval never touches.
    assert row["video_path"] == str(paths.video_mp4)
    assert row["thumbnail_path"] == str(paths.thumbnail_png)
    assert "debug_audit" not in row["video_path"]
    assert "debug_audit" not in row["thumbnail_path"]


def test_production_pipeline_sends_the_real_thumbnail_file_to_telegram(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send,
):
    paths = _produce()
    assert len(fake_telegram_send["photos"]) == 1
    assert fake_telegram_send["photos"][0]["image_path"] == paths.thumbnail_png


# ---------------------------------------------------------------------------
# 3/4 - the legacy telegram_bot.py notification is gone; exactly one
# approval message (Telegram Approval V1) is ever created for a video.
# ---------------------------------------------------------------------------

def test_pipeline_never_calls_the_legacy_telegram_bot_notification(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send, monkeypatch,
):
    def _must_not_be_called(*a, **k):
        raise AssertionError("telegram_bot.send_for_approval must NOT be called from produce_video() anymore")
    monkeypatch.setattr(tb, "send_for_approval", _must_not_be_called)

    _produce()  # would raise via the guard above if the old path were still wired in


def test_only_one_approval_message_is_created_per_video(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send,
):
    paths = _produce()

    approvals = [row for row in ta.list_approvals(limit=50) if row["slug"] == paths.slug]
    assert len(approvals) == 1
    assert len(fake_telegram_send["photos"]) == 1


# ---------------------------------------------------------------------------
# 7 - English-only mode: an explicit override, never a global config change.
# ---------------------------------------------------------------------------

def test_english_only_mode_skips_localization_without_touching_global_config(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send, monkeypatch,
):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    # Default behavior (languages=None) still honors the configured "tr" -
    # proves the override doesn't change what "default" means.
    default_paths = pipeline_module.produce_video("Default Localization Topic")
    assert default_paths.script_localized_json("tr").exists()
    assert settings.LANGUAGE_LOCALIZATIONS == ["tr"]  # untouched by the default call itself

    # English-only override: same global settings, but this ONE call skips "tr".
    english_only_paths = pipeline_module.produce_video("English Only Topic", languages=[])
    assert not english_only_paths.script_localized_json("tr").exists()
    assert not english_only_paths.narration_mp3_for("tr").exists()
    assert not english_only_paths.metadata_json_for("tr").exists()

    # The global config itself was never mutated by the override.
    assert settings.LANGUAGE_LOCALIZATIONS == ["tr"]


# ---------------------------------------------------------------------------
# 8/9/10 - approval state machine, exercised against a production-pipeline-
# sourced approval specifically (the state machine's own exhaustive test
# matrix already lives in tests/test_telegram_approval.py - this confirms
# it behaves identically when video_path/thumbnail_path come from the
# production pipeline instead of the manual demo CLI).
# ---------------------------------------------------------------------------

def test_no_youtube_upload_happens_before_approve(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send, fake_youtube,
):
    _produce()
    assert fake_youtube.video_calls == []
    assert fake_youtube.thumbnail_calls == []


def test_approve_uploads_using_the_persisted_production_paths(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send, fake_youtube,
):
    paths = _produce()
    row = ta.get_latest_approval_by_slug(paths.slug)

    tap.approve(row["id"])
    result = tap.process_approval(row["id"])

    assert result["video_id"] == "prodVideoId789"
    assert fake_youtube.video_calls[0][0] == paths.video_mp4
    assert fake_youtube.thumbnail_calls[0][1] == paths.thumbnail_png
    uploaded_row = ta.get_approval(row["id"])
    assert uploaded_row["status"] == tap.STATUS_UPLOADED
    assert uploaded_row["thumbnail_status"] == tap.THUMBNAIL_STATUS_SUCCESS


def test_reject_never_uploads(
    isolated_storage, stubbed_pipeline_stages, telegram_configured, fake_telegram_send, fake_youtube,
):
    paths = _produce()
    row = ta.get_latest_approval_by_slug(paths.slug)

    tap.reject(row["id"], reason="not this one")

    rejected_row = ta.get_approval(row["id"])
    assert rejected_row["status"] == tap.STATUS_REJECTED
    assert fake_youtube.video_calls == []
    assert fake_youtube.thumbnail_calls == []
    with pytest.raises(tap.ApprovalStateError):
        tap.process_approval(row["id"])  # rejected -> uploading is not an allowed transition
