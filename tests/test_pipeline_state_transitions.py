"""
Telegram approval state machine (spec Round 2 section 4):

    PENDING_APPROVAL -> APPROVED -> upload -> PUBLISHED
    PENDING_APPROVAL -> rejected (with feedback reason stored)
    PENDING_APPROVAL -> regenerate (new video_id produced, old one rejected)
    PUBLISHED -> approve again must NOT trigger a second YouTube upload
"""
import pytest

import pipeline as pipeline_module
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import uploader


def _make_pending_approval_project(topic="State Machine Topic"):
    paths, _ = ps.create_project(topic)
    ps.save_json(paths.metadata_json, {"title": "T", "description": "d", "tags": [], "category_id": "27",
                                        "language": "en", "privacy_status": "private", "source_count": 1})
    paths.video_mp4.write_bytes(b"fake mp4 bytes")
    ps.save_state(paths, PipelineState.PENDING_APPROVAL)
    return paths


def test_approve_uploads_and_reaches_published(isolated_storage, monkeypatch):
    paths = _make_pending_approval_project()

    calls = []

    def fake_upload_video(video_path, metadata, channel_key=None):
        calls.append((video_path, metadata["title"]))
        return {"video_id": "yt123", "url": "https://www.youtube.com/watch?v=yt123"}

    monkeypatch.setattr(uploader, "upload_video", fake_upload_video)

    result = pipeline_module.handle_approve(paths.video_id)

    assert result["url"] == "https://www.youtube.com/watch?v=yt123"
    assert len(calls) == 1
    assert ps.load_state(paths) == PipelineState.PUBLISHED
    row = db.get_video(paths.video_id)
    assert row["youtube_video_id"] == "yt123"
    assert row["youtube_url"] == "https://www.youtube.com/watch?v=yt123"


def test_reject_stores_feedback_reason_and_fails_state(isolated_storage):
    paths = _make_pending_approval_project()

    pipeline_module.handle_reject(paths.video_id, reason="thumbnail looked off-brand")

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "telegram_approval"
    assert "thumbnail looked off-brand" in data["error"]


def test_regenerate_creates_new_video_and_rejects_old_one(isolated_storage, monkeypatch):
    paths = _make_pending_approval_project("Regenerate Me Topic")

    # produce_video will run the full stage plan for the new video; stub every
    # stage so this test stays focused on the state-transition behavior, not
    # on research/script/etc (those have their own tests).
    def fake_produce(topic, category=None, project_id=None, video_id=None, force_new=False, background_music=False):
        new_paths, _ = ps.create_project(topic, category=category, force_new=True)
        ps.save_state(new_paths, PipelineState.PENDING_APPROVAL)
        return new_paths

    monkeypatch.setattr(pipeline_module, "produce_video", fake_produce)

    new_paths = pipeline_module.handle_regenerate(paths.video_id)

    assert new_paths.video_id != paths.video_id
    assert ps.load_state(paths) == PipelineState.FAILED  # old one marked rejected
    old_state_data = ps.load_json(paths.state_json)
    assert "regenerate" in old_state_data["error"]
    assert ps.load_state(new_paths) == PipelineState.PENDING_APPROVAL


def test_approving_an_already_published_video_does_not_reupload(isolated_storage, monkeypatch):
    paths = _make_pending_approval_project("Already Published Topic")

    calls = []

    def fake_upload_video(video_path, metadata, channel_key=None):
        calls.append(1)
        return {"video_id": "yt999", "url": "https://www.youtube.com/watch?v=yt999"}

    monkeypatch.setattr(uploader, "upload_video", fake_upload_video)

    first = pipeline_module.handle_approve(paths.video_id)
    assert len(calls) == 1
    assert first["url"] == "https://www.youtube.com/watch?v=yt999"

    # Simulate a double-tap / duplicate Telegram callback on the same button.
    second = pipeline_module.handle_approve(paths.video_id)

    assert len(calls) == 1  # NOT called again
    assert second["already_published"] is True
    assert second["url"] == "https://www.youtube.com/watch?v=yt999"
    assert ps.load_state(paths) == PipelineState.PUBLISHED  # state untouched, not bounced through APPROVED


def test_uploader_run_and_save_also_refuses_duplicate_directly(isolated_storage, monkeypatch):
    """Defense-in-depth: even calling youtube.uploader directly (bypassing
    pipeline.handle_approve) must refuse to re-upload a PUBLISHED video."""
    paths = _make_pending_approval_project("Direct Uploader Call Topic")
    ps.save_state(paths, PipelineState.PUBLISHED, extra={"youtube_video_id": "ytABC", "youtube_url": "https://www.youtube.com/watch?v=ytABC"})
    db.update_video_youtube(paths.video_id, "ytABC", "https://www.youtube.com/watch?v=ytABC")

    calls = []
    monkeypatch.setattr(uploader, "upload_video", lambda *a, **k: calls.append(1))

    result = uploader.run_and_save(paths)

    assert calls == []
    assert result["already_published"] is True
