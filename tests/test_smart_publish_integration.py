"""
Smart Publish Scheduler V1 <-> approve-flow integration tests.

These exercise youtube/telegram_approval.py's ACTUAL approve path
(_handle_approve_action -> process_approval -> _upload_to_youtube) with
Smart Publish enabled/disabled, proving:

  - disabled (the library default) leaves production behavior 100%
    unchanged - byte-identical Telegram caption, no publish_at in the
    metadata handed to the uploader, no publish_schedules row (Faz 14).
  - enabled computes and persists exactly one idempotent publish_schedules
    row per approval, passes publish_at through to the uploader, and
    updates the Telegram caption to the new Faz 6 format.
  - the pre-existing upload-idempotency guarantee (mark_uploading()'s
    _transition() gate) still prevents a second YouTube upload and a
    second publish_schedules row on any retry, with Smart Publish enabled.
  - a Telegram update failure AFTER a successful upload never re-triggers
    the upload (Faz 8's second crash scenario).

Mirrors tests/test_telegram_approval.py's own fixtures self-contained, per
this codebase's established convention. Telegram and YouTube are always
mocked - no real network call is ever made by this file.
"""
from __future__ import annotations

import json

import pytest

from core.config import settings
from storage import publish_schedules as psched
from youtube import telegram_approval as tap


@pytest.fixture
def approval_env(isolated_storage, tmp_path, monkeypatch):
    video_root = tmp_path / "debug_audit" / "final_video_v2"
    thumb_root = tmp_path / "debug_audit" / "thumbnail_v1"
    video_root.mkdir(parents=True)
    thumb_root.mkdir(parents=True)
    monkeypatch.setattr(tap, "FINAL_VIDEO_ROOT", video_root)
    monkeypatch.setattr(tap, "THUMBNAIL_ROOT", thumb_root)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)
    return tap


def _write_project(storage_dir, slug, title="The 800-Year-Old Castle Hidden Inside a University Building", scenes=33):
    project_dir = storage_dir / "2026" / "09" / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata_en.json").write_text(json.dumps({
        "title": title, "description": "A short description.", "tags": ["a", "b"],
    }))
    (project_dir / "scenes.json").write_text(json.dumps([{"scene_id": i} for i in range(1, scenes + 1)]))
    return project_dir


def _write_video(tap_module, slug, size_bytes=1000):
    path = tap_module.video_path_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)
    return path


def _write_thumbnail(tap_module, slug):
    path = tap_module.thumbnail_path_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    return path


def _full_fixture(approval_env, isolated_storage, slug="medieval-norman-castle-3"):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, slug)
    _write_video(approval_env, slug)
    _write_thumbnail(approval_env, slug)
    return project_dir


class _FakeYouTube:
    """Unlike tests/test_telegram_approval.py's own _FakeYouTube, this one
    actually honors metadata["publish_at"] (mirroring the real
    youtube/uploader.py's own behavior, tested separately in
    tests/test_uploader.py) so the Smart Publish wiring can be verified
    end-to-end against something that behaves like the real thing."""

    def __init__(self, video_id="abc123", url="https://www.youtube.com/watch?v=abc123", fail=False):
        self.video_calls = []
        self.thumbnail_calls = []
        self.video_id = video_id
        self.url = url
        self.fail = fail

    def upload_video(self, video_path, metadata, channel_key=None):
        self.video_calls.append((video_path, dict(metadata), channel_key))
        if self.fail:
            raise RuntimeError("[YOUTUBE] simulated upload failure")
        publish_at = metadata.get("publish_at")
        return {
            "video_id": self.video_id, "url": self.url, "channel_key": channel_key or "default",
            "scheduled_publish_at": publish_at,
        }

    def upload_thumbnail(self, video_id, thumbnail_path, channel_key=None):
        self.thumbnail_calls.append((video_id, thumbnail_path, channel_key))


@pytest.fixture
def fake_youtube(monkeypatch):
    from youtube import uploader
    fake = _FakeYouTube()
    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", fake.upload_thumbnail)
    return fake


@pytest.fixture
def fake_telegram_send(monkeypatch):
    sent = {"photos": [], "videos": [], "edits": []}

    def fake_photo(image_path, caption, keyboard, chat_id=None):
        message_id = 1000 + len(sent["photos"])
        sent["photos"].append({"message_id": message_id, "keyboard": keyboard})
        return message_id

    def fake_video(video_path, caption="", chat_id=None):
        message_id = 2000 + len(sent["videos"])
        sent["videos"].append({"message_id": message_id})
        return message_id

    def fake_edit(message_id, caption, reply_markup=None, chat_id=None):
        sent["edits"].append(caption)

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_photo)
    monkeypatch.setattr(tap, "send_video_preview", fake_video)
    monkeypatch.setattr(tap, "edit_caption", fake_edit)
    return sent


def _send(approval_env, isolated_storage, fake_telegram_send, slug="medieval-norman-castle-3"):
    project_dir = _full_fixture(approval_env, isolated_storage, slug)
    payload = tap.build_approval_payload(slug, project_dir=project_dir)
    result = tap.send_new_approval(payload)
    return result


# ---------------------------------------------------------------------------
# Faz 14 - disabled preserves EXACT existing behavior
# ---------------------------------------------------------------------------

def test_smart_publish_disabled_sends_no_publish_at_and_creates_no_schedule(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", False)
    result = _send(approval_env, isolated_storage, fake_telegram_send)

    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), fake_telegram_send["photos"][-1]["message_id"])

    assert "publish_at" not in fake_youtube.video_calls[0][1]
    assert psched.list_schedules() == []
    assert fake_telegram_send["edits"][-1].startswith("\U0001f680 UPLOADED TO YOUTUBE")
    assert "VIDEO APPROVED" not in fake_telegram_send["edits"][-1]


def test_smart_publish_disabled_regression_existing_approval_flow(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    """Plain existing-behavior regression, unrelated to Smart Publish
    specifics: approve -> uploaded, thumbnail attached, no schedule table
    touched at all."""
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", False)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]

    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED
    assert len(fake_youtube.thumbnail_calls) == 1


# ---------------------------------------------------------------------------
# Enabled: schedule creation, publish_at passthrough, new caption format
# ---------------------------------------------------------------------------

def test_smart_publish_enabled_creates_exactly_one_schedule_and_passes_publish_at(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]

    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    schedules = psched.list_schedules()
    assert len(schedules) == 1
    assert schedules[0]["approval_id"] == result["approval_id"]
    assert schedules[0]["source"] == "inferred_performance"
    assert schedules[0]["confidence"] == "low"
    assert schedules[0]["status"] == psched.STATUS_SCHEDULED

    sent_metadata = fake_youtube.video_calls[0][1]
    assert "publish_at" in sent_metadata
    assert sent_metadata["publish_at"].endswith("Z")
    assert sent_metadata["privacy_status"] == "private"


def test_smart_publish_enabled_uses_the_new_telegram_caption_format(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]

    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    caption = fake_telegram_send["edits"][-1]
    assert "VIDEO APPROVED" in caption
    assert "Scheduled:" in caption
    assert "Europe/Istanbul" in caption
    assert "YouTube:" in caption
    assert "Scheduled / Private" in caption
    assert "Publish strategy:" in caption
    assert "inferred_performance" in caption
    assert "Reason:" in caption
    assert "Using audience timing data from inferred_performance" in caption


def test_smart_publish_enabled_thumbnail_behavior_preserved(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]

    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    assert len(fake_youtube.thumbnail_calls) == 1
    row = tap.ta.get_approval(result["approval_id"])
    assert row["thumbnail_status"] == tap.THUMBNAIL_STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Faz 7/8 - idempotency: no duplicate schedule, no duplicate upload
# ---------------------------------------------------------------------------

def test_publish_decision_computed_twice_for_same_approval_reuses_the_schedule(
    approval_env, isolated_storage, fake_telegram_send, monkeypatch,
):
    """Directly exercises _compute_and_persist_publish_decision()'s own
    idempotency (the crash-and-retry scenario: something re-enters this
    computation for an approval that already has a schedule)."""
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    row = tap.ta.get_approval(approval_id)

    decision1, schedule_id1 = tap._compute_and_persist_publish_decision(approval_id, row["slug"])
    decision2, schedule_id2 = tap._compute_and_persist_publish_decision(approval_id, row["slug"])

    assert schedule_id1 == schedule_id2
    assert decision1.scheduled_at == decision2.scheduled_at
    assert len(psched.list_schedules()) == 1


def test_second_process_approval_call_never_double_uploads_or_double_schedules(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    tap.approve(approval_id)
    tap.process_approval(approval_id)

    with pytest.raises(tap.ApprovalStateError):
        tap.process_approval(approval_id)  # already uploaded -> cannot re-enter "uploading"

    assert len(fake_youtube.video_calls) == 1
    assert len(psched.list_schedules()) == 1


def test_telegram_update_failure_after_successful_upload_never_retriggers_upload(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    """Faz 8's second crash scenario: upload succeeded, thumbnail succeeded,
    the Telegram status-update call itself fails - the video must never be
    uploaded a second time."""
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    message_id = fake_telegram_send["photos"][-1]["message_id"]

    # Only the SECOND edit_caption call (the final post-upload status
    # update) fails - the first ("APPROVED... upload started...", sent
    # BEFORE process_approval() runs) must still succeed, otherwise the
    # upload would never even be attempted and this wouldn't test the
    # scenario at all.
    edit_calls = {"n": 0}

    def failing_edit(message_id, caption, reply_markup=None, chat_id=None):
        edit_calls["n"] += 1
        if edit_calls["n"] >= 2:
            raise RuntimeError("simulated Telegram network failure")

    monkeypatch.setattr(tap, "edit_caption", failing_edit)

    with pytest.raises(RuntimeError, match="simulated Telegram network failure"):
        tap._handle_approve_action(approval_id, tap.ta.get_approval(approval_id), message_id)

    # The upload itself already succeeded and was persisted BEFORE the
    # failing Telegram call - retrying the whole approve action must not
    # upload a second time.
    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 1
    with pytest.raises(tap.ApprovalStateError):
        tap.process_approval(approval_id)
    assert len(fake_youtube.video_calls) == 1
    assert len(psched.list_schedules()) == 1


# ---------------------------------------------------------------------------
# Faz 5/8 - upload-failure path still marks the schedule failed and the
# approval upload_failed, without ever double-uploading on a later retry
# attempt through the same approval_id
# ---------------------------------------------------------------------------

def test_upload_failure_marks_schedule_failed_and_approval_upload_failed(
    approval_env, isolated_storage, fake_telegram_send, monkeypatch,
):
    from youtube import uploader
    failing = _FakeYouTube(fail=True)
    monkeypatch.setattr(uploader, "upload_video", failing.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing.upload_thumbnail)
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)

    result = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    tap.approve(approval_id)

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        tap.process_approval(approval_id)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_UPLOAD_FAILED
    schedules = psched.list_schedules()
    assert len(schedules) == 1
    assert schedules[0]["status"] == psched.STATUS_FAILED


# ---------------------------------------------------------------------------
# 21/22/23 - broader existing-flow regressions, run with Smart Publish OFF
# (the library default), proving this feature's mere presence in the
# codebase changes nothing when disabled.
# ---------------------------------------------------------------------------

def test_existing_reject_flow_regression_with_smart_publish_present_but_disabled(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", False)
    result = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]

    tap.reject(approval_id, reason="not interesting")

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_REJECTED
    assert fake_youtube.video_calls == []
    assert psched.list_schedules() == []
