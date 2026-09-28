"""
youtube/telegram_approval.py tests - Telegram Approval V1 core logic.

Covers (of the spec's 18 scenarios):
    1  approval creation
    2  approval status = ready_for_approval
    3  approve callback
    4  reject callback
    5  duplicate approve protection
    6  approve after reject
    7  reject after approve
    8  invalid callback
    9  invalid slug
   10  wrong chat_id
   11  wrong message_id
   12  missing Telegram config
   13  YouTube upload failure
   14  successful YouTube upload
   15  upload failure state
   16  thumbnail attachment/path validation
   17  video file missing
plus caption/keyboard content, static no-LLM-call/no-storage-write checks,
and existing telegram_bot.py/youtube.uploader.py behavior untouched.

Telegram and YouTube are always mocked - no real network call is ever made
by this file.
"""
from __future__ import annotations

import ast
import json

import pytest

from core.config import settings
from youtube import telegram_approval as tap


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

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
    # Duration probing shells out to ffprobe on the real video bytes - fix
    # it to a deterministic value so tests don't depend on ffmpeg/ffprobe
    # being installed or on real video content.
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)  # -> 09:39
    return tap


def _write_project(storage_dir, slug, title="The Wow! Signal: Astronomy's Greatest Unsolved Mystery", scenes=33):
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


def _full_fixture(approval_env, isolated_storage, slug="the-wow-signal-6"):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, slug)
    _write_video(approval_env, slug)
    _write_thumbnail(approval_env, slug)
    return project_dir


class _FakeYouTube:
    """Stand-in for youtube.uploader - records calls, never touches the
    network. Installed via monkeypatch.setattr(youtube.uploader, ...)."""

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
        return {"video_id": self.video_id, "url": self.url, "channel_key": channel_key or "default"}

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
    """Mocks the two network-hitting send functions; records calls."""
    sent = {"photos": [], "videos": []}

    def fake_photo(image_path, caption, keyboard, chat_id=None):
        message_id = 1000 + len(sent["photos"])
        sent["photos"].append({"image_path": image_path, "caption": caption, "keyboard": keyboard,
                                "chat_id": chat_id, "message_id": message_id})
        return message_id

    def fake_video(video_path, caption="", chat_id=None):
        message_id = 2000 + len(sent["videos"])
        sent["videos"].append({"video_path": video_path, "caption": caption, "chat_id": chat_id,
                                "message_id": message_id})
        return message_id

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_photo)
    monkeypatch.setattr(tap, "send_video_preview", fake_video)
    return sent


@pytest.fixture
def fake_edit_text(monkeypatch):
    """Named after the concept it fakes (editing an approval message), not
    the specific Telegram endpoint - approval messages are sendPhoto-based,
    so the real code now calls tap.edit_caption() (editMessageCaption), not
    tb.edit_text() (editMessageText, which fails on a photo message)."""
    edits = []
    monkeypatch.setattr(tap, "edit_caption", lambda message_id, caption, reply_markup=None, chat_id=None: edits.append((message_id, caption)))
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": None)
    return edits


def _send(approval_env, isolated_storage, fake_telegram_send, slug="the-wow-signal-6", **kwargs):
    project_dir = _full_fixture(approval_env, isolated_storage, slug)
    payload = tap.build_approval_payload(slug, project_dir=project_dir, **kwargs)
    result = tap.send_new_approval(payload)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    return result, message_id


def _callback(action, slug, chat_id=999, message_id=None, cbq_id="cbq1"):
    return {
        "id": cbq_id,
        "data": f"approval:{action}:{slug}",
        "message": {"message_id": message_id, "chat": {"id": chat_id}},
    }


# ---------------------------------------------------------------------------
# caption / keyboard content
# ---------------------------------------------------------------------------

def test_build_caption_contains_required_fields():
    caption = tap.build_caption("My Title", "09:39", 33, "en", video_sendable=True)
    assert "My Title" in caption
    assert "09:39" in caption
    assert "33" in caption
    assert "English" in caption
    assert "Private" in caption
    assert "WAITING FOR APPROVAL" in caption
    assert "Video preview: unavailable" not in caption


def test_build_caption_notes_unavailable_preview_when_not_sendable():
    caption = tap.build_caption("Title", "01:00", 10, "en", video_sendable=False)
    assert "Video preview: unavailable" in caption


def test_build_keyboard_has_approve_and_reject_with_slug_callback_data():
    kb = tap.build_keyboard("the-wow-signal-6")
    buttons = kb["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "approval:approve:the-wow-signal-6"
    assert buttons[1]["callback_data"] == "approval:reject:the-wow-signal-6"


# ---------------------------------------------------------------------------
# 2026-09 production incident: sendPhoto returned HTTP 400
# {"ok":false,"error_code":400,"description":"Bad Request: BUTTON_DATA_INVALID"}
# for approval_id=3 (slug=medieval-norman-castle-discovered-hidden-inside-3).
# Root cause: build_keyboard(slug) embeds the FULL slug in callback_data,
# and Telegram's callback_data has a hard 64-BYTE limit - for this real
# slug, "approval:approve:<slug>" is 66 bytes and "approval:reject:<slug>"
# is 65, both over the limit, so Telegram rejected the whole sendPhoto
# call outright (the message was never even created). These tests pin the
# fix: build_keyboard_by_id() (short, deterministic, ID-based) is what
# every real send now actually transmits, and handle_approval_callback()
# accepts both the new ID format and the old slug format (for any
# already-sent message whose buttons still carry it).
# ---------------------------------------------------------------------------

_REAL_INCIDENT_SLUG = "medieval-norman-castle-discovered-hidden-inside-3"


def test_legacy_build_keyboard_exceeds_telegram_limit_for_the_real_incident_slug():
    """Confirms the actual reported failure mode against the real slug from
    the incident, not a hypothetical - if this ever stops being true (e.g.
    Telegram's limit changes), the test itself should be revisited."""
    kb = tap.build_keyboard(_REAL_INCIDENT_SLUG)
    lengths = [len(b["callback_data"].encode("utf-8")) for b in kb["inline_keyboard"][0]]
    assert any(length > 64 for length in lengths), (
        f"expected the legacy slug-based keyboard to exceed Telegram's 64-byte "
        f"callback_data limit for this real slug, got lengths={lengths}"
    )


def test_build_keyboard_by_id_produces_short_deterministic_callback_data():
    kb = tap.build_keyboard_by_id(3)
    buttons = kb["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "approval:approve:3"
    assert buttons[1]["callback_data"] == "approval:reject:3"
    for b in buttons:
        assert len(b["callback_data"].encode("utf-8")) <= 64


def test_build_keyboard_by_id_stays_within_telegram_limit_for_large_ids():
    kb = tap.build_keyboard_by_id(999_999_999_999)
    for b in kb["inline_keyboard"][0]:
        assert len(b["callback_data"].encode("utf-8")) <= 64


def test_send_new_approval_transmits_id_based_keyboard_within_limit_for_long_slug(
    approval_env, isolated_storage, fake_telegram_send,
):
    """The actual fix, exercised end-to-end: even for the exact slug that
    triggered BUTTON_DATA_INVALID, the keyboard send_new_approval() hands
    to send_photo_with_keyboard() must be the short ID-based one, never
    the slug-based preview build_approval_payload() initially built."""
    project_dir = _write_project(isolated_storage.STORAGE_DIR, _REAL_INCIDENT_SLUG)
    _write_video(approval_env, _REAL_INCIDENT_SLUG)
    _write_thumbnail(approval_env, _REAL_INCIDENT_SLUG)
    payload = tap.build_approval_payload(_REAL_INCIDENT_SLUG, project_dir=project_dir)

    result = tap.send_new_approval(payload)

    sent_keyboard = fake_telegram_send["photos"][-1]["keyboard"]
    buttons = sent_keyboard["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == f"approval:approve:{result['approval_id']}"
    assert buttons[1]["callback_data"] == f"approval:reject:{result['approval_id']}"
    for b in buttons:
        assert len(b["callback_data"].encode("utf-8")) <= 64
        assert _REAL_INCIDENT_SLUG not in b["callback_data"]


def test_handle_approval_callback_parses_new_id_based_format_and_approves(
    approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube,
):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send, "id-format-approve-slug")
    approval_id = result["approval_id"]
    callback = {
        "id": "cbq1", "data": f"approval:approve:{approval_id}",
        "message": {"message_id": message_id, "chat": {"id": 999}},
    }

    tap.handle_approval_callback(callback)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 1


def test_handle_approval_callback_parses_new_id_based_format_and_rejects(
    approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube,
):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send, "id-format-reject-slug")
    approval_id = result["approval_id"]
    callback = {
        "id": "cbq1", "data": f"approval:reject:{approval_id}",
        "message": {"message_id": message_id, "chat": {"id": 999}},
    }

    tap.handle_approval_callback(callback)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_REJECTED
    assert fake_youtube.video_calls == []


def test_handle_approval_callback_still_parses_legacy_slug_based_format(
    approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube,
):
    """A real send now transmits an ID-based keyboard, but the listener must
    still honor a manually-crafted or pre-fix legacy slug-based
    callback_data for the SAME approval - full backward compatibility, not
    just for old already-sent messages but for the identifier format
    itself."""
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send, "legacy-format-slug")
    callback = {
        "id": "cbq1", "data": "approval:approve:legacy-format-slug",
        "message": {"message_id": message_id, "chat": {"id": 999}},
    }

    tap.handle_approval_callback(callback)

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED


def test_handle_approval_callback_id_format_rejects_wrong_chat_or_message(
    approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube,
):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send, "id-format-stale-slug")
    approval_id = result["approval_id"]
    stale_callback = {
        "id": "cbq1", "data": f"approval:approve:{approval_id}",
        "message": {"message_id": 999999, "chat": {"id": 999}},
    }

    tap.handle_approval_callback(stale_callback)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_READY  # unchanged - never approved
    assert fake_youtube.video_calls == []


# ---------------------------------------------------------------------------
# 16 / 17 - thumbnail / video path validation
# ---------------------------------------------------------------------------

def test_build_payload_missing_video_raises(approval_env, isolated_storage):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, "no-video-slug")
    _write_thumbnail(approval_env, "no-video-slug")
    with pytest.raises(tap.TelegramApprovalError, match="final video not found"):
        tap.build_approval_payload("no-video-slug", project_dir=project_dir)


def test_build_payload_missing_thumbnail_raises(approval_env, isolated_storage):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, "no-thumb-slug")
    _write_video(approval_env, "no-thumb-slug")
    with pytest.raises(tap.TelegramApprovalError, match="thumbnail not found"):
        tap.build_approval_payload("no-thumb-slug", project_dir=project_dir)


def test_build_payload_missing_metadata_raises(approval_env, isolated_storage):
    project_dir = isolated_storage.STORAGE_DIR / "2026" / "09" / "no-meta-slug"
    project_dir.mkdir(parents=True)
    _write_video(approval_env, "no-meta-slug")
    _write_thumbnail(approval_env, "no-meta-slug")
    with pytest.raises(tap.TelegramApprovalError, match="metadata_en.json"):
        tap.build_approval_payload("no-meta-slug", project_dir=project_dir)


def test_build_payload_missing_project_dir_raises(approval_env, isolated_storage):
    with pytest.raises(tap.TelegramApprovalError, match="no project directory"):
        tap.build_approval_payload("completely-unknown-slug")


def test_video_sendable_false_when_over_size_limit(approval_env, isolated_storage, monkeypatch):
    monkeypatch.setattr(tap.tb, "MAX_UPLOAD_BYTES", 10)
    project_dir = _write_project(isolated_storage.STORAGE_DIR, "big-video-slug")
    _write_video(approval_env, "big-video-slug", size_bytes=1000)
    _write_thumbnail(approval_env, "big-video-slug")
    payload = tap.build_approval_payload("big-video-slug", project_dir=project_dir)
    assert payload.video_sendable is False
    assert "unavailable" in payload.caption


# ---------------------------------------------------------------------------
# 1 / 2 - approval creation / ready_for_approval
# ---------------------------------------------------------------------------

def test_send_new_approval_creates_row_in_ready_for_approval(approval_env, isolated_storage, fake_telegram_send):
    result, _ = _send(approval_env, isolated_storage, fake_telegram_send)
    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_READY
    assert row["slug"] == "the-wow-signal-6"
    assert row["telegram_message_id"] == result["message_id"]


def test_send_new_approval_sends_photo_with_thumbnail_and_keyboard(approval_env, isolated_storage, fake_telegram_send):
    _send(approval_env, isolated_storage, fake_telegram_send)
    sent_photo = fake_telegram_send["photos"][0]
    assert sent_photo["image_path"] == tap.thumbnail_path_for("the-wow-signal-6")
    assert "APPROVE" in json.dumps(sent_photo["keyboard"])
    assert "REJECT" in json.dumps(sent_photo["keyboard"])


def test_send_new_approval_also_sends_video_preview_when_sendable(approval_env, isolated_storage, fake_telegram_send):
    _send(approval_env, isolated_storage, fake_telegram_send)
    assert len(fake_telegram_send["videos"]) == 1
    assert fake_telegram_send["videos"][0]["video_path"] == tap.video_path_for("the-wow-signal-6")


def test_send_new_approval_skips_video_preview_when_not_sendable(approval_env, isolated_storage, fake_telegram_send, monkeypatch):
    monkeypatch.setattr(tap.tb, "MAX_UPLOAD_BYTES", 10)
    _send(approval_env, isolated_storage, fake_telegram_send)
    assert len(fake_telegram_send["videos"]) == 0


# ---------------------------------------------------------------------------
# 12 - missing Telegram config
# ---------------------------------------------------------------------------

def test_send_new_approval_raises_on_missing_bot_token(approval_env, isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    project_dir = _full_fixture(approval_env, isolated_storage, "token-missing-slug")
    payload = tap.build_approval_payload("token-missing-slug", project_dir=project_dir)
    with pytest.raises(tap.TelegramApprovalError, match="Telegram bot token missing"):
        tap.send_new_approval(payload)


def test_send_new_approval_raises_on_missing_chat_id(approval_env, isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "")
    project_dir = _full_fixture(approval_env, isolated_storage, "chatid-missing-slug")
    payload = tap.build_approval_payload("chatid-missing-slug", project_dir=project_dir)
    with pytest.raises(tap.TelegramApprovalError, match="Telegram chat id missing"):
        tap.send_new_approval(payload)


def test_missing_config_never_creates_a_db_row(approval_env, isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    project_dir = _full_fixture(approval_env, isolated_storage, "no-row-slug")
    payload = tap.build_approval_payload("no-row-slug", project_dir=project_dir)
    with pytest.raises(tap.TelegramApprovalError):
        tap.send_new_approval(payload)
    assert tap.ta.get_latest_approval_by_slug("no-row-slug") is None


# ---------------------------------------------------------------------------
# 3 / 14 - approve callback / successful YouTube upload
# ---------------------------------------------------------------------------

def test_approve_callback_uploads_and_marks_uploaded(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    cbq = _callback("approve", "the-wow-signal-6", chat_id=999, message_id=message_id)

    tap.handle_approval_callback(cbq)

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED
    assert row["youtube_video_id"] == "abc123"
    assert row["youtube_url"] == "https://www.youtube.com/watch?v=abc123"
    assert len(fake_youtube.video_calls) == 1
    assert len(fake_youtube.thumbnail_calls) == 1


def test_approve_callback_forces_privacy_private(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_UPLOAD_PRIVACY", "public")  # even if misconfigured...
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    cbq = _callback("approve", "the-wow-signal-6", chat_id=999, message_id=message_id)

    tap.handle_approval_callback(cbq)

    _, metadata, _ = fake_youtube.video_calls[0]
    assert metadata["privacy_status"] == "private"  # ...V1 always forces private


def test_approve_callback_edits_message_to_uploaded_notice(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    _, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    texts = [t for _, t in fake_edit_text]
    assert any("APPROVED" in t for t in texts)
    assert any("UPLOADED TO YOUTUBE" in t and "abc123" in t for t in texts)


# ---------------------------------------------------------------------------
# 4 - reject callback
# ---------------------------------------------------------------------------

def test_reject_callback_marks_rejected_and_never_uploads(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("reject", "the-wow-signal-6", message_id=message_id))

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_REJECTED
    assert fake_youtube.video_calls == []
    assert any("REJECTED" in t for _, t in fake_edit_text)
    assert any("No YouTube upload was performed" in t for _, t in fake_edit_text)


# ---------------------------------------------------------------------------
# 5 - duplicate approve protection
# ---------------------------------------------------------------------------

def test_duplicate_approve_click_never_uploads_twice(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    cbq = _callback("approve", "the-wow-signal-6", message_id=message_id)

    tap.handle_approval_callback(cbq)
    tap.handle_approval_callback(cbq)
    tap.handle_approval_callback(cbq)

    assert len(fake_youtube.video_calls) == 1
    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED


def test_approve_state_error_raised_directly_on_second_call(approval_env, isolated_storage, fake_telegram_send, fake_youtube):
    result, _ = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.approve(result["approval_id"])
    with pytest.raises(tap.ApprovalStateError):
        tap.approve(result["approval_id"])


# ---------------------------------------------------------------------------
# 6 / 7 - approve after reject / reject after approve
# ---------------------------------------------------------------------------

def test_approve_after_reject_is_rejected(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("reject", "the-wow-signal-6", message_id=message_id))

    fake_edit_text.clear()
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_REJECTED  # unchanged
    assert fake_youtube.video_calls == []
    assert any("Already rejected" in t for _, t in fake_edit_text)


def test_reject_after_approve_is_rejected(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    fake_edit_text.clear()
    tap.handle_approval_callback(_callback("reject", "the-wow-signal-6", message_id=message_id))

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED  # unchanged - still the successful upload
    assert any("Already approved" in t for _, t in fake_edit_text)


# ---------------------------------------------------------------------------
# 8 - invalid callback
# ---------------------------------------------------------------------------

def test_invalid_callback_data_shape_is_ignored_safely(approval_env, isolated_storage, fake_edit_text, monkeypatch):
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))
    tap.handle_approval_callback({"id": "cbq1", "data": "garbage", "message": {"message_id": 1, "chat": {"id": 1}}})
    assert answered == ["Invalid button data"]


def test_invalid_callback_action_is_ignored_safely(approval_env, isolated_storage, monkeypatch):
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))
    tap.handle_approval_callback({"id": "cbq1", "data": "approval:frobnicate:some-slug", "message": {"message_id": 1, "chat": {"id": 1}}})
    assert answered == ["Invalid button data"]


def test_callback_missing_data_key_does_not_crash(approval_env, isolated_storage, monkeypatch):
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": None)
    tap.handle_approval_callback({"id": "cbq1", "message": {"message_id": 1, "chat": {"id": 1}}})  # no "data" at all - must not raise


# ---------------------------------------------------------------------------
# 9 - invalid slug
# ---------------------------------------------------------------------------

def test_callback_for_unknown_slug_is_rejected(approval_env, isolated_storage, monkeypatch):
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))
    tap.handle_approval_callback(_callback("approve", "never-created-slug"))
    assert answered == ["Unknown project"]


# ---------------------------------------------------------------------------
# 10 / 11 - wrong chat_id / wrong message_id
# ---------------------------------------------------------------------------

def test_callback_wrong_chat_id_is_rejected(approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))

    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", chat_id=666, message_id=message_id))

    assert answered == ["This approval message is no longer valid"]
    assert tap.ta.get_approval(result["approval_id"])["status"] == tap.STATUS_READY
    assert fake_youtube.video_calls == []


def test_callback_wrong_message_id_is_rejected(approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))

    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", chat_id=999, message_id=message_id + 12345))

    assert answered == ["This approval message is no longer valid"]
    assert tap.ta.get_approval(result["approval_id"])["status"] == tap.STATUS_READY
    assert fake_youtube.video_calls == []


def test_callback_from_a_superseded_older_message_is_rejected(approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch):
    """A second approval sent for the same slug (e.g. after a restart)
    supersedes the first - a button press on the FIRST (now stale)
    message must not be able to act on the row anymore."""
    project_dir = _full_fixture(approval_env, isolated_storage, "resend-slug")
    payload = tap.build_approval_payload("resend-slug", project_dir=project_dir)

    first = tap.send_new_approval(payload)
    first_message_id = fake_telegram_send["photos"][0]["message_id"]

    second = tap.send_new_approval(payload)
    assert second["approval_id"] != first["approval_id"]

    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))
    tap.handle_approval_callback(_callback("approve", "resend-slug", chat_id=999, message_id=first_message_id))

    assert answered == ["This approval message is no longer valid"]
    assert fake_youtube.video_calls == []


# ---------------------------------------------------------------------------
# 13 / 15 - YouTube upload failure / upload failure state
# ---------------------------------------------------------------------------

def test_upload_failure_marks_upload_failed_and_never_crashes(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, monkeypatch):
    from youtube import uploader
    failing = _FakeYouTube(fail=True)
    monkeypatch.setattr(uploader, "upload_video", failing.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing.upload_thumbnail)

    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))  # must not raise

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOAD_FAILED
    assert row["upload_error"] and "simulated upload failure" in row["upload_error"]
    assert any("failed" in t.lower() for _, t in fake_edit_text)


def test_process_approval_reraises_on_upload_failure_for_direct_callers(approval_env, isolated_storage, fake_telegram_send, monkeypatch):
    from youtube import uploader
    failing = _FakeYouTube(fail=True)
    monkeypatch.setattr(uploader, "upload_video", failing.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing.upload_thumbnail)

    result, _ = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.approve(result["approval_id"])
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        tap.process_approval(result["approval_id"])
    assert tap.ta.get_approval(result["approval_id"])["status"] == tap.STATUS_UPLOAD_FAILED


def test_thumbnail_upload_failure_does_not_fail_the_whole_approve(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, monkeypatch):
    """A thumbnail-set failure on YouTube's side must not un-publish the
    video or flip the approval back to a failed state - matches
    youtube/uploader.py's own run_and_save() precedent."""
    from youtube import uploader
    fake = _FakeYouTube()

    def failing_thumbnail(video_id, thumbnail_path, channel_key=None):
        raise RuntimeError("thumbnail API hiccup")

    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing_thumbnail)

    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED
    assert row["youtube_video_id"] == "abc123"


def test_thumbnail_upload_failure_persists_thumbnail_status_failed(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, monkeypatch):
    """thumbnail_status/thumbnail_error (storage/telegram_approvals.py's
    migrated columns) must be persisted separately from the video's own
    upload outcome - a failed thumbnail is tracked, never conflated with
    upload_error or STATUS_UPLOAD_FAILED."""
    from youtube import uploader
    fake = _FakeYouTube()

    def failing_thumbnail(video_id, thumbnail_path, channel_key=None):
        raise RuntimeError("thumbnail API hiccup")

    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing_thumbnail)

    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED  # video upload is NOT considered failed
    assert row["thumbnail_status"] == "failed"
    assert row["thumbnail_error"] and "thumbnail API hiccup" in row["thumbnail_error"]
    assert row["upload_error"] is None  # the thumbnail failure is not the video's upload_error


def test_thumbnail_upload_failure_shown_in_final_telegram_message(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, monkeypatch):
    from youtube import uploader
    fake = _FakeYouTube()

    def failing_thumbnail(video_id, thumbnail_path, channel_key=None):
        raise RuntimeError("thumbnail API hiccup")

    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing_thumbnail)

    _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][0]["message_id"]
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    texts = [t for _, t in fake_edit_text]
    assert any("Thumbnail: FAILED" in t and "thumbnail API hiccup" in t for t in texts)


def test_thumbnail_upload_success_persists_thumbnail_status_success(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", "the-wow-signal-6", message_id=message_id))

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED
    assert row["thumbnail_status"] == "success"
    assert row["thumbnail_error"] is None

    texts = [t for _, t in fake_edit_text]
    assert any("Thumbnail: OK" in t for t in texts)


# ---------------------------------------------------------------------------
# edit_caption() itself - editMessageText -> editMessageCaption fix
# ---------------------------------------------------------------------------

class _RecordingHttpClient:
    """Fake httpx.Client that records exactly what was POSTed and returns a
    canned success response - used to assert edit_caption() talks to the
    right Telegram endpoint with the right payload shape, without any real
    network call."""
    calls: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, **kwargs):
        _RecordingHttpClient.calls.append((url, kwargs.get("json")))
        import httpx as _httpx
        request = _httpx.Request("POST", url)
        return _httpx.Response(200, request=request, json={"ok": True, "result": {}})


def test_edit_caption_calls_editmessagecaption_not_editmessagetext(approval_env, isolated_storage, monkeypatch):
    """The exact reported bug: approval messages are sendPhoto-based, so
    updating one must hit editMessageCaption, never editMessageText (which
    fails with "Bad Request: there is no text in the message to edit" on a
    photo message)."""
    _RecordingHttpClient.calls = []
    monkeypatch.setattr(tap.httpx, "Client", _RecordingHttpClient)

    tap.edit_caption(1234, "hello caption")

    assert len(_RecordingHttpClient.calls) == 1
    url, payload = _RecordingHttpClient.calls[0]
    assert url.endswith("/editMessageCaption")
    assert "editMessageText" not in url
    assert payload["caption"] == "hello caption"
    assert "text" not in payload
    assert payload["message_id"] == 1234


def test_edit_caption_never_raises_on_http_failure(approval_env, isolated_storage, monkeypatch, caplog):
    """Mirrors telegram_bot.edit_text()'s best-effort contract: a Telegram
    hiccup while updating a status message must be logged, not raised -
    callers (handle_approval_callback's own except branches) must never
    crash because the follow-up status edit itself failed."""
    import logging

    class _FailingClient(_RecordingHttpClient):
        def post(self, url, **kwargs):
            import httpx as _httpx
            request = _httpx.Request("POST", url)
            return _httpx.Response(400, request=request, text="Bad Request: there is no text in the message to edit")

    monkeypatch.setattr(tap.httpx, "Client", _FailingClient)

    with caplog.at_level(logging.WARNING, logger="ycf.telegram_approval"):
        tap.edit_caption(1234, "hello caption")  # must not raise

    assert "editMessageCaption failed" in caplog.text


def test_edit_caption_never_leaks_token_on_http_error(approval_env, isolated_storage, monkeypatch, caplog):
    """Mirrors telegram_bot.py's own token-redaction tests: a raised httpx
    exception embeds the full request URL, which for every Telegram Bot API
    call contains the live bot token - edit_caption() must redact it (via
    tb._redact_token, exactly as telegram_bot.edit_text() does) before it
    ever reaches a log line."""
    import logging

    secret_token = "123456:SECRET-TOKEN-VALUE"
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", secret_token)

    class _HttpErrorClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **kwargs):
            import httpx as _httpx
            request = _httpx.Request("POST", url)
            response = _httpx.Response(401, request=request, text="Unauthorized")
            raise _httpx.HTTPStatusError("error 401", request=request, response=response)

    monkeypatch.setattr(tap.httpx, "Client", _HttpErrorClient)

    with caplog.at_level(logging.WARNING, logger="ycf.telegram_approval"):
        tap.edit_caption(1234, "hello caption")  # must not raise

    assert secret_token not in caplog.text


# ---------------------------------------------------------------------------
# state machine unit checks
# ---------------------------------------------------------------------------

def test_reject_unknown_approval_id_raises_not_found(isolated_storage):
    with pytest.raises(tap.ApprovalNotFoundError):
        tap.approve(999999)


def test_reject_records_reason(approval_env, isolated_storage, fake_telegram_send):
    result, _ = _send(approval_env, isolated_storage, fake_telegram_send)
    tap.reject(result["approval_id"], reason="looks off-topic")
    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_REJECTED
    assert row["rejection_reason"] == "looks off-topic"


def test_draft_cannot_be_approved_directly(isolated_storage):
    approval_id = tap.ta.create_approval("draft-slug", "Draft Title")  # status=draft, never sent
    with pytest.raises(tap.ApprovalStateError):
        tap.approve(approval_id)


# ---------------------------------------------------------------------------
# static checks: no LLM/research/TTS calls, existing files untouched
# ---------------------------------------------------------------------------

def _imported_top_level_modules(path) -> set[str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_module_never_imports_llm_client_researcher_or_tts():
    banned = {"llm_client", "researcher", "tts"}
    imported = _imported_top_level_modules(tap.__file__)
    assert not (imported & banned), f"unexpected import(s): {imported & banned}"


def test_module_does_not_reimplement_youtube_oauth():
    source = open(tap.__file__, encoding="utf-8").read()
    assert "InstalledAppFlow" not in source
    assert "googleapiclient" not in source
    assert "from youtube import uploader" in source or "import youtube.uploader" in source


def test_module_reuses_telegram_bot_helpers_rather_than_duplicating():
    source = open(tap.__file__, encoding="utf-8").read()
    assert "import telegram_bot" in source
    for helper in ("_url", "_redact_token", "_probe_duration_seconds", "_format_duration", "MAX_UPLOAD_BYTES", "_answer_callback"):
        assert f"tb.{helper}" in source, f"expected reuse of telegram_bot.{helper}"
    # edit_text() is deliberately NOT reused: approval messages are
    # sendPhoto-based, so their caption must be updated via
    # editMessageCaption (this module's own edit_caption()), not
    # telegram_bot.edit_text()'s editMessageText - see edit_caption()'s
    # docstring. telegram_bot.py's edit_text() itself is untouched.
    assert "def edit_caption(" in source
    assert "tb.edit_text" not in source


def test_send_new_approval_never_writes_into_the_project_directory(approval_env, isolated_storage, fake_telegram_send):
    """The approval row legitimately lives in storage/factory.db (same
    sqlite convention as storage/db.py's own `videos` table) - what must
    NEVER happen is a write into the per-project artifact directory itself
    (research.json/script_en.json/scenes.json/metadata_en.json etc.)."""
    project_dir = _write_project(isolated_storage.STORAGE_DIR, "readonly-check-slug")
    _write_video(approval_env, "readonly-check-slug")
    _write_thumbnail(approval_env, "readonly-check-slug")
    before = {p: p.read_bytes() for p in sorted(project_dir.iterdir())}

    payload = tap.build_approval_payload("readonly-check-slug", project_dir=project_dir)
    tap.send_new_approval(payload)

    after = {p: p.read_bytes() for p in sorted(project_dir.iterdir())}
    assert after == before, "the source project directory must never be modified"


# ---------------------------------------------------------------------------
# 2026-09 production incident: sendPhoto returned HTTP 400 with no visible
# reason (pipeline.py logged only "[TELEGRAM] approval notification
# skipped: [TELEGRAM] sendPhoto failed: HTTP 400 Bad Request"). Root cause:
# httpx.HTTPStatusError's str() never includes Telegram's own JSON response
# body (the "description" field with the real reason) - only
# e.response.text does, and the old code discarded it. These tests pin the
# fix: the real Telegram description now reaches both the log and the
# raised error, still with the bot token redacted and never including the
# chat_id we sent.
# ---------------------------------------------------------------------------

def _tiny_valid_png_bytes() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 3), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_describe_image_for_diagnostics_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.png"
    desc = tap._describe_image_for_diagnostics(missing)
    assert "exists=False" in desc
    assert str(missing) in desc


def test_describe_image_for_diagnostics_valid_image(tmp_path):
    path = tmp_path / "thumbnail.png"
    path.write_bytes(_tiny_valid_png_bytes())
    desc = tap._describe_image_for_diagnostics(path)
    assert "exists=True" in desc
    assert "format=PNG" in desc
    assert "dimensions=2x3" in desc


def test_describe_image_for_diagnostics_corrupt_file_never_raises(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not a real image, just garbage bytes" * 5)
    desc = tap._describe_image_for_diagnostics(path)  # must not raise
    assert "exists=True" in desc
    assert "UNRECOGNIZED" in desc


class _FakeStatusErrorClient:
    """Fake httpx.Client whose post() always raises HTTPStatusError with a
    real httpx.Response carrying the given status/body - mirrors this
    file's own _HttpErrorClient pattern used for edit_caption()'s
    token-redaction test."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, **kwargs):
        import httpx as _httpx
        request = _httpx.Request("POST", url)
        response = _httpx.Response(self.status_code, request=request, text=self.body)
        raise _httpx.HTTPStatusError(f"error {self.status_code}", request=request, response=response)


def test_send_photo_with_keyboard_surfaces_telegram_response_body_on_400(approval_env, isolated_storage, monkeypatch, tmp_path):
    """The exact reported bug: a bare 'HTTP 400 Bad Request' with no detail
    must become the real Telegram description once this fix is in place."""
    thumb = tmp_path / "thumbnail.png"
    thumb.write_bytes(_tiny_valid_png_bytes())
    real_description = '{"ok":false,"error_code":400,"description":"Bad Request: IMAGE_PROCESS_FAILED"}'
    monkeypatch.setattr(tap.httpx, "Client", _FakeStatusErrorClient(400, real_description))

    with pytest.raises(RuntimeError) as excinfo:
        tap.send_photo_with_keyboard(thumb, "a caption", {"inline_keyboard": [[]]})

    assert "IMAGE_PROCESS_FAILED" in str(excinfo.value)
    assert "HTTP 400" in str(excinfo.value)


def test_send_photo_with_keyboard_logs_telegram_response_body_on_failure(approval_env, isolated_storage, monkeypatch, tmp_path, caplog):
    import logging

    thumb = tmp_path / "thumbnail.png"
    thumb.write_bytes(_tiny_valid_png_bytes())
    monkeypatch.setattr(tap.httpx, "Client", _FakeStatusErrorClient(400, "Bad Request: message caption is too long"))

    with caplog.at_level(logging.WARNING, logger="ycf.telegram_approval"):
        with pytest.raises(RuntimeError):
            tap.send_photo_with_keyboard(thumb, "a caption", {"inline_keyboard": [[]]})

    assert "message caption is too long" in caplog.text


def test_send_photo_with_keyboard_never_leaks_token_on_http_status_error(approval_env, isolated_storage, monkeypatch, tmp_path, caplog):
    import logging

    secret_token = "123456:SECRET-TOKEN-VALUE"
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", secret_token)
    thumb = tmp_path / "thumbnail.png"
    thumb.write_bytes(_tiny_valid_png_bytes())
    # Pathological case: even if Telegram's response body somehow echoed the
    # token back, it must still be redacted before it can reach a log/error.
    monkeypatch.setattr(tap.httpx, "Client", _FakeStatusErrorClient(400, f"Bad Request: token {secret_token} rejected"))

    with caplog.at_level(logging.WARNING, logger="ycf.telegram_approval"):
        with pytest.raises(RuntimeError) as excinfo:
            tap.send_photo_with_keyboard(thumb, "a caption", {"inline_keyboard": [[]]})

    assert secret_token not in caplog.text
    assert secret_token not in str(excinfo.value)


class _FakeConnectErrorClient:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, **kwargs):
        import httpx as _httpx
        raise _httpx.ConnectError(f"connection failed for {url}")


def test_send_photo_with_keyboard_never_leaks_token_on_connection_error(approval_env, isolated_storage, monkeypatch, tmp_path):
    """Non-HTTP-status errors (no response body available) still go through
    the pre-existing str(e) + tb._redact_token() path - confirms that path
    is unchanged by this fix."""
    secret_token = "123456:SECRET-TOKEN-VALUE"
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", secret_token)
    thumb = tmp_path / "thumbnail.png"
    thumb.write_bytes(_tiny_valid_png_bytes())
    monkeypatch.setattr(tap.httpx, "Client", _FakeConnectErrorClient)

    with pytest.raises(RuntimeError) as excinfo:
        tap.send_photo_with_keyboard(thumb, "a caption", {"inline_keyboard": [[]]})

    assert secret_token not in str(excinfo.value)


def test_send_photo_with_keyboard_logs_diagnostics_before_sending(approval_env, isolated_storage, monkeypatch, tmp_path, caplog):
    """Pre-flight diagnostic logging (path/exists/size/format/dimensions/
    caption length) must be present, and must never include chat_id or the
    bot token."""
    import logging

    secret_token = "123456:SECRET-TOKEN-VALUE"
    chat_id_value = "999888777"
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", secret_token)
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", chat_id_value)
    thumb = tmp_path / "thumbnail.png"
    thumb.write_bytes(_tiny_valid_png_bytes())

    class _SuccessClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **kwargs):
            import httpx as _httpx
            request = _httpx.Request("POST", url)
            return _httpx.Response(200, request=request, json={"ok": True, "result": {"message_id": 42}})

    monkeypatch.setattr(tap.httpx, "Client", _SuccessClient)

    with caplog.at_level(logging.INFO, logger="ycf.telegram_approval"):
        message_id = tap.send_photo_with_keyboard(thumb, "a short caption", {"inline_keyboard": [[]]})

    assert message_id == 42
    assert "format=PNG" in caplog.text
    assert "dimensions=2x3" in caplog.text
    assert "caption_len=15" in caplog.text
    assert chat_id_value not in caplog.text
    assert secret_token not in caplog.text
