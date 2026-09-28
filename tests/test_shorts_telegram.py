"""
youtube/telegram_approval.py tests - Shorts Engine V1's ONE-package-approval
flow (Long + 3 Shorts, single APPROVE/REJECT).

Covers spec scenarios:
   11  Short thumbnail generation (payload building)
   12  thumbnail-only Telegram preview
   13  no Telegram video upload (neither Long nor any Short)
   14  single approval
   15  APPROVE approves all four
   16  REJECT rejects all four
   17  duplicate APPROVE is idempotent
   18  duplicate callback does not create duplicate uploads
   21  scheduling of Long + 3 Shorts
   22  configurable Short times
   23  YouTube ID persistence
   24  retry/idempotent upload behavior

Telegram and YouTube are always mocked - no real network call is ever made
by this file (matches tests/test_telegram_approval.py's own convention).
"""
from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from pathlib import Path

import pytest
from PIL import Image

from core.config import settings
from storage import project_store as ps
from storage import shorts as shorts_store
from storage import telegram_approvals as ta
from youtube import telegram_approval as tap


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def approval_env(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 480.0)
    return tap


@pytest.fixture
def fake_telegram_send(monkeypatch):
    sent = {"photos": [], "videos": []}

    def fake_photo(image_path, caption, keyboard, chat_id=None):
        message_id = 1000 + len(sent["photos"])
        sent["photos"].append({"image_path": image_path, "caption": caption, "keyboard": keyboard,
                                "chat_id": chat_id, "message_id": message_id})
        return message_id

    def fake_video(video_path, caption="", chat_id=None):
        sent["videos"].append({"video_path": video_path, "chat_id": chat_id})
        return 2000 + len(sent["videos"])

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_photo)
    monkeypatch.setattr(tap, "send_video_preview", fake_video)
    return sent


@pytest.fixture
def fake_edit_text(monkeypatch):
    edits = []
    monkeypatch.setattr(tap, "edit_caption", lambda message_id, caption, reply_markup=None, chat_id=None: edits.append((message_id, caption)))
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": None)
    return edits


class _FakeYouTube:
    def __init__(self):
        self.video_calls = []
        self.thumbnail_calls = []
        self._n = 0
        self.fail_titles = set()

    def upload_video(self, video_path, metadata, channel_key=None):
        self.video_calls.append((str(video_path), dict(metadata), channel_key))
        if metadata.get("title") in self.fail_titles:
            raise RuntimeError("[YOUTUBE] simulated upload failure")
        self._n += 1
        video_id = f"vid{self._n}"
        return {"video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
                "channel_key": channel_key or "default", "scheduled_publish_at": metadata.get("publish_at")}

    def upload_thumbnail(self, video_id, thumbnail_path, channel_key=None):
        self.thumbnail_calls.append((video_id, str(thumbnail_path), channel_key))


@pytest.fixture
def fake_youtube(monkeypatch):
    from youtube import uploader
    fake = _FakeYouTube()
    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", fake.upload_thumbnail)
    return fake


def _make_long_project(isolated_storage, slug="a-mystery-video"):
    paths, _ = ps.create_project(slug.replace("-", " ").title())
    paths.video_mp4.write_bytes(b"x" * 1000)
    Image.new("RGB", (1280, 720)).save(paths.thumbnail_png)
    ps.save_json(paths.metadata_en_json, {"title": "The Mystery Video", "description": "d", "tags": []})
    return paths


def _make_shorts_for(paths, n=3):
    rows = []
    for i in range(1, n + 1):
        short_id = shorts_store.create_short(
            parent_video_id=paths.video_id, short_number=i, status=shorts_store.STATUS_READY,
            title=f"Short {i} Title", script=f"Narration text for short {i}.",
        )
        video_path = paths.short_video_path(i)
        thumb_path = paths.short_thumbnail_path(i)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x" * 500)
        Image.new("RGB", (1080, 1920)).save(thumb_path)
        shorts_store.update_fields(short_id, video_path=str(video_path), thumbnail_path=str(thumb_path))
        rows.append(shorts_store.get_short(short_id))
    return rows


def _package_send(approval_env, isolated_storage, fake_telegram_send, slug="a-mystery-video", n_shorts=3):
    paths = _make_long_project(isolated_storage, slug)
    shorts_rows = _make_shorts_for(paths, n=n_shorts)
    payload = tap.build_package_approval_payload(
        paths.slug, "The Mystery Video", paths.video_id,
        long_video_path=paths.video_mp4, long_thumbnail_path=paths.thumbnail_png, shorts_rows=shorts_rows,
    )
    result = tap.send_new_package_approval(payload)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    return paths, result["approval_id"], message_id


def _callback(action, approval_id, chat_id=999, message_id=None, cbq_id="cbq1"):
    return {"id": cbq_id, "data": f"approval:{action}:{approval_id}",
            "message": {"message_id": message_id, "chat": {"id": chat_id}}}


# ---------------------------------------------------------------------------
# 11/12/13 - collage/payload building: thumbnails only, no video ever sent
# ---------------------------------------------------------------------------

def test_build_package_collage_includes_long_and_every_short_thumbnail(isolated_storage, tmp_path):
    long_thumb = tmp_path / "long.png"
    Image.new("RGB", (1280, 720), (10, 10, 10)).save(long_thumb)
    short_thumbs = []
    for i in range(1, 4):
        p = tmp_path / f"short_{i}.png"
        Image.new("RGB", (1080, 1920), (i * 10, 0, 0)).save(p)
        short_thumbs.append((i, p))

    dest = tmp_path / "collage.png"
    result = tap.build_package_collage(long_thumb, short_thumbs, dest)
    assert result.exists()
    with Image.open(result) as im:
        assert im.width == tap.COLLAGE_SHORT_CELL_WIDTH * 3
        assert im.height > tap.COLLAGE_SHORT_CELL_HEIGHT  # long row + shorts row


def test_send_new_package_approval_sends_exactly_one_photo_no_video(approval_env, isolated_storage, fake_telegram_send):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)
    assert len(fake_telegram_send["photos"]) == 1  # thumbnail-only preview (13/12)
    assert fake_telegram_send["videos"] == []  # never sends the Long/Short video to Telegram
    sent = fake_telegram_send["photos"][0]
    assert "collage" in str(sent["image_path"])


def test_package_approval_never_references_a_video_file_as_the_sent_photo(approval_env, isolated_storage, fake_telegram_send):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)
    sent = fake_telegram_send["photos"][0]
    assert not str(sent["image_path"]).endswith(".mp4")


# ---------------------------------------------------------------------------
# 14 - single approval: every Short attached to the SAME approval_id, no
# separate approval rows
# ---------------------------------------------------------------------------

def test_send_new_package_approval_creates_exactly_one_approval_row(approval_env, isolated_storage, fake_telegram_send):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)
    assert len(ta.list_approvals()) == 1
    shorts_rows = shorts_store.list_shorts_for_approval(approval_id)
    assert len(shorts_rows) == 3
    assert all(r["approval_id"] == approval_id for r in shorts_rows)


# ---------------------------------------------------------------------------
# 15/21/22/23 - APPROVE approves all four, schedules Long + 3 Shorts at
# their configured fixed slots, and persists each YouTube video id
# ---------------------------------------------------------------------------

def test_approve_uploads_long_and_all_three_shorts(approval_env, isolated_storage, fake_telegram_send,
                                                     fake_edit_text, fake_youtube):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)

    tap.handle_approval_callback(_callback("approve", approval_id, message_id=message_id))

    approval_row = ta.get_approval(approval_id)
    assert approval_row["status"] == tap.STATUS_UPLOADED

    shorts_rows = shorts_store.list_shorts_for_approval(approval_id)
    assert len(shorts_rows) == 3
    for row in shorts_rows:
        assert row["status"] == shorts_store.STATUS_UPLOADED
        assert row["youtube_video_id"]  # 23 - persisted
        assert row["youtube_url"]
        assert row["scheduled_at"]  # 21 - scheduled


def test_approve_schedules_shorts_at_their_configured_fixed_hours(approval_env, isolated_storage, fake_telegram_send,
                                                                    fake_edit_text, fake_youtube, monkeypatch):
    monkeypatch.setattr(settings, "SHORT_1_DEFAULT_HOUR", 12)
    monkeypatch.setattr(settings, "SHORT_2_DEFAULT_HOUR", 16)
    monkeypatch.setattr(settings, "SHORT_3_DEFAULT_HOUR", 22)
    monkeypatch.setattr(settings, "SHORTS_TIMEZONE", "Europe/Istanbul")

    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", approval_id, message_id=message_id))

    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Istanbul")
    rows = {r["short_number"]: r for r in shorts_store.list_shorts_for_approval(approval_id)}
    assert datetime.fromisoformat(rows[1]["scheduled_at"]).astimezone(tz).hour == 12
    assert datetime.fromisoformat(rows[2]["scheduled_at"]).astimezone(tz).hour == 16
    assert datetime.fromisoformat(rows[3]["scheduled_at"]).astimezone(tz).hour == 22


def test_approve_never_uploads_public_immediately(approval_env, isolated_storage, fake_telegram_send,
                                                    fake_edit_text, fake_youtube):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)
    tap.handle_approval_callback(_callback("approve", approval_id, message_id=message_id))
    for _video_path, metadata, _channel in fake_youtube.video_calls:
        assert metadata["privacy_status"] == "private"


# ---------------------------------------------------------------------------
# 16 - REJECT rejects all four
# ---------------------------------------------------------------------------

def test_reject_rejects_long_and_all_shorts(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)

    tap.handle_approval_callback(_callback("reject", approval_id, message_id=message_id))

    assert ta.get_approval(approval_id)["status"] == tap.STATUS_REJECTED
    shorts_rows = shorts_store.list_shorts_for_approval(approval_id)
    assert all(r["status"] == shorts_store.STATUS_REJECTED for r in shorts_rows)
    assert fake_youtube.video_calls == []  # no upload was ever attempted


# ---------------------------------------------------------------------------
# 17/18 - duplicate APPROVE callback is idempotent: no duplicate uploads
# ---------------------------------------------------------------------------

def test_duplicate_approve_callback_never_double_uploads(approval_env, isolated_storage, fake_telegram_send,
                                                           fake_edit_text, fake_youtube):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)

    tap.handle_approval_callback(_callback("approve", approval_id, message_id=message_id))
    first_call_count = len(fake_youtube.video_calls)
    assert first_call_count == 4  # 1 Long + 3 Shorts

    tap.handle_approval_callback(_callback("approve", approval_id, message_id=message_id))  # duplicate tap
    assert len(fake_youtube.video_calls) == first_call_count  # no new uploads at all

    shorts_rows = shorts_store.list_shorts_for_approval(approval_id)
    assert all(r["status"] == shorts_store.STATUS_UPLOADED for r in shorts_rows)


# ---------------------------------------------------------------------------
# 24 - retry/idempotent upload behavior: process_shorts_for_approval() is
# independently re-callable and only retries what actually failed/is not
# yet uploaded, never re-uploading an already-uploaded Short
# ---------------------------------------------------------------------------

def test_process_shorts_for_approval_retries_only_the_failed_short(approval_env, isolated_storage,
                                                                     fake_telegram_send, fake_youtube):
    paths, approval_id, message_id = _package_send(approval_env, isolated_storage, fake_telegram_send)

    fake_youtube.fail_titles = {"Short 2 Title"}
    results = tap.process_shorts_for_approval(approval_id)
    statuses = {r["short_number"]: r["status"] for r in results}
    assert statuses[1] == shorts_store.STATUS_UPLOADED
    assert statuses[2] == shorts_store.STATUS_UPLOAD_FAILED
    assert statuses[3] == shorts_store.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 3

    # fix the "failure" and retry - only Short 2 should be (re)attempted
    fake_youtube.fail_titles = set()
    results2 = tap.process_shorts_for_approval(approval_id)
    by_number = {r["short_number"]: r for r in results2}
    assert by_number[1].get("skipped") is True
    assert by_number[3].get("skipped") is True
    assert by_number[2]["status"] == shorts_store.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 4  # exactly one new call (Short 2's retry)


def test_process_shorts_for_approval_is_a_safe_noop_with_no_attached_shorts(approval_env, isolated_storage,
                                                                             fake_telegram_send, fake_youtube):
    """An ordinary, non-package single-video approval (no attached Shorts
    rows) must never error - process_shorts_for_approval() returns []."""
    paths = _make_long_project(isolated_storage, "solo-video")
    payload = tap.build_approval_payload(paths.slug, project_dir=paths.root, video_path=paths.video_mp4, thumbnail_path=paths.thumbnail_png)
    result = tap.send_new_approval(payload)
    results = tap.process_shorts_for_approval(result["approval_id"])
    assert results == []
    assert fake_youtube.video_calls == []
