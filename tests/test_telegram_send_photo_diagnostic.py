"""
youtube/telegram_send_photo_diagnostic.py tests.

2026-09 production incident: video_id=5's Telegram approval notification
failed with a bare "HTTP 400 Bad Request" (see
tests/test_telegram_approval.py's own sendPhoto-diagnostics tests for the
underlying fix in youtube/telegram_approval.py). This module is the
standalone reproduction tool for that failure - these tests pin down that
it never re-runs any pipeline stage, never creates a video or an approval
row, and never makes a network call unless --send is explicitly given.

Telegram is always mocked (via monkeypatching tap.send_photo_with_keyboard
itself) or never called at all - no real network call is ever made here.
"""
from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from core.config import settings
from storage import project_store as ps
from storage import telegram_approvals as ta
from youtube import telegram_approval as tap
from youtube import telegram_send_photo_diagnostic as diag


def _tiny_valid_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def real_project(isolated_storage, monkeypatch):
    """A real, already-produced project (video.mp4 + thumbnail.png +
    metadata_en.json + scenes.json under storage/<slug>/) - mirrors exactly
    what the production pipeline leaves on disk for an already-completed
    video, so diagnose() can resolve it the same way pipeline.py's
    _notify_telegram() does (via storage/project_store, not the
    debug_audit/ demo-tool convention)."""
    paths, _created = ps.create_project("Medieval Norman Castle Discovered Hidden Inside", category="History")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.video_mp4.write_bytes(b"x" * 1000)
    paths.thumbnail_png.write_bytes(_tiny_valid_png_bytes())
    paths.metadata_en_json.write_text(json.dumps({
        "title": "Medieval Norman Castle Discovered Hidden Inside a Mountain",
        "description": "A short description.", "tags": ["a", "b"],
    }))
    paths.scenes_json.write_text(json.dumps([{"scene_id": i} for i in range(1, 19)]))
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)
    return paths


def test_diagnose_resolves_by_video_id(real_project):
    result = diag.diagnose(video_id=real_project.video_id)
    assert result["video_id"] == real_project.video_id
    assert result["slug"] == real_project.slug
    assert result["thumbnail_exists"] is True
    assert result["thumbnail_size_bytes"] > 0
    assert "format=PNG" in result["thumbnail_diagnostics"]
    assert result["bot_token_configured"] is True
    assert result["chat_id_configured"] is True
    assert result["sent"] is False


def test_diagnose_resolves_by_slug(real_project):
    result = diag.diagnose(slug=real_project.slug)
    assert result["video_id"] == real_project.video_id
    assert result["slug"] == real_project.slug


def test_diagnose_unknown_video_id_raises(isolated_storage):
    with pytest.raises(tap.TelegramApprovalError):
        diag.diagnose(video_id=999999)


def test_diagnose_unknown_slug_raises(isolated_storage):
    with pytest.raises(tap.TelegramApprovalError):
        diag.diagnose(slug="no-such-slug")


def test_diagnose_without_send_makes_no_network_call(real_project, monkeypatch):
    def _must_not_be_called(*a, **k):
        raise AssertionError("send_photo_with_keyboard must not be called without --send/send=True")

    monkeypatch.setattr(tap, "send_photo_with_keyboard", _must_not_be_called)

    result = diag.diagnose(video_id=real_project.video_id, send=False)

    assert result["sent"] is False
    assert "send_error" not in result
    assert "message_id" not in result


def test_diagnose_with_send_calls_send_photo_with_keyboard_exactly_once(real_project, monkeypatch):
    calls = []

    def fake_send(image_path, caption, keyboard, chat_id=None):
        calls.append((image_path, caption, keyboard, chat_id))
        return 4242

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_send)

    result = diag.diagnose(video_id=real_project.video_id, send=True)

    assert len(calls) == 1
    assert calls[0][0] == real_project.thumbnail_png
    assert result["sent"] is True
    assert result["message_id"] == 4242


def test_diagnose_with_send_reports_failure_without_raising(real_project, monkeypatch):
    monkeypatch.setattr(
        tap, "send_photo_with_keyboard",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("[TELEGRAM] sendPhoto failed: HTTP 400 - Bad Request: IMAGE_PROCESS_FAILED")),
    )

    result = diag.diagnose(video_id=real_project.video_id, send=True)  # must not raise

    assert result["sent"] is False
    assert "IMAGE_PROCESS_FAILED" in result["send_error"]


def test_diagnose_never_creates_an_approval_row(real_project, monkeypatch):
    """The whole point of this tool is to reproduce the ONE network call
    without touching the approval state machine at all - create_approval()
    must never be called, whether or not --send is given."""
    def _must_not_be_called(*a, **k):
        raise AssertionError("diagnose() must never create a telegram_approvals row")

    monkeypatch.setattr(ta, "create_approval", _must_not_be_called)
    monkeypatch.setattr(tap, "send_photo_with_keyboard", lambda *a, **k: 1)

    diag.diagnose(video_id=real_project.video_id, send=True)
    diag.diagnose(video_id=real_project.video_id, send=False)


def test_diagnose_never_generates_a_new_video_or_project(real_project, monkeypatch):
    """No research/script/image/TTS/assembly stage may be invoked - assert
    no new project directory or DB row is created by diagnose()."""
    from storage import db

    before_count = len(db.list_videos()) if hasattr(db, "list_videos") else None
    monkeypatch.setattr(tap, "send_photo_with_keyboard", lambda *a, **k: 1)

    diag.diagnose(video_id=real_project.video_id, send=True)

    if before_count is not None:
        assert len(db.list_videos()) == before_count
    # The only project directory that exists is the one the fixture created.
    year_dirs = list(settings.STORAGE_DIR.glob("*/*/*"))
    assert len(year_dirs) == 1
    assert year_dirs[0] == real_project.root


def test_main_prints_diagnostics_without_send_and_returns_0(real_project, capsys):
    exit_code = diag.main(["--video-id", str(real_project.video_id)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"video_id: {real_project.video_id}" in captured.out
    assert real_project.slug in captured.out
    assert "diagnostics only - no network call made" in captured.out


def test_main_with_send_reports_success(real_project, monkeypatch, capsys):
    monkeypatch.setattr(tap, "send_photo_with_keyboard", lambda *a, **k: 555)

    exit_code = diag.main(["--video-id", str(real_project.video_id), "--send"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK - sendPhoto succeeded" in captured.out
    assert "message_id: 555" in captured.out


def test_main_with_send_reports_failure_with_exit_code_1(real_project, monkeypatch, capsys):
    monkeypatch.setattr(
        tap, "send_photo_with_keyboard",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("[TELEGRAM] sendPhoto failed: HTTP 400 - Bad Request: IMAGE_PROCESS_FAILED")),
    )

    exit_code = diag.main(["--video-id", str(real_project.video_id), "--send"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "IMAGE_PROCESS_FAILED" in captured.err


def test_main_requires_video_id_or_slug():
    with pytest.raises(SystemExit):
        diag.main([])


def test_main_rejects_both_video_id_and_slug():
    with pytest.raises(SystemExit):
        diag.main(["--video-id", "5", "--slug", "some-slug"])


def test_main_warns_when_caption_exceeds_telegram_limit(real_project, capsys, monkeypatch):
    monkeypatch.setattr(tap, "build_caption", lambda *a, **k: "x" * 1025)

    exit_code = diag.main(["--video-id", str(real_project.video_id)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WARNING: caption_length exceeds Telegram's 1024-character" in captured.out
