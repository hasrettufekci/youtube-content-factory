"""
youtube/smart_publish_demo.py tests - Smart Publish Scheduler V1's
diagnostic CLI (Faz 9).

This tool is read-only by construction: diagnose()/main() never touch
YouTube, Telegram, or the DB beyond the plain project lookup every other
diagnostic CLI in this codebase already does (see
tests/test_telegram_send_photo_diagnostic.py's own real_project fixture,
mirrored here). --dry-run is accepted but the tool behaves identically
with or without it - there is no non-dry-run action to take.
"""
from __future__ import annotations

import json

import pytest

from core.config import settings
from storage import project_store as ps
from storage import publish_schedules as psched
from storage import telegram_approvals as ta
from youtube import smart_publish_demo as demo


@pytest.fixture
def real_project(isolated_storage):
    paths, _created = ps.create_project("Medieval Norman Castle Discovered Hidden Inside", category="History")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.metadata_en_json.write_text(json.dumps({
        "title": "The 800-Year-Old Castle Hidden Inside a University Building",
        "description": "A short description.", "tags": ["a", "b"],
    }))
    return paths


def test_diagnose_resolves_by_video_id_and_returns_a_decision(real_project):
    result = demo.diagnose(video_id=real_project.video_id, now=None, timezone="Europe/Istanbul")
    assert result["video_id"] == real_project.video_id
    assert result["slug"] == real_project.slug
    assert result["title"] == "The 800-Year-Old Castle Hidden Inside a University Building"
    assert result["decision"].source == "bootstrap"


def test_diagnose_resolves_by_slug(real_project):
    result = demo.diagnose(slug=real_project.slug, timezone="Europe/Istanbul")
    assert result["video_id"] == real_project.video_id


def test_diagnose_unknown_video_id_raises(isolated_storage):
    with pytest.raises(ValueError):
        demo.diagnose(video_id=999999)


def test_diagnose_unknown_slug_raises(isolated_storage):
    with pytest.raises(ValueError):
        demo.diagnose(slug="no-such-slug")


def test_diagnose_requires_video_id_or_slug(isolated_storage):
    with pytest.raises(ValueError):
        demo.diagnose()


def test_diagnose_respects_explicit_now_override(real_project):
    from datetime import datetime, timezone as dt_timezone
    fixed_now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    result = demo.diagnose(video_id=real_project.video_id, now=fixed_now, timezone="Europe/Istanbul")
    assert result["decision"].scheduled_at > fixed_now.astimezone(result["decision"].scheduled_at.tzinfo)


# ---------------------------------------------------------------------------
# 14/15/16 - dry-run makes no DB write, no YouTube call, no Telegram call
# ---------------------------------------------------------------------------

def test_diagnose_makes_no_publish_schedules_write(real_project):
    demo.diagnose(video_id=real_project.video_id, timezone="Europe/Istanbul")
    assert psched.list_schedules() == []


def test_diagnose_makes_no_approval_row(real_project):
    demo.diagnose(video_id=real_project.video_id, timezone="Europe/Istanbul")
    assert ta.list_approvals() == []


def test_diagnose_never_calls_youtube_uploader(real_project, monkeypatch):
    from youtube import uploader

    def _must_not_be_called(*a, **k):
        raise AssertionError("smart_publish_demo.diagnose() must never call the YouTube uploader")

    monkeypatch.setattr(uploader, "upload_video", _must_not_be_called)
    monkeypatch.setattr(uploader, "upload_thumbnail", _must_not_be_called)

    demo.diagnose(video_id=real_project.video_id, timezone="Europe/Istanbul")  # must not raise


def test_diagnose_never_sends_a_telegram_message(real_project, monkeypatch):
    from youtube import telegram_approval as tap

    def _must_not_be_called(*a, **k):
        raise AssertionError("smart_publish_demo.diagnose() must never send a Telegram message")

    monkeypatch.setattr(tap, "send_photo_with_keyboard", _must_not_be_called)
    monkeypatch.setattr(tap, "send_video_preview", _must_not_be_called)

    demo.diagnose(video_id=real_project.video_id, timezone="Europe/Istanbul")  # must not raise


def test_diagnose_never_creates_a_new_project_or_video_row(real_project):
    from storage import db

    before = len(db.list_videos())
    demo.diagnose(video_id=real_project.video_id, timezone="Europe/Istanbul")
    assert len(db.list_videos()) == before


# ---------------------------------------------------------------------------
# CLI (main())
# ---------------------------------------------------------------------------

def test_main_prints_the_expected_report_shape(real_project, capsys):
    exit_code = demo.main(["--video-id", str(real_project.video_id), "--now", "2026-09-27T10:00:00+00:00",
                           "--timezone", "Europe/Istanbul"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "VIDEO:" in captured.out
    assert str(real_project.video_id) in captured.out
    assert "The 800-Year-Old Castle Hidden Inside a University Building" in captured.out
    assert "PUBLISH DECISION" in captured.out
    assert "scheduled_at: 2026-09-27 20:00" in captured.out
    assert "timezone: Europe/Istanbul" in captured.out
    assert "source: bootstrap" in captured.out
    assert "confidence: bootstrap" in captured.out
    assert "REASON:" in captured.out
    assert "Channel does not yet have sufficient audience timing data." in captured.out
    assert "DRY RUN:" in captured.out
    assert "No YouTube action" in captured.out
    assert "No Telegram action" in captured.out
    assert "No DB mutation" in captured.out


def test_main_dry_run_flag_is_accepted_and_behaves_identically(real_project, capsys):
    exit_code = demo.main(["--video-id", str(real_project.video_id), "--dry-run",
                           "--now", "2026-09-27T10:00:00+00:00", "--timezone", "Europe/Istanbul"])
    assert exit_code == 0
    assert "scheduled_at: 2026-09-27 20:00" in capsys.readouterr().out


def test_main_requires_video_id_or_slug(capsys):
    exit_code = demo.main([])
    assert exit_code == 1
    assert "FAILED" in capsys.readouterr().err


def test_main_reports_unknown_video_id_cleanly(isolated_storage, capsys):
    exit_code = demo.main(["--video-id", "999999"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.err
    assert "Traceback" not in captured.err


def test_main_rejects_malformed_now(real_project, capsys):
    exit_code = demo.main(["--video-id", str(real_project.video_id), "--now", "not-a-date"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.err


def test_main_never_mutates_state(real_project, capsys):
    demo.main(["--video-id", str(real_project.video_id)])
    assert psched.list_schedules() == []
    assert ta.list_approvals() == []
