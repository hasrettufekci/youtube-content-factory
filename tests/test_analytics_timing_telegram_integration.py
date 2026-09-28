"""
YouTube Analytics Timing V1 <-> Smart Publish <-> Telegram approve-flow
integration tests.

Builds directly on tests/test_smart_publish_integration.py's own fixtures
(approval_env/fake_youtube/fake_telegram_send/_send) - this file only adds
the NEW YouTube Analytics Timing V1 wiring on top: when
YOUTUBE_ANALYTICS_ENABLED, youtube/telegram_approval.py's
_compute_and_persist_publish_decision() passes a real
CompositeAudienceTimingProvider into smart_publish.calculate_publish_time(),
and the resulting decision (analytics-inferred or bootstrap-fallback) flows
through to the persisted publish_schedules row and the Telegram caption
exactly like a Smart-Publish-V1-only bootstrap decision already did.

Covers spec scenarios 19/20 (Smart Publish selects the real
inferred_performance slot when available), 21 (falls back to bootstrap when
Analytics has insufficient data), 22 (scheduled_at still never in the
past), 23 (minimum delay still respected), 26 (existing approval flow
unaffected when this feature is off), 27 (existing Telegram callback
unaffected by a provider construction failure - defensive fallback layer).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone as dt_timezone

import pytest

from core.config import settings
from storage import publish_schedules as psched
from youtube import analytics_client as ac
from youtube import analytics_timing as at
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
    return tap.send_new_approval(payload)


def _video(video_id, published_at_utc, views):
    return {"video_id": video_id, "published_at": published_at_utc, "title": "t",
            "view_count": views, "like_count": 0, "comment_count": 0}


@pytest.fixture(autouse=True)
def _default_analytics_settings(monkeypatch):
    monkeypatch.setattr(settings, "SMART_PUBLISH_DEFAULT_HOUR", 20)
    monkeypatch.setattr(settings, "SMART_PUBLISH_DEFAULT_MINUTE", 0)
    monkeypatch.setattr(settings, "SMART_PUBLISH_MIN_DELAY_MINUTES", 60)
    monkeypatch.setattr(settings, "SMART_PUBLISH_LOOKAHEAD_DAYS", 7)
    monkeypatch.setattr(settings, "SMART_PUBLISH_TIMEZONE", "Europe/Istanbul")
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_CACHE_TTL_HOURS", 24)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_LOOKBACK_DAYS", 90)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_VIDEOS", 5)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_TIMEZONE", "Europe/Istanbul")
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW", 5)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_SAMPLE_MEDIUM", 10)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_SAMPLE_HIGH", 20)


# ---------------------------------------------------------------------------
# 19/20 - Smart Publish selects the real inferred_performance slot
# ---------------------------------------------------------------------------

def test_approve_flow_uses_inferred_performance_slot_when_analytics_available(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]  # Friday 20:00 Istanbul
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    schedules = psched.list_schedules()
    assert len(schedules) == 1
    assert schedules[0]["source"] == "inferred_performance"
    assert schedules[0]["confidence"] in ("low", "medium", "high")

    caption = fake_telegram_send["edits"][-1]
    assert "Inferred from video performance" in caption


# ---------------------------------------------------------------------------
# 21 - Smart Publish falls back to bootstrap when Analytics has insufficient data
# ---------------------------------------------------------------------------

def test_approve_flow_falls_back_to_bootstrap_when_analytics_has_too_few_videos(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: [_video("v1", "2026-09-04T17:00:00+00:00", 100)])

    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    schedules = psched.list_schedules()
    assert schedules[0]["source"] == "bootstrap"
    assert schedules[0]["confidence"] == "bootstrap"
    assert "Bootstrap fallback" in fake_telegram_send["edits"][-1]


# ---------------------------------------------------------------------------
# 22/23 - scheduled_at never in the past, minimum delay still respected, even
# with a real analytics-inferred slot in play
# ---------------------------------------------------------------------------

def test_inferred_performance_schedule_is_never_in_the_past_and_respects_min_delay(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    # Best-performing slot is "right now" (whatever hour we happen to run
    # at) - the decision must still land at least MIN_DELAY_MINUTES ahead.
    now_utc = datetime.now(dt_timezone.utc)
    videos = [_video(f"v{i}", now_utc.isoformat(), 100) for i in range(6)]
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    row = tap.ta.get_approval(approval_id)
    decision, _schedule_id = tap._compute_and_persist_publish_decision(approval_id, row["slug"])

    assert decision.scheduled_at > now_utc.astimezone(decision.scheduled_at.tzinfo)
    min_gap = decision.scheduled_at - now_utc.astimezone(decision.scheduled_at.tzinfo)
    assert min_gap.total_seconds() >= settings.SMART_PUBLISH_MIN_DELAY_MINUTES * 60


# ---------------------------------------------------------------------------
# 26 - existing approval flow unaffected when this feature is off
# ---------------------------------------------------------------------------

def test_approve_flow_identical_to_smart_publish_v1_baseline_when_analytics_disabled(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)
    monkeypatch.setattr(
        ac, "list_recent_uploads",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the Analytics API when disabled")),
    )

    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    schedules = psched.list_schedules()
    assert schedules[0]["source"] == "bootstrap"
    assert schedules[0]["confidence"] == "bootstrap"


# ---------------------------------------------------------------------------
# 27 - existing Telegram callback unaffected by a provider construction
# failure (defensive fallback layer in _compute_and_persist_publish_decision)
# ---------------------------------------------------------------------------

def test_approve_flow_survives_analytics_provider_construction_failure(
    approval_env, isolated_storage, fake_telegram_send, fake_youtube, monkeypatch,
):
    monkeypatch.setattr(settings, "SMART_PUBLISH_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)

    def _explode(*a, **k):
        raise RuntimeError("provider construction blew up")

    monkeypatch.setattr(at, "get_audience_timing_provider", _explode)

    result = _send(approval_env, isolated_storage, fake_telegram_send)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    # Must not raise - degrades to the bootstrap-only path exactly as if
    # YOUTUBE_ANALYTICS_ENABLED were False.
    tap._handle_approve_action(result["approval_id"], tap.ta.get_approval(result["approval_id"]), message_id)

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED
    schedules = psched.list_schedules()
    assert schedules[0]["source"] == "bootstrap"
