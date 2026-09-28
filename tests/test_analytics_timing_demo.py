"""
youtube/analytics_timing_demo.py tests - YouTube Analytics Timing V1's
read-only diagnostic CLI.

Covers spec scenarios 7 (cache refresh, via --refresh), 24 (dry-run: no DB
mutation beyond what a real approval's own caching would do), 25 (dry-run:
never calls the YouTube uploader / sends Telegram).
"""
from __future__ import annotations

from core.config import settings
from youtube import analytics_client as ac
from youtube import analytics_timing as at
from youtube import analytics_timing_demo as demo


def _video(video_id, published_at_utc, views):
    return {"video_id": video_id, "published_at": published_at_utc, "title": "t",
            "view_count": views, "like_count": 0, "comment_count": 0}


def test_diagnose_reports_disabled_state_and_makes_no_api_call(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)
    monkeypatch.setattr(
        ac, "list_recent_uploads",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the API when disabled")),
    )
    result = demo.diagnose()
    assert result["enabled"] is False
    assert result["slots"] == []


def test_diagnose_reports_unavailable_when_no_usable_data(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_VIDEOS", 5)
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: [])
    result = demo.diagnose(timezone="Europe/Istanbul", days=90)
    assert result["enabled"] is True
    assert result["source"] == "unavailable"
    assert result["slots"] == []


def test_diagnose_returns_real_slots_when_available(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_VIDEOS", 5)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE", True)
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = demo.diagnose(timezone="Europe/Istanbul", days=90)
    assert result["source"] == "inferred_performance"
    assert result["sample_size"] == 6
    assert len(result["slots"]) == 1


# ---------------------------------------------------------------------------
# 7 - cache refresh
# ---------------------------------------------------------------------------

def test_refresh_flag_clears_any_cached_result_before_looking_up(isolated_storage, monkeypatch):
    from storage import analytics_timing_cache as cache

    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_VIDEOS", 5)
    key = cache.make_cache_key(None, "Europe/Istanbul", 90)
    cache.set_cache(cache_key=key, channel_key=None, timezone_name="Europe/Istanbul", lookback_days=90,
                     source="inferred_performance", sample_size=99, confidence="high", slots=[], ttl_hours=24)

    call_count = {"n": 0}

    def _list_recent_uploads(**k):
        call_count["n"] += 1
        return [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]

    monkeypatch.setattr(ac, "list_recent_uploads", _list_recent_uploads)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = demo.diagnose(timezone="Europe/Istanbul", days=90, refresh=True)
    assert call_count["n"] == 1  # actually re-fetched, not served from the stale cached "99"
    assert result["sample_size"] == 6


# ---------------------------------------------------------------------------
# 24/25 - dry-run: no DB mutation, no YouTube/Telegram call
# ---------------------------------------------------------------------------

def test_main_never_calls_youtube_uploader(isolated_storage, monkeypatch):
    from youtube import uploader

    def _must_not_be_called(*a, **k):
        raise AssertionError("analytics_timing_demo must never call the YouTube uploader")

    monkeypatch.setattr(uploader, "upload_video", _must_not_be_called)
    monkeypatch.setattr(uploader, "upload_thumbnail", _must_not_be_called)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)

    exit_code = demo.main(["--dry-run"])
    assert exit_code == 0


def test_main_never_sends_a_telegram_message(isolated_storage, monkeypatch):
    from youtube import telegram_approval as tap

    def _must_not_be_called(*a, **k):
        raise AssertionError("analytics_timing_demo must never send a Telegram message")

    monkeypatch.setattr(tap, "send_photo_with_keyboard", _must_not_be_called)
    monkeypatch.setattr(tap, "send_video_preview", _must_not_be_called)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)

    exit_code = demo.main([])
    assert exit_code == 0


def test_main_never_creates_a_publish_schedule_or_approval_row(isolated_storage, monkeypatch):
    from storage import publish_schedules as psched
    from storage import telegram_approvals as ta

    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)
    demo.main(["--dry-run"])
    assert psched.list_schedules() == []
    assert ta.list_approvals() == []


def test_main_prints_expected_shape_when_disabled(isolated_storage, monkeypatch, capsys):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)
    exit_code = demo.main(["--days", "90", "--timezone", "Europe/Istanbul"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "YOUTUBE ANALYTICS TIMING" in captured.out
    assert "STATUS: disabled" in captured.out
    assert "No YouTube action" in captured.out
    assert "No Telegram action" in captured.out
    assert "No DB mutation" in captured.out


def test_main_prints_best_slots_when_available(isolated_storage, monkeypatch, capsys):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_VIDEOS", 5)
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    exit_code = demo.main(["--days", "90", "--timezone", "Europe/Istanbul"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "source: inferred_performance" in captured.out
    assert "videos analyzed: 6" in captured.out
    assert "BEST SLOTS:" in captured.out
    assert "INFERRED from past video performance" in captured.out


def test_main_handles_an_unexpected_exception_cleanly(isolated_storage, monkeypatch, capsys):
    def _explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(demo, "diagnose", _explode)
    exit_code = demo.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.err
