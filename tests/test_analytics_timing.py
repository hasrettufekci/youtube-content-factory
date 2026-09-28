"""
youtube/analytics_timing.py tests - YouTube Analytics Timing V1's core
provider logic: DirectAudienceTimingProvider (always honestly
"unavailable"), PerformanceTimingProvider (real per-video performance
bucketed by weekday/hour), confidence_for_video_count(), and
CompositeAudienceTimingProvider (priority order + cache + "never raises").

No real network access anywhere - youtube.analytics_client's functions are
monkeypatched directly (list_recent_uploads/query_video_engagement), the
same style tests/test_analytics_client.py itself uses for the layer below.

Covers spec scenarios 1 (disabled), 8-18.
"""
from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

import pytest

from core.config import settings
from youtube import analytics_client as ac
from youtube import analytics_timing as at
from youtube.smart_publish import SOURCE_INFERRED_PERFORMANCE


@pytest.fixture(autouse=True)
def _default_analytics_settings(monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE", True)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_CACHE_TTL_HOURS", 24)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_LOOKBACK_DAYS", 90)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_VIDEOS", 5)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_TIMEZONE", "Europe/Istanbul")
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW", 5)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_SAMPLE_MEDIUM", 10)
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_MIN_SAMPLE_HIGH", 20)


def _video(video_id, published_at_utc, views):
    return {"video_id": video_id, "published_at": published_at_utc, "title": "t",
            "view_count": views, "like_count": 0, "comment_count": 0}


# ---------------------------------------------------------------------------
# 1 - analytics disabled
# ---------------------------------------------------------------------------

def test_composite_provider_returns_no_slots_and_makes_no_call_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_ENABLED", False)
    monkeypatch.setattr(
        ac, "list_recent_uploads",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the API when disabled")),
    )
    provider = at.CompositeAudienceTimingProvider()
    assert provider.get_best_publish_slots() == []


# ---------------------------------------------------------------------------
# 8 - direct audience timing provider (always honestly unavailable)
# ---------------------------------------------------------------------------

def test_direct_provider_always_reports_unavailable():
    result = at.DirectAudienceTimingProvider().get_audience_timing(
        channel_key=None, start_date="2026-01-01", end_date="2026-09-27", timezone_name="Europe/Istanbul",
    )
    assert result.slots == []
    assert result.source == at.SOURCE_UNAVAILABLE
    assert result.sample_size == 0
    assert "hour-of-day dimension" in result.reason


# ---------------------------------------------------------------------------
# 9 - performance inference provider
# ---------------------------------------------------------------------------

def test_performance_provider_returns_slots_from_real_videos(monkeypatch):
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]  # Friday 20:00 Istanbul
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = at.PerformanceTimingProvider().get_audience_timing(
        start_date="2026-06-01", end_date="2026-09-27", timezone_name="Europe/Istanbul",
    )
    assert result.source == SOURCE_INFERRED_PERFORMANCE
    assert result.sample_size == 6
    assert len(result.slots) == 1
    slot = result.slots[0]
    assert slot.weekday == 4  # Friday
    assert slot.hour == 20
    assert slot.sample_size == 6  # total videos analyzed, not per-bucket count
    assert slot.source == SOURCE_INFERRED_PERFORMANCE


# ---------------------------------------------------------------------------
# 10 - insufficient sample
# ---------------------------------------------------------------------------

def test_performance_provider_reports_insufficient_below_min_videos(monkeypatch):
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: [_video("v1", "2026-09-04T17:00:00+00:00", 10)])
    result = at.PerformanceTimingProvider().get_audience_timing(
        start_date="2026-06-01", end_date="2026-09-27", timezone_name="Europe/Istanbul",
    )
    assert result.slots == []
    assert result.sample_size == 1
    assert result.confidence == at.CONFIDENCE_INSUFFICIENT
    assert "below YOUTUBE_ANALYTICS_MIN_VIDEOS" in result.reason


# ---------------------------------------------------------------------------
# 11/12/13 - confidence tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count,expected", [
    (0, at.CONFIDENCE_INSUFFICIENT),
    (4, at.CONFIDENCE_INSUFFICIENT),
    (5, at.CONFIDENCE_LOW),
    (9, at.CONFIDENCE_LOW),
    (10, at.CONFIDENCE_MEDIUM),
    (19, at.CONFIDENCE_MEDIUM),
    (20, at.CONFIDENCE_HIGH),
    (100, at.CONFIDENCE_HIGH),
])
def test_confidence_for_video_count_thresholds(count, expected):
    assert at.confidence_for_video_count(count) == expected


# ---------------------------------------------------------------------------
# 14 - timezone conversion
# ---------------------------------------------------------------------------

def test_bucketing_converts_published_at_into_target_timezone(monkeypatch):
    # 22:00 UTC on a Thursday = 01:00 Friday in Europe/Istanbul (UTC+3, no DST in Sep 2026 - see below).
    videos = [_video(f"v{i}", "2026-09-03T22:00:00+00:00", 10) for i in range(5)]
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = at.PerformanceTimingProvider().get_audience_timing(
        start_date="2026-06-01", end_date="2026-09-27", timezone_name="Europe/Istanbul",
    )
    assert len(result.slots) == 1
    slot = result.slots[0]
    assert slot.hour == 1
    assert slot.weekday == 4  # Thursday 22:00 UTC -> Friday 01:00 local (weekday 4 = Friday)


# ---------------------------------------------------------------------------
# 15 - DST
# ---------------------------------------------------------------------------

def test_bucketing_handles_dst_spring_forward_in_target_timezone(monkeypatch):
    """America/New_York springs forward on 2026-03-08 (2am -> 3am). A video
    published well after the transition must bucket by its correct local
    hour, not a naively-offset one."""
    videos = [_video(f"v{i}", "2026-03-09T18:00:00+00:00", 10) for i in range(5)]  # 14:00 EDT (UTC-4)
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = at.PerformanceTimingProvider().get_audience_timing(
        start_date="2026-01-01", end_date="2026-09-27", timezone_name="America/New_York",
    )
    assert len(result.slots) == 1
    assert result.slots[0].hour == 14


# ---------------------------------------------------------------------------
# 16 - weekday/hour grouping
# ---------------------------------------------------------------------------

def test_videos_are_grouped_into_separate_buckets_by_weekday_and_hour(monkeypatch):
    videos = (
        [_video(f"fri20-{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(3)]  # Friday 20:00 Istanbul
        + [_video(f"mon10-{i}", "2026-08-31T07:00:00+00:00", 50) for i in range(3)]  # Monday 10:00 Istanbul
    )
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    result = at.PerformanceTimingProvider().get_audience_timing(
        start_date="2026-06-01", end_date="2026-09-27", timezone_name="Europe/Istanbul",
    )
    buckets = {(s.weekday, s.hour) for s in result.slots}
    assert (4, 20) in buckets  # Friday
    assert (0, 10) in buckets  # Monday
    assert len(result.slots) == 2
    # Every slot's sample_size is the TOTAL videos analyzed (6), not the
    # per-bucket count (3) - see module docstring.
    assert all(s.sample_size == 6 for s in result.slots)


# ---------------------------------------------------------------------------
# 17 - normalized performance score
# ---------------------------------------------------------------------------

def test_normalized_performance_score_is_bounded_and_deterministic():
    video_top = {"view_count": 100}
    video_half = {"view_count": 50}
    assert at._normalized_performance_score(video_top, None, max_views=100) == 1.0
    assert at._normalized_performance_score(video_half, None, max_views=100) == 0.5
    assert at._normalized_performance_score(video_top, None, max_views=0) == 0.0
    # Calling it twice with identical inputs gives identical output - no randomness.
    assert (at._normalized_performance_score(video_half, None, max_views=100)
            == at._normalized_performance_score(video_half, None, max_views=100))


def test_normalized_performance_score_blends_real_engagement_when_present():
    video = {"view_count": 100}
    engagement = {"average_view_percentage": 60.0}
    score = at._normalized_performance_score(video, engagement, max_views=100)
    # view_component=1.0, engagement_component=0.6 -> average=0.8
    assert score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 18 - no fabricated metrics (engagement API failure degrades gracefully)
# ---------------------------------------------------------------------------

def test_engagement_failure_degrades_to_view_count_only_scoring(monkeypatch):
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(5)]
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)

    def _raise(*a, **k):
        raise ac.AnalyticsUnavailableError("engagement API down")

    monkeypatch.setattr(ac, "query_video_engagement", _raise)

    result = at.PerformanceTimingProvider().get_audience_timing(
        start_date="2026-06-01", end_date="2026-09-27", timezone_name="Europe/Istanbul",
    )
    # Never raises, never fabricates an engagement figure - still produces a
    # real, view-count-only-scored result.
    assert result.sample_size == 5
    assert len(result.slots) == 1
    assert result.slots[0].score == 1.0  # all videos tied at max_views -> 1.0 view-only score


# ---------------------------------------------------------------------------
# CompositeAudienceTimingProvider - priority order, caching, "never raises"
# ---------------------------------------------------------------------------

def test_composite_provider_falls_back_to_bootstrap_shape_when_inference_disabled(monkeypatch, isolated_storage):
    monkeypatch.setattr(settings, "YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE", False)
    monkeypatch.setattr(
        ac, "list_recent_uploads",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the API when inference is off")),
    )
    provider = at.CompositeAudienceTimingProvider()
    assert provider.get_best_publish_slots() == []


def test_composite_provider_returns_real_slots_from_performance_inference(monkeypatch, isolated_storage):
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]
    monkeypatch.setattr(ac, "list_recent_uploads", lambda **k: videos)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    provider = at.CompositeAudienceTimingProvider(timezone_name="Europe/Istanbul")
    slots = provider.get_best_publish_slots()
    assert len(slots) == 1
    assert slots[0].source == SOURCE_INFERRED_PERFORMANCE


def test_composite_provider_uses_cached_result_without_recalling_the_api(monkeypatch, isolated_storage):
    videos = [_video(f"v{i}", "2026-09-04T17:00:00+00:00", 100) for i in range(6)]
    call_count = {"n": 0}

    def _list_recent_uploads(**k):
        call_count["n"] += 1
        return videos

    monkeypatch.setattr(ac, "list_recent_uploads", _list_recent_uploads)
    monkeypatch.setattr(ac, "query_video_engagement", lambda *a, **k: {})

    provider = at.CompositeAudienceTimingProvider(timezone_name="Europe/Istanbul")
    first = provider.get_best_publish_slots()
    second = provider.get_best_publish_slots()
    assert call_count["n"] == 1
    assert len(first) == len(second) == 1
    assert first[0].weekday == second[0].weekday and first[0].hour == second[0].hour


def test_composite_provider_never_raises_on_an_unexpected_api_exception(monkeypatch, isolated_storage):
    def _explode(**k):
        raise ac.AnalyticsUnavailableError("channel not found")

    monkeypatch.setattr(ac, "list_recent_uploads", _explode)
    provider = at.CompositeAudienceTimingProvider()
    assert provider.get_best_publish_slots() == []  # never raises


def test_composite_provider_never_raises_even_on_a_totally_unexpected_error(monkeypatch, isolated_storage):
    """Defense-in-depth (last line, in get_best_publish_slots() itself): even
    a non-AnalyticsUnavailableError exception from deep inside
    list_recent_uploads (a genuine programming bug, say) must never escape
    and break the approval flow - it degrades to an empty slot list exactly
    like a documented AnalyticsUnavailableError would."""
    def _explode(**k):
        raise ValueError("totally unexpected")

    monkeypatch.setattr(ac, "list_recent_uploads", _explode)
    provider = at.CompositeAudienceTimingProvider()
    assert provider.get_best_publish_slots() == []  # must not raise


def test_get_audience_timing_provider_factory_returns_a_composite_provider():
    provider = at.get_audience_timing_provider(channel_key="english")
    assert isinstance(provider, at.CompositeAudienceTimingProvider)
    assert provider.channel_key == "english"
