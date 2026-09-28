"""
youtube/smart_publish.py tests - Smart Publish Scheduler V1's decision
engine (Faz 2/3/4/11/12).

V1 honesty rule under test throughout: with no real audience-timing data
(both DefaultAudienceTimingProvider and YouTubeAnalyticsAudienceTimingProvider
always return an empty slot list), calculate_publish_time() must ALWAYS
fall back to the deterministic bootstrap slot - source="bootstrap",
confidence="bootstrap" - and must never fabricate a higher confidence or a
score. No network/DB access anywhere in this module or these tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo

import pytest

from core.config import settings
from youtube import smart_publish as sp


@pytest.fixture(autouse=True)
def _default_smart_publish_settings(monkeypatch):
    """Deterministic defaults for every test in this file, independent of
    whatever .env/.env.example ships - explicit > implicit."""
    monkeypatch.setattr(settings, "SMART_PUBLISH_DEFAULT_HOUR", 20)
    monkeypatch.setattr(settings, "SMART_PUBLISH_DEFAULT_MINUTE", 0)
    monkeypatch.setattr(settings, "SMART_PUBLISH_MIN_DELAY_MINUTES", 60)
    monkeypatch.setattr(settings, "SMART_PUBLISH_LOOKAHEAD_DAYS", 7)


class _FakeAnalyticsProvider:
    def __init__(self, slots):
        self._slots = slots

    def get_best_publish_slots(self):
        return list(self._slots)


# ---------------------------------------------------------------------------
# 1/2 - future publish time / no past time ever
# ---------------------------------------------------------------------------

def test_scheduled_at_is_always_in_the_future():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert decision.scheduled_at > now


@pytest.mark.parametrize("hour,minute", [(0, 0), (12, 0), (19, 59), (20, 0), (20, 1), (23, 59)])
def test_scheduled_at_never_in_the_past_across_many_now_values(hour, minute):
    now = datetime(2026, 9, 27, hour, minute, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert decision.scheduled_at >= now  # never past, and min-delay-checked separately


# ---------------------------------------------------------------------------
# 3 - timezone correctness
# ---------------------------------------------------------------------------

def test_timezone_field_matches_the_requested_timezone():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="America/New_York")
    assert decision.timezone == "America/New_York"
    assert decision.scheduled_at.tzinfo is not None
    assert decision.scheduled_at.utcoffset() == ZoneInfo("America/New_York").utcoffset(decision.scheduled_at.replace(tzinfo=None))


def test_scheduled_at_local_wall_clock_hour_matches_default_hour():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert decision.scheduled_at.hour == 20
    assert decision.scheduled_at.minute == 0
    assert decision.scheduled_at.tzinfo.key == "Europe/Istanbul"


def test_now_without_tzinfo_raises():
    naive_now = datetime(2026, 9, 27, 10, 0)
    with pytest.raises(ValueError):
        sp.calculate_publish_time(now=naive_now, timezone="Europe/Istanbul")


# ---------------------------------------------------------------------------
# 4 - minimum delay
# ---------------------------------------------------------------------------

def test_scheduled_at_respects_minimum_delay_minutes(monkeypatch):
    monkeypatch.setattr(settings, "SMART_PUBLISH_MIN_DELAY_MINUTES", 180)
    # "now" is already past today's default hour, in Istanbul time (20:00
    # +03:00 = 17:00 UTC) - only 30 minutes before it, well under the
    # 180-minute minimum delay, so today's slot must be rejected in favor
    # of tomorrow's.
    now = datetime(2026, 9, 27, 16, 30, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert decision.scheduled_at - now.astimezone(ZoneInfo("Europe/Istanbul")) >= timedelta(minutes=180)
    assert decision.scheduled_at.date() == datetime(2026, 9, 28).date()


def test_min_delay_pushes_to_next_day_when_default_hour_is_too_soon():
    # 19:30 Istanbul local time - only 30 minutes before the 20:00 default,
    # under the (default, from the autouse fixture) 60-minute minimum delay.
    now_local = datetime(2026, 9, 27, 19, 30, tzinfo=ZoneInfo("Europe/Istanbul"))
    now_utc = now_local.astimezone(dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now_utc, timezone="Europe/Istanbul")
    assert decision.scheduled_at.date() == datetime(2026, 9, 28).date()
    assert decision.scheduled_at.hour == 20


def test_min_delay_keeps_today_when_default_hour_is_far_enough_ahead():
    # 10:00 Istanbul local time - hours before the 20:00 default, well past
    # the 60-minute minimum delay, so today's slot must be kept.
    now_local = datetime(2026, 9, 27, 10, 0, tzinfo=ZoneInfo("Europe/Istanbul"))
    now_utc = now_local.astimezone(dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now_utc, timezone="Europe/Istanbul")
    assert decision.scheduled_at.date() == datetime(2026, 9, 27).date()
    assert decision.scheduled_at.hour == 20


# ---------------------------------------------------------------------------
# 5/6/7 - default fallback / bootstrap source / confidence bootstrap
# ---------------------------------------------------------------------------

def test_no_analytics_uses_bootstrap_source_and_confidence():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert decision.source == sp.SOURCE_BOOTSTRAP
    assert decision.confidence == sp.CONFIDENCE_BOOTSTRAP
    assert decision.reason == sp.BOOTSTRAP_REASON


def test_default_provider_returns_no_fabricated_slots():
    assert sp.DefaultAudienceTimingProvider().get_best_publish_slots() == []


def test_youtube_analytics_provider_returns_no_fabricated_slots_in_v1():
    """V1 placeholder - must never invent scores/sample sizes."""
    assert sp.YouTubeAnalyticsAudienceTimingProvider().get_best_publish_slots() == []


def test_analytics_provider_present_but_empty_still_uses_bootstrap():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(
        now=now, timezone="Europe/Istanbul", analytics=sp.DefaultAudienceTimingProvider(),
    )
    assert decision.source == sp.SOURCE_BOOTSTRAP
    assert decision.confidence == sp.CONFIDENCE_BOOTSTRAP


def test_confidence_never_fabricated_high_for_bootstrap_path():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert decision.confidence != sp.CONFIDENCE_HIGH
    assert decision.confidence != sp.CONFIDENCE_MEDIUM
    assert decision.confidence != sp.CONFIDENCE_LOW


# ---------------------------------------------------------------------------
# 8/9 - Monday slot / Sunday slot (analytics-driven path)
# ---------------------------------------------------------------------------

def test_analytics_slot_picks_the_correct_next_monday():
    # 2026-09-27 is a Sunday.
    now = datetime(2026, 9, 27, 6, 0, tzinfo=dt_timezone.utc)
    monday_slot = sp.PublishSlot(weekday=0, hour=18, timezone="Europe/Istanbul", score=0.9, sample_size=42, source="test")
    decision = sp.calculate_publish_time(
        now=now, timezone="Europe/Istanbul", analytics=_FakeAnalyticsProvider([monday_slot]),
    )
    assert decision.scheduled_at.weekday() == 0  # Monday
    assert decision.scheduled_at.hour == 18
    assert decision.scheduled_at.date() == datetime(2026, 9, 28).date()
    assert decision.source == sp.SOURCE_ANALYTICS
    assert decision.confidence == sp.CONFIDENCE_HIGH  # sample_size=42 >= 30


def test_analytics_slot_picks_the_correct_next_sunday():
    # 2026-09-27 is itself a Sunday, at an hour already past the requested
    # slot hour, so the NEXT Sunday (7 days later) must be chosen.
    now = datetime(2026, 9, 27, 20, 0, tzinfo=dt_timezone.utc)
    sunday_slot = sp.PublishSlot(weekday=6, hour=15, timezone="Europe/Istanbul", score=0.8, sample_size=15, source="test")
    decision = sp.calculate_publish_time(
        now=now, timezone="Europe/Istanbul", analytics=_FakeAnalyticsProvider([sunday_slot]),
    )
    assert decision.scheduled_at.weekday() == 6  # Sunday
    assert decision.scheduled_at.hour == 15
    assert decision.scheduled_at.date() == datetime(2026, 10, 4).date()
    assert decision.confidence == sp.CONFIDENCE_MEDIUM  # sample_size=15 -> medium


def test_analytics_picks_the_highest_scoring_usable_slot():
    now = datetime(2026, 9, 27, 6, 0, tzinfo=dt_timezone.utc)
    low = sp.PublishSlot(weekday=1, hour=10, timezone="Europe/Istanbul", score=0.2, sample_size=50, source="test")
    high = sp.PublishSlot(weekday=2, hour=14, timezone="Europe/Istanbul", score=0.95, sample_size=50, source="test")
    decision = sp.calculate_publish_time(
        now=now, timezone="Europe/Istanbul", analytics=_FakeAnalyticsProvider([low, high]),
    )
    assert decision.scheduled_at.weekday() == 2
    assert decision.scheduled_at.hour == 14


def test_analytics_slot_with_zero_sample_size_is_never_trusted():
    """A provider bug (or a malicious/broken future implementation) handing
    back a slot with sample_size=0 must never be used - that would be
    exactly the fabricated-data case the spec forbids."""
    now = datetime(2026, 9, 27, 6, 0, tzinfo=dt_timezone.utc)
    fake_slot = sp.PublishSlot(weekday=0, hour=9, timezone="Europe/Istanbul", score=1.0, sample_size=0, source="test")
    decision = sp.calculate_publish_time(
        now=now, timezone="Europe/Istanbul", analytics=_FakeAnalyticsProvider([fake_slot]),
    )
    assert decision.source == sp.SOURCE_BOOTSTRAP


def test_analytics_slot_beyond_lookahead_falls_back_to_bootstrap(monkeypatch):
    monkeypatch.setattr(settings, "SMART_PUBLISH_LOOKAHEAD_DAYS", 2)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=dt_timezone.utc)  # Sunday
    # Friday is 5 days away - beyond a 2-day lookahead window.
    far_slot = sp.PublishSlot(weekday=4, hour=12, timezone="Europe/Istanbul", score=0.99, sample_size=100, source="test")
    decision = sp.calculate_publish_time(
        now=now, timezone="Europe/Istanbul", analytics=_FakeAnalyticsProvider([far_slot]),
    )
    assert decision.source == sp.SOURCE_BOOTSTRAP


# ---------------------------------------------------------------------------
# 10 - midnight boundary
# ---------------------------------------------------------------------------

def test_midnight_boundary_default_hour_zero(monkeypatch):
    monkeypatch.setattr(settings, "SMART_PUBLISH_DEFAULT_HOUR", 0)
    monkeypatch.setattr(settings, "SMART_PUBLISH_DEFAULT_MINUTE", 0)
    # 23:30 local, with the (default, from the autouse fixture) 60-minute
    # minimum delay: "today's" 00:00 slot (Sep 27) is already in the past,
    # and even "tomorrow's" 00:00 (Sep 28, only 30 minutes away) is under
    # the 60-minute minimum delay - so the correct next slot is Sep 29 00:00.
    now_local = datetime(2026, 9, 27, 23, 30, tzinfo=ZoneInfo("Europe/Istanbul"))
    decision = sp.calculate_publish_time(now=now_local.astimezone(dt_timezone.utc), timezone="Europe/Istanbul")
    assert decision.scheduled_at.hour == 0
    assert decision.scheduled_at.minute == 0
    assert decision.scheduled_at.date() == datetime(2026, 9, 29).date()
    assert decision.scheduled_at > now_local
    assert decision.scheduled_at - now_local >= timedelta(minutes=60)


def test_now_exactly_at_default_slot_time_moves_to_next_day():
    # "now" is EXACTLY 20:00 Istanbul local time - the default slot itself
    # is not >= min_delay ahead of "now" (it IS "now"), so it must roll to
    # the next day, never selecting a slot equal to "now" itself.
    now_local = datetime(2026, 9, 27, 20, 0, tzinfo=ZoneInfo("Europe/Istanbul"))
    decision = sp.calculate_publish_time(now=now_local.astimezone(dt_timezone.utc), timezone="Europe/Istanbul")
    assert decision.scheduled_at.date() == datetime(2026, 9, 28).date()


# ---------------------------------------------------------------------------
# 11 - DST / timezone edge case
# ---------------------------------------------------------------------------

def test_dst_spring_forward_boundary_keeps_correct_wall_clock_hour():
    """US DST 2027 spring-forward is 2027-03-14 (clocks jump 02:00 -> 03:00).
    Scheduling from just before it must still land on the correct LOCAL
    wall-clock default hour the next day, not an hour shifted by the DST
    change (which a naive fixed-UTC-offset implementation could get wrong)."""
    now_local = datetime(2027, 3, 13, 22, 0, tzinfo=ZoneInfo("America/New_York"))
    decision = sp.calculate_publish_time(now=now_local.astimezone(dt_timezone.utc), timezone="America/New_York")
    assert decision.scheduled_at.hour == 20
    assert decision.scheduled_at.minute == 0
    assert decision.scheduled_at.date() == datetime(2027, 3, 14).date()
    assert decision.scheduled_at > now_local


def test_dst_fall_back_boundary_keeps_correct_wall_clock_hour():
    """US DST 2026 fall-back is 2026-11-01 (clocks fall 02:00 -> 01:00)."""
    now_local = datetime(2026, 10, 31, 22, 0, tzinfo=ZoneInfo("America/New_York"))
    decision = sp.calculate_publish_time(now=now_local.astimezone(dt_timezone.utc), timezone="America/New_York")
    assert decision.scheduled_at.hour == 20
    assert decision.scheduled_at.minute == 0
    assert decision.scheduled_at.date() == datetime(2026, 11, 1).date()
    assert decision.scheduled_at > now_local


def test_istanbul_has_no_dst_and_stays_at_a_fixed_utc_offset():
    """Turkey abolished DST in 2016 - a useful contrast case confirming the
    engine doesn't assume every timezone has a DST transition to handle."""
    winter = sp.calculate_publish_time(now=datetime(2026, 1, 15, 6, 0, tzinfo=dt_timezone.utc), timezone="Europe/Istanbul")
    summer = sp.calculate_publish_time(now=datetime(2026, 7, 15, 6, 0, tzinfo=dt_timezone.utc), timezone="Europe/Istanbul")
    assert winter.scheduled_at.utcoffset() == timedelta(hours=3)
    assert summer.scheduled_at.utcoffset() == timedelta(hours=3)


# ---------------------------------------------------------------------------
# Faz 12 - confidence tier thresholds (pure function, reused by a future
# real analytics provider)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_size,expected", [
    (0, sp.CONFIDENCE_LOW), (1, sp.CONFIDENCE_LOW), (9, sp.CONFIDENCE_LOW),
    (10, sp.CONFIDENCE_MEDIUM), (29, sp.CONFIDENCE_MEDIUM),
    (30, sp.CONFIDENCE_HIGH), (1000, sp.CONFIDENCE_HIGH),
])
def test_confidence_for_sample_size_thresholds(sample_size, expected):
    assert sp.confidence_for_sample_size(sample_size) == expected


# ---------------------------------------------------------------------------
# PublishDecision / PublishSlot are plain, inspectable dataclasses
# ---------------------------------------------------------------------------

def test_publish_decision_has_all_required_fields():
    now = datetime(2026, 9, 27, 10, 0, tzinfo=dt_timezone.utc)
    decision = sp.calculate_publish_time(now=now, timezone="Europe/Istanbul")
    assert hasattr(decision, "scheduled_at")
    assert hasattr(decision, "timezone")
    assert hasattr(decision, "source")
    assert hasattr(decision, "confidence")
    assert hasattr(decision, "reason")
