"""
AI FOUND THIS - YouTube Analytics Timing V1.

Extends Smart Publish Scheduler V1 (youtube/smart_publish.py) with a REAL,
opt-in (YOUTUBE_ANALYTICS_ENABLED) source of audience-timing data, instead
of Smart Publish V1's permanent bootstrap-only fallback. Purely additive:
with the feature disabled (the library default), get_audience_timing_provider()
still exists but CompositeAudienceTimingProvider.get_best_publish_slots()
returns [] immediately without a single API call - calculate_publish_time()
then falls back to the exact same bootstrap behavior Smart Publish V1 has
always had.

WHY THIS IS "inferred_performance", NOT A DIRECT AUDIENCE-TIMING READING
-------------------------------------------------------------------------
The spec asked this module to examine the REAL YouTube Analytics API's
supported dimensions/metrics rather than assume anything. Verified via live
WebFetch against https://developers.google.com/youtube/analytics/dimensions
and .../metrics on 2026-09-27: the API's only time-related dimensions are
`day` and `month`. There is NO hour-of-day / time-of-day dimension. YouTube
Studio's own "When your viewers are on YouTube" chart is generated from
data that is simply not exposed by this public API.

That makes a genuine "direct" audience-online-by-hour provider structurally
impossible to build against the real API - DirectAudienceTimingProvider
below exists to say so honestly (it always reports unavailable) rather than
silently omit the idea or, worse, fake an hour-level reading that was never
really measured.

What IS real and available: each video's own `publishedAt` timestamp (Data
API v3) and per-video performance TOTALS - views, estimatedMinutesWatched,
averageViewDuration, averageViewPercentage, subscribersGained (Analytics
API v2, `dimensions="video"`, no time dimension). PerformanceTimingProvider
below uses exactly this: it buckets a channel's own upload history by the
weekday/hour its videos were PUBLISHED (converted into the target
timezone), scores each video by its real, unweighted performance, and
reports the best-performing bucket(s) as the inferred "good time to
publish". This is a real, defensible signal (better-performing videos
cluster around better publish times, all else equal) - but it is an
INFERENCE from performance, not a direct measurement of when the audience
is online, and every slot this provider emits is labeled
source=SOURCE_INFERRED_PERFORMANCE (never SOURCE_ANALYTICS) so nothing
downstream can mistake one for the other.

CONFIDENCE / SAMPLE SIZE
-------------------------------------------------------------------------
Two DISTINCT, deliberately un-conflated scales exist in this codebase:
  - smart_publish.confidence_for_sample_size() (10/30 thresholds) - unchanged,
    answers "how many observations support THIS ONE SLOT". Still what
    ultimately sets PublishDecision.confidence for every source, including
    inferred_performance (see smart_publish.calculate_publish_time()).
  - confidence_for_video_count() below (5/10/20 thresholds, configurable via
    YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW/MEDIUM/HIGH) - answers "how many TOTAL
    videos were analyzed for this channel-wide inference", and is what
    AudienceTimingResult.confidence reports (a diagnostic-level summary, not
    what feeds PublishDecision.confidence).

By design, every PublishSlot a PerformanceTimingProvider run emits carries
`sample_size = total videos analyzed` (not a per-bucket count) - so
whichever weekday/hour bucket calculate_publish_time() picks, its resulting
confidence reflects the overall statistical weight behind the whole
inference, matching this feature's worked example: "sample_size=12,
confidence=medium" for "Videos analyzed: 12".

NEVER FABRICATED
-------------------------------------------------------------------------
Every number in every AudienceTimingResult/PublishSlot this module produces
traces back to a real API response. Any failure to obtain real data (no
OAuth credentials, insufficient scope, network/API error, a channel with
fewer than YOUTUBE_ANALYTICS_MIN_VIDEOS usable videos) results in an EMPTY
slot list and an honest "unavailable" reason - never a guessed, interpolated,
or synthetic slot. CompositeAudienceTimingProvider.get_best_publish_slots()
never raises: any exception from analytics_client is caught, logged, and
degrades to bootstrap - "API kullanilamiyorsa sistem durmamali."
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo

from core.config import settings
from core.logging_setup import get_logger
from youtube import analytics_client
from youtube.smart_publish import PublishSlot, SOURCE_INFERRED_PERFORMANCE

logger = get_logger("analytics_timing")

SOURCE_ANALYTICS_DIRECT = "analytics_direct"
SOURCE_UNAVAILABLE = "unavailable"

CONFIDENCE_INSUFFICIENT = "insufficient"
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"


@dataclass
class AudienceTimingResult:
    """The full outcome of one audience-timing lookup - returned by both
    DirectAudienceTimingProvider and PerformanceTimingProvider, and what
    gets serialized into storage/analytics_timing_cache.py's cache rows."""
    slots: list[PublishSlot]
    source: str
    sample_size: int
    confidence: str
    fetched_at: datetime
    start_date: str
    end_date: str
    timezone: str
    reason: str = ""


def confidence_for_video_count(video_count: int) -> str:
    """YouTube Analytics Timing V1's own video-COUNT confidence scale -
    distinct from smart_publish.confidence_for_sample_size() (see module
    docstring). Thresholds are configurable
    (YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW/MEDIUM/HIGH, defaults 5/10/20)."""
    if video_count < settings.YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW:
        return CONFIDENCE_INSUFFICIENT
    if video_count < settings.YOUTUBE_ANALYTICS_MIN_SAMPLE_MEDIUM:
        return CONFIDENCE_LOW
    if video_count < settings.YOUTUBE_ANALYTICS_MIN_SAMPLE_HIGH:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_HIGH


class DirectAudienceTimingProvider:
    """Always reports unavailable - see this module's docstring for why a
    genuine direct (hour-level, "audience online now") reading cannot be
    obtained from the real, documented YouTube Analytics API v2 (only
    day/month time dimensions exist). Kept as an explicit class, rather
    than simply omitted, so:
      (a) the priority order the spec asked for (direct -> inference ->
          bootstrap) is honestly represented in code, not silently skipped;
      (b) if a future YouTube API version ever adds a real hour-of-day
          dimension, only this one class needs to change - nothing else in
          this module or in smart_publish.py assumes anything about HOW a
          slot was obtained.
    Never makes a network call - there is nothing this could honestly ask
    the API for."""

    def get_audience_timing(
        self, *, channel_key: str | None, start_date: str, end_date: str, timezone_name: str,
    ) -> AudienceTimingResult:
        return AudienceTimingResult(
            slots=[], source=SOURCE_UNAVAILABLE, sample_size=0, confidence=CONFIDENCE_INSUFFICIENT,
            fetched_at=datetime.now(dt_timezone.utc), start_date=start_date, end_date=end_date,
            timezone=timezone_name,
            reason=(
                "The YouTube Analytics API v2 has no hour-of-day dimension (only day/month) - "
                "direct audience-online-by-hour timing cannot be obtained from the real API."
            ),
        )


def _normalized_performance_score(video: dict, engagement: dict | None, max_views: int) -> float:
    """Deterministic, reproducible score in [0.0, 1.0] for one video - never
    randomized, never a placeholder. Two real signals are blended when both
    are available:
      - view_count / max_views_in_analyzed_set (always available - Data
        API v3 statistics.viewCount requires no Analytics scope at all).
      - average_view_percentage / 100 (Analytics API v2 - only included
        when query_video_engagement() actually returned a row for this
        video; if the Analytics API call failed or returned nothing for
        this video, the score degrades gracefully to view-count-only
        rather than inventing an engagement figure).
    When both are present, the score is their simple average. This formula
    is intentionally simple and fully documented rather than a tuned/opaque
    ranking - the spec requires the scoring to be explainable, not
    optimal."""
    view_component = (video["view_count"] / max_views) if max_views > 0 else 0.0
    if engagement is not None and "average_view_percentage" in engagement:
        engagement_component = max(0.0, min(1.0, engagement["average_view_percentage"] / 100.0))
        return (view_component + engagement_component) / 2.0
    return view_component


class PerformanceTimingProvider:
    """Buckets a channel's own upload history by the (weekday, hour) its
    videos were published, in the target timezone, and scores each bucket
    by the real, normalized performance of the videos published in it. See
    module docstring for the full "why inferred_performance, not analytics"
    reasoning and the sample_size convention."""

    def __init__(self, channel_key: str | None = None):
        self.channel_key = channel_key

    def get_audience_timing(
        self, *, channel_key: str | None = None, start_date: str, end_date: str, timezone_name: str,
    ) -> AudienceTimingResult:
        channel_key = channel_key if channel_key is not None else self.channel_key
        now_dt = datetime.now(dt_timezone.utc)
        lookback_days = (
            datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)
        ).days or settings.YOUTUBE_ANALYTICS_LOOKBACK_DAYS

        videos = analytics_client.list_recent_uploads(
            channel_key=channel_key, lookback_days=lookback_days,
        )

        if len(videos) < settings.YOUTUBE_ANALYTICS_MIN_VIDEOS:
            return AudienceTimingResult(
                slots=[], source=SOURCE_INFERRED_PERFORMANCE, sample_size=len(videos),
                confidence=CONFIDENCE_INSUFFICIENT, fetched_at=now_dt,
                start_date=start_date, end_date=end_date, timezone=timezone_name,
                reason=(
                    f"Only {len(videos)} video(s) with usable stats in the last {lookback_days} "
                    f"day(s) - below YOUTUBE_ANALYTICS_MIN_VIDEOS={settings.YOUTUBE_ANALYTICS_MIN_VIDEOS}, "
                    "refusing to infer a publish-time preference from too small a sample."
                ),
            )

        engagement: dict[str, dict] = {}
        if settings.YOUTUBE_ANALYTICS_ENABLED:
            try:
                engagement = analytics_client.query_video_engagement(
                    channel_key, [v["video_id"] for v in videos], start_date, end_date,
                )
            except analytics_client.AnalyticsUnavailableError as e:
                # Engagement metrics are an enrichment, not a requirement -
                # degrade gracefully to view-count-only scoring rather than
                # failing the whole inference (spec: API unavailable must
                # not stop the pipeline).
                logger.info("[ANALYTICS] engagement metrics unavailable, scoring by view count only: %s", e)
                engagement = {}

        tz = ZoneInfo(timezone_name)
        max_views = max((v["view_count"] for v in videos), default=0)
        buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
        for video in videos:
            published_at = datetime.fromisoformat(video["published_at"]).astimezone(tz)
            score = _normalized_performance_score(video, engagement.get(video["video_id"]), max_views)
            buckets[(published_at.weekday(), published_at.hour)].append(score)

        total_analyzed = len(videos)
        slots = [
            PublishSlot(
                weekday=weekday, hour=hour, minute=0, timezone=timezone_name,
                score=sum(scores) / len(scores),
                # Deliberately the TOTAL videos analyzed, not this bucket's
                # own count - see module docstring's "CONFIDENCE / SAMPLE
                # SIZE" section for why.
                sample_size=total_analyzed,
                source=SOURCE_INFERRED_PERFORMANCE,
            )
            for (weekday, hour), scores in buckets.items()
        ]

        return AudienceTimingResult(
            slots=slots, source=SOURCE_INFERRED_PERFORMANCE, sample_size=total_analyzed,
            confidence=confidence_for_video_count(total_analyzed), fetched_at=now_dt,
            start_date=start_date, end_date=end_date, timezone=timezone_name,
            reason=f"Inferred from {total_analyzed} video(s) published in the last {lookback_days} day(s).",
        )


def _slot_to_dict(slot: PublishSlot) -> dict:
    return {
        "weekday": slot.weekday, "hour": slot.hour, "minute": slot.minute,
        "timezone": slot.timezone, "score": slot.score, "sample_size": slot.sample_size,
        "source": slot.source,
    }


def _slot_from_dict(data: dict) -> PublishSlot:
    return PublishSlot(
        weekday=data["weekday"], hour=data["hour"], minute=data.get("minute", 0),
        timezone=data["timezone"], score=data["score"], sample_size=data["sample_size"],
        source=data["source"],
    )


class CompositeAudienceTimingProvider:
    """Implements youtube.smart_publish.AudienceTimingProvider's protocol
    (get_best_publish_slots() -> list[PublishSlot]) - this is the object
    youtube/telegram_approval.py actually passes as calculate_publish_time()'s
    `analytics=` argument when YOUTUBE_ANALYTICS_ENABLED. Orchestrates the
    spec's priority order (direct -> performance inference -> [caller falls
    back to bootstrap when this returns []]) and the TTL cache, and NEVER
    raises - any failure anywhere below is caught, logged, and results in an
    empty list so the approval/upload flow is never at risk from an
    Analytics API problem."""

    def __init__(
        self,
        channel_key: str | None = None,
        timezone_name: str | None = None,
        lookback_days: int | None = None,
    ):
        self.channel_key = channel_key
        self.timezone_name = timezone_name or settings.YOUTUBE_ANALYTICS_TIMEZONE
        self.lookback_days = lookback_days or settings.YOUTUBE_ANALYTICS_LOOKBACK_DAYS

    def get_best_publish_slots(self) -> list[PublishSlot]:
        if not settings.YOUTUBE_ANALYTICS_ENABLED:
            return []

        from storage import analytics_timing_cache as cache

        cache_key = cache.make_cache_key(self.channel_key, self.timezone_name, self.lookback_days)
        try:
            cached = cache.get_fresh(cache_key)
            if cached is not None:
                return [_slot_from_dict(s) for s in cached["slots"]]
        except Exception as e:
            # A cache read failure is never fatal - just re-derive.
            logger.info("[ANALYTICS] cache read failed for key=%s, re-fetching: %s", cache_key, e)

        now_dt = datetime.now(dt_timezone.utc)
        try:
            result = self._fetch_fresh()
        except Exception as e:
            # Absolute last line of defense (spec: "API kullanilamiyorsa
            # sistem durmamali" - an Analytics/Data API problem must never
            # break the approval/upload flow). PerformanceTimingProvider
            # itself only ever raises analytics_client.AnalyticsUnavailableError
            # by design, but this catches literally anything else too (a
            # genuine bug, a malformed API response shape, etc.) so
            # get_best_publish_slots() truly never raises.
            logger.info("[ANALYTICS] unexpected failure computing audience timing, falling back to bootstrap: %s", e)
            start_date = (now_dt - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            result = AudienceTimingResult(
                slots=[], source=SOURCE_UNAVAILABLE, sample_size=0, confidence=CONFIDENCE_INSUFFICIENT,
                fetched_at=now_dt, start_date=start_date, end_date=now_dt.strftime("%Y-%m-%d"),
                timezone=self.timezone_name, reason=str(e),
            )

        try:
            cache.set_cache(
                cache_key=cache_key, channel_key=self.channel_key, timezone_name=self.timezone_name,
                lookback_days=self.lookback_days, source=result.source, sample_size=result.sample_size,
                confidence=result.confidence, slots=[_slot_to_dict(s) for s in result.slots],
                ttl_hours=settings.YOUTUBE_ANALYTICS_CACHE_TTL_HOURS, fetched_at=result.fetched_at,
            )
        except Exception as e:
            logger.info("[ANALYTICS] cache write failed for key=%s: %s", cache_key, e)

        return result.slots

    def _fetch_fresh(self) -> AudienceTimingResult:
        now_dt = datetime.now(dt_timezone.utc)
        start_date = (now_dt - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        end_date = now_dt.strftime("%Y-%m-%d")

        direct = DirectAudienceTimingProvider().get_audience_timing(
            channel_key=self.channel_key, start_date=start_date, end_date=end_date,
            timezone_name=self.timezone_name,
        )
        if direct.slots:
            return direct

        if not settings.YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE:
            return direct

        try:
            return PerformanceTimingProvider(self.channel_key).get_audience_timing(
                start_date=start_date, end_date=end_date, timezone_name=self.timezone_name,
            )
        except analytics_client.AnalyticsUnavailableError as e:
            logger.info("[ANALYTICS] performance inference unavailable, falling back to bootstrap: %s", e)
            return AudienceTimingResult(
                slots=[], source=SOURCE_UNAVAILABLE, sample_size=0, confidence=CONFIDENCE_INSUFFICIENT,
                fetched_at=now_dt, start_date=start_date, end_date=end_date, timezone=self.timezone_name,
                reason=str(e),
            )


def get_audience_timing_provider(channel_key: str | None = None) -> CompositeAudienceTimingProvider:
    """Factory used by youtube/telegram_approval.py and
    youtube/analytics_timing_demo.py - the one place either module needs to
    know how a CompositeAudienceTimingProvider is constructed."""
    return CompositeAudienceTimingProvider(channel_key=channel_key)
