"""
AI FOUND THIS - Smart Publish Scheduler V1.

The gap this fills: today, Telegram APPROVE drives an immediate PRIVATE
YouTube upload with no publish-time strategy at all. This module adds a
publish-TIME decision engine only - it does not itself touch the upload
call, the approval state machine, or any production pipeline stage (see
youtube/telegram_approval.py's _upload_to_youtube()/process_approval() for
the one, narrow place this is consulted, and core/config.py's
SMART_PUBLISH_ENABLED for the opt-in flag that keeps this 100% inert when
off).

V1 HONESTY RULE (spec: "confidence hicbir zaman sahte sekilde yuksek
gosterilmemeli" / "Kesinlikle uydurma Analytics verisi uretme"): this
channel does not yet have real YouTube Analytics audience-timing data.
Every PublishDecision produced by calculate_publish_time() in V1 therefore
comes from the deterministic SMART_PUBLISH_DEFAULT_HOUR/MINUTE fallback -
source="bootstrap", confidence="bootstrap" - never a fabricated score or a
falsely-high confidence. The AudienceTimingProvider abstraction below exists
so a REAL analytics-backed provider can be dropped in later (Faz 12) without
touching calculate_publish_time()'s calling contract; until then, both
providers shipped here always return an empty slot list.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from core.config import settings
from core.logging_setup import get_logger

logger = get_logger("smart_publish")

SOURCE_BOOTSTRAP = "bootstrap"
SOURCE_ANALYTICS = "analytics"
# YouTube Analytics Timing V1 (additive): a slot inferred from historical
# per-video performance (views/watch-time bucketed by the weekday/hour of
# each video's OWN publishedAt), used because the real YouTube Analytics
# API v2 has no hour-of-day dimension (only day/month - verified against
# https://developers.google.com/youtube/analytics/dimensions) and therefore
# cannot supply genuine "audience online by hour" data directly. See
# youtube/analytics_timing.py's module docstring for the full explanation.
# Kept textually distinct from SOURCE_ANALYTICS so a caller/report can never
# confuse a real direct-audience-timing reading (which V1's shipped
# providers never produce) with this inference.
SOURCE_INFERRED_PERFORMANCE = "inferred_performance"

CONFIDENCE_BOOTSTRAP = "bootstrap"
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"

BOOTSTRAP_REASON = "Channel does not yet have sufficient audience timing data."


@dataclass
class PublishSlot:
    """One candidate weekday/hour(/minute) publish slot, as a real (never
    fabricated) audience-timing provider would report it. weekday follows
    Python's date.weekday() convention: 0=Monday ... 6=Sunday.

    `minute` (YouTube Analytics Timing V1, additive): defaults to 0 so every
    existing caller/test that constructs a PublishSlot without it keeps
    working unchanged - the bootstrap and original analytics paths were
    always effectively minute=0 anyway."""
    weekday: int
    hour: int
    timezone: str
    score: float
    sample_size: int
    source: str
    minute: int = 0


@dataclass
class PublishDecision:
    scheduled_at: datetime  # timezone-aware, in `timezone`
    timezone: str
    source: str
    confidence: str
    reason: str
    # YouTube Analytics Timing V1 (additive, optional): the winning slot's
    # own score/sample_size, carried through for callers/diagnostics that
    # want to show *why* this time was picked - never set (stays None) for
    # a bootstrap decision, exactly like before this feature existed.
    sample_size: int | None = None
    score: float | None = None


class AudienceTimingProvider(Protocol):
    """Faz 3 abstraction: anything that can suggest candidate publish slots.
    calculate_publish_time() only ever trusts a slot whose sample_size > 0 -
    a provider that has no real data must return an empty list, never a
    slot with a made-up score/sample_size."""

    def get_best_publish_slots(self) -> list[PublishSlot]:
        ...


class DefaultAudienceTimingProvider:
    """V1's actual default. Deliberately dumb: this channel has no audience-
    timing history yet, so there is nothing honest to report - an empty
    list tells calculate_publish_time() to fall back to the deterministic
    bootstrap slot rather than inventing one here."""

    def get_best_publish_slots(self) -> list[PublishSlot]:
        return []


class YouTubeAnalyticsAudienceTimingProvider:
    """V1 placeholder, kept for backward compatibility. Always returns an
    empty list, exactly like DefaultAudienceTimingProvider - nothing
    constructs or wires this class up to calculate_publish_time() in
    practice.

    YouTube Analytics Timing V1 (AI FOUND THIS) implemented the REAL
    analytics-backed provider as a separate module instead of filling in
    this class: see youtube/analytics_timing.py's
    CompositeAudienceTimingProvider (constructed via
    get_audience_timing_provider()), which implements this same
    AudienceTimingProvider protocol and is what
    youtube/telegram_approval.py actually passes as calculate_publish_time()'s
    `analytics=` argument when settings.YOUTUBE_ANALYTICS_ENABLED. This
    class is left inert and untouched so nothing that already references it
    changes behavior."""

    def get_best_publish_slots(self) -> list[PublishSlot]:
        return []


def confidence_for_sample_size(sample_size: int) -> str:
    """Faz 12's stated thresholds - kept as a pure function so a future
    analytics provider's ranking can reuse it verbatim. Only ever applied to
    a REAL sample_size from a real provider; calculate_publish_time() never
    calls this for the bootstrap path (which always reports
    confidence=CONFIDENCE_BOOTSTRAP)."""
    if sample_size < 10:
        return CONFIDENCE_LOW
    if sample_size < 30:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_HIGH


def _next_bootstrap_slot(local_now: datetime, earliest: datetime, hour: int, minute: int) -> datetime:
    """The next occurrence of `hour:minute` (local wall-clock time in
    local_now's own tzinfo) that is >= `earliest`. This is a DAILY slot (no
    weekday constraint), so at most one extra day is ever needed."""
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    while candidate < earliest:
        candidate += timedelta(days=1)
    return candidate


def _next_weekday_slot(local_now: datetime, earliest: datetime, weekday: int, hour: int, minute: int) -> datetime:
    """The next occurrence of a specific (weekday, hour:minute) combination
    that is >= `earliest`. Used for an analytics-provided slot, which - unlike
    the daily bootstrap default - only recurs weekly."""
    days_ahead = (weekday - local_now.weekday()) % 7
    candidate = (local_now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    while candidate < earliest:
        candidate += timedelta(days=7)
    return candidate


def calculate_publish_time(
    *,
    now: datetime,
    timezone: str,
    analytics: AudienceTimingProvider | None = None,
    topic_category: str | None = None,
    language: str = "en",
) -> PublishDecision:
    """Faz 2's decision engine. `now` must be a timezone-aware datetime
    (typically datetime.now(timezone.utc)) - this never guesses "now" for
    itself, so callers/tests fully control it. `timezone` is an IANA zone
    name (e.g. "Europe/Istanbul").

    Guarantees (spec, non-negotiable in V1):
      - scheduled_at is NEVER in the past relative to `now`.
      - scheduled_at is always at least SMART_PUBLISH_MIN_DELAY_MINUTES
        ahead of `now` (upload headroom).
      - confidence is never fabricated as artificially high - the bootstrap
        fallback (the only path V1's shipped providers ever take, since
        both return an empty slot list) always reports
        confidence=CONFIDENCE_BOOTSTRAP, never "high"/"medium"/"low".

    `topic_category`/`language` are accepted now (Faz 12 abstraction) but
    unused by the bootstrap path - reserved for a future analytics provider
    that scores slots per-category/per-language."""
    if now.tzinfo is None:
        raise ValueError("calculate_publish_time(): `now` must be timezone-aware")

    tz = ZoneInfo(timezone)
    local_now = now.astimezone(tz)
    min_delay = timedelta(minutes=settings.SMART_PUBLISH_MIN_DELAY_MINUTES)
    lookahead_limit = local_now + timedelta(days=settings.SMART_PUBLISH_LOOKAHEAD_DAYS)
    earliest = local_now + min_delay

    slots = analytics.get_best_publish_slots() if analytics is not None else []
    # Never trust a slot without a genuine, positive sample_size - guards
    # against a future provider bug (or a test double) smuggling in a
    # fabricated-looking slot with no real data behind it.
    usable_slots = [s for s in slots if s.sample_size > 0]

    if usable_slots:
        best = max(usable_slots, key=lambda s: (s.score, s.sample_size))
        candidate = _next_weekday_slot(local_now, earliest, best.weekday, best.hour, best.minute)
        if candidate <= lookahead_limit:
            # Source-resolution (YouTube Analytics Timing V1, additive):
            # only the two recognized "real" sources are ever surfaced
            # as-is - anything else (including the literal "test" source
            # tests/test_smart_publish.py's own fake provider uses) still
            # resolves to the original generic SOURCE_ANALYTICS label, so
            # every pre-existing test/caller keeps its exact behavior.
            resolved_source = (
                best.source if best.source in (SOURCE_ANALYTICS, SOURCE_INFERRED_PERFORMANCE)
                else SOURCE_ANALYTICS
            )
            decision = PublishDecision(
                scheduled_at=candidate,
                timezone=timezone,
                source=resolved_source,
                confidence=confidence_for_sample_size(best.sample_size),
                reason=(
                    f"Using audience timing data from {best.source} "
                    f"(sample_size={best.sample_size}, weekday={best.weekday}, hour={best.hour})."
                ),
                sample_size=best.sample_size,
                score=best.score,
            )
            _log_decision(decision)
            return decision
        logger.info(
            "[PUBLISH] best analytics slot weekday=%d hour=%d falls outside the %d-day lookahead window - "
            "falling back to the bootstrap default",
            best.weekday, best.hour, settings.SMART_PUBLISH_LOOKAHEAD_DAYS,
        )

    candidate = _next_bootstrap_slot(
        local_now, earliest, settings.SMART_PUBLISH_DEFAULT_HOUR, settings.SMART_PUBLISH_DEFAULT_MINUTE,
    )
    decision = PublishDecision(
        scheduled_at=candidate,
        timezone=timezone,
        source=SOURCE_BOOTSTRAP,
        confidence=CONFIDENCE_BOOTSTRAP,
        reason=BOOTSTRAP_REASON,
    )
    _log_decision(decision)
    return decision


def _log_decision(decision: PublishDecision) -> None:
    """Faz 13 logging - one line per field, video_id is logged by the
    caller (youtube/telegram_approval.py), which is the only place that
    actually knows it."""
    logger.info("[PUBLISH] scheduled_at=%s", decision.scheduled_at.isoformat())
    logger.info("[PUBLISH] timezone=%s", decision.timezone)
    logger.info("[PUBLISH] source=%s", decision.source)
    logger.info("[PUBLISH] confidence=%s", decision.confidence)
    logger.info("[PUBLISH] reason=%s", decision.reason)
