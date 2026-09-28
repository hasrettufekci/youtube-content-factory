"""
AI FOUND THIS - YouTube Analytics Timing V1 diagnostic CLI.

Read-only, ALWAYS: never uploads to YouTube, never sends a Telegram
message, never mutates the DB (no publish_schedules row, no
analytics_timing_cache write beyond what the normal --refresh-aware caching
path would do for a real approval anyway, no approval/video row). Exists so
a human can see exactly what CompositeAudienceTimingProvider would report
right now - real data, an honest "unavailable", or the disabled/off state -
without touching production state.

Usage:
    python -m youtube.analytics_timing_demo --days 90 --timezone Europe/Istanbul
    python -m youtube.analytics_timing_demo --channel-id english --days 30 --refresh
    python -m youtube.analytics_timing_demo --dry-run
"""
from __future__ import annotations

import argparse
import sys

from core.config import settings
from core.logging_setup import get_logger

logger = get_logger("analytics_timing_demo")


def diagnose(channel_key: str | None = None, days: int | None = None,
             timezone: str | None = None, refresh: bool = False) -> dict:
    """Pure/read-only. Returns a dict describing exactly what would be used
    for a real publish-time decision right now:
        {"enabled": bool, "channel_key": str | None, "timezone": str,
         "lookback_days": int, "source": str, "sample_size": int,
         "confidence": str, "reason": str, "slots": list[PublishSlot]}

    When YOUTUBE_ANALYTICS_ENABLED is False, no API call (and no cache
    read/write) happens at all - this mirrors exactly what
    CompositeAudienceTimingProvider.get_best_publish_slots() does for a
    real approval."""
    from youtube import analytics_timing

    resolved_timezone = timezone or settings.YOUTUBE_ANALYTICS_TIMEZONE
    resolved_days = days if days is not None else settings.YOUTUBE_ANALYTICS_LOOKBACK_DAYS

    if not settings.YOUTUBE_ANALYTICS_ENABLED:
        return {
            "enabled": False, "channel_key": channel_key, "timezone": resolved_timezone,
            "lookback_days": resolved_days, "source": "disabled", "sample_size": 0,
            "confidence": "n/a",
            "reason": "YOUTUBE_ANALYTICS_ENABLED=false - Smart Publish uses the bootstrap default only.",
            "slots": [],
        }

    if refresh:
        from storage import analytics_timing_cache as cache
        cache_key = cache.make_cache_key(channel_key, resolved_timezone, resolved_days)
        try:
            with cache.connect() as conn:
                conn.execute("DELETE FROM analytics_timing_cache WHERE cache_key = ?", (cache_key,))
        except Exception as e:
            logger.info("[ANALYTICS_DEMO] cache invalidation failed (continuing anyway): %s", e)

    provider = analytics_timing.CompositeAudienceTimingProvider(
        channel_key=channel_key, timezone_name=resolved_timezone, lookback_days=resolved_days,
    )
    # get_best_publish_slots() never raises (see youtube/analytics_timing.py) -
    # any real failure already degraded to an empty list internally, so this
    # tool has nothing further to catch.
    slots = provider.get_best_publish_slots()

    if not slots:
        return {
            "enabled": True, "channel_key": channel_key, "timezone": resolved_timezone,
            "lookback_days": resolved_days, "source": "unavailable", "sample_size": 0,
            "confidence": "insufficient",
            "reason": (
                "No usable audience-timing data right now (insufficient sample, missing/insufficient "
                "OAuth scope, or an API error) - Smart Publish would fall back to the bootstrap default."
            ),
            "slots": [],
        }

    best = max(slots, key=lambda s: (s.score, s.sample_size))
    return {
        "enabled": True, "channel_key": channel_key, "timezone": resolved_timezone,
        "lookback_days": resolved_days, "source": best.source, "sample_size": best.sample_size,
        "confidence": analytics_timing.confidence_for_video_count(best.sample_size),
        "reason": f"{len(slots)} candidate slot(s) inferred from {best.sample_size} video(s).",
        "slots": sorted(slots, key=lambda s: s.score, reverse=True),
    }


_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview YouTube Analytics Timing V1's audience-timing data - read-only, never "
                    "touches YouTube upload/Telegram/production DB rows."
    )
    parser.add_argument("--channel-id", dest="channel_key", default=None,
                         help="channel key from YOUTUBE_CHANNELS (default: the legacy single-channel config)")
    parser.add_argument("--days", type=int, default=None,
                         help="lookback window in days (default: settings.YOUTUBE_ANALYTICS_LOOKBACK_DAYS)")
    parser.add_argument("--timezone", default=None,
                         help="IANA timezone (default: settings.YOUTUBE_ANALYTICS_TIMEZONE)")
    parser.add_argument("--refresh", action="store_true",
                         help="bypass and clear any cached result before looking up fresh data")
    parser.add_argument("--dry-run", action="store_true",
                         help="accepted for symmetry with the project's other diagnostic CLIs - this tool "
                              "is ALWAYS read-only, with or without this flag")
    args = parser.parse_args(argv)

    try:
        result = diagnose(channel_key=args.channel_key, days=args.days,
                           timezone=args.timezone, refresh=args.refresh)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print("YOUTUBE ANALYTICS TIMING")
    print("------------------------")
    print(f"channel: {result['channel_key'] or '(default)'}")
    print(f"period: last {result['lookback_days']} day(s)")
    print(f"timezone: {result['timezone']}")
    print()
    if not result["enabled"]:
        print("STATUS: disabled (YOUTUBE_ANALYTICS_ENABLED=false)")
        print(result["reason"])
        print()
        print("DRY RUN:" if args.dry_run else "NOTE:")
        print("No YouTube action")
        print("No Telegram action")
        print("No DB mutation")
        return 0

    print(f"source: {result['source']}")
    print(f"videos analyzed: {result['sample_size']}")
    print(f"confidence: {result['confidence']}")
    print()
    if result["slots"]:
        print("BEST SLOTS:")
        for slot in result["slots"][:5]:
            print(f"  {_WEEKDAY_NAMES[slot.weekday]} {slot.hour:02d}:{slot.minute:02d} "
                  f"{slot.timezone} - score={slot.score:.3f} sample_size={slot.sample_size}")
        print()
        print("NOTE:")
        print(
            "These are INFERRED from past video performance (view count / engagement), not a direct "
            "reading of when the audience is online - the real YouTube Analytics API v2 has no "
            "hour-of-day dimension, so a direct reading is not obtainable (see "
            "youtube/analytics_timing.py's module docstring)."
        )
    else:
        print("BEST SLOTS: (none)")
        print()
        print("NOTE:")
        print(result["reason"])
    print()
    print("DRY RUN:" if args.dry_run else "NOTE:")
    print("No YouTube action")
    print("No Telegram action")
    print("No DB mutation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
