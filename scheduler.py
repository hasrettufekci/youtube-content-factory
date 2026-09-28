"""
Daily automated production (spec sections 37 PHASE 5, 43 "Her gun yeni
video hazirla"). Intentionally minimal for MVP - a blocking loop, not a
service/cron system (no Celery/Kubernetes, per spec section 38).

Each day at the configured time: pull the top-scored candidate from
topic_scout, produce it with pipeline.py, and stop there - upload still
requires a human Telegram approval, same as manual runs.

    python scheduler.py --daily-at 09:00
    python scheduler.py --once            # run one production cycle now and exit (useful for testing/cron)

For real production use, prefer wiring `python scheduler.py --once` into
the OS's own cron/Task Scheduler rather than leaving this process running
indefinitely - simpler to supervise/restart than a long-lived Python loop.

DAILY TOPIC DISCOVERY (Phase A/B - added alongside the above, not replacing
it): run_topic_discovery_once()/run_daily_topic_discovery() below are a
SEPARATE schedule from run_once()/run_daily() above - they only discover,
score, persist, and send a Telegram topic-selection message (see
youtube/topic_discovery.py); they NEVER call pipeline.produce_video()
themselves. Production only starts once a human taps one of the 5 buttons
(youtube/telegram_approval_listener.py -> topic_discovery.
handle_topic_select_callback()). Configured via DAILY_TOPIC_SCOUT_ENABLED/
DAILY_TOPIC_SCOUT_TIME/DAILY_TOPIC_SCOUT_TIMEZONE (core/config.py),
defaulting to 09:00 Europe/Istanbul - never hardcoded.

    python scheduler.py --discover-once    # run discovery now and exit (useful for testing/cron)
    python scheduler.py --discover-daily   # blocking loop at DAILY_TOPIC_SCOUT_TIME/TIMEZONE
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from zoneinfo import ZoneInfo

from core.config import settings
from core.logging_setup import get_logger
from storage import db, project_store as ps
from youtube import topic_scout

logger = get_logger("scheduler")


def run_once(category: str | None = None) -> None:
    import pipeline as pipeline_module

    candidate = topic_scout.top_candidate()
    if not candidate:
        logger.warning("[SCHEDULER] no topic candidates available - add one with `python -m youtube.topic_scout --topic \"...\"`")
        return

    logger.info("[SCHEDULER] producing top candidate: %r (score=%s)", candidate["topic"], candidate["score"])
    try:
        paths = pipeline_module.produce_video(candidate["topic"], category=candidate["category"] or category)
        db.mark_topic_used(ps.normalize_topic(candidate["topic"]))
        logger.info("[SCHEDULER] video_id=%d ready for Telegram approval", paths.video_id)
    except Exception:
        logger.exception("[SCHEDULER] production cycle failed")


def run_daily(hour: int, minute: int, category: str | None = None) -> None:
    logger.info("[SCHEDULER] running daily at %02d:%02d (local time)", hour, minute)
    last_run_date = None
    while True:
        now = dt.datetime.now()
        if now.hour == hour and now.minute == minute and now.date() != last_run_date:
            run_once(category=category)
            last_run_date = now.date()
        time.sleep(20)


def run_topic_discovery_once() -> None:
    """Discovery-only cycle (Phase A/B): discover -> score -> persist a
    batch -> send Telegram topic selection. Deliberately does NOT call
    pipeline.produce_video() or anything that reaches it - production only
    starts from a human's button tap (see youtube/topic_discovery.py's
    handle_topic_select_callback(), dispatched by the existing listener)."""
    from youtube import topic_discovery

    if not settings.DAILY_TOPIC_SCOUT_ENABLED:
        logger.info("[TOPIC_DISCOVERY] DAILY_TOPIC_SCOUT_ENABLED is false - skipping")
        return
    try:
        result = topic_discovery.run_daily_discovery(triggered_by="scheduler")
        if result is None:
            logger.warning("[TOPIC_DISCOVERY] scheduled discovery found no valid candidates today")
    except Exception:
        logger.exception("[TOPIC_DISCOVERY] scheduled discovery cycle failed")


def run_daily_topic_discovery(hour: int, minute: int, timezone_name: str) -> None:
    """Blocking loop, deliberately mirroring run_daily() above exactly
    (same simple "check every 20s, fire once per calendar date" pattern)
    but on its own schedule/timezone and calling run_topic_discovery_once()
    instead of run_once() - so this never starts production, only ever
    discovery. `timezone_name` is resolved via the stdlib zoneinfo (no new
    dependency) so DAILY_TOPIC_SCOUT_TIMEZONE can be any IANA zone, not
    just the machine's local time."""
    tz = ZoneInfo(timezone_name)
    logger.info("[TOPIC_DISCOVERY] running daily at %02d:%02d %s", hour, minute, timezone_name)
    last_run_date = None
    while True:
        if not settings.DAILY_TOPIC_SCOUT_ENABLED:
            logger.info("[TOPIC_DISCOVERY] DAILY_TOPIC_SCOUT_ENABLED is false - stopping the discovery loop")
            return
        now = dt.datetime.now(tz)
        if now.hour == hour and now.minute == minute and now.date() != last_run_date:
            run_topic_discovery_once()
            last_run_date = now.date()
        time.sleep(20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily automated video production (future automation, spec Phase 5).")
    parser.add_argument("--daily-at", metavar="HH:MM", help="Run every day at this local time (blocking)")
    parser.add_argument("--once", action="store_true", help="Run a single production cycle now and exit")
    parser.add_argument("--category", default=None)
    parser.add_argument("--discover-once", action="store_true",
                         help="Run daily topic discovery now and exit (discovery + Telegram send only - never starts production)")
    parser.add_argument("--discover-daily", action="store_true",
                         help="Blocking loop: run daily topic discovery at DAILY_TOPIC_SCOUT_TIME/DAILY_TOPIC_SCOUT_TIMEZONE every day")
    args = parser.parse_args(argv)

    if args.once:
        run_once(category=args.category)
        return 0

    if args.daily_at:
        hour, minute = (int(x) for x in args.daily_at.split(":"))
        run_daily(hour, minute, category=args.category)
        return 0

    if args.discover_once:
        run_topic_discovery_once()
        return 0

    if args.discover_daily:
        hour, minute = (int(x) for x in settings.DAILY_TOPIC_SCOUT_TIME.split(":"))
        run_daily_topic_discovery(hour, minute, settings.DAILY_TOPIC_SCOUT_TIMEZONE)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
