"""
AI FOUND THIS - Smart Publish Scheduler V1 diagnostic CLI (Faz 9).

Read-only, ALWAYS: resolves an existing, already-produced project (by
--video-id or --slug, via storage/project_store.py - the same DB-backed
resolution youtube/telegram_send_photo_diagnostic.py and
youtube/telegram_approval_demo.py already use) and prints the
PublishDecision youtube.smart_publish.calculate_publish_time() would
produce for it right now. This tool never makes a YouTube call, never
makes a Telegram call, and never writes to the DB (no publish_schedules
row, no approval row, no video row) - it exists purely so a human can
sanity-check a publish decision without touching any real state.

Usage:
    python -m youtube.smart_publish_demo --video-id 5
    python -m youtube.smart_publish_demo --slug the-wow-signal-6
    python -m youtube.smart_publish_demo --video-id 5 --now 2026-09-27T10:00:00+00:00 --timezone Europe/Istanbul
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone as dt_timezone

from core.config import settings
from core.logging_setup import get_logger
from storage import project_store as ps
from youtube import smart_publish

logger = get_logger("smart_publish_demo")


def diagnose(video_id: int | None = None, slug: str | None = None,
             now: datetime | None = None, timezone: str | None = None) -> dict:
    """Pure/read-only: no YouTube/Telegram call, no DB write of any kind
    (publish_schedules.create_schedule() is never called here - only
    youtube.smart_publish.calculate_publish_time(), which itself makes no
    DB or network call either)."""
    if video_id is None and slug is None:
        raise ValueError("one of video_id or slug is required")

    paths = ps.get_project(video_id) if video_id is not None else ps.get_project_by_slug(slug)
    if paths is None:
        raise ValueError(f"no project found for video_id={video_id!r} slug={slug!r}")

    title = None
    if paths.metadata_en_json.exists():
        try:
            title = json.loads(paths.metadata_en_json.read_text()).get("title")
        except (json.JSONDecodeError, OSError):
            title = None

    resolved_now = now if now is not None else datetime.now(dt_timezone.utc)
    resolved_timezone = timezone or settings.SMART_PUBLISH_TIMEZONE
    decision = smart_publish.calculate_publish_time(now=resolved_now, timezone=resolved_timezone)

    return {
        "video_id": paths.video_id,
        "slug": paths.slug,
        "title": title,
        "decision": decision,
    }


def _parse_now(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview the Smart Publish Scheduler V1 decision for an existing project - "
                    "read-only, never touches YouTube/Telegram/DB."
    )
    parser.add_argument("--video-id", type=int, help="video_id of an already-produced project (resolved via the DB)")
    parser.add_argument("--slug", help="slug of an already-produced project (resolved via the DB)")
    parser.add_argument("--dry-run", action="store_true",
                         help="accepted for symmetry with the project's other diagnostic CLIs - this tool is "
                              "ALWAYS read-only: no YouTube call, no Telegram call, no DB mutation, with or "
                              "without this flag")
    parser.add_argument("--now", help="override 'now' (ISO 8601, e.g. 2026-09-27T10:00:00+00:00) - "
                                       "defaults to the real current time")
    parser.add_argument("--timezone", help="override the publish timezone (default: settings.SMART_PUBLISH_TIMEZONE)")
    args = parser.parse_args(argv)

    if args.video_id is None and args.slug is None:
        print("FAILED: one of --video-id or --slug is required", file=sys.stderr)
        return 1

    now = None
    if args.now:
        try:
            now = _parse_now(args.now)
        except ValueError:
            print(f"FAILED: --now must be ISO 8601, got {args.now!r}", file=sys.stderr)
            return 1

    try:
        result = diagnose(video_id=args.video_id, slug=args.slug, now=now, timezone=args.timezone)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    decision = result["decision"]
    print("VIDEO:")
    print(result["video_id"])
    if result["title"]:
        print(result["title"])
    print()
    print("PUBLISH DECISION")
    print("----------------")
    print(f"scheduled_at: {decision.scheduled_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"timezone: {decision.timezone}")
    print(f"source: {decision.source}")
    print(f"confidence: {decision.confidence}")
    print()
    print("REASON:")
    print(decision.reason)
    print()
    print("DRY RUN:")
    print("No YouTube action")
    print("No Telegram action")
    print("No DB mutation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
