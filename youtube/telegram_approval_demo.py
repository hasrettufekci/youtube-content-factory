"""
YOUTUBE_CONTENT_FACTORY - SAFE TELEGRAM APPROVAL DEMO (Telegram Approval V1).

Sends the approval message (thumbnail photo + caption + APPROVE/REJECT
inline keyboard, then a best-effort video preview) for an EXISTING,
already-produced project.

Resolution order (see youtube/telegram_approval.py's
resolve_project_for_tooling()): a real, DB-backed project record
(storage/project_store.py's `videos` table - the same source of truth
pipeline.py itself writes to, e.g. storage/2026/09/<slug>/video.mp4 +
thumbnail.png) is tried first, by either --video-id or --slug; only when
no DB row exists at all does this fall back to the original debug_audit/
demo convention (final_video_v2/<slug>-v2.mp4 + thumbnail_v1/<slug>/
thumbnail.png), for full backward compatibility with the original
the-wow-signal-6 manual demo flow.

Reads ONLY metadata_<lang>.json / scenes.json from the resolved project
directory (read-only - nothing under storage/ is ever written by a plain
send or --dry-run). A plain send's only side effects are one new row in
the telegram_approvals table (storage/factory.db) and, unless --dry-run,
real Telegram API calls.

RESEND (--resend-approval-id): recovers an approval that was created but
never finished sending (e.g. the 2026-09 sendPhoto HTTP 400 incident,
where approval_id=3 was left stuck in status=draft because
send_photo_with_keyboard() raised before the row could be marked
ready_for_approval). Resend re-resolves the SAME project's real artifacts
and retries the ONE sendPhoto call - it NEVER creates a new approval row
(no ta.create_approval() call) and NEVER re-runs any pipeline stage (no
research/script/images/TTS/assembly, no pipeline.produce_video()). Only
usable while the approval hasn't been decided yet (status is draft or
ready_for_approval) - see youtube.telegram_approval.resend_existing_approval().

COST/API SAFETY: makes zero Claude/OpenAI/research/FAL calls, for either a
plain send or a resend. --dry-run makes zero Telegram calls too (it only
builds and prints the payload that WOULD be sent, and touches no DB row)
- this is the mode every automated test uses; a real run needs
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID configured, exactly like any other
live stage in this project.

This demo only SENDS/resends the approval message - it does not itself
listen for the APPROVE/REJECT button press (see
youtube/telegram_approval_listener.py, a separate long-running process,
for that).

Usage:
    python -m youtube.telegram_approval_demo --video-id 5 --dry-run          # preview, no network, no DB write
    python -m youtube.telegram_approval_demo --video-id 5                    # real send (creates a NEW approval row)
    python -m youtube.telegram_approval_demo --slug the-wow-signal-6 --dry-run
    python -m youtube.telegram_approval_demo --resend-approval-id 3          # resend/repair an existing approval - NEVER creates a new row
    python -m youtube.telegram_approval_demo --resend-approval-id 3 --dry-run  # preview the resend only, no network, no DB write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.logging_setup import get_logger
from youtube import telegram_approval as tap

logger = get_logger("telegram_approval_demo")


def run(slug: str | None = None, video_id: int | None = None, language: str = "en",
        project_dir: Path | None = None, dry_run: bool = False, chat_id: str | None = None) -> dict:
    resolved_project_dir, resolved_slug, video_path, thumbnail_path = tap.resolve_project_for_tooling(
        slug=slug, video_id=video_id, project_dir=project_dir,
    )

    payload = tap.build_approval_payload(
        resolved_slug, language=language, project_dir=resolved_project_dir,
        video_path=video_path, thumbnail_path=thumbnail_path,
    )

    if dry_run:
        logger.info("[TELEGRAM] DRY RUN - no Telegram call made, no approval row created")
        return {
            "dry_run": True,
            "slug": payload.slug,
            "title": payload.title,
            "caption": payload.caption,
            "keyboard": payload.keyboard,
            "video_path": str(payload.video_path),
            "thumbnail_path": str(payload.thumbnail_path),
            "video_sendable": payload.video_sendable,
        }

    send_result = tap.send_new_approval(payload, chat_id=chat_id)
    return {
        "dry_run": False,
        "slug": payload.slug,
        "title": payload.title,
        "caption": payload.caption,
        **send_result,
    }


def resend(approval_id: int, dry_run: bool = False, chat_id: str | None = None) -> dict:
    """--resend-approval-id path: repairs/resends an existing approval row -
    never creates a new one, never re-runs any pipeline stage. --dry-run
    previews the rebuilt payload without any network call or DB write,
    exactly like the plain-send --dry-run above (see
    tap.resend_existing_approval()'s own dry_run contract)."""
    if dry_run:
        logger.info("[TELEGRAM] DRY RUN - resend preview only, no Telegram call made, no DB row touched")
    result = tap.resend_existing_approval(approval_id, chat_id=chat_id, dry_run=dry_run)
    return {"dry_run": dry_run, **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send (or resend) a Telegram approval message for an existing project's video+thumbnail."
    )
    # --video-id/--slug/--project-dir may be combined exactly as before
    # (--project-dir "overrides --slug lookup", per its own help text below) -
    # only --resend-approval-id is a genuinely separate mode, checked for
    # manually below rather than via a mutually-exclusive group, since a
    # group would also (wrongly) forbid combining --slug with --project-dir.
    parser.add_argument("--video-id", type=int, help="video_id of an already-produced project (resolved via the DB)")
    parser.add_argument("--slug", help="slug of an already-produced project (resolved via the DB, or the legacy "
                                        "debug_audit/ convention if no DB row exists)")
    parser.add_argument("--project-dir", help="Explicit source project directory (overrides --slug lookup)")
    parser.add_argument("--resend-approval-id", type=int,
                         help="Resend/repair an EXISTING approval by id - never creates a new approval row, "
                              "never re-runs any pipeline stage. Cannot be combined with --video-id/--slug/--project-dir")
    parser.add_argument("--language", default="en", help="Metadata language to read (default: en)")
    parser.add_argument("--chat-id", default=None, help="Override TELEGRAM_CHAT_ID for this send")
    parser.add_argument("--dry-run", action="store_true",
                         help="Build and print the approval payload only - no Telegram call, no DB write")
    args = parser.parse_args(argv)

    if args.resend_approval_id is not None and any([args.slug, args.video_id, args.project_dir]):
        print("FAILED: --resend-approval-id cannot be combined with --video-id/--slug/--project-dir", file=sys.stderr)
        return 1
    if not any([args.slug, args.video_id, args.project_dir, args.resend_approval_id]):
        print("FAILED: one of --slug, --video-id, --project-dir, or --resend-approval-id is required", file=sys.stderr)
        return 1
    # --project-dir alone (no --slug) infers the slug from the directory's
    # own name - unchanged from the original CLI contract.
    slug = args.slug or (Path(args.project_dir).name if args.project_dir else None)

    try:
        if args.resend_approval_id is not None:
            result = resend(args.resend_approval_id, dry_run=args.dry_run, chat_id=args.chat_id)
        else:
            project_dir = Path(args.project_dir) if args.project_dir else None
            result = run(slug=slug, video_id=args.video_id, language=args.language, project_dir=project_dir,
                         dry_run=args.dry_run, chat_id=args.chat_id)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print("OK")
    print(f"slug: {result['slug']}")
    if "title" in result:
        print(f"title: {result['title']}")
    print()
    if "caption" in result:
        print(result["caption"])
        print()
    if result["dry_run"]:
        print("(DRY RUN - nothing was sent, no approval row was created or modified)")
        if "video_path" in result:
            print(f"video_path: {result['video_path']}")
            print(f"thumbnail_path: {result['thumbnail_path']}")
            print(f"video_sendable: {result['video_sendable']}")
    else:
        print(f"approval_id: {result['approval_id']}")
        print(f"message_id: {result['message_id']}")
        print(f"video_message_id: {result['video_message_id']}")
        if args.resend_approval_id is not None:
            print("(RESEND - existing approval row reused, no new approval row created)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
