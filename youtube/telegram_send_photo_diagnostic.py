"""
YOUTUBE_CONTENT_FACTORY - Telegram sendPhoto diagnostic (2026-09 production
incident: video_id=5 / approval_id=3 / slug=medieval-norman-castle-
discovered-hidden-inside-3 - pipeline.py's _notify_telegram() logged
"[TELEGRAM] approval notification skipped: [TELEGRAM] sendPhoto failed:
HTTP 400 Bad Request" with no further detail).

Reproduces ONLY the one Telegram sendPhoto call the production pipeline
already made for an existing, already-produced video - using the SAME
real artifacts (storage/<slug>/video.mp4 + thumbnail.png, resolved via
storage/project_store.get_project()/get_project_by_slug(), exactly like
pipeline.py's _notify_telegram()) and the SAME tap.build_approval_payload()
+ tap.send_photo_with_keyboard() functions the production pipeline itself
uses. This script does not modify either function - it only calls them.

COST/API SAFETY - this is deliberately the smallest possible reproduction:
    - Makes ZERO Claude/OpenAI/FAL calls (no research/script/image/TTS/
      assembly is ever re-run - it only reads files already on disk).
    - Creates NO new video, NO new storage/<slug>/ project, and writes
      NOTHING under storage/ or debug_audit/.
    - Creates NO telegram_approvals DB row and calls neither
      tap.approve()/reject() nor any other state-machine transition -
      tap.build_approval_payload() is pure (no network, no DB write) and
      this script never calls tap.send_new_approval()/ta.create_approval().
    - Without --send: makes ZERO network calls at all - just resolves the
      real artifact paths and prints the same safe diagnostics
      send_photo_with_keyboard() now logs internally (path/exists/size/
      decoded image format+dimensions/caption length), so the file itself
      can be inspected before ever touching the network.
    - With --send: makes exactly ONE real Telegram API call (sendPhoto),
      identical to what the production pipeline already attempted - never
      more than one, never a batch/retry loop.

Never touches TELEGRAM_BOT_TOKEN/chat_id values in its own output (they are
config, not something this script prints) - only booleans for whether each
is configured, matching the "never log token/chat ID" rule this whole
Telegram integration follows (see telegram_bot._redact_token(),
tap.send_photo_with_keyboard()'s own diagnostic logging).

Usage:
    python -m youtube.telegram_send_photo_diagnostic --video-id 5
    python -m youtube.telegram_send_photo_diagnostic --slug medieval-norman-castle-discovered-hidden-inside-3
    python -m youtube.telegram_send_photo_diagnostic --video-id 5 --send   # makes the one real API call
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import settings
from core.logging_setup import get_logger
from storage import project_store as ps
from youtube import telegram_approval as tap

logger = get_logger("telegram_send_photo_diagnostic")


def _resolve_paths(video_id: int | None, slug: str | None) -> ps.ProjectPaths:
    if video_id is not None:
        paths = ps.get_project(video_id)
        if paths is None:
            raise tap.TelegramApprovalError(f"no project found for video_id={video_id}")
        return paths
    paths = ps.get_project_by_slug(slug)
    if paths is None:
        raise tap.TelegramApprovalError(f"no project found for slug={slug!r}")
    return paths


def diagnose(video_id: int | None = None, slug: str | None = None, language: str = "en", send: bool = False) -> dict:
    paths = _resolve_paths(video_id, slug)

    # Pure, side-effect-free (no network, no DB write) - the exact same call
    # pipeline.py's _notify_telegram() makes, with the exact same explicit
    # video_path/thumbnail_path override (the real production artifacts,
    # never the debug_audit/ demo-tool convention).
    payload = tap.build_approval_payload(
        paths.slug, language=language, project_dir=paths.root,
        video_path=paths.video_mp4, thumbnail_path=paths.thumbnail_png,
    )

    caption_len = len(payload.caption)
    result: dict = {
        "video_id": paths.video_id,
        "slug": paths.slug,
        "thumbnail_path": str(payload.thumbnail_path),
        "thumbnail_exists": payload.thumbnail_path.exists(),
        "thumbnail_size_bytes": payload.thumbnail_path.stat().st_size if payload.thumbnail_path.exists() else None,
        "thumbnail_diagnostics": tap._describe_image_for_diagnostics(payload.thumbnail_path),
        "caption_length": caption_len,
        "caption_exceeds_telegram_limit": caption_len > 1024,  # Telegram sendPhoto caption hard limit
        "bot_token_configured": bool(settings.TELEGRAM_BOT_TOKEN),
        "chat_id_configured": bool(settings.TELEGRAM_CHAT_ID),
        "sent": False,
    }

    if send:
        try:
            message_id = tap.send_photo_with_keyboard(payload.thumbnail_path, payload.caption, payload.keyboard)
        except RuntimeError as e:
            result["sent"] = False
            result["send_error"] = str(e)  # already redacted/sanitized by send_photo_with_keyboard() itself
        else:
            result["sent"] = True
            result["message_id"] = message_id
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce/diagnose a Telegram sendPhoto failure for an EXISTING, already-produced video - "
                    "no pipeline stage is re-run, no video is created, no FAL/Claude credits are spent."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", type=int, help="video_id of an already-produced project (e.g. 5)")
    group.add_argument("--slug", help="slug of an already-produced project")
    parser.add_argument("--language", default="en", help="Metadata language to read (default: en)")
    parser.add_argument("--send", action="store_true",
                         help="Also make the one real sendPhoto call (default: diagnostics only, no network call)")
    args = parser.parse_args(argv)

    try:
        result = diagnose(video_id=args.video_id, slug=args.slug, language=args.language, send=args.send)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print(f"video_id: {result['video_id']}")
    print(f"slug: {result['slug']}")
    print(f"thumbnail_path: {result['thumbnail_path']}")
    print(f"thumbnail_exists: {result['thumbnail_exists']}")
    print(f"thumbnail_size_bytes: {result['thumbnail_size_bytes']}")
    print(f"thumbnail_diagnostics: {result['thumbnail_diagnostics']}")
    print(f"caption_length: {result['caption_length']} (Telegram limit: 1024)")
    if result["caption_exceeds_telegram_limit"]:
        print("WARNING: caption_length exceeds Telegram's 1024-character sendPhoto caption limit")
    print(f"bot_token_configured: {result['bot_token_configured']}")
    print(f"chat_id_configured: {result['chat_id_configured']}")
    print()

    if not args.send:
        print("(diagnostics only - no network call made; re-run with --send to reproduce the real sendPhoto call)")
        return 0

    if result["sent"]:
        print("OK - sendPhoto succeeded")
        print(f"message_id: {result['message_id']}")
        return 0
    print(f"FAILED: {result['send_error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
