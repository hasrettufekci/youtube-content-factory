"""
Telegram approval bot (spec section 23).

No video leaves this system for YouTube without a human tapping APPROVE
here first (enforced by pipeline.py, which only uploads from
handle_approve). Uses the raw Telegram Bot API over HTTPS (no extra
framework/webhook server - just long polling via `--listen`, which keeps
this dependency-light per spec section 38).

Message format:
    🎬 NEW VIDEO READY
    Title: ...
    Duration: 08:42
    Scenes: 32
    Research sources: 7
    [▶ Preview] [✅ Approve] [🔄 Regenerate] [❌ Reject]

Run the approval listener with:
    python telegram_bot.py --listen
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import httpx

from core.config import settings, require
from core.logging_setup import get_logger
from storage import db, project_store as ps

logger = get_logger("telegram")

API_BASE = "https://api.telegram.org/bot{token}"
MAX_UPLOAD_BYTES = 49 * 1024 * 1024  # Telegram bot API practical limit (~50MB)
OFFSET_FILE = settings.STORAGE_DIR / ".telegram_offset"


def _url(method: str) -> str:
    return f"{API_BASE.format(token=require('TELEGRAM_BOT_TOKEN'))}/{method}"


def _redact_token(text: str) -> str:
    """Strip the live bot token out of a string before it can reach a log
    file. Every Telegram Bot API URL embeds the token directly
    (https://api.telegram.org/bot<TOKEN>/method), and httpx's own
    exceptions (HTTPStatusError, ConnectError, TimeoutException, ...)
    include the full request URL in their str()/repr() - so without this,
    a single API error (bad token, rate limit, a brief Telegram outage)
    would write the live bot token straight into logs/app.log the moment
    anything logs that exception (spec Round 2 section 15: "Telegram
    token never logged")."""
    token = settings.TELEGRAM_BOT_TOKEN
    if token and token in text:
        text = text.replace(token, "***REDACTED***")
    return text


def _probe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _keyboard(video_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "▶ Preview", "callback_data": f"preview:{video_id}"},
            {"text": "✅ Approve", "callback_data": f"approve:{video_id}"},
            {"text": "\U0001f504 Regenerate", "callback_data": f"regenerate:{video_id}"},
            {"text": "❌ Reject", "callback_data": f"reject:{video_id}"},
        ]]
    }


def send_text(text: str, reply_markup: dict | None = None, chat_id: str | None = None) -> int | None:
    payload = {"chat_id": chat_id or require("TELEGRAM_CHAT_ID"), "text": text, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(_url("sendMessage"), json=payload)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        # Never let a raw httpx exception escape this function: it embeds
        # the token-bearing request URL (see _redact_token above), and
        # `from None` keeps it out of any traceback a caller might log.
        raise RuntimeError(f"[TELEGRAM] sendMessage failed: {_redact_token(str(e))}") from None
    return resp.json().get("result", {}).get("message_id")


def edit_text(message_id: int, text: str, reply_markup: dict | None = None) -> None:
    payload = {"chat_id": require("TELEGRAM_CHAT_ID"), "message_id": message_id, "text": text, "parse_mode": "HTML"}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(_url("editMessageText"), json=payload)
    except httpx.HTTPError as e:
        logger.warning("[TELEGRAM] editMessageText failed: %s", _redact_token(str(e)))
        return
    if resp.status_code >= 400:
        logger.warning("[TELEGRAM] editMessageText failed: %s", resp.text[:300])


def _answer_callback(callback_query_id: str, text: str = "") -> None:
    try:
        with httpx.Client(timeout=15) as client:
            client.post(_url("answerCallbackQuery"), json={"callback_query_id": callback_query_id, "text": text})
    except httpx.HTTPError as e:
        logger.warning("[TELEGRAM] answerCallbackQuery failed: %s", _redact_token(str(e)))


def _relative_media_path(paths: ps.ProjectPaths) -> str:
    """This video's storage folder, relative to STORAGE_DIR - used only to
    build a PUBLIC_BASE_URL preview link. The absolute local filesystem path
    (paths.root) must never be sent to Telegram - it's private-server detail,
    not something a remote Telegram user can do anything with anyway."""
    return paths.root.relative_to(settings.STORAGE_DIR).as_posix()


def _public_preview_url(paths: ps.ProjectPaths) -> str | None:
    """Public HTTPS preview link, IF PUBLIC_BASE_URL is configured. This
    project does not run its own media server (kept out of scope - see
    README "Video preview for large files"); PUBLIC_BASE_URL is expected to
    already publicly serve the storage/ folder (e.g. a reverse proxy or
    static file host pointed at it) at this same relative path."""
    if not settings.PUBLIC_BASE_URL:
        return None
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    return f"{base}/{_relative_media_path(paths)}/video.mp4"


def _send_video_or_link(paths: ps.ProjectPaths, caption: str, reply_markup: dict | None = None) -> int | None:
    """Give the Telegram user something they can actually watch: the real
    video inline when it fits Telegram's upload limit, otherwise a public
    HTTPS preview URL (PUBLIC_BASE_URL) when configured. NEVER a local
    filesystem path - a Telegram user on their phone can't open one, and it
    leaks server layout (spec section 15/23)."""
    video_size = paths.video_mp4.stat().st_size if paths.video_mp4.exists() else 0

    if 0 < video_size <= MAX_UPLOAD_BYTES:
        data = {"chat_id": require("TELEGRAM_CHAT_ID"), "caption": caption, "parse_mode": "HTML"}
        if reply_markup is not None:
            data["reply_markup"] = _to_json(reply_markup)
        try:
            with httpx.Client(timeout=180) as client:
                with open(paths.video_mp4, "rb") as f:
                    resp = client.post(_url("sendVideo"), data=data, files={"video": (paths.video_mp4.name, f, "video/mp4")})
            if resp.status_code < 400:
                return resp.json().get("result", {}).get("message_id")
            logger.warning("[TELEGRAM] sendVideo failed (%s), falling back to link/text notice", resp.status_code)
        except httpx.HTTPError as e:
            logger.warning("[TELEGRAM] sendVideo failed (%s), falling back to link/text notice", _redact_token(str(e)))

    if video_size == 0:
        caption += "\n\n(video file not found - it may not have finished assembling)"
    else:
        preview_url = _public_preview_url(paths)
        size_mb = video_size / 1e6
        if preview_url:
            caption += f"\n\nVideo is {size_mb:.0f}MB - too large for Telegram's inline preview.\nPreview: {preview_url}"
        else:
            caption += (
                f"\n\nVideo is {size_mb:.0f}MB - too large for Telegram's inline preview, and "
                "PUBLIC_BASE_URL isn't configured, so no remote preview link is available. Set "
                "PUBLIC_BASE_URL to a URL that publicly serves the storage/ folder to enable this, or "
                "approve/reject/regenerate from the machine running the pipeline: "
                "`python pipeline.py --approve/--reject/--regenerate <video_id>`."
            )
    return send_text(caption, reply_markup=reply_markup)


def send_for_approval(paths: ps.ProjectPaths) -> int:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured")

    script = ps.load_json(paths.script_json) or {}
    scenes = ps.load_json(paths.scenes_json) or []
    research = ps.load_json(paths.research_json) or {}
    duration = _probe_duration_seconds(paths.video_mp4) if paths.video_mp4.exists() else 0.0

    caption = (
        "\U0001f3ac <b>NEW VIDEO READY</b>\n\n"
        f"Title: {script.get('title', '(untitled)')}\n"
        f"Duration: {_format_duration(duration)}\n"
        f"Scenes: {len(scenes)}\n"
        f"Research sources: {len(research.get('sources', []))}\n"
        f"video_id: {paths.video_id}"
    )

    message_id = _send_video_or_link(paths, caption, reply_markup=_keyboard(paths.video_id))
    logger.info("[TELEGRAM] sent draft for video_id=%d (message_id=%s)", paths.video_id, message_id)
    return message_id


def _to_json(value: dict) -> str:
    import json
    return json.dumps(value)


def _handle_callback(callback_query: dict) -> None:
    import pipeline as pipeline_module

    data = callback_query.get("data", "")
    if data.startswith("approval:"):
        # Telegram Approval V1's own callback namespace (see
        # youtube/telegram_approval_listener.py's "approval:<approve|reject>:
        # <slug>" format) - never parsed or answered here. This legacy,
        # video_id-based listener has no business with a V1 callback: its
        # own `action, _, video_id_str = data.partition(":")` below would
        # read "approval" as the action and "approve:<slug>" as the
        # video_id, fail int() with ValueError, and answer the button
        # "Invalid button data" - a false, misleading response to a press
        # meant for the other listener, and (if this poller happened to be
        # the one Telegram handed the update to) the V1 listener would then
        # never see that update at all. Silently ignoring it here - no
        # answerCallbackQuery, no log noise - leaves it entirely for
        # youtube.telegram_approval_listener to receive and process
        # normally, exactly as if this legacy listener were not running.
        return
    message = callback_query.get("message", {})
    message_id = message.get("message_id")
    action, _, video_id_str = data.partition(":")
    try:
        video_id = int(video_id_str)
    except ValueError:
        _answer_callback(callback_query["id"], "Invalid button data")
        return

    _answer_callback(callback_query["id"], f"{action}...")

    try:
        if action == "approve":
            result = pipeline_module.handle_approve(video_id)
            if message_id:
                edit_text(message_id, f"✅ Published: {result['url']}")
        elif action == "reject":
            pipeline_module.handle_reject(video_id)
            if message_id:
                edit_text(message_id, f"❌ Rejected (video_id={video_id})")
        elif action == "regenerate":
            if message_id:
                edit_text(message_id, f"\U0001f504 Regenerating (video_id={video_id})...")
            new_paths = pipeline_module.handle_regenerate(video_id)
            send_text(f"New draft ready: video_id={new_paths.video_id}")
        elif action == "preview":
            preview_paths = ps.get_project(video_id)
            if preview_paths is None:
                send_text(f"No project found for video_id={video_id}")
            else:
                script = ps.load_json(preview_paths.script_json) or {}
                row = db.get_video(video_id)
                caption = (
                    f"video_id={video_id}\n"
                    f"Title: {script.get('title', '(untitled)')}\n"
                    f"state: {row['state'] if row else 'unknown'}"
                )
                _send_video_or_link(preview_paths, caption)
        else:
            logger.warning("[TELEGRAM] unknown callback action: %s", action)
    except Exception:
        # Never forward raw exception text to Telegram: many of this
        # project's error messages embed local filesystem paths (e.g.
        # "no metadata.json found at /home/.../storage/...") or other
        # server-side detail that must never leave this machine (spec
        # Round 2 section 15). The full exception is already captured
        # below via logger.exception() - Telegram only ever gets a
        # generic, path-free notice.
        logger.exception("[TELEGRAM] callback handling failed (action=%s video_id=%d)", action, video_id)
        if message_id:
            edit_text(message_id, f"⚠️ {action} failed for video_id={video_id} - check server logs for details")


def poll_forever(poll_interval: float = 2.0) -> None:
    offset = int(OFFSET_FILE.read_text()) if OFFSET_FILE.exists() else 0
    logger.info("[TELEGRAM] listening for approval button presses (offset=%d)...", offset)
    while True:
        try:
            with httpx.Client(timeout=35) as client:
                resp = client.get(_url("getUpdates"), params={"offset": offset, "timeout": 30})
            resp.raise_for_status()
            updates = resp.json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                OFFSET_FILE.write_text(str(offset))
                if "callback_query" in update:
                    _handle_callback(update["callback_query"])
        except httpx.HTTPError as e:
            # str(e) on an httpx error embeds the full request URL, which
            # for every Telegram call contains the live bot token - redact
            # it before it can reach logs/app.log (spec Round 2 section 15).
            logger.warning("[TELEGRAM] poll error: %s", _redact_token(str(e)))
            time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telegram approval bot.")
    parser.add_argument("--listen", action="store_true", help="Long-poll for Approve/Reject/Regenerate button presses")
    parser.add_argument("--send", type=int, metavar="VIDEO_ID", help="(Re)send the approval message for a video")
    args = parser.parse_args(argv)

    if args.send:
        paths = ps.get_project(args.send)
        if not paths:
            print(f"No project with video_id={args.send}")
            return 1
        send_for_approval(paths)
        return 0

    if args.listen:
        poll_forever()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
