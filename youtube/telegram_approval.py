"""
Telegram Approval V1 (spec: "YOUTUBE_CONTENT_FACTORY_NEW - TELEGRAM APPROVAL
V1").

    VIDEO + THUMBNAIL
            |
    TELEGRAM APPROVAL
            |
      [APPROVE] [REJECT]
            |
    APPROVE -> YOUTUBE PRIVATE UPLOAD
    REJECT  -> STATUS = rejected

This is a NEW, slug-based approval flow, separate from the existing
video_id-based telegram_bot.py/pipeline.py flow (which is untouched by
this module - not imported, not modified, not called). It targets the
artifacts already produced by this session's other V2 pipeline stages:

    final video: debug_audit/final_video_v2/<slug>-v2.mp4
    thumbnail:   debug_audit/thumbnail_v1/<slug>/thumbnail.png

REUSE, NOT DUPLICATION:
  - Telegram HTTP plumbing: `_url()` / `_redact_token()` / `send_text()` /
    `_answer_callback()` / `_probe_duration_seconds()` /
    `_format_duration()` / `MAX_UPLOAD_BYTES` are all imported from the
    existing telegram_bot.py rather than reimplemented. Only sendPhoto, a
    size-guarded sendVideo-preview, and `edit_caption()` are new here:
    telegram_bot.py doesn't have sendPhoto/sendVideo at all, and its own
    `edit_text()` posts to editMessageText, which only works on plain-text
    messages - every approval message here is sent via sendPhoto, so its
    caption must be updated with editMessageCaption instead (see
    `edit_caption()` below - this is NOT a modification of telegram_bot.py
    or its edit_text(), which is left exactly as-is for its own flow).
  - YouTube upload: `youtube.uploader.upload_video()` /
    `upload_thumbnail()` are called directly (both already accept a plain
    file path + a metadata dict - no ProjectPaths/DB "videos" row
    required), so no new OAuth code is written here. Metadata is read
    as-is from an existing metadata_<lang>.json - never regenerated.

STATE MACHINE (persisted in storage/telegram_approvals.py's
`telegram_approvals` table):

    draft -> ready_for_approval -> approved -> uploading -> uploaded
                                 \\-> rejected            \\-> upload_failed

`_transition()` is the single gate every status change goes through, and
it is what makes a double APPROVE tap (or REJECT-after-APPROVE / APPROVE-
after-REJECT) a safe no-op instead of a second YouTube upload: once a row
has left 'ready_for_approval', every further approve()/reject() call finds
the target status is not in ALLOWED_TRANSITIONS[current] and raises
ApprovalStateError, which handle_approval_callback() turns into a plain
"already processed" reply rather than acting again.

COST/API SAFETY: this module makes no Claude/OpenAI/research calls at
all. It reads already-generated files and (only on a real, once-per-
approval APPROVE tap) makes Telegram HTTP calls plus exactly one YouTube
upload call.

Production usage:
    python -m youtube.telegram_approval_demo --slug the-wow-signal-6
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

import telegram_bot as tb
from core.config import require, settings
from core.logging_setup import get_logger
from storage import telegram_approvals as ta

logger = get_logger("telegram_approval")

# --- artifact locations (fixed conventions - match final_video_assembly_demo.py
# and thumbnail_demo.py's own OUTPUT_ROOT defaults exactly) ------------------
FINAL_VIDEO_ROOT = Path("debug_audit") / "final_video_v2"
THUMBNAIL_ROOT = Path("debug_audit") / "thumbnail_v1"


def video_path_for(slug: str) -> Path:
    return FINAL_VIDEO_ROOT / f"{slug}-v2.mp4"


def thumbnail_path_for(slug: str) -> Path:
    return THUMBNAIL_ROOT / slug / "thumbnail.png"


# --- state machine -----------------------------------------------------------
STATUS_DRAFT = "draft"
STATUS_READY = "ready_for_approval"
STATUS_APPROVED = "approved"
STATUS_UPLOADING = "uploading"
STATUS_UPLOADED = "uploaded"
STATUS_UPLOAD_FAILED = "upload_failed"
STATUS_REJECTED = "rejected"

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATUS_DRAFT: {STATUS_READY},
    STATUS_READY: {STATUS_APPROVED, STATUS_REJECTED},
    STATUS_APPROVED: {STATUS_UPLOADING},
    STATUS_UPLOADING: {STATUS_UPLOADED, STATUS_UPLOAD_FAILED},
    STATUS_UPLOADED: set(),
    STATUS_UPLOAD_FAILED: set(),
    STATUS_REJECTED: set(),
}

UPLOAD_VISIBILITY_LABEL = "Private"  # EN ONEMLI KURAL: V1 never auto-publishes public
_LANGUAGE_LABELS = {"en": "English", "tr": "Turkish"}

# Thumbnail outcome tracking - deliberately NOT a second state machine (no
# ALLOWED_TRANSITIONS, no _transition() gate): a video's own approval
# status (STATUS_* above) is the only thing that governs upload behavior.
# This is just a plain result label persisted alongside it, so a failed
# thumbnail can never turn a successfully-uploaded video into
# upload_failed.
THUMBNAIL_STATUS_PENDING = "pending"
THUMBNAIL_STATUS_SUCCESS = "success"
THUMBNAIL_STATUS_FAILED = "failed"


class TelegramApprovalError(RuntimeError):
    """Precondition failure (missing artifact/config) - never a partial send."""


class ApprovalNotFoundError(RuntimeError):
    pass


class ApprovalStateError(RuntimeError):
    """Raised by _transition() when the requested status change is not a
    legal move from the row's current status - this is what makes a
    duplicate/late button press a safe no-op instead of a second upload."""


class CallbackValidationError(RuntimeError):
    """Malformed callback_data, or a slug/chat_id/message_id combination
    that does not match a known, live approval row (spec section 11)."""


# --- pure helpers --------------------------------------------------------

def _language_label(language: str) -> str:
    return _LANGUAGE_LABELS.get(language, language.upper())


def _find_project_dir(slug: str) -> Path | None:
    """Read-only search under storage/<year>/<month>/<slug> - mirrors
    youtube/thumbnail_demo.py's / final_video_assembly_demo.py's own
    lookup exactly. Never writes or modifies anything it finds."""
    storage_root = settings.STORAGE_DIR
    if not storage_root.exists():
        return None
    matches = sorted(storage_root.glob(f"*/*/{slug}"))
    return matches[0] if matches else None


def _load_json_or_none(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_keyboard(slug: str) -> dict:
    """Only APPROVE/REJECT (spec section 2) - a narrower, distinct
    callback_data namespace ("approval:...") from telegram_bot.py's own
    "approve:<video_id>" etc., so the two flows never collide even if both
    happen to be polled by the same bot token.

    LEGACY/PREVIEW ONLY as of the 2026-09 BUTTON_DATA_INVALID fix below -
    build_approval_payload() (called before any approval_id exists, e.g.
    for --dry-run or by pipeline.py's _notify_telegram()) still uses this
    for its payload.keyboard preview, but the actual real send
    (send_new_approval()/resend_existing_approval()) always overwrites it
    with build_keyboard_by_id()'s short, deterministic, ID-based
    callback_data before transmission - see that function's own docstring
    for why. handle_approval_callback() below still parses THIS format too
    (for any already-sent message whose buttons still carry it)."""
    return {
        "inline_keyboard": [[
            {"text": "✅ APPROVE", "callback_data": f"approval:approve:{slug}"},
            {"text": "❌ REJECT", "callback_data": f"approval:reject:{slug}"},
        ]]
    }


def build_keyboard_by_id(approval_id: int) -> dict:
    """Short, deterministic, ID-based callback_data - "approval:approve:<id>"
    / "approval:reject:<id>" - the actual keyboard every real send uses.

    2026-09 production incident: build_keyboard(slug) above embeds the
    FULL slug into callback_data. Telegram's callback_data has a hard
    64-BYTE limit (not just a friendly cap - it's rejected outright), and
    a long-enough slug pushes the combined string over it: for slug=
    "medieval-norman-castle-discovered-hidden-inside-3", "approval:approve:
    <slug>" is 66 bytes and "approval:reject:<slug>" is 65 - both over the
    limit - so Telegram's sendPhoto call itself failed with "Bad Request:
    BUTTON_DATA_INVALID" (the message is never even created; this is a
    request-validation error, not a delivery failure). An integer
    approval_id is always short regardless of slug length, so this can
    never happen again. Same "approval:<action>:..." prefix/action names
    as build_keyboard() above - only the identifier changes - and
    handle_approval_callback() below accepts both formats."""
    return {
        "inline_keyboard": [[
            {"text": "✅ APPROVE", "callback_data": f"approval:approve:{approval_id}"},
            {"text": "❌ REJECT", "callback_data": f"approval:reject:{approval_id}"},
        ]]
    }


def build_caption(title: str, duration_label: str, scenes_count: int, language: str, video_sendable: bool) -> str:
    lines = [
        "\U0001f3ac <b>AI FOUND THIS — VIDEO READY</b>",
        "",
        title,
        "",
        f"⏱ Duration: {duration_label}",
        f"\U0001f39e Scenes: {scenes_count}",
        "\U0001f5bc Thumbnail: Ready",
        f"\U0001f30d Language: {_language_label(language)}",
        f"\U0001f4fa Upload: YouTube / {UPLOAD_VISIBILITY_LABEL}",
    ]
    if not video_sendable:
        lines.append("\U0001f3a5 Video preview: unavailable")
    lines += ["", "STATUS: WAITING FOR APPROVAL"]
    return "\n".join(lines)


@dataclass
class ApprovalPayload:
    slug: str
    title: str
    duration_label: str
    scenes_count: int
    language: str
    video_path: Path
    thumbnail_path: Path
    video_sendable: bool
    caption: str
    keyboard: dict
    channel_key: str | None = None


def build_approval_payload(slug: str, language: str | None = None, project_dir: Path | None = None,
                            video_path: Path | None = None, thumbnail_path: Path | None = None,
                            channel_key: str | None = None) -> ApprovalPayload:
    """Pure/side-effect-free: no network, no DB write. Resolves the source
    project (for title/scene-count only), validates the video+thumbnail
    artifacts exist, and builds the exact caption/keyboard that would be
    sent - safe to call from a --dry-run CLI path or a test with zero
    mocking.

    ARTIFACT SOURCE (architecture note): video_path/thumbnail_path are
    normally given explicitly by the caller - the production pipeline
    (pipeline.py) passes its OWN real, just-produced artifacts
    (ProjectPaths.video_mp4 / ProjectPaths.thumbnail_png, i.e.
    storage/<slug>/video.mp4 + thumbnail.png - the single, real
    production source of truth; NOT youtube/final_video_assembly_demo.py's
    output, which is a narrow, the-wow-signal-6-specific 33-scene
    remediation pass and was never meant to be a general video source).
    When neither is given (the existing --dry-run/manual telegram_approval_
    demo.py CLI path, unchanged), this falls back to the original
    video_path_for(slug)/thumbnail_path_for(slug) debug_audit/ convention
    for full backward compatibility - every existing caller/test keeps
    working exactly as before."""
    language = language or settings.LANGUAGE_MASTER
    resolved_project_dir = project_dir or _find_project_dir(slug)
    if not resolved_project_dir or not resolved_project_dir.exists():
        raise TelegramApprovalError(
            f"no project directory found for slug={slug!r} under {settings.STORAGE_DIR}"
        )

    metadata = _load_json_or_none(resolved_project_dir / f"metadata_{language}.json")
    if not metadata or not metadata.get("title"):
        raise TelegramApprovalError(
            f"no metadata_{language}.json (with a title) found at {resolved_project_dir} - run metadata.py first"
        )
    title = metadata["title"]

    scenes = _load_json_or_none(resolved_project_dir / "scenes.json")
    scenes_count = len(scenes) if isinstance(scenes, list) else 0

    resolved_video_path = Path(video_path) if video_path is not None else video_path_for(slug)
    resolved_thumbnail_path = Path(thumbnail_path) if thumbnail_path is not None else thumbnail_path_for(slug)
    if not resolved_video_path.exists():
        raise TelegramApprovalError(f"final video not found at {resolved_video_path}")
    if not resolved_thumbnail_path.exists():
        raise TelegramApprovalError(f"thumbnail not found at {resolved_thumbnail_path}")

    duration_label = tb._format_duration(tb._probe_duration_seconds(resolved_video_path))

    video_size = resolved_video_path.stat().st_size
    video_sendable = 0 < video_size <= tb.MAX_UPLOAD_BYTES

    caption = build_caption(title, duration_label, scenes_count, language, video_sendable)
    keyboard = build_keyboard(slug)

    return ApprovalPayload(
        slug=slug, title=title, duration_label=duration_label, scenes_count=scenes_count,
        language=language, video_path=resolved_video_path, thumbnail_path=resolved_thumbnail_path,
        video_sendable=video_sendable, caption=caption, keyboard=keyboard, channel_key=channel_key,
    )


# --- Telegram network calls (new here - telegram_bot.py has no sendPhoto) ---

def _describe_image_for_diagnostics(image_path: Path) -> str:
    """Safe, no-secrets description of the file about to be uploaded - path,
    existence, size, and (via Pillow, already a project dependency) the
    actual decoded format/dimensions, NOT just the file extension. Used
    only for diagnostic logging (2026-09 production incident: sendPhoto
    returned HTTP 400 with no visible reason) - never raises, so a broken
    image can never prevent the real error from being logged. Deliberately
    excludes chat_id/bot token - nothing here ever touches Telegram
    request/response data."""
    if not image_path.exists():
        return f"path={image_path} exists=False"
    size_bytes = image_path.stat().st_size
    try:
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(image_path) as img:
                return (
                    f"path={image_path} exists=True size_bytes={size_bytes} "
                    f"format={img.format} dimensions={img.width}x{img.height} mode={img.mode}"
                )
        except UnidentifiedImageError:
            return f"path={image_path} exists=True size_bytes={size_bytes} format=UNRECOGNIZED (not a valid image file)"
    except Exception as e:  # pragma: no cover - Pillow itself unavailable/corrupt install
        return f"path={image_path} exists=True size_bytes={size_bytes} format=<could not inspect: {e}>"


def send_photo_with_keyboard(image_path: Path, caption: str, keyboard: dict, chat_id: str | None = None) -> int | None:
    logger.info("[TELEGRAM] sendPhoto: %s (caption_len=%d)", _describe_image_for_diagnostics(image_path), len(caption))
    data = {
        "chat_id": chat_id or require("TELEGRAM_CHAT_ID"),
        "caption": caption,
        "parse_mode": "HTML",
        "reply_markup": json.dumps(keyboard),
    }
    try:
        with httpx.Client(timeout=60) as client:
            with open(image_path, "rb") as f:
                resp = client.post(tb._url("sendPhoto"), data=data, files={"photo": (image_path.name, f, "image/png")})
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # The previous behavior here (str(e) only) throws away exactly the
        # information needed to diagnose a sendPhoto 400 - Telegram's own
        # JSON body (e.g. {"description": "Bad Request: <the real reason>"})
        # is never in httpx.HTTPStatusError's str(), only in e.response.text.
        # Surface it (redacted/truncated, never the request payload we sent
        # - so chat_id is never included) instead of a bare "HTTP 400 Bad
        # Request" that gives no actionable detail.
        status = e.response.status_code if e.response is not None else "?"
        body = tb._redact_token(e.response.text[:500]) if e.response is not None else ""
        logger.warning("[TELEGRAM] sendPhoto failed: HTTP %s - %s", status, body)
        raise RuntimeError(f"[TELEGRAM] sendPhoto failed: HTTP {status} - {body}") from None
    except httpx.HTTPError as e:
        # Non-HTTP-status errors (connection/timeout) have no response body
        # to show - mirrors telegram_bot.send_text()'s own redaction of the
        # token-bearing request URL that httpx embeds in these exceptions.
        raise RuntimeError(f"[TELEGRAM] sendPhoto failed: {tb._redact_token(str(e))}") from None
    return resp.json().get("result", {}).get("message_id")


def send_video_preview(video_path: Path, caption: str = "", chat_id: str | None = None) -> int | None:
    """Best-effort only (spec section 4): a failure here must never break
    the approval flow, which has already been sent by the time this is
    called - callers treat None as "no preview, that's fine"."""
    data: dict = {"chat_id": chat_id or require("TELEGRAM_CHAT_ID")}
    if caption:
        data["caption"] = caption
        data["parse_mode"] = "HTML"
    try:
        with httpx.Client(timeout=180) as client:
            with open(video_path, "rb") as f:
                resp = client.post(tb._url("sendVideo"), data=data, files={"video": (video_path.name, f, "video/mp4")})
        if resp.status_code >= 400:
            logger.warning("[TELEGRAM] sendVideo (preview) failed (%s)", resp.status_code)
            return None
        return resp.json().get("result", {}).get("message_id")
    except httpx.HTTPError as e:
        logger.warning("[TELEGRAM] sendVideo (preview) failed: %s", tb._redact_token(str(e)))
        return None


def edit_caption(message_id: int, caption: str, reply_markup: dict | None = None, chat_id: str | None = None) -> None:
    """Updates an already-sent approval message. Every approval message is
    sent via send_photo_with_keyboard() above (Telegram sendPhoto), and a
    photo message's text lives in its CAPTION, not a "text" field - Telegram
    rejects editMessageText on it with "Bad Request: there is no text in
    the message to edit" (the exact reported bug). This calls
    editMessageCaption instead, with a "caption" field in place of "text".

    Mirrors telegram_bot.edit_text()'s own contract exactly (best-effort:
    logs a warning and returns on any failure, never raises - a Telegram
    hiccup while updating a status message must never turn into an
    unhandled exception mid-callback) and reuses its URL/token-redaction
    plumbing (tb._url / tb._redact_token). telegram_bot.py itself, and its
    own edit_text(), are not modified."""
    payload = {
        "chat_id": chat_id or require("TELEGRAM_CHAT_ID"),
        "message_id": message_id,
        "caption": caption,
        "parse_mode": "HTML",
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(tb._url("editMessageCaption"), json=payload)
    except httpx.HTTPError as e:
        logger.warning("[TELEGRAM] editMessageCaption failed: %s", tb._redact_token(str(e)))
        return
    if resp.status_code >= 400:
        logger.warning("[TELEGRAM] editMessageCaption failed: %s", resp.text[:300])


def send_new_approval(payload: ApprovalPayload, chat_id: str | None = None) -> dict:
    """The one network+DB-writing entry point that creates and sends a new
    approval. Raises TelegramApprovalError with the exact phrasing spec
    section 12 asks for if config is missing - before any network call."""
    if not settings.TELEGRAM_BOT_TOKEN:
        raise TelegramApprovalError("Telegram bot token missing")
    resolved_chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not resolved_chat_id:
        raise TelegramApprovalError("Telegram chat id missing")

    approval_id = ta.create_approval(payload.slug, payload.title, status=STATUS_DRAFT)
    logger.info("[TELEGRAM] approval created slug=%s approval_id=%d", payload.slug, approval_id)
    logger.info("[TELEGRAM] thumbnail ready")
    logger.info("[TELEGRAM] video ready" if payload.video_sendable else "[TELEGRAM] video preview unavailable")

    # Overwrite the preview keyboard (slug-based, built before approval_id
    # existed) with the short, deterministic, ID-based one - see
    # build_keyboard_by_id()'s docstring for why this is required, not
    # cosmetic (a long slug's callback_data can exceed Telegram's 64-byte
    # limit and get the whole sendPhoto call rejected).
    payload.keyboard = build_keyboard_by_id(approval_id)
    message_id = send_photo_with_keyboard(payload.thumbnail_path, payload.caption, payload.keyboard, chat_id=resolved_chat_id)
    # Persist the full approval record reliably (spec: "Approval sistemi bu
    # bilgileri DB'ye guvenilir sekilde kaydetmeli") - video_path/
    # thumbnail_path in particular are not just informational: process_approval()
    # below reads them straight back off this row at upload time, so the
    # artifact that gets uploaded is always exactly the one shown in this
    # Telegram message, never whatever happens to be at a fixed convention
    # path by the time APPROVE is tapped. channel_key=None is stored as
    # NULL, meaning "the default/legacy YouTube channel" - the same
    # meaning None already has everywhere else in this codebase.
    ta.update_fields(
        approval_id, status=STATUS_READY,
        telegram_chat_id=str(resolved_chat_id), telegram_message_id=message_id,
        video_path=str(payload.video_path), thumbnail_path=str(payload.thumbnail_path),
        duration_label=payload.duration_label, scene_count=payload.scenes_count,
        language=payload.language, privacy=UPLOAD_VISIBILITY_LABEL,
        channel_key=payload.channel_key,
    )
    logger.info("[TELEGRAM] approval message sent")
    logger.info("[TELEGRAM] waiting for approval")

    video_message_id = None
    if payload.video_sendable:
        video_message_id = send_video_preview(payload.video_path, chat_id=resolved_chat_id)

    return {"approval_id": approval_id, "message_id": message_id, "video_message_id": video_message_id}


# --- Shorts Engine V1: ONE package approval for Long + 3 Shorts ------------
#
# CRITICAL (spec, non-negotiable): there must be ONLY ONE approval for the
# whole content package. Telegram must NOT upload/send the Long video file
# or any Short video file - only thumbnails + metadata + ONE APPROVE/REJECT
# pair. This reuses the SAME telegram_approvals table/row as a normal
# single-video approval (the Long IS that one parent approval row - no new
# approval table) and the SAME send_photo_with_keyboard()/build_keyboard_by_id()
# plumbing, completely unmodified; the only new thing is WHAT gets sent as
# the one photo (a composed collage of the Long + each Short's thumbnail,
# since Telegram's Bot API has no way to attach inline buttons to a
# multi-photo sendMediaGroup message).

COLLAGE_SHORT_CELL_WIDTH = 360
COLLAGE_SHORT_CELL_HEIGHT = 640


def _cover_resize(img, target_w: int, target_h: int):
    """Cover-crop-then-resize to an exact (target_w, target_h), same
    technique as youtube/thumbnail.py's compose_thumbnail() cover-crop step
    - never a non-uniform stretch."""
    w, h = img.size
    target_ratio = target_w / target_h
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    from PIL import Image
    return img.resize((target_w, target_h), Image.LANCZOS)


def build_package_collage(long_thumbnail_path: Path, short_thumbnails: list[tuple[int, Path]], dest_path: Path) -> Path:
    """ONE image: the Long's own thumbnail on top, each Short's own
    thumbnail in a row underneath (in short_number order) - sent through
    the completely unmodified send_photo_with_keyboard(), which is what
    lets a single Telegram message carry inline APPROVE/REJECT buttons
    alongside a preview of everything in the package (see module note
    above)."""
    from PIL import Image

    n_shorts = max(len(short_thumbnails), 1)
    collage_width = COLLAGE_SHORT_CELL_WIDTH * n_shorts
    long_height = int(collage_width * 9 / 16)
    border = 6
    collage_height = long_height + border + COLLAGE_SHORT_CELL_HEIGHT

    collage = Image.new("RGB", (collage_width, collage_height), (18, 18, 18))
    with Image.open(long_thumbnail_path) as im:
        collage.paste(_cover_resize(im.convert("RGB"), collage_width, long_height), (0, 0))

    y = long_height + border
    for i, (_short_number, thumb_path) in enumerate(short_thumbnails):
        with Image.open(thumb_path) as im:
            cell = _cover_resize(im.convert("RGB"), COLLAGE_SHORT_CELL_WIDTH, COLLAGE_SHORT_CELL_HEIGHT)
        collage.paste(cell, (i * COLLAGE_SHORT_CELL_WIDTH, y))

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(dest_path, format="PNG")
    return dest_path


def build_package_caption(long_title: str, long_duration_label: str, shorts_info: list[dict]) -> str:
    lines = [
        "\U0001f3ac <b>AI FOUND THIS — DAILY CONTENT READY</b>",
        "",
        long_title,
        "",
        "<b>LONG</b>",
        f"⏱ Duration: {long_duration_label}",
        "",
        "<b>SHORTS</b>",
    ]
    for s in shorts_info:
        lines.append(f"  {s['short_number']}. {s['title']}")
    lines += ["", "STATUS: WAITING FOR APPROVAL"]
    return "\n".join(lines)


@dataclass
class PackageApprovalPayload:
    slug: str
    long_title: str
    parent_video_id: int
    long_duration_label: str
    long_video_path: Path
    long_thumbnail_path: Path
    shorts: list[dict]  # [{"short_number", "title"}, ...] - display info only
    collage_path: Path
    caption: str
    keyboard: dict
    language: str
    channel_key: str | None = None


def build_package_approval_payload(
    slug: str, long_title: str, parent_video_id: int,
    long_video_path: Path, long_thumbnail_path: Path,
    shorts_rows, language: str | None = None, channel_key: str | None = None,
) -> PackageApprovalPayload:
    """Pure/side-effect-free (no network, no DB write) - mirrors
    build_approval_payload()'s own contract. `shorts_rows` is an iterable of
    storage.shorts rows (or plain dicts) each with at least short_number/
    title/thumbnail_path - normally storage.shorts.list_shorts_for_parent(
    parent_video_id), always in short_number order, always already
    STATUS_READY (QC-passed) by the time the caller builds this - see
    pipeline.py's _notify_telegram()."""
    language = language or settings.LANGUAGE_MASTER
    long_video_path = Path(long_video_path)
    long_thumbnail_path = Path(long_thumbnail_path)
    if not long_video_path.exists():
        raise TelegramApprovalError(f"final video not found at {long_video_path}")
    if not long_thumbnail_path.exists():
        raise TelegramApprovalError(f"thumbnail not found at {long_thumbnail_path}")
    for s in shorts_rows:
        if not s["thumbnail_path"] or not Path(s["thumbnail_path"]).exists():
            raise TelegramApprovalError(f"short thumbnail not found for short_number={s['short_number']}")

    duration_label = tb._format_duration(tb._probe_duration_seconds(long_video_path))
    shorts_info = [{"short_number": s["short_number"], "title": s["title"]} for s in shorts_rows]
    caption = build_package_caption(long_title, duration_label, shorts_info)

    collage_dest = long_thumbnail_path.parent / "shorts" / "package_collage.png"
    short_thumbs = [(s["short_number"], Path(s["thumbnail_path"])) for s in shorts_rows]
    collage_path = build_package_collage(long_thumbnail_path, short_thumbs, collage_dest)

    return PackageApprovalPayload(
        slug=slug, long_title=long_title, parent_video_id=parent_video_id,
        long_duration_label=duration_label, long_video_path=long_video_path,
        long_thumbnail_path=long_thumbnail_path, shorts=shorts_info,
        collage_path=collage_path, caption=caption, keyboard=build_keyboard(slug),
        language=language, channel_key=channel_key,
    )


def send_shorts_production_failed_notice(long_title: str, slug: str, shorts_results: list[dict],
                                          chat_id: str | None = None) -> int | None:
    """Spec: "If ANY Short fails QC: DO NOT send the package as
    APPROVE-ready. The Telegram package should show that production
    failed and provide the relevant failure information." No APPROVE/
    REJECT keyboard at all (there is nothing safe to approve yet) - a
    plain text notice via telegram_bot.send_text(), reused unchanged.
    Best-effort like every other Telegram send here: a notification
    failure must never raise back into the pipeline."""
    from storage import shorts as shorts_store

    lines = [
        "⚠️ AI FOUND THIS — SHORTS PRODUCTION FAILED",
        "",
        long_title,
        f"(slug={slug})",
        "",
        "The Long video is ready, but one or more Shorts did not pass QC - "
        "no package was sent for approval.",
        "",
    ]
    for s in shorts_results or []:
        status = s.get("status")
        if status == shorts_store.STATUS_READY:
            continue
        errors = "; ".join(s.get("qc_errors") or []) or s.get("error") or "unknown error"
        lines.append(f"Short {s.get('short_number')}: {status} — {errors[:200]}")
    text = "\n".join(lines)
    try:
        return tb.send_text(text, chat_id=chat_id or settings.TELEGRAM_CHAT_ID)
    except Exception:
        logger.exception("[TELEGRAM] failed to send Shorts production-failed notice")
        return None


def send_new_package_approval(payload: PackageApprovalPayload, chat_id: str | None = None) -> dict:
    """The package equivalent of send_new_approval() above - creates and
    sends ONE approval (the same telegram_approvals row a single-video
    approval would use - no new approval table) whose photo is the
    collage, then attaches every currently-produced Short for this Long to
    THIS approval_id (storage.shorts.set_approval_id_for_parent) so a
    single APPROVE/REJECT later governs all four pieces together (spec
    CRITICAL: ONE approval, never four)."""
    if not settings.TELEGRAM_BOT_TOKEN:
        raise TelegramApprovalError("Telegram bot token missing")
    resolved_chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not resolved_chat_id:
        raise TelegramApprovalError("Telegram chat id missing")

    approval_id = ta.create_approval(payload.slug, payload.long_title, status=STATUS_DRAFT)
    logger.info("[TELEGRAM] package approval created slug=%s approval_id=%d shorts=%d",
                payload.slug, approval_id, len(payload.shorts))

    payload.keyboard = build_keyboard_by_id(approval_id)
    message_id = send_photo_with_keyboard(payload.collage_path, payload.caption, payload.keyboard, chat_id=resolved_chat_id)

    ta.update_fields(
        approval_id, status=STATUS_READY,
        telegram_chat_id=str(resolved_chat_id), telegram_message_id=message_id,
        video_path=str(payload.long_video_path), thumbnail_path=str(payload.long_thumbnail_path),
        duration_label=payload.long_duration_label, language=payload.language,
        privacy=UPLOAD_VISIBILITY_LABEL, channel_key=payload.channel_key,
    )

    from storage import shorts as shorts_store
    shorts_store.set_approval_id_for_parent(payload.parent_video_id, approval_id)

    logger.info("[TELEGRAM] package approval message sent (thumbnails only - no Long/Short video sent to Telegram)")
    logger.info("[TELEGRAM] waiting for approval")
    return {"approval_id": approval_id, "message_id": message_id}


# --- project resolution for manual/recovery tooling -------------------------
#
# 2026-09 production incident: youtube.telegram_approval_demo's --slug
# resolution relied solely on _find_project_dir()'s filesystem glob
# (STORAGE_DIR/*/*/{slug}), which failed to locate a real, already-produced
# pipeline project and sent build_approval_payload() looking for
# metadata_en.json in the wrong directory entirely. The DB (`videos` table,
# storage/project_store.py) is the actual source of truth pipeline.py
# itself writes to and reads back from (see storage_path column) - this
# resolves against it first, and only falls back to the old debug_audit/
# glob convention when there is no DB row at all (e.g. the original
# the-wow-signal-6 manual demo flow, produced before this project tracked
# videos in the `videos` table).

def resolve_project_for_tooling(
    slug: str | None = None, video_id: int | None = None, project_dir: Path | None = None,
) -> tuple[Path, str, Path | None, Path | None]:
    """Returns (project_dir, slug, video_path, thumbnail_path) for
    youtube.telegram_approval_demo and resend_existing_approval() below.

    video_path/thumbnail_path are None when resolution falls back to the
    old debug_audit/ convention - build_approval_payload() already knows
    how to fall back to video_path_for(slug)/thumbnail_path_for(slug) for
    that case (unchanged), so this function only ever ADDS a more reliable
    resolution path, never removes the existing one."""
    from storage import project_store as ps

    if project_dir is not None:
        if slug is None:
            raise TelegramApprovalError("--project-dir requires --slug (to name the project)")
        return Path(project_dir), slug, None, None

    if video_id is not None:
        paths = ps.get_project(video_id)
        if paths is None:
            raise TelegramApprovalError(f"no project found for video_id={video_id}")
        return paths.root, paths.slug, paths.video_mp4, paths.thumbnail_png

    if slug is None:
        raise TelegramApprovalError("--slug, --video-id, or --project-dir is required")

    paths = ps.get_project_by_slug(slug)
    if paths is not None:
        return paths.root, paths.slug, paths.video_mp4, paths.thumbnail_png

    legacy_dir = _find_project_dir(slug)
    if legacy_dir is None:
        raise TelegramApprovalError(f"no project directory found for slug={slug!r} under {settings.STORAGE_DIR}")
    return legacy_dir, slug, None, None


# --- resend/repair an existing approval (never creates a new row) ----------

RESENDABLE_STATUSES = {STATUS_DRAFT, STATUS_READY}


def resend_existing_approval(approval_id: int, chat_id: str | None = None, dry_run: bool = False) -> dict:
    """Re-sends the Telegram approval message for an EXISTING approval row -
    the recovery path for a send_new_approval() that raised before it could
    finish (2026-09 incident: approval_id=3 was created by
    ta.create_approval() and then send_photo_with_keyboard() raised HTTP
    400, so the row was left stuck in STATUS_DRAFT with no
    telegram_message_id/video_path/thumbnail_path ever persisted on it -
    see the ordering in send_new_approval() above).

    NEVER creates a new approval row (no ta.create_approval() call here)
    and NEVER re-runs any pipeline stage - research/script/images/TTS/
    assembly are never touched, and pipeline.produce_video() is never
    called. This only rebuilds the approval payload from the project's
    already-produced artifacts (via resolve_project_for_tooling() above)
    and retries the ONE send_photo_with_keyboard() call the original
    attempt already made - reusing that function completely unchanged,
    diagnostics/HTTPStatusError-body-surfacing included.

    dry_run=True does everything up to and including building the payload
    (so a broken resolution/state is still caught) but makes no Telegram
    call and no DB write - mirrors send_new_approval()'s own --dry-run
    contract in youtube/telegram_approval_demo.py.

    Refuses to resend once the approval has already been decided (approved/
    uploading/uploaded/upload_failed/rejected) - re-prompting for a video
    that has already been approved or rejected would be misleading, not a
    recovery, and this never goes through _transition()/ALLOWED_TRANSITIONS
    (draft->ready is bookkeeping, not a human decision - exactly like
    send_new_approval() above, which updates status the same direct way)."""
    row = _get_or_raise(approval_id)
    if row["status"] not in RESENDABLE_STATUSES:
        raise TelegramApprovalError(
            f"cannot resend approval_id={approval_id} (slug={row['slug']!r}): status={row['status']!r} "
            f"has already been decided - resend only applies to {sorted(RESENDABLE_STATUSES)}"
        )

    project_dir, slug, video_path, thumbnail_path = resolve_project_for_tooling(slug=row["slug"])
    payload = build_approval_payload(
        slug, language=row["language"] or settings.LANGUAGE_MASTER, project_dir=project_dir,
        video_path=video_path, thumbnail_path=thumbnail_path, channel_key=row["channel_key"],
    )
    # approval_id is already known here (unlike send_new_approval(), which
    # only learns it after creating the row) - always use the short,
    # deterministic ID-based keyboard, never the slug-based preview one
    # (see build_keyboard_by_id()'s docstring: a long slug's callback_data
    # can exceed Telegram's 64-byte limit - the exact real-world
    # BUTTON_DATA_INVALID this fixes).
    payload.keyboard = build_keyboard_by_id(approval_id)

    if dry_run:
        return {
            "approval_id": approval_id, "slug": slug, "title": payload.title, "caption": payload.caption,
            "video_path": str(payload.video_path), "thumbnail_path": str(payload.thumbnail_path),
            "video_sendable": payload.video_sendable,
        }

    if not settings.TELEGRAM_BOT_TOKEN:
        raise TelegramApprovalError("Telegram bot token missing")
    resolved_chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not resolved_chat_id:
        raise TelegramApprovalError("Telegram chat id missing")

    message_id = send_photo_with_keyboard(payload.thumbnail_path, payload.caption, payload.keyboard, chat_id=resolved_chat_id)
    # Same fields, same direct ta.update_fields() call send_new_approval()
    # makes on its own freshly-created row - here it repairs/refreshes the
    # EXISTING row instead, self-healing one left incomplete by a failed
    # first attempt.
    ta.update_fields(
        approval_id, status=STATUS_READY,
        telegram_chat_id=str(resolved_chat_id), telegram_message_id=message_id,
        video_path=str(payload.video_path), thumbnail_path=str(payload.thumbnail_path),
        duration_label=payload.duration_label, scene_count=payload.scenes_count,
        language=payload.language, privacy=UPLOAD_VISIBILITY_LABEL,
        channel_key=payload.channel_key,
    )
    logger.info("[TELEGRAM] approval_id=%d resent (slug=%s) - no new approval row created", approval_id, slug)

    video_message_id = None
    if payload.video_sendable:
        video_message_id = send_video_preview(payload.video_path, chat_id=resolved_chat_id)

    return {"approval_id": approval_id, "slug": slug, "message_id": message_id, "video_message_id": video_message_id}


# --- state-machine transitions -------------------------------------------

def _get_or_raise(approval_id: int) -> sqlite3.Row:
    row = ta.get_approval(approval_id)
    if row is None:
        raise ApprovalNotFoundError(f"no approval with id={approval_id}")
    return row


def _transition(approval_id: int, to_status: str, **extra_fields) -> sqlite3.Row:
    row = _get_or_raise(approval_id)
    current = row["status"]
    if to_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ApprovalStateError(
            f"cannot move approval_id={approval_id} (slug={row['slug']!r}) "
            f"from status={current!r} to status={to_status!r}"
        )
    ta.update_fields(approval_id, status=to_status, **extra_fields)
    return ta.get_approval(approval_id)


def approve(approval_id: int) -> sqlite3.Row:
    return _transition(approval_id, STATUS_APPROVED)


def reject(approval_id: int, reason: str = "") -> sqlite3.Row:
    return _transition(approval_id, STATUS_REJECTED, rejection_reason=reason)


def mark_uploading(approval_id: int) -> sqlite3.Row:
    return _transition(approval_id, STATUS_UPLOADING)


def mark_uploaded(approval_id: int, youtube_video_id: str, youtube_url: str,
                   thumbnail_status: str | None = None, thumbnail_error: str | None = None) -> sqlite3.Row:
    """A video reaching STATUS_UPLOADED means the YouTube upload itself
    succeeded - thumbnail_status/thumbnail_error are recorded alongside it
    as plain informational fields, never as a reason to withhold or change
    this transition (see THUMBNAIL_STATUS_* above: a failed thumbnail must
    never turn a successful video upload into upload_failed)."""
    extra: dict = {"youtube_video_id": youtube_video_id, "youtube_url": youtube_url}
    if thumbnail_status is not None:
        extra["thumbnail_status"] = thumbnail_status
    if thumbnail_error is not None:
        extra["thumbnail_error"] = thumbnail_error
    return _transition(approval_id, STATUS_UPLOADED, **extra)


def mark_upload_failed(approval_id: int, error: str) -> sqlite3.Row:
    return _transition(approval_id, STATUS_UPLOAD_FAILED, upload_error=str(error)[:2000])


# --- YouTube upload trigger (reuses youtube.uploader - no new OAuth code) ---

def _upload_to_youtube(slug: str, video_path: Path, thumbnail_path: Path,
                        language: str | None = None, channel_key: str | None = None,
                        publish_decision=None) -> dict:
    """video_path/thumbnail_path are the artifacts to actually upload -
    always the ones persisted on the approval row at send_new_approval()
    time (see process_approval() below), NEVER recomputed from
    video_path_for(slug)/thumbnail_path_for(slug) here: what gets
    uploaded must always be exactly what was shown for approval, whatever
    its real source (production pipeline.py, or the manual demo CLI).

    publish_decision (Smart Publish Scheduler V1, additive/optional): a
    youtube.smart_publish.PublishDecision, only ever passed when
    settings.SMART_PUBLISH_ENABLED (see process_approval() below) - None
    for every pre-existing call site, in which case behavior is byte-for-
    byte unchanged from before this feature existed."""
    from youtube import uploader as yt_uploader

    language = language or settings.LANGUAGE_MASTER
    project_dir = _find_project_dir(slug)
    if project_dir is None:
        raise TelegramApprovalError(f"no project directory found for slug={slug!r} - cannot load metadata for upload")
    metadata = _load_json_or_none(project_dir / f"metadata_{language}.json")
    if not metadata:
        raise TelegramApprovalError(f"no metadata_{language}.json found at {project_dir} - cannot upload without metadata")

    metadata = dict(metadata)
    # EN ONEMLI KURAL (spec): V1 never auto-publishes public - force
    # private regardless of what the metadata file or env config says.
    metadata["privacy_status"] = "private"
    metadata.setdefault("category_id", settings.YOUTUBE_CATEGORY_ID)
    if publish_decision is not None:
        # RFC3339 UTC, the format the YouTube Data API's status.publishAt
        # actually requires (see youtube/uploader.py's upload_video(), which
        # forces privacyStatus="private" whenever this is set - already true
        # here regardless, per the rule just above).
        metadata["publish_at"] = publish_decision.scheduled_at.astimezone(dt_timezone.utc).isoformat().replace("+00:00", "Z")

    video_path = Path(video_path)
    thumbnail_path = Path(thumbnail_path)
    if not video_path.exists():
        raise TelegramApprovalError(f"final video not found at {video_path}")

    logger.info("[YOUTUBE] upload started")
    result = yt_uploader.upload_video(video_path, metadata, channel_key=channel_key)
    logger.info("[YOUTUBE] upload completed")
    logger.info("[YOUTUBE] visibility=private")

    # Thumbnail failure must NEVER un-publish or fail the video upload -
    # the video is already live on YouTube by this point. upload_thumbnail()
    # itself already classifies and redacts its own errors (scope vs.
    # channel-verification vs. generic - see youtube/uploader.py), so the
    # str(e) captured here is already safe to persist/display (no
    # client_secret/refresh_token can appear in it).
    thumbnail_status = THUMBNAIL_STATUS_PENDING
    thumbnail_error: str | None = None
    if thumbnail_path.exists():
        try:
            yt_uploader.upload_thumbnail(result["video_id"], thumbnail_path, channel_key=channel_key)
        except Exception as e:
            thumbnail_status = THUMBNAIL_STATUS_FAILED
            thumbnail_error = str(e)[:2000]
            logger.warning("[YOUTUBE] thumbnail upload failed (video still published): %s", e)
        else:
            thumbnail_status = THUMBNAIL_STATUS_SUCCESS
    else:
        thumbnail_status = THUMBNAIL_STATUS_FAILED
        thumbnail_error = f"thumbnail file not found at {thumbnail_path}"
        logger.warning("[YOUTUBE] thumbnail upload skipped: %s", thumbnail_error)

    result = dict(result)
    result["thumbnail_status"] = thumbnail_status
    result["thumbnail_error"] = thumbnail_error
    return result


def _compute_and_persist_publish_decision(approval_id: int, slug: str):
    """Smart Publish Scheduler V1 (additive, opt-in - see
    core.config.settings.SMART_PUBLISH_ENABLED). Returns (decision,
    schedule_id), or (None, None) when the feature is disabled - in which
    case process_approval()'s upload call is untouched, exactly as before
    this feature existed (Faz 14).

    IDEMPOTENT (Faz 7/8): storage.publish_schedules.create_schedule() is
    itself idempotent by approval_id - calling this twice for the same
    approval_id (e.g. a crashed-and-restarted process re-entering this
    function before mark_uploading()'s own state-machine guard would even
    apply) reuses the SAME schedule row and never computes/persists a
    second, different scheduled_at for it."""
    if not settings.SMART_PUBLISH_ENABLED:
        return None, None

    from storage import publish_schedules as psched
    from youtube import smart_publish

    existing = psched.get_schedule_by_approval_id(approval_id)
    if existing is not None:
        # Re-entry for an approval that already has a schedule (e.g. a
        # crash/restart) - reuse its persisted decision verbatim rather
        # than recomputing a new one (which could land on a different
        # scheduled_at than whatever was already logged/shown for this
        # approval).
        decision = smart_publish.PublishDecision(
            scheduled_at=datetime.fromisoformat(existing["scheduled_at"]),
            timezone=existing["timezone"], source=existing["source"],
            confidence=existing["confidence"], reason=existing["reason"] or "",
        )
        return decision, existing["id"]

    video_row = None
    try:
        from storage import db as _db
        video_row = _db.get_video_by_slug(slug)
    except Exception:
        # Best-effort only - a manual demo/test project may have no
        # corresponding storage/db.py `videos` row at all; publish_schedules
        # tolerates a NULL video_id (see storage/publish_schedules.py).
        video_row = None
    video_id = video_row["id"] if video_row is not None else None

    analytics_provider = None
    if settings.YOUTUBE_ANALYTICS_ENABLED:
        try:
            from youtube import analytics_timing
            analytics_provider = analytics_timing.get_audience_timing_provider()
        except Exception as e:
            # YouTube Analytics Timing V1 (additive, opt-in): any failure
            # constructing the provider must never block the existing
            # Smart Publish flow - fall back to the bootstrap-only path
            # exactly as if the feature were disabled. The provider itself
            # also never raises from get_best_publish_slots() (see
            # youtube/analytics_timing.py), so this is a defensive second
            # layer, not the primary safety net.
            logger.info("[PUBLISH] YouTube Analytics Timing provider unavailable, using bootstrap only: %s", e)
            analytics_provider = None

    decision = smart_publish.calculate_publish_time(
        now=datetime.now(dt_timezone.utc), timezone=settings.SMART_PUBLISH_TIMEZONE,
        analytics=analytics_provider,
    )
    logger.info("[PUBLISH] video_id=%s approval_id=%d slug=%s", video_id, approval_id, slug)
    schedule_id = psched.create_schedule(
        approval_id=approval_id, video_id=video_id,
        scheduled_at=decision.scheduled_at.isoformat(), timezone=decision.timezone,
        source=decision.source, confidence=decision.confidence, reason=decision.reason,
        status=psched.STATUS_PLANNED,
    )
    return decision, schedule_id


def _short_publish_decision(short_number: int):
    """Fixed daily publish slot for one Short (spec: recommended default
    cadence Short1=12:00/Short2=16:00/Short3=22:00, Europe/Istanbul,
    entirely config-driven via SHORT_<n>_DEFAULT_HOUR/MINUTE - see
    core/config.py's short_default_hour_minute()). Reuses
    youtube.smart_publish's existing, already-parameterized
    `_next_bootstrap_slot()` helper directly - NOT
    smart_publish.calculate_publish_time() itself, and NOT
    settings.SMART_PUBLISH_DEFAULT_HOUR/MINUTE - so this never touches or
    depends on Smart Publish/Analytics Timing being enabled at all (spec:
    "Do NOT modify Analytics Timing just for Shorts. Long scheduling should
    continue to use the existing Smart Publish Scheduler"). Returns a
    smart_publish.PublishDecision with a distinct source="shorts_fixed_slot"
    literal (never added to smart_publish.py itself) so a Short's schedule
    is clearly labeled without touching that module."""
    from youtube import smart_publish

    tz_name = settings.SHORTS_TIMEZONE
    tz = ZoneInfo(tz_name)
    local_now = datetime.now(dt_timezone.utc).astimezone(tz)
    hour, minute = settings.short_default_hour_minute(short_number)
    earliest = local_now + timedelta(minutes=settings.SMART_PUBLISH_MIN_DELAY_MINUTES)
    candidate = smart_publish._next_bootstrap_slot(local_now, earliest, hour, minute)
    return smart_publish.PublishDecision(
        scheduled_at=candidate, timezone=tz_name, source="shorts_fixed_slot", confidence="configured",
        reason=f"Fixed daily Shorts publish slot (short_number={short_number}, {hour:02d}:{minute:02d} {tz_name}).",
    )


def _build_short_metadata(row: sqlite3.Row, decision) -> dict:
    """Spec LONG/SHORT LINKING: if the parent Long already has a YouTube
    URL (storage/db.py's `videos.youtube_url`), reference it in the
    Short's description; scheduling may upload a Short before the Long, in
    which case a generic CTA is used instead - never a broken/guessed URL.
    V1 never auto-publishes public (mirrors _upload_to_youtube()'s own
    rule) - privacy_status is always forced "private" here regardless of
    settings.YOUTUBE_UPLOAD_PRIVACY, exactly like the Long's own upload."""
    from storage import db as _db

    long_url = None
    try:
        parent = _db.get_video(row["parent_video_id"])
        if parent is not None and parent["youtube_url"]:
            long_url = parent["youtube_url"]
    except Exception:
        long_url = None

    cta = f"Watch the full story: {long_url}" if long_url else "Watch the full story on AI FOUND THIS."
    description = f"{(row['script'] or '').strip()}\n\n{cta}".strip()

    return {
        "title": row["title"] or f"Short {row['short_number']}",
        "description": description,
        "tags": ["shorts", "documentary", "mystery", "aifoundthis"],
        "category_id": settings.YOUTUBE_CATEGORY_ID,
        "language": settings.LANGUAGE_MASTER,
        "privacy_status": "private",
        "publish_at": decision.scheduled_at.astimezone(dt_timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def process_shorts_for_approval(approval_id: int) -> list[dict]:
    """Uploads every Short attached to this ONE package approval_id that
    isn't already fully uploaded (storage.shorts.UPLOAD_ELIGIBLE_STATUSES),
    using the existing YouTube uploader (youtube.uploader - no second OAuth
    system) with a fixed daily publish slot per Short
    (_short_publish_decision() above).

    IDEMPOTENT AND INDEPENDENTLY RE-CALLABLE (spec CRITICAL): unlike the
    Long's own process_approval(), which can only ever run its upload once
    per approval (mark_uploading()'s ALLOWED_TRANSITIONS gate), this
    function is deliberately NOT gated that way - each Short's OWN status
    column is its idempotency guard, so calling this function any number
    of times (normally once, automatically, from inside process_approval()
    right after the Long's own upload succeeds - but also independently,
    e.g. for a manual retry after a partial failure) only ever
    (re)attempts Shorts that are not yet STATUS_UPLOADED, and never
    re-uploads one that already is. Returns [] for an approval_id with no
    attached Shorts (an ordinary, non-package single-video approval) - a
    safe no-op, never an error."""
    from storage import shorts as shorts_store
    from youtube import uploader as yt_uploader

    rows = shorts_store.list_shorts_for_approval(approval_id)
    results = []
    for row in rows:
        if row["status"] not in shorts_store.UPLOAD_ELIGIBLE_STATUSES:
            results.append({"short_number": row["short_number"], "status": row["status"], "skipped": True})
            continue
        try:
            shorts_store.update_fields(row["id"], status=shorts_store.STATUS_UPLOADING)
            decision = _short_publish_decision(row["short_number"])
            metadata = _build_short_metadata(row, decision)

            video_path = Path(row["video_path"]) if row["video_path"] else None
            if not video_path or not video_path.exists():
                raise TelegramApprovalError(f"short video not found at {video_path} (short_number={row['short_number']})")

            upload_result = yt_uploader.upload_video(video_path, metadata, channel_key=row["channel_key"])

            thumb_path = Path(row["thumbnail_path"]) if row["thumbnail_path"] else None
            if thumb_path and thumb_path.exists():
                try:
                    yt_uploader.upload_thumbnail(upload_result["video_id"], thumb_path, channel_key=row["channel_key"])
                except Exception as e:
                    logger.warning("[SHORTS][YOUTUBE] short_number=%d thumbnail upload failed (video still "
                                    "published): %s", row["short_number"], e)

            shorts_store.update_fields(
                row["id"], status=shorts_store.STATUS_UPLOADED,
                youtube_video_id=upload_result["video_id"], youtube_url=upload_result["url"],
                scheduled_at=decision.scheduled_at.isoformat(), timezone=decision.timezone,
                publish_source=decision.source, publish_confidence=decision.confidence,
                publish_reason=decision.reason, error=None,
            )
            logger.info("[SHORTS][YOUTUBE] short_number=%d uploaded: %s", row["short_number"], upload_result["url"])
            results.append({
                "short_number": row["short_number"], "status": shorts_store.STATUS_UPLOADED,
                "youtube_video_id": upload_result["video_id"], "youtube_url": upload_result["url"],
            })
        except Exception as e:
            shorts_store.update_fields(row["id"], status=shorts_store.STATUS_UPLOAD_FAILED, error=str(e)[:2000])
            logger.warning("[SHORTS][YOUTUBE] short_number=%d upload failed: %s", row["short_number"], e)
            results.append({"short_number": row["short_number"], "status": shorts_store.STATUS_UPLOAD_FAILED, "error": str(e)[:300]})
    return results


def _reject_shorts_for_approval(approval_id: int) -> None:
    """Spec: REJECT rejects all four. Only ever reached once per approval -
    reject(approval_id) itself raises ApprovalStateError on any repeat tap
    (STATUS_REJECTED has no outgoing ALLOWED_TRANSITIONS), so a duplicate
    REJECT callback never re-enters this function - handle_approval_callback()'s
    existing ApprovalStateError handling already makes that a safe no-op."""
    from storage import shorts as shorts_store

    for row in shorts_store.list_shorts_for_approval(approval_id):
        if row["status"] != shorts_store.STATUS_UPLOADED:
            shorts_store.update_fields(row["id"], status=shorts_store.STATUS_REJECTED)


def process_approval(approval_id: int, language: str | None = None) -> dict:
    """approved -> uploading -> uploaded|upload_failed. Only reachable once
    per approval: mark_uploading()'s own _transition() call raises
    ApprovalStateError on a second attempt, so this never double-uploads.

    video_path/thumbnail_path/channel_key are read straight off THIS
    approval's own row (persisted by send_new_approval() at send time) -
    never recomputed from a fixed path convention - so what gets uploaded
    is always exactly the artifact that was shown for approval.

    Smart Publish Scheduler V1 (additive/opt-in): when
    settings.SMART_PUBLISH_ENABLED, a publish time is computed and
    persisted (idempotently, per approval_id) BEFORE the YouTube call, and
    passed through so the upload itself carries YouTube's own
    publishAt/privacyStatus=private scheduling instead of an immediate
    private-forever upload. When disabled (the default), this whole block
    is skipped and the call below is identical to before this feature
    existed."""
    row = mark_uploading(approval_id)
    publish_decision, schedule_id = _compute_and_persist_publish_decision(approval_id, row["slug"])
    try:
        result = _upload_to_youtube(
            row["slug"], Path(row["video_path"]), Path(row["thumbnail_path"]),
            language=language, channel_key=row["channel_key"], publish_decision=publish_decision,
        )
    except Exception as e:
        if schedule_id is not None:
            from storage import publish_schedules as psched
            psched.update_fields(schedule_id, status=psched.STATUS_FAILED)
        logger.warning("[PUBLISH] scheduling failed")
        mark_upload_failed(approval_id, str(e))
        raise
    if schedule_id is not None:
        from storage import publish_schedules as psched
        new_status = psched.STATUS_SCHEDULED if result.get("scheduled_publish_at") else psched.STATUS_UPLOADED
        psched.update_fields(schedule_id, status=new_status, youtube_video_id=result["video_id"])
    mark_uploaded(
        approval_id, result["video_id"], result["url"],
        thumbnail_status=result.get("thumbnail_status"),
        thumbnail_error=result.get("thumbnail_error"),
    )
    result = dict(result)
    result["publish_decision"] = publish_decision

    # Shorts Engine V1 (additive): process every Short attached to this
    # approval right alongside the Long's own upload, so ONE APPROVE tap
    # does everything (spec: "After APPROVE... the system schedules all 4
    # pieces automatically"). Returns [] (falsy) for an approval_id with no
    # attached Shorts - a normal, pre-existing single-video approval - so
    # result["shorts"] is empty and every caller's existing behavior is
    # unchanged. Wrapped so an unexpected bug here can never lose the
    # Long's own already-successful upload result.
    try:
        result["shorts"] = process_shorts_for_approval(approval_id)
    except Exception as e:
        logger.exception("[SHORTS] processing shorts for approval_id=%d failed unexpectedly", approval_id)
        result["shorts"] = None
        result["shorts_error"] = str(e)[:500]
    return result


# --- Telegram callback handling -------------------------------------------

def _already_processed_text(row: sqlite3.Row) -> str:
    status = row["status"]
    title = row["title"]
    if status == STATUS_UPLOADED:
        return f"✅ Already approved & uploaded\n\n{title}\n\nYouTube URL:\n{row['youtube_url']}"
    if status in (STATUS_APPROVED, STATUS_UPLOADING):
        return f"✅ Already approved\n\n{title}\n\nUpload in progress or already completed."
    if status == STATUS_REJECTED:
        return f"❌ Already rejected\n\n{title}"
    if status == STATUS_UPLOAD_FAILED:
        return f"⚠️ Upload previously failed for\n\n{title}\n\nCheck server logs."
    return f"This approval for {title} has already been processed (status={status})."


def _thumbnail_status_line(thumbnail_status: str | None, thumbnail_error: str | None) -> str:
    """The final approval message's thumbnail line. thumbnail_error is
    already redacted/classified by upload_thumbnail() itself (see
    youtube/uploader.py) before it ever reaches here, so it is safe to
    show - still truncated defensively to keep the Telegram message short
    and readable."""
    if thumbnail_status == THUMBNAIL_STATUS_SUCCESS:
        return "\U0001f5bc Thumbnail: OK"
    short = thumbnail_error or "unknown error"
    if len(short) > 150:
        short = short[:147] + "..."
    return f"\U0001f5bc Thumbnail: FAILED — {short}"


def _smart_publish_strategy_label(source: str) -> str:
    from youtube import smart_publish
    if source == smart_publish.SOURCE_BOOTSTRAP:
        return "Bootstrap fallback"
    if source == smart_publish.SOURCE_ANALYTICS:
        return "Audience timing analytics"
    if source == smart_publish.SOURCE_INFERRED_PERFORMANCE:
        # YouTube Analytics Timing V1 (additive): labeled distinctly from
        # "Audience timing analytics" above, since this is a real-data
        # inference from video performance, not a direct audience-online
        # reading - see youtube/analytics_timing.py's module docstring.
        return "Inferred from video performance"
    return source


def _smart_publish_caption(title: str, result: dict) -> str:
    """Faz 6's approved-message format, used only when Smart Publish
    Scheduler V1 is enabled (result["publish_decision"] is not None - see
    process_approval()). Video production language is untouched (still
    English) - this is the Telegram admin caption, a separate concern from
    the Turkish Telegram Topic Selector management UI, which this does not
    touch at all."""
    decision = result["publish_decision"]
    youtube_state = "Scheduled / Private" if result.get("scheduled_publish_at") else "Private"
    thumbnail_line = _thumbnail_status_line(result.get("thumbnail_status"), result.get("thumbnail_error"))
    return (
        "\U0001f3ac VIDEO APPROVED\n\n"
        f"Title:\n{title}\n\n"
        f"\U0001f4c5 Scheduled:\n{decision.scheduled_at.strftime('%d %B %Y')}\n"
        f"{decision.scheduled_at.strftime('%H:%M')} {decision.timezone}\n\n"
        f"\U0001f4fa YouTube:\n{youtube_state}\n\n"
        f"\U0001f9e0 Publish strategy:\n{_smart_publish_strategy_label(decision.source)}\n\n"
        f"Reason:\n{decision.reason}\n\n"
        f"{thumbnail_line}"
    )


def _package_approved_caption(long_title: str, result: dict) -> str:
    """Shorts Engine V1: caption used only when this approval had attached
    Shorts (result["shorts"] non-empty - see process_approval()). Shows the
    Long's own upload result plus every Short's own outcome, never a
    partial/misleading "all done" when one failed."""
    shorts_results = result.get("shorts") or []
    lines = [
        "\U0001f680 PACKAGE APPROVED",
        "",
        f"LONG:\n{long_title}",
        f"YouTube: {result.get('url', '?')}",
        "",
        "SHORTS:",
    ]
    for s in shorts_results:
        status = s.get("status")
        if status == "uploaded":
            lines.append(f"  {s['short_number']}. ✅ {s.get('youtube_url', '')}")
        elif s.get("skipped"):
            lines.append(f"  {s['short_number']}. (already {status})")
        else:
            lines.append(f"  {s['short_number']}. ⚠️ {status}: {s.get('error', '')[:80]}")
    return "\n".join(lines)


def _handle_approve_action(approval_id: int, row: sqlite3.Row, message_id: int | None) -> None:
    approve(approval_id)
    logger.info("[TELEGRAM] approval received slug=%s", row["slug"])
    if message_id:
        edit_caption(message_id, f"✅ APPROVED\n\n{row['title']}\n\nYouTube upload started...")
    result = process_approval(approval_id)
    if message_id:
        if result.get("shorts"):
            # Shorts Engine V1: this approval had one or more attached
            # Shorts - show the combined package result rather than the
            # Long-only caption below.
            edit_caption(message_id, _package_approved_caption(row["title"], result))
        elif result.get("publish_decision") is not None:
            # Smart Publish Scheduler V1 enabled - Faz 6 caption.
            edit_caption(message_id, _smart_publish_caption(row["title"], result))
        else:
            # Smart Publish Scheduler V1 disabled (default) - EXACT original
            # message, byte-for-byte, per Faz 14's production-safety rule.
            thumbnail_line = _thumbnail_status_line(result.get("thumbnail_status"), result.get("thumbnail_error"))
            edit_caption(
                message_id,
                "\U0001f680 UPLOADED TO YOUTUBE\n\n"
                f"Visibility: PRIVATE\n\nVideo ID:\n{result['video_id']}\n\n"
                f"YouTube URL:\n{result['url']}\n\n{thumbnail_line}\n\nUpload status:\nSUCCESS",
            )
    logger.info("[TELEGRAM] approval message updated")


def _handle_reject_action(approval_id: int, row: sqlite3.Row, message_id: int | None) -> None:
    reject(approval_id)
    # Shorts Engine V1: reject every attached Short too (spec: "REJECT
    # rejects all four") - a safe no-op (empty list) for an ordinary,
    # non-package approval with no attached Shorts.
    _reject_shorts_for_approval(approval_id)
    if message_id:
        edit_caption(message_id, f"❌ REJECTED\n\n{row['title']}\n\nNo YouTube upload was performed.")


def handle_approval_callback(callback_query: dict) -> None:
    """Entry point for one Telegram callback_query dict (the payload of an
    update's "callback_query" key - see telegram_bot.poll_forever() for the
    shape). Mirrors telegram_bot._handle_callback()'s overall structure
    (validate -> answerCallbackQuery -> dispatch -> editMessageText) for the
    "approval:<approve|reject>:<identifier>" callback_data namespace and
    its own chat_id/message_id validation (spec section 11).

    <identifier> is accepted in TWO formats (2026-09 BUTTON_DATA_INVALID
    fix - see build_keyboard_by_id()'s docstring): a numeric approval_id
    (every real send now uses this - short and deterministic regardless of
    slug length) or, for backward compatibility with any already-sent
    message whose buttons still carry the old format, a slug (looked up
    via ta.get_latest_approval_by_slug(), exactly as before this fix)."""
    data = callback_query.get("data", "") or ""
    message = callback_query.get("message", {}) or {}
    message_id = message.get("message_id")
    chat = message.get("chat", {}) or {}
    chat_id = chat.get("id")
    cbq_id = callback_query.get("id")

    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "approval" or parts[1] not in ("approve", "reject"):
        logger.warning("[TELEGRAM] invalid approval callback data: %r", data)
        if cbq_id:
            tb._answer_callback(cbq_id, "Invalid button data")
        return

    _, action, identifier = parts
    try:
        approval_id = int(identifier)
        row = ta.get_approval(approval_id)
    except ValueError:
        # Legacy slug-based callback_data (pre-fix message still in Telegram).
        row = ta.get_latest_approval_by_slug(identifier)
        approval_id = row["id"] if row is not None else None

    if row is None:
        # 2026-09 production incident: an approval that genuinely exists
        # (confirmed via `telegram_approval_demo --resend-approval-id`) was
        # reported "unknown" here. Direct code/schema/test review found the
        # parsing and ta.get_approval()/ta.get_latest_approval_by_slug()
        # lookups themselves correct (see the dedicated tests in
        # tests/test_telegram_approval.py and
        # tests/test_telegram_approval_listener.py exercising this exact
        # path end-to-end). The most likely real-world cause of an
        # otherwise-valid identifier resolving to nothing is this listener
        # process reading a DIFFERENT storage/DB_PATH than whatever
        # created/resent that approval (each process resolves
        # STORAGE_DIR/DB_PATH purely from its OWN environment at startup -
        # see core/config.py). Logging the resolved DB path here (never a
        # secret) makes that mismatch immediately visible instead of
        # requiring another round of guessing.
        logger.warning(
            "[TELEGRAM] approval callback for unknown identifier=%s (db_path=%s) - if this approval "
            "is known to exist, confirm this listener process was started with the SAME "
            "STORAGE_DIR/DB_PATH environment as whatever created/resent it",
            identifier, settings.DB_PATH,
        )
        if cbq_id:
            tb._answer_callback(cbq_id, "Unknown project")
        return

    slug = row["slug"]

    # Security (spec section 11): the callback must match the SAME chat +
    # message this specific approval row was sent to. A forged
    # callback_data naming a real approval_id/slug, or a stale button on a
    # superseded message, is rejected here rather than acted on.
    if str(row["telegram_chat_id"]) != str(chat_id) or row["telegram_message_id"] != message_id:
        logger.warning(
            "[TELEGRAM] approval callback chat/message mismatch for slug=%s "
            "(expected chat_id=%s message_id=%s, got chat_id=%s message_id=%s)",
            slug, row["telegram_chat_id"], row["telegram_message_id"], chat_id, message_id,
        )
        if cbq_id:
            tb._answer_callback(cbq_id, "This approval message is no longer valid")
        return

    if cbq_id:
        tb._answer_callback(cbq_id, f"{action}...")

    try:
        if action == "approve":
            _handle_approve_action(approval_id, row, message_id)
        else:
            _handle_reject_action(approval_id, row, message_id)
    except ApprovalStateError:
        current = ta.get_approval(approval_id)
        logger.info(
            "[TELEGRAM] duplicate/late %s for slug=%s (approval_id=%d, already status=%s) - ignoring",
            action, slug, approval_id, current["status"],
        )
        if message_id:
            edit_caption(message_id, _already_processed_text(current))
    except Exception:
        # Never forward raw exception text to Telegram (mirrors
        # telegram_bot._handle_callback's own rule) - many error messages
        # here embed local filesystem paths.
        logger.exception("[TELEGRAM] approval callback handling failed (action=%s slug=%s)", action, slug)
        if message_id:
            edit_caption(message_id, f"⚠️ {action} failed for {slug} - check server logs for details")

# NOTE: the long-poll network loop used to live here as poll_forever(), but
# it was dead code (nothing called it, nothing tested it) and is now
# superseded by the dedicated youtube/telegram_approval_listener.py module
# (see its "python -m youtube.telegram_approval_listener" CLI), which
# dispatches to handle_approval_callback() above completely unchanged.
