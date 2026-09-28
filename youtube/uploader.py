"""
YouTube upload (spec sections 24, 30; extended for multi-channel OAuth in
Round 3.2).

Uses the YouTube Data API v3 (google-api-python-client). Requires an OAuth
refresh token per channel - get one once with:

    python -m youtube.uploader --authorize                    # legacy/default channel
    python -m youtube.uploader --authorize --channel english   # a configured channel

(opens a local browser consent flow, then prints the refresh token to put
in .env). After that, uploads are fully non-interactive.

MULTI-CHANNEL (Round 3.2): the content-generation pipeline stays shared -
ONE research, ONE master script, ONE shared scene plan/visuals (see
pipeline.py) - only the YouTube DESTINATION varies per channel. A channel
is a stable internal key (e.g. "english", "turkish") with its own OAuth
client/refresh token, resolved entirely from environment variables (never
hard-coded/committed secrets - see `ChannelConfig`/`resolve_channel()`
below). Every OAuth-using function in this module (`upload_video`,
`upload_thumbnail`, `get_authenticated_channel`, `verify_channel_identity`,
`run_and_save`) takes an optional `channel_key`:

    - channel_key=None (the default) -> legacy/default single-channel
      behavior: YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET /
      YOUTUBE_REFRESH_TOKEN, exactly as before Round 3.2. Every pre-Round-
      3.2 caller (including pipeline.py's handle_approve()) keeps working
      completely unchanged.
    - channel_key="english" (etc.) -> that channel's
      YOUTUBE_CHANNEL_ENGLISH_CLIENT_ID / _CLIENT_SECRET / _REFRESH_TOKEN
      (see `load_channel_config()`), only used once that channel key is
      listed in YOUTUBE_CHANNELS=english,turkish,... - see .env.example
      and README.md ("Multi-channel configuration") for the full list of
      per-channel env vars and a worked example.

OAUTH SCOPES (Round 3.2 fix): SCOPES now includes both upload and a
read-only channel-info scope, because a refresh token issued with ONLY
`youtube.upload` gets a 403 "insufficient authentication scopes" from
`channels().list()` (this is exactly the real-world failure this round
fixes). A refresh token issued BEFORE this change only has the old scope -
it is never silently assumed to have the new one; `--authorize` must be
re-run (per channel) to get a token covering both scopes. This module
never proceeds on a guess: any scope/identity problem surfaces as a clear,
fail-safe RuntimeError rather than a partial or wrong-channel action.

OAUTH REFRESH BUG FIX (Round 3.2.1): SCOPES is only ever sent to Google on
a NEW authorization (authorize_interactively(), via InstalledAppFlow).
`_credentials()`, which builds credentials for an EXISTING refresh token,
deliberately does NOT pass scopes=SCOPES - forcing that on every refresh
asked Google to widen a token's already-granted scope, which the refresh-
token grant rejects outright with `invalid_scope: Bad Request` (real-world
failure hit via `--check-channel` on an old token). See `_credentials()`
for the full explanation. A token that turns out to lack a needed scope
now fails only the specific call that needs it, with a clear message to
re-run `--authorize` (see `_is_scope_error()` / `get_authenticated_channel()`).

CHANNEL SAFETY: `get_authenticated_channel()` is read-only - it NEVER
uploads anything. `verify_channel_identity()` additionally checks the
authenticated channel_id (the only trustworthy identity - never a display
name) against that channel's configured `expected_channel_id`, when one is
set, and raises rather than proceeding on a mismatch. `upload_video()`
runs that same check automatically before touching the upload API whenever
`expected_channel_id` is configured for the resolved channel.

MVP uploads as private/unlisted (YOUTUBE_UPLOAD_PRIVACY, default "private",
overridable per channel - see ChannelConfig.privacy_status). Never uploads
without APPROVED state - enforced by pipeline.py, not here, so this module
stays a plain "upload this file with this metadata, to this channel"
utility that's independently testable.

SECRETS: never logged or printed. `_redact()` strips every known secret
string for the resolved channel out of any log/exception text before it is
emitted (mirrors telegram_bot.py's `_redact_token`, generalized to
per-channel OAuth secrets/refresh tokens instead of one Telegram token).

Standalone test (needs real YouTube OAuth credentials):
    python -m youtube.uploader --video-id 1
    python -m youtube.uploader --video-id 1 --channel english
    python -m youtube.uploader --check-channel
    python -m youtube.uploader --check-channel --channel english
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from core.config import settings, require
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps

logger = get_logger("youtube")

# Upload + read-only channel identity (Round 3.2: the read scope is what
# get_authenticated_channel()/verify_channel_identity() need for
# channels().list(mine=True) - a token issued with only youtube.upload
# gets a 403 "insufficient authentication scopes" from that call, which is
# exactly the real-world failure that motivated this scope expansion).
# Minimum appropriate scopes only - nothing broader is requested.
#
# yt-analytics.readonly (YouTube Analytics Timing V1, additive) - needed by
# youtube/analytics_client.py's read-only YouTube Analytics API v2 calls
# (see that module). Exactly the same pattern as the youtube.readonly
# addition above: an EXISTING refresh token issued before this scope was
# added simply doesn't have it yet, and every Analytics API call fails
# with the same insufficient-scope shape _is_scope_error() already
# recognizes below - youtube/analytics_timing.py treats that as "analytics
# unavailable" and falls back to Smart Publish's bootstrap default rather
# than raising. Re-running --authorize (this module's existing one-time
# flow - no new OAuth system) is what grants it; nothing about the upload
# path below changes or requires it.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

_BOOL_TRUE = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ChannelConfig:
    """One YouTube destination's resolved OAuth credentials + optional
    defaults. Never constructed with a literal secret in source code -
    always built from environment variables by `resolve_channel()` /
    `load_channel_config()` below."""

    key: str
    display_name: str
    client_id: str
    client_secret: str
    refresh_token: str
    enabled: bool = True
    language: str | None = None
    privacy_status: str | None = None
    category_id: str | None = None
    expected_channel_id: str | None = None

    def secrets(self) -> set[str]:
        """Every literal secret string that must never appear in a log
        line or raised error for this channel."""
        return {s for s in (self.client_secret, self.refresh_token) if s}


def _redact(text: str, channel: ChannelConfig) -> str:
    for secret in channel.secrets():
        text = text.replace(secret, "***REDACTED***")
    return text


def list_configured_channels() -> list[str]:
    """Channel keys configured via YOUTUBE_CHANNELS (comma-separated, e.g.
    YOUTUBE_CHANNELS=english,turkish - see .env.example). Empty/unset means
    no multi-channel config exists yet; callers fall back to the legacy
    single-channel env vars instead (see resolve_channel())."""
    raw = os.getenv("YOUTUBE_CHANNELS", "")
    return [c.strip() for c in raw.split(",") if c.strip()]


def _channel_env(channel_key: str, field_name: str, default: str = "") -> str:
    return os.getenv(f"YOUTUBE_CHANNEL_{channel_key.upper()}_{field_name}", default)


def load_channel_config(channel_key: str) -> ChannelConfig:
    """Build one channel's full config from its YOUTUBE_CHANNEL_<KEY>_*
    environment variables (never from source code - spec Round 3.2: no
    plaintext secrets in source, no duplicating client secrets into
    Python). Raises RuntimeError - never returns a partially-configured
    channel - if a required OAuth field is missing, naming exactly which
    env var(s) to set."""
    client_id = _channel_env(channel_key, "CLIENT_ID")
    client_secret = _channel_env(channel_key, "CLIENT_SECRET")
    refresh_token = _channel_env(channel_key, "REFRESH_TOKEN")
    missing = [
        f"YOUTUBE_CHANNEL_{channel_key.upper()}_{name}"
        for name, value in (("CLIENT_ID", client_id), ("CLIENT_SECRET", client_secret), ("REFRESH_TOKEN", refresh_token))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"[YOUTUBE] channel {channel_key!r} is missing required environment variable(s): "
            + ", ".join(missing)
            + " (run `python -m youtube.uploader --authorize --channel "
            + channel_key + "` after setting CLIENT_ID/CLIENT_SECRET to obtain the refresh token)"
        )

    return ChannelConfig(
        key=channel_key,
        display_name=_channel_env(channel_key, "DISPLAY_NAME", channel_key),
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
        enabled=_channel_env(channel_key, "ENABLED", "true").strip().lower() in _BOOL_TRUE,
        language=_channel_env(channel_key, "LANGUAGE") or None,
        privacy_status=_channel_env(channel_key, "PRIVACY") or None,
        category_id=_channel_env(channel_key, "CATEGORY_ID") or None,
        expected_channel_id=_channel_env(channel_key, "EXPECTED_CHANNEL_ID") or None,
    )


def _legacy_channel_config() -> ChannelConfig:
    """Backward compatibility (spec Round 3.2, REQUIRED): the pre-Round-
    3.2 single-channel env vars (YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN),
    wrapped as a ChannelConfig so every function below has exactly one
    code path regardless of whether the caller ever heard of multi-channel
    config. Existing users/scripts that never set YOUTUBE_CHANNELS keep
    working completely unchanged."""
    return ChannelConfig(
        key="default",
        display_name="default",
        client_id=require("YOUTUBE_CLIENT_ID"),
        client_secret=require("YOUTUBE_CLIENT_SECRET"),
        refresh_token=require("YOUTUBE_REFRESH_TOKEN"),
        enabled=True,
    )


def resolve_channel(channel_key: str | None = None) -> ChannelConfig:
    """The single entry point every OAuth-using function goes through.

        - channel_key=None -> legacy/default channel (YOUTUBE_CLIENT_ID/
          SECRET/REFRESH_TOKEN). This is the required backward-compat
          default: every caller written before Round 3.2 omits
          channel_key and keeps working unchanged.
        - channel_key="english" (etc.) -> that channel's own env vars.
          Raises a clear RuntimeError (never a silent fallback to another
          channel) if the key isn't listed in YOUTUBE_CHANNELS, or if that
          channel is disabled.
    """
    if channel_key is None:
        return _legacy_channel_config()

    configured = list_configured_channels()
    if channel_key not in configured:
        raise RuntimeError(
            f"[YOUTUBE] unknown channel {channel_key!r} - configured channels: "
            f"{configured if configured else '(none - set YOUTUBE_CHANNELS in .env, e.g. YOUTUBE_CHANNELS=english,turkish)'}"
        )
    channel = load_channel_config(channel_key)
    if not channel.enabled:
        raise RuntimeError(
            f"[YOUTUBE] channel {channel_key!r} is disabled "
            f"(YOUTUBE_CHANNEL_{channel_key.upper()}_ENABLED=false) - refusing to authenticate or upload"
        )
    return channel


def _credentials(channel: ChannelConfig):
    """Build Credentials for an EXISTING refresh token.

    Round 3.2 OAuth bug fix (real-world failure: `--check-channel` on
    Windows raised `('invalid_scope: Bad Request', ...)`): do NOT pass
    scopes=SCOPES here. Per RFC 6749 SS6, the `scope` parameter on a
    refresh-token grant is a request to narrow the token's ALREADY-
    GRANTED scope, not a way to add new permissions to it - google-auth's
    refresh_grant() only sends a `scope` value in the refresh request when
    Credentials.scopes is truthy. Forcing scopes=SCOPES here made every
    refresh of an EXISTING token ask Google for youtube.readonly even on
    tokens that were never granted it, and Google's token endpoint
    rejected the whole refresh with `invalid_scope: Bad Request` - even
    for a call that didn't need the missing scope at all.

    Leaving scopes unset means the refresh always succeeds, returning
    whatever scope the token already has. If that scope then turns out to
    be insufficient for the specific API call being made, THAT call fails
    on its own with a normal 403 - which get_authenticated_channel() below
    turns into a clear "re-run --authorize" message instead of a raw
    low-level invalid_scope/insufficient-scope error.

    SCOPES is unchanged and is still exactly what authorize_interactively()
    (--authorize) requests when creating a NEW token from scratch - a
    fresh consent grant is a different OAuth flow (authorization_code, not
    refresh_token) where requesting the full scope list is correct and
    required.
    """
    from google.oauth2.credentials import Credentials

    return Credentials(
        token=None,
        refresh_token=channel.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=channel.client_id,
        client_secret=channel.client_secret,
    )


def _service(channel: ChannelConfig):
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=_credentials(channel))


# Substrings that show up (case-insensitively) in the various shapes a
# missing/insufficient OAuth scope error can take here: an HttpError from
# an API call made with a token that lacks the needed scope
# ("insufficientPermissions" / "insufficient authentication scopes" -
# googleapiclient.errors.HttpError), or an invalid_scope rejection from the
# token refresh itself (google.auth.exceptions.RefreshError, or any
# similarly-shaped error - "invalid_scope" / "insufficientScope"). Any of
# these means the SAME thing to a user: the current refresh token doesn't
# cover what this call needs, and re-authorizing is the fix - so they all
# get the one clear message below instead of a raw low-level string.
_SCOPE_ERROR_MARKERS = (
    "invalid_scope",
    "insufficientpermissions",
    "insufficient authentication scopes",
    "insufficientscope",
)


def _is_scope_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _SCOPE_ERROR_MARKERS)


# Markers for the OTHER common real-world thumbnails.set() 403: the channel
# itself isn't enabled for custom thumbnails (YouTube requires phone/channel
# verification for this feature, independent of OAuth scope entirely - a
# token can have every scope it needs and this will still fail). Re-running
# --authorize does NOT fix this, so it gets its own distinct message rather
# than being folded into _is_scope_error()'s reauth guidance, which would be
# actively misleading here.
_THUMBNAIL_VERIFICATION_MARKERS = (
    "youtubesignuprequired",
    "not enabled for this feature",
)


def _is_thumbnail_verification_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _THUMBNAIL_VERIFICATION_MARKERS)


def _reauthorize_hint(channel_key: str | None) -> str:
    return (
        "python -m youtube.uploader --authorize"
        if channel_key is None
        else f"python -m youtube.uploader --authorize --channel {channel_key}"
    )


def authorize_interactively(channel_key: str | None = None) -> str:
    """One-time interactive OAuth consent flow. Requires a local browser -
    run this on your own machine, not on a headless server.

    Re-authorization (Round 3.2): this must be re-run whenever SCOPES
    changes (e.g. the youtube.readonly scope added this round) - an old
    refresh token keeps only the scope(s) it was originally issued with,
    and this module never assumes otherwise. Channel-aware: with
    --channel, only that channel's CLIENT_ID/CLIENT_SECRET need to be set
    beforehand (the refresh token is exactly what this flow produces, so
    it is intentionally NOT required yet - unlike load_channel_config(),
    used everywhere else, which does require it)."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if channel_key is None:
        client_id = require("YOUTUBE_CLIENT_ID")
        client_secret = require("YOUTUBE_CLIENT_SECRET")
        env_hint = "YOUTUBE_REFRESH_TOKEN"
    else:
        client_id = _channel_env(channel_key, "CLIENT_ID")
        client_secret = _channel_env(channel_key, "CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                f"[YOUTUBE] set YOUTUBE_CHANNEL_{channel_key.upper()}_CLIENT_ID and "
                f"YOUTUBE_CHANNEL_{channel_key.upper()}_CLIENT_SECRET before authorizing channel {channel_key!r}"
            )
        env_hint = f"YOUTUBE_CHANNEL_{channel_key.upper()}_REFRESH_TOKEN"

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)
    print(f"\nAdd this to your .env as {env_hint}:\n")
    print(creds.refresh_token)
    return creds.refresh_token


def get_authenticated_channel(channel_key: str | None = None) -> dict:
    """Read-only channel identity check - NEVER uploads or modifies
    anything (spec Round 3.2 requirement: "The function MUST NOT upload
    anything"). Requires the youtube.readonly scope (included in SCOPES
    above); a refresh token issued before that scope existed fails here
    with a clear error rather than any partial/guessed result - this is
    the exact real-world failure ("403 insufficient authentication
    scopes") this round fixes, now surfaced clearly instead of crashing
    the whole upload flow."""
    channel = resolve_channel(channel_key)
    service = _service(channel)
    try:
        response = service.channels().list(part="snippet,contentDetails,statistics", mine=True).execute()
    except Exception as e:
        if _is_scope_error(e):
            # Round 3.2 OAuth bug fix: this is the exact real-world failure
            # (an existing refresh token that predates the youtube.readonly
            # scope, or a refresh rejected as invalid_scope) - never leave
            # the user with just that raw low-level error. The redacted
            # original is still included for anyone who wants to see it.
            raise RuntimeError(
                f"[YOUTUBE] channel identity check failed for channel={channel.key!r}: "
                "the current refresh token does not cover the youtube.readonly scope "
                "(it was likely issued before that scope was added, or a scope "
                "mismatch was rejected on refresh). Re-authorize this channel to get "
                f"a token with both scopes: `{_reauthorize_hint(channel_key)}` "
                f"(original error: {_redact(str(e), channel)})"
            ) from None
        raise RuntimeError(
            f"[YOUTUBE] channel identity check failed for channel={channel.key!r}: {_redact(str(e), channel)}"
        ) from None

    items = response.get("items") or []
    if not items:
        raise RuntimeError(
            f"[YOUTUBE] channel identity check for channel={channel.key!r} returned no channels - "
            f"is this Google account associated with a YouTube channel?"
        )

    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    return {
        "channel_key": channel.key,
        "channel_id": item.get("id"),
        "title": snippet.get("title"),
        "custom_url": snippet.get("customUrl"),
        "subscriber_count": statistics.get("subscriberCount"),
        "video_count": statistics.get("videoCount"),
        "view_count": statistics.get("viewCount"),
    }


def verify_channel_identity(channel_key: str | None = None) -> dict:
    """Like get_authenticated_channel(), but additionally enforces that
    channel's configured expected_channel_id (YOUTUBE_CHANNEL_<KEY>_
    EXPECTED_CHANNEL_ID), when one is set - spec Round 3.2 CHANNEL SAFETY:
    "Use the authenticated channel ID as the authoritative identity...
    If the identity does not match: FAIL SAFE. Do not upload." Never
    matches on display name - only the API-returned channel_id counts."""
    channel = resolve_channel(channel_key)
    identity = get_authenticated_channel(channel_key)
    if channel.expected_channel_id and identity["channel_id"] != channel.expected_channel_id:
        raise RuntimeError(
            f"[YOUTUBE] channel identity MISMATCH for channel={channel.key!r}: authenticated as "
            f"channel_id={identity['channel_id']!r} (title={identity.get('title')!r}) but expected "
            f"channel_id={channel.expected_channel_id!r} - refusing to proceed (no upload attempted)"
        )
    return identity


def upload_video(video_path, metadata: dict, channel_key: str | None = None) -> dict:
    channel = resolve_channel(channel_key)

    # Channel safety (spec Round 3.2, IMPORTANT): if this channel has a
    # configured expected_channel_id, verify the authenticated identity
    # BEFORE touching the upload API at all. A mismatch raises and nothing
    # is uploaded - fail safe, never a silent wrong-channel upload.
    if channel.expected_channel_id:
        verify_channel_identity(channel_key)

    privacy_status = metadata.get("privacy_status") or channel.privacy_status or settings.YOUTUBE_UPLOAD_PRIVACY
    category_id = metadata.get("category_id") or channel.category_id or settings.YOUTUBE_CATEGORY_ID
    language = metadata.get("language") or channel.language or "en"

    # Smart Publish Scheduler V1 (AI FOUND THIS, additive/optional): a
    # caller (youtube/telegram_approval.py's _upload_to_youtube(), only when
    # settings.SMART_PUBLISH_ENABLED) may set metadata["publish_at"] to an
    # RFC3339 UTC timestamp. metadata.get("publish_at") is None for every
    # pre-existing caller, so this is a no-op change for them - body["status"]
    # comes out byte-identical to before. When set, this follows the YouTube
    # Data API's own actual, real requirement (not invented here): publishAt
    # is only honored when privacyStatus is "private" at insert time, so
    # privacy_status is forced to "private" in that case, exactly like the
    # pre-existing spec rule that V1 never auto-publishes public.
    publish_at = metadata.get("publish_at")
    if publish_at:
        privacy_status = "private"

    logger.info(
        "[YOUTUBE] uploading %s (channel=%s privacy=%s%s)", video_path, channel.key, privacy_status,
        f" publish_at={publish_at}" if publish_at else "",
    )

    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    # Named status_body, deliberately NOT `status` - the upload-progress loop
    # a few lines below already uses that name for next_chunk()'s own return
    # value (a MediaUploadProgress object), which is something completely
    # different; reusing it here would shadow that loop variable.
    status_body: dict = {
        "privacyStatus": privacy_status,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        status_body["publishAt"] = publish_at

    body = {
        "snippet": {
            "title": metadata["title"],
            "description": metadata["description"],
            "tags": metadata.get("tags", []),
            "categoryId": category_id,
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": status_body,
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True, chunksize=1024 * 1024 * 8)
    service = _service(channel)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    last_progress = -1
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504):
                logger.warning("[YOUTUBE] transient upload error %s, retrying chunk", e.resp.status)
                continue
            if _is_scope_error(e):
                raise RuntimeError(
                    f"[YOUTUBE] upload failed for channel={channel.key!r}: the current refresh "
                    "token does not cover a required scope. Re-authorize this channel: "
                    f"`{_reauthorize_hint(channel_key)}` (original error: {_redact(str(e), channel)})"
                ) from None
            raise RuntimeError(f"[YOUTUBE] upload failed: {_redact(str(e), channel)}") from e
        if status:
            pct = int(status.progress() * 100)
            if pct != last_progress:
                logger.info("[YOUTUBE] upload progress: %d%%", pct)
                last_progress = pct

    video_id = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info("[YOUTUBE] upload complete: %s (channel=%s)", url, channel.key)
    # scheduled_publish_at is a purely additive field (None for every
    # pre-existing caller that never set metadata["publish_at"]) - it lets
    # youtube/telegram_approval.py know whether YouTube actually received a
    # publishAt without having to re-derive it from `metadata` itself.
    return {"video_id": video_id, "url": url, "channel_key": channel.key, "scheduled_publish_at": publish_at}


def upload_thumbnail(video_id: str, thumbnail_path, channel_key: str | None = None) -> None:
    """Sets a video's custom thumbnail. Uses the SAME scope as upload_video()
    (youtube.upload, already in SCOPES) - no separate OAuth path, no new
    consent flow. Previously this had no error handling at all, so any
    failure (a real scope problem, or - far more common in practice, since
    it triggers even with a fully-scoped token - the channel simply not
    being enabled for custom thumbnails yet) surfaced as an opaque,
    unclassified exception. Now mirrors upload_video()'s own error
    classification: a genuine scope shortfall gets the --authorize hint,
    while a channel-verification 403 gets a distinct message that does NOT
    suggest re-authorizing (it wouldn't help)."""
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    channel = resolve_channel(channel_key)
    service = _service(channel)
    try:
        service.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
        ).execute()
    except HttpError as e:
        if _is_scope_error(e):
            raise RuntimeError(
                f"[YOUTUBE] thumbnail upload failed for channel={channel.key!r}: the current refresh "
                "token does not cover a required scope. Re-authorize this channel: "
                f"`{_reauthorize_hint(channel_key)}` (original error: {_redact(str(e), channel)})"
            ) from None
        if _is_thumbnail_verification_error(e):
            raise RuntimeError(
                f"[YOUTUBE] thumbnail upload failed for channel={channel.key!r}: YouTube reports this "
                "channel is not enabled for custom thumbnails (this is a channel verification "
                "requirement, not an OAuth scope problem - re-authorizing will not fix it; verify the "
                "channel, usually via phone number, in YouTube Studio, then retry). "
                f"(original error: {_redact(str(e), channel)})"
            ) from None
        raise RuntimeError(
            f"[YOUTUBE] thumbnail upload failed for channel={channel.key!r}: {_redact(str(e), channel)}"
        ) from None
    logger.info("[YOUTUBE] thumbnail set for %s (channel=%s)", video_id, channel.key)


def run_and_save(paths: ps.ProjectPaths, channel_key: str | None = None) -> dict:
    # Defense in depth against a duplicate upload: pipeline.handle_approve()
    # is the normal entry point and already guards this, but uploader.py is
    # independently callable (its own CLI, or direct import), so re-check
    # here too rather than trusting every caller to have checked state first.
    current_state = ps.load_state(paths)
    if current_state == PipelineState.PUBLISHED:
        row = db.get_video(paths.video_id)
        logger.warning("[YOUTUBE] video_id=%d is already PUBLISHED (%s) - refusing to upload again",
                        paths.video_id, row["youtube_url"] if row else "unknown url")
        return {
            "video_id": row["youtube_video_id"] if row else None,
            "url": row["youtube_url"] if row else None,
            "already_published": True,
        }
    if current_state == PipelineState.UPLOADING:
        raise RuntimeError(f"[YOUTUBE] video_id={paths.video_id} is already uploading (state=UPLOADING) - refusing a concurrent upload")

    metadata = ps.load_json(paths.metadata_json)
    if not metadata:
        raise RuntimeError(f"[YOUTUBE] no metadata.json found at {paths.metadata_json} - run metadata.py first")
    if not paths.video_mp4.exists():
        raise RuntimeError(f"[YOUTUBE] video file not found at {paths.video_mp4}")

    ps.save_state(paths, PipelineState.UPLOADING)
    run_id = db.start_run(paths.video_id, "youtube_upload")
    try:
        result = upload_video(paths.video_mp4, metadata, channel_key=channel_key)
        if paths.thumbnail_png.exists():
            try:
                upload_thumbnail(result["video_id"], paths.thumbnail_png, channel_key=channel_key)
            except Exception as e:
                logger.warning("[YOUTUBE] thumbnail upload failed (video still published): %s", e)

        db.update_video_youtube(paths.video_id, result["video_id"], result["url"])
        ps.save_state(paths, PipelineState.PUBLISHED, extra={
            "youtube_video_id": result["video_id"], "youtube_url": result["url"],
        })
        db.finish_run(run_id, "success")
        return result
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "youtube_upload", str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload the final video to YouTube.")
    parser.add_argument("--video-id", type=int, help="Video to upload")
    parser.add_argument("--authorize", action="store_true", help="Run the one-time interactive OAuth consent flow")
    parser.add_argument("--check-channel", action="store_true",
                         help="Read-only: verify the authenticated YouTube channel identity without uploading anything")
    parser.add_argument("--channel", default=None,
                         help="Channel key from YOUTUBE_CHANNELS (e.g. english). Omit to use the legacy "
                              "single-channel YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN configuration.")
    args = parser.parse_args(argv)

    if args.authorize:
        authorize_interactively(channel_key=args.channel)
        return 0

    if args.check_channel:
        try:
            identity = verify_channel_identity(args.channel)
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 1
        print("Authenticated YouTube channel:")
        print(f"Name: {identity.get('title')}")
        print(f"Channel ID: {identity.get('channel_id')}")
        if identity.get("custom_url"):
            print(f"Handle: {identity['custom_url']}")
        if identity.get("subscriber_count") is not None:
            print(f"Subscribers: {identity['subscriber_count']}")
        if identity.get("video_count") is not None:
            print(f"Videos: {identity['video_count']}")
        return 0

    if not args.video_id:
        print("FAILED: --video-id is required (or use --authorize / --check-channel)", file=sys.stderr)
        return 1

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        result = run_and_save(paths, channel_key=args.channel)
        print(f"OK - published: {result['url']}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
