"""
AI FOUND THIS - YouTube Analytics Timing V1: thin, read-only client wrapper
around the real YouTube Data API v3 (video listing/stats) and YouTube
Analytics API v2 (per-video engagement metrics).

Deliberately reuses youtube/uploader.py's existing OAuth/client
architecture rather than inventing a second one (spec requirement: "yeni
bir OAuth sistemi kurma, mumkunse mevcut credential'lari kullan"):
resolve_channel()/ChannelConfig/_credentials()/_is_scope_error()/_redact()
are all imported from that module unchanged. The ONLY change made to
uploader.py itself for this feature is the additive third OAuth scope,
`yt-analytics.readonly`, added to its SCOPES list (see the comment block
there) - exactly the same "Round 3.2" pattern already used for
`youtube.readonly`.

REAL API CAPABILITIES (verified via live WebFetch against
https://developers.google.com/youtube/analytics/dimensions and .../metrics
on 2026-09-27, per the spec's explicit "assume etme, gercek API'yi
incele" requirement - never assumed):

    - The YouTube Analytics API v2 has exactly TWO time-related dimensions:
      `day` and `month`. There is NO hour-of-day / time-of-day dimension.
      This means the "When your viewers are on YouTube" chart shown in
      YouTube Studio is NOT reproducible via this public API - any provider
      claiming to read real hour-level audience-online data directly from
      it would be lying. See youtube/analytics_timing.py's module
      docstring for how this codebase handles that honestly (performance
      inference from each video's own publishedAt hour, instead of a
      fabricated direct reading).
    - Per-video TOTALS (no time dimension at all - `dimensions="video"`,
      filtered to specific video IDs) ARE available and genuinely useful:
      estimatedMinutesWatched, averageViewDuration, averageViewPercentage,
      subscribersGained, subscribersLost, views. query_video_engagement()
      below uses exactly this shape.
    - Video-level view/like/comment counts and publishedAt timestamps come
      from the (older, simpler, already-scoped-for) Data API v3 instead -
      channels().list()/playlistItems().list()/videos().list() - which is
      what list_recent_uploads() below uses.

Every public function here converts ANY failure (missing credentials,
insufficient scope, network error, malformed response, the channel simply
having no videos yet) into AnalyticsUnavailableError with a redacted
message - callers (youtube/analytics_timing.py) treat that one exception
type uniformly as "no real Analytics data available right now, fall back"
and NEVER let it propagate up into the approval/upload flow (spec: "API
kullanilamiyorsa sistem durmamali").

No OAuth token, refresh token, or client secret is ever included in a log
line or exception message raised from this module - uploader._redact()
strips them from any error text before it's used.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone

from core.logging_setup import get_logger
from youtube import uploader

logger = get_logger("analytics_client")


class AnalyticsUnavailableError(RuntimeError):
    """Raised for any reason real Analytics/Data-API-backed video
    performance data could not be obtained right now - missing/insufficient
    OAuth scope, no configured channel, a network/API error, or a channel
    with zero uploaded videos. Always safe to catch broadly and fall back
    to the bootstrap default; never raised for "the data says nothing
    useful" (that case returns an empty/low-sample result instead - see
    youtube/analytics_timing.py)."""


def _data_service(channel: uploader.ChannelConfig):
    """YouTube Data API v3 service. A thin, separately-monkeypatchable
    wrapper (mirrors uploader._service()'s own pattern) so tests never need
    the real googleapiclient package installed."""
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=uploader._credentials(channel))


def _analytics_service(channel: uploader.ChannelConfig):
    """YouTube Analytics API v2 service - a distinct API/service name from
    the Data API v3 used by _data_service()/uploader._service(), requiring
    the additive yt-analytics.readonly scope."""
    from googleapiclient.discovery import build

    return build("youtubeAnalytics", "v2", credentials=uploader._credentials(channel))


def _wrap_error(exc: Exception, channel: uploader.ChannelConfig, action: str) -> AnalyticsUnavailableError:
    if uploader._is_scope_error(exc):
        return AnalyticsUnavailableError(
            f"[ANALYTICS] {action} failed for channel={channel.key!r}: the current refresh token "
            "does not cover the yt-analytics.readonly scope (it was likely issued before this "
            "feature was added). Re-authorize this channel to grant it: "
            f"`python -m youtube.uploader --authorize"
            f"{'' if channel.key == 'default' else ' --channel ' + channel.key}` "
            f"(original error: {uploader._redact(str(exc), channel)})"
        )
    return AnalyticsUnavailableError(
        f"[ANALYTICS] {action} failed for channel={channel.key!r}: {uploader._redact(str(exc), channel)}"
    )


def list_recent_uploads(
    channel_key: str | None = None,
    lookback_days: int = 90,
    max_results: int = 50,
) -> list[dict]:
    """Read-only: the channel's own most recent uploaded videos, each with
    its real publishedAt timestamp and view/like/comment counts (YouTube
    Data API v3 - no Analytics API call here). Returns a list of dicts:
        {"video_id": str, "published_at": str (ISO-8601 UTC),
         "title": str, "view_count": int, "like_count": int,
         "comment_count": int}
    ordered newest-first. Returns [] for a channel with no uploads at all
    (that is not an error - a caller with zero videos legitimately has
    nothing to infer from). Never returns a fabricated/partial row: a video
    missing statistics entirely is skipped rather than backfilled with 0.

    Raises AnalyticsUnavailableError (never a raw googleapiclient/auth
    exception) on any credential, scope, or API failure - including missing
    OAuth credentials themselves (resolve_channel() raising because
    YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN aren't set is just as much an
    "Analytics unavailable right now" situation as an insufficient-scope
    API error, so it is caught and wrapped here too, not left to propagate
    as a bare RuntimeError)."""
    try:
        channel = uploader.resolve_channel(channel_key)
    except Exception as e:
        raise AnalyticsUnavailableError(
            f"[ANALYTICS] list_recent_uploads failed: could not resolve channel "
            f"{channel_key!r}: {e}"
        ) from None

    cutoff = datetime.now(dt_timezone.utc) - timedelta(days=lookback_days)

    try:
        service = _data_service(channel)
        channels_response = service.channels().list(part="contentDetails", mine=True).execute()
        items = channels_response.get("items") or []
        if not items:
            raise AnalyticsUnavailableError(
                f"[ANALYTICS] list_recent_uploads failed for channel={channel.key!r}: "
                "no channel found for the authenticated account (mine=True returned no items)"
            )
        uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        video_ids: list[str] = []
        page_token = None
        while len(video_ids) < max_results:
            playlist_response = service.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=page_token,
            ).execute()
            for item in playlist_response.get("items", []):
                video_ids.append(item["contentDetails"]["videoId"])
            page_token = playlist_response.get("nextPageToken")
            if not page_token:
                break

        if not video_ids:
            return []

        videos: list[dict] = []
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start:start + 50]
            videos_response = service.videos().list(
                part="snippet,statistics", id=",".join(batch),
            ).execute()
            for item in videos_response.get("items", []):
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                published_at_raw = snippet.get("publishedAt")
                if not published_at_raw:
                    continue
                published_at = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
                if published_at < cutoff:
                    continue
                videos.append({
                    "video_id": item.get("id"),
                    "published_at": published_at.isoformat(),
                    "title": snippet.get("title"),
                    "view_count": int(statistics.get("viewCount", 0) or 0),
                    "like_count": int(statistics.get("likeCount", 0) or 0),
                    "comment_count": int(statistics.get("commentCount", 0) or 0),
                })
    except AnalyticsUnavailableError:
        raise
    except Exception as e:
        raise _wrap_error(e, channel, "list_recent_uploads") from None

    videos.sort(key=lambda v: v["published_at"], reverse=True)
    return videos


def query_video_engagement(
    channel_key: str | None,
    video_ids: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, dict]:
    """Read-only: per-video engagement TOTALS over [start_date, end_date]
    (YYYY-MM-DD strings) from the real YouTube Analytics API v2 -
    `dimensions="video"` with no time dimension at all, which is exactly
    the shape the real API documents as supported for per-video totals
    (see this module's docstring for why a time-of-day dimension is NOT
    used/available). Returns {video_id: {"estimated_minutes_watched": float,
    "average_view_duration": float, "average_view_percentage": float,
    "subscribers_gained": int}} - only for video IDs the API actually
    returned a row for for; a video with no data in this window is simply
    absent from the result, never backfilled with zeros.

    Returns {} for an empty video_ids list (no call made). Raises
    AnalyticsUnavailableError on any credential/scope/API failure -
    callers must treat that as "engagement data unavailable, fall back to
    view-count-only scoring" (see PerformanceTimingProvider), never as a
    reason to stop."""
    if not video_ids:
        return {}

    try:
        channel = uploader.resolve_channel(channel_key)
    except Exception as e:
        raise AnalyticsUnavailableError(
            f"[ANALYTICS] query_video_engagement failed: could not resolve channel {channel_key!r}: {e}"
        ) from None

    metrics = "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained"

    try:
        service = _analytics_service(channel)
        response = service.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=metrics,
            dimensions="video",
            filters=f"video=={','.join(video_ids)}",
            maxResults=len(video_ids),
        ).execute()
    except Exception as e:
        raise _wrap_error(e, channel, "query_video_engagement") from None

    headers = [h.get("name") for h in response.get("columnHeaders", [])]
    rows = response.get("rows") or []
    if "video" not in headers:
        return {}

    video_index = headers.index("video")
    result: dict[str, dict] = {}
    for row in rows:
        video_id = row[video_index]
        by_name = dict(zip(headers, row))
        result[video_id] = {
            "views": int(by_name.get("views", 0) or 0),
            "estimated_minutes_watched": float(by_name.get("estimatedMinutesWatched", 0) or 0),
            "average_view_duration": float(by_name.get("averageViewDuration", 0) or 0),
            "average_view_percentage": float(by_name.get("averageViewPercentage", 0) or 0),
            "subscribers_gained": int(by_name.get("subscribersGained", 0) or 0),
        }
    return result
