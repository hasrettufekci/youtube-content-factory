"""
youtube/analytics_client.py tests - YouTube Analytics Timing V1's thin
read-only wrapper around the real YouTube Data API v3 (video listing/stats)
and YouTube Analytics API v2 (per-video engagement).

No real network/Google package access anywhere here - every test
monkeypatches analytics_client._data_service()/_analytics_service()
directly (mirrors tests/test_uploader.py's own established pattern of
monkeypatching uploader._service() rather than faking the whole
googleapiclient package), and channel credentials come from
core.config.settings, monkeypatched per test.

Covers spec scenarios 2 (no OAuth credentials), 3 (API unavailable),
4 (API returns no data), plus the underlying data-shaping this feature's
scoring/bucketing logic depends on.
"""
from __future__ import annotations

import pytest

from core.config import settings
from youtube import analytics_client as ac


def _set_legacy_env(monkeypatch, client_id="legacy-id", client_secret="legacy-secret", refresh_token="legacy-token"):
    monkeypatch.setattr(settings, "YOUTUBE_CLIENT_ID", client_id)
    monkeypatch.setattr(settings, "YOUTUBE_CLIENT_SECRET", client_secret)
    monkeypatch.setattr(settings, "YOUTUBE_REFRESH_TOKEN", refresh_token)


def _clear_legacy_env(monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_CLIENT_ID", "")
    monkeypatch.setattr(settings, "YOUTUBE_CLIENT_SECRET", "")
    monkeypatch.setattr(settings, "YOUTUBE_REFRESH_TOKEN", "")


# ---------------------------------------------------------------------------
# Fakes for the Data API v3 shapes list_recent_uploads() calls.
# ---------------------------------------------------------------------------

class _FakeExecutable:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def execute(self):
        if self._error is not None:
            raise self._error
        return self._response


class _FakeDataService:
    """Mimics googleapiclient's youtube v3 service just enough for
    list_recent_uploads(): .channels().list().execute(),
    .playlistItems().list().execute(), .videos().list().execute()."""

    def __init__(self, channels_response=None, playlist_pages=None, videos_response=None, error=None):
        self._channels_response = channels_response
        self._playlist_pages = list(playlist_pages or [])
        self._videos_response = videos_response
        self._error = error

    def channels(self):
        return self

    def playlistItems(self):
        return self

    def videos(self):
        return self

    def list(self, **kwargs):
        if self._error is not None:
            return _FakeExecutable(error=self._error)
        if "playlistId" in kwargs:
            page = self._playlist_pages.pop(0) if self._playlist_pages else {"items": []}
            return _FakeExecutable(response=page)
        if "id" in kwargs:
            return _FakeExecutable(response=self._videos_response or {"items": []})
        return _FakeExecutable(response=self._channels_response)


_CHANNEL_WITH_UPLOADS = {
    "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UUplaylist123"}}}]
}


def _video_item(video_id, published_at, views, likes=0, comments=0, title="A title"):
    return {
        "id": video_id,
        "snippet": {"publishedAt": published_at, "title": title},
        "statistics": {"viewCount": str(views), "likeCount": str(likes), "commentCount": str(comments)},
    }


# ---------------------------------------------------------------------------
# 2 - no OAuth credentials
# ---------------------------------------------------------------------------

def test_list_recent_uploads_raises_unavailable_when_credentials_missing(monkeypatch):
    _clear_legacy_env(monkeypatch)
    with pytest.raises(ac.AnalyticsUnavailableError):
        ac.list_recent_uploads()


def test_query_video_engagement_raises_unavailable_when_credentials_missing(monkeypatch):
    _clear_legacy_env(monkeypatch)
    with pytest.raises(ac.AnalyticsUnavailableError):
        ac.query_video_engagement(None, ["v1"], "2026-01-01", "2026-09-27")


# ---------------------------------------------------------------------------
# 3 - API unavailable (network/API error, and insufficient-scope specifically)
# ---------------------------------------------------------------------------

def test_list_recent_uploads_wraps_a_generic_api_error(monkeypatch):
    _set_legacy_env(monkeypatch)
    monkeypatch.setattr(ac, "_data_service", lambda channel: _FakeDataService(error=RuntimeError("boom")))
    with pytest.raises(ac.AnalyticsUnavailableError, match="boom"):
        ac.list_recent_uploads()


def test_list_recent_uploads_scope_error_mentions_reauthorize(monkeypatch):
    _set_legacy_env(monkeypatch)
    monkeypatch.setattr(
        ac, "_data_service",
        lambda channel: _FakeDataService(error=RuntimeError("insufficient authentication scopes")),
    )
    with pytest.raises(ac.AnalyticsUnavailableError, match="--authorize"):
        ac.list_recent_uploads()


def test_query_video_engagement_wraps_api_error(monkeypatch):
    _set_legacy_env(monkeypatch)

    class _FailingAnalyticsService:
        def reports(self):
            return self

        def query(self, **kwargs):
            return _FakeExecutable(error=RuntimeError("quota exceeded"))

    monkeypatch.setattr(ac, "_analytics_service", lambda channel: _FailingAnalyticsService())
    with pytest.raises(ac.AnalyticsUnavailableError, match="quota exceeded"):
        ac.query_video_engagement(None, ["v1"], "2026-01-01", "2026-09-27")


def test_list_recent_uploads_no_channel_found_raises_unavailable(monkeypatch):
    _set_legacy_env(monkeypatch)
    monkeypatch.setattr(ac, "_data_service", lambda channel: _FakeDataService(channels_response={"items": []}))
    with pytest.raises(ac.AnalyticsUnavailableError):
        ac.list_recent_uploads()


# ---------------------------------------------------------------------------
# 4 - API returns no data (channel exists, zero uploaded videos)
# ---------------------------------------------------------------------------

def test_list_recent_uploads_returns_empty_list_for_channel_with_no_videos(monkeypatch):
    _set_legacy_env(monkeypatch)
    fake = _FakeDataService(channels_response=_CHANNEL_WITH_UPLOADS, playlist_pages=[{"items": []}])
    monkeypatch.setattr(ac, "_data_service", lambda channel: fake)
    assert ac.list_recent_uploads() == []


def test_query_video_engagement_returns_empty_dict_for_empty_video_id_list(monkeypatch):
    _set_legacy_env(monkeypatch)
    # No service ever constructed - short-circuits before any API call.
    monkeypatch.setattr(ac, "_analytics_service", lambda channel: (_ for _ in ()).throw(AssertionError("should not be called")))
    assert ac.query_video_engagement(None, [], "2026-01-01", "2026-09-27") == {}


# ---------------------------------------------------------------------------
# Real data shaping - video listing
# ---------------------------------------------------------------------------

def test_list_recent_uploads_returns_real_fields_for_each_video(monkeypatch):
    _set_legacy_env(monkeypatch)
    fake = _FakeDataService(
        channels_response=_CHANNEL_WITH_UPLOADS,
        playlist_pages=[{"items": [{"contentDetails": {"videoId": "v1"}}], "nextPageToken": None}],
        videos_response={"items": [_video_item("v1", "2026-09-01T20:00:00Z", views=100, likes=10, comments=2, title="My Video")]},
    )
    monkeypatch.setattr(ac, "_data_service", lambda channel: fake)
    videos = ac.list_recent_uploads(lookback_days=90)
    assert len(videos) == 1
    assert videos[0]["video_id"] == "v1"
    assert videos[0]["title"] == "My Video"
    assert videos[0]["view_count"] == 100
    assert videos[0]["like_count"] == 10
    assert videos[0]["comment_count"] == 2
    assert videos[0]["published_at"].startswith("2026-09-01")


def test_list_recent_uploads_excludes_videos_older_than_lookback(monkeypatch):
    _set_legacy_env(monkeypatch)
    fake = _FakeDataService(
        channels_response=_CHANNEL_WITH_UPLOADS,
        playlist_pages=[{"items": [
            {"contentDetails": {"videoId": "recent"}},
            {"contentDetails": {"videoId": "old"}},
        ], "nextPageToken": None}],
        videos_response={"items": [
            _video_item("recent", "2026-09-20T20:00:00Z", views=50),
            _video_item("old", "2020-01-01T20:00:00Z", views=999),
        ]},
    )
    monkeypatch.setattr(ac, "_data_service", lambda channel: fake)
    videos = ac.list_recent_uploads(lookback_days=30)
    ids = [v["video_id"] for v in videos]
    assert "recent" in ids
    assert "old" not in ids


def test_list_recent_uploads_skips_videos_missing_published_at(monkeypatch):
    _set_legacy_env(monkeypatch)
    broken_item = {"id": "broken", "snippet": {"title": "No date"}, "statistics": {"viewCount": "5"}}
    fake = _FakeDataService(
        channels_response=_CHANNEL_WITH_UPLOADS,
        playlist_pages=[{"items": [{"contentDetails": {"videoId": "broken"}}], "nextPageToken": None}],
        videos_response={"items": [broken_item]},
    )
    monkeypatch.setattr(ac, "_data_service", lambda channel: fake)
    assert ac.list_recent_uploads() == []


def test_list_recent_uploads_orders_newest_first(monkeypatch):
    _set_legacy_env(monkeypatch)
    fake = _FakeDataService(
        channels_response=_CHANNEL_WITH_UPLOADS,
        playlist_pages=[{"items": [
            {"contentDetails": {"videoId": "older"}},
            {"contentDetails": {"videoId": "newer"}},
        ], "nextPageToken": None}],
        videos_response={"items": [
            _video_item("older", "2026-08-01T20:00:00Z", views=1),
            _video_item("newer", "2026-09-01T20:00:00Z", views=1),
        ]},
    )
    monkeypatch.setattr(ac, "_data_service", lambda channel: fake)
    videos = ac.list_recent_uploads()
    assert [v["video_id"] for v in videos] == ["newer", "older"]


# ---------------------------------------------------------------------------
# Real data shaping - engagement query
# ---------------------------------------------------------------------------

def test_query_video_engagement_parses_rows_by_column_headers(monkeypatch):
    _set_legacy_env(monkeypatch)

    class _FakeAnalyticsService:
        def reports(self):
            return self

        def query(self, **kwargs):
            return _FakeExecutable(response={
                "columnHeaders": [
                    {"name": "video"}, {"name": "views"}, {"name": "estimatedMinutesWatched"},
                    {"name": "averageViewDuration"}, {"name": "averageViewPercentage"},
                    {"name": "subscribersGained"},
                ],
                "rows": [["v1", 100, 250.0, 90.0, 55.5, 3]],
            })

    monkeypatch.setattr(ac, "_analytics_service", lambda channel: _FakeAnalyticsService())
    result = ac.query_video_engagement(None, ["v1"], "2026-01-01", "2026-09-27")
    assert result["v1"]["views"] == 100
    assert result["v1"]["average_view_percentage"] == 55.5
    assert result["v1"]["subscribers_gained"] == 3


def test_query_video_engagement_omits_videos_with_no_returned_row(monkeypatch):
    _set_legacy_env(monkeypatch)

    class _FakeAnalyticsService:
        def reports(self):
            return self

        def query(self, **kwargs):
            return _FakeExecutable(response={
                "columnHeaders": [{"name": "video"}, {"name": "views"}],
                "rows": [["v1", 10]],
            })

    monkeypatch.setattr(ac, "_analytics_service", lambda channel: _FakeAnalyticsService())
    result = ac.query_video_engagement(None, ["v1", "v2"], "2026-01-01", "2026-09-27")
    assert "v1" in result
    assert "v2" not in result  # never backfilled with a fabricated zero row
