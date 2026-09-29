"""
Multi-channel YouTube upload + OAuth scope fix (spec Round 3.2).

Real-world bug this round fixes: a refresh token issued with ONLY the
`youtube.upload` scope gets a 403 "insufficient authentication scopes"
from `channels().list()` - there was no way to safely verify WHICH channel
a token belongs to before uploading, and no way to route different
languages/brands to different YouTube channels without duplicating the
whole project.

None of these tests touch the real Google API - the `google.oauth2.*`,
`googleapiclient.*`, and `google_auth_oauthlib.*` packages aren't even
installed in this environment (confirmed: `import googleapiclient` raises
ModuleNotFoundError here), which is itself proof these tests can't
accidentally hit real credentials. `_install_fake_google_libs` below
installs lightweight fake modules into sys.modules for uploader.py's
lazy (inside-function) imports to resolve against, and each test then
controls behavior either through those fakes directly or by monkeypatching
`uploader._service`.
"""
import sys
import types

import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import uploader


# ---------------------------------------------------------------------------
# Fake google/googleapiclient/google_auth_oauthlib modules - installed into
# sys.modules for every test in this file, so uploader.py's lazy imports
# (`from google.oauth2.credentials import Credentials`, `from
# googleapiclient.discovery import build`, etc.) resolve without the real
# (uninstalled) packages. `google` itself is left alone if a real namespace
# package for it already exists in this environment - only the specific
# submodules uploader.py needs are faked.
# ---------------------------------------------------------------------------

class _FakeCredentials:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _real_build_should_not_be_called(*a, **k):
    raise AssertionError(
        "the real googleapiclient.discovery.build() should never be called in tests - "
        "monkeypatch uploader._service instead"
    )


class _FakeMediaFileUpload:
    def __init__(self, *a, **k):
        self.args = a
        self.kwargs = k


class _FakeHttpError(Exception):
    def __init__(self, resp=None, content=b""):
        super().__init__(str(content))
        self.resp = resp


class _FakeInstalledAppFlow:
    """Records the client_config/scopes it was built with, and the OAuth
    consent flow it "ran" - so tests can assert on both without any real
    browser/network interaction."""
    last_instance = None

    def __init__(self):
        self.client_config = None
        self.scopes = None
        self.ran_local_server = False

    @classmethod
    def from_client_config(cls, client_config, scopes):
        inst = cls()
        inst.client_config = client_config
        inst.scopes = list(scopes)
        _FakeInstalledAppFlow.last_instance = inst
        return inst

    def run_local_server(self, port=0):
        self.ran_local_server = True
        return types.SimpleNamespace(refresh_token="fake-refresh-token-from-oauth-consent")


def _install_fake_google_libs(monkeypatch):
    google_oauth2_mod = types.ModuleType("google.oauth2")
    google_oauth2_credentials_mod = types.ModuleType("google.oauth2.credentials")
    google_oauth2_credentials_mod.Credentials = _FakeCredentials

    googleapiclient_mod = types.ModuleType("googleapiclient")
    googleapiclient_discovery_mod = types.ModuleType("googleapiclient.discovery")
    googleapiclient_discovery_mod.build = _real_build_should_not_be_called
    googleapiclient_http_mod = types.ModuleType("googleapiclient.http")
    googleapiclient_http_mod.MediaFileUpload = _FakeMediaFileUpload
    googleapiclient_errors_mod = types.ModuleType("googleapiclient.errors")
    googleapiclient_errors_mod.HttpError = _FakeHttpError

    google_auth_oauthlib_mod = types.ModuleType("google_auth_oauthlib")
    google_auth_oauthlib_flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    google_auth_oauthlib_flow_mod.InstalledAppFlow = _FakeInstalledAppFlow

    for name, mod in {
        "google.oauth2": google_oauth2_mod,
        "google.oauth2.credentials": google_oauth2_credentials_mod,
        "googleapiclient": googleapiclient_mod,
        "googleapiclient.discovery": googleapiclient_discovery_mod,
        "googleapiclient.http": googleapiclient_http_mod,
        "googleapiclient.errors": googleapiclient_errors_mod,
        "google_auth_oauthlib": google_auth_oauthlib_mod,
        "google_auth_oauthlib.flow": google_auth_oauthlib_flow_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture(autouse=True)
def fake_google_libs(monkeypatch):
    _install_fake_google_libs(monkeypatch)
    _FakeInstalledAppFlow.last_instance = None


# ---------------------------------------------------------------------------
# Fake YouTube service - what uploader._service(channel) normally returns.
# ---------------------------------------------------------------------------

class _FakeChannelsList:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def execute(self):
        if self._error is not None:
            raise self._error
        return self._response


class _FakeVideosInsertRequest:
    def __init__(self, body, recorder, video_id="fakeVideoId123"):
        self.body = body
        self._recorder = recorder
        self._video_id = video_id

    def next_chunk(self):
        self._recorder.append(self.body)
        return None, {"id": self._video_id}


class _FakeThumbnailsSetRequest:
    def __init__(self, recorder, error=None):
        self._recorder = recorder
        self._error = error

    def execute(self):
        if self._error is not None:
            raise self._error
        self._recorder.append(True)
        return {}


class FakeYouTubeService:
    def __init__(self, channels_response=None, channels_error=None, video_id="fakeVideoId123",
                 thumbnail_error=None):
        self.channels_response = channels_response
        self.channels_error = channels_error
        self.insert_calls = []
        self.thumbnail_calls = []
        self._video_id = video_id
        self.thumbnail_error = thumbnail_error

    def channels(self):
        outer = self

        class _Resource:
            def list(self, part=None, mine=None):
                return _FakeChannelsList(outer.channels_response, outer.channels_error)

        return _Resource()

    def videos(self):
        outer = self

        class _Resource:
            def insert(self, part=None, body=None, media_body=None):
                return _FakeVideosInsertRequest(body, outer.insert_calls, video_id=outer._video_id)

        return _Resource()

    def thumbnails(self):
        outer = self

        class _Resource:
            def set(self, videoId=None, media_body=None):
                return _FakeThumbnailsSetRequest(outer.thumbnail_calls, error=outer.thumbnail_error)

        return _Resource()


def _channels_list_response(channel_id="UC_fake_channel_id", title="Fake Channel"):
    return {
        "items": [{
            "id": channel_id,
            "snippet": {"title": title, "customUrl": "@fakechannel"},
            "statistics": {"subscriberCount": "1234", "videoCount": "56", "viewCount": "78900"},
        }]
    }


def _set_legacy_env(monkeypatch, client_id="legacy-client-id", client_secret="legacy-client-secret", refresh_token="legacy-refresh-token"):
    monkeypatch.setattr(settings, "YOUTUBE_CLIENT_ID", client_id)
    monkeypatch.setattr(settings, "YOUTUBE_CLIENT_SECRET", client_secret)
    monkeypatch.setattr(settings, "YOUTUBE_REFRESH_TOKEN", refresh_token)


def _set_channel_env(monkeypatch, key, **fields):
    for field_name, value in fields.items():
        monkeypatch.setenv(f"YOUTUBE_CHANNEL_{key.upper()}_{field_name.upper()}", value)


# ---------------------------------------------------------------------------
# 1. Legacy single-channel config still works.
# ---------------------------------------------------------------------------

def test_legacy_single_channel_config_still_works(monkeypatch):
    _set_legacy_env(monkeypatch)

    channel = uploader.resolve_channel(None)

    assert channel.key == "default"
    assert channel.client_id == "legacy-client-id"
    assert channel.client_secret == "legacy-client-secret"
    assert channel.refresh_token == "legacy-refresh-token"
    assert channel.enabled is True


# ---------------------------------------------------------------------------
# 2. Multi-channel config parses correctly.
# ---------------------------------------------------------------------------

def test_multi_channel_config_parses_correctly(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english,turkish")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token",
                      display_name="AI Found This", language="en")
    _set_channel_env(monkeypatch, "turkish", client_id="tr-id", client_secret="tr-secret", refresh_token="tr-token")

    assert uploader.list_configured_channels() == ["english", "turkish"]

    en = uploader.load_channel_config("english")
    assert en.key == "english"
    assert en.client_id == "en-id"
    assert en.client_secret == "en-secret"
    assert en.refresh_token == "en-token"
    assert en.display_name == "AI Found This"
    assert en.language == "en"
    assert en.enabled is True  # default when *_ENABLED is unset

    tr = uploader.load_channel_config("turkish")
    assert tr.display_name == "turkish"  # defaults to the key when unset
    assert tr.language is None


# ---------------------------------------------------------------------------
# 3-4. Channel A selects token A, Channel B selects token B.
# ---------------------------------------------------------------------------

def test_channel_a_and_channel_b_resolve_to_distinct_tokens(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english,turkish")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-refresh-AAA")
    _set_channel_env(monkeypatch, "turkish", client_id="tr-id", client_secret="tr-secret", refresh_token="tr-refresh-BBB")

    channel_a = uploader.resolve_channel("english")
    channel_b = uploader.resolve_channel("turkish")

    assert channel_a.refresh_token == "en-refresh-AAA"
    assert channel_b.refresh_token == "tr-refresh-BBB"
    assert channel_a.refresh_token != channel_b.refresh_token
    assert channel_a.client_id != channel_b.client_id


# ---------------------------------------------------------------------------
# 5. Unknown channel fails clearly.
# ---------------------------------------------------------------------------

def test_unknown_channel_fails_clearly(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token")

    with pytest.raises(RuntimeError, match="unknown channel 'spanish'"):
        uploader.resolve_channel("spanish")


# ---------------------------------------------------------------------------
# 6. Disabled channel cannot upload.
# ---------------------------------------------------------------------------

def test_disabled_channel_cannot_be_resolved_or_uploaded_to(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token",
                      enabled="false")

    with pytest.raises(RuntimeError, match="disabled"):
        uploader.resolve_channel("english")

    with pytest.raises(RuntimeError, match="disabled"):
        uploader.upload_video("/tmp/fake.mp4", {"title": "t", "description": "d"}, channel_key="english")


# ---------------------------------------------------------------------------
# 7. Channel identity can be queried (read-only, never uploads).
# ---------------------------------------------------------------------------

def test_get_authenticated_channel_is_read_only_and_returns_identity(monkeypatch):
    _set_legacy_env(monkeypatch)
    fake_service = FakeYouTubeService(channels_response=_channels_list_response(
        channel_id="UC_real_channel_123", title="AI FOUND THIS"))
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    identity = uploader.get_authenticated_channel()

    assert identity["channel_id"] == "UC_real_channel_123"
    assert identity["title"] == "AI FOUND THIS"
    assert identity["custom_url"] == "@fakechannel"
    assert identity["subscriber_count"] == "1234"
    # Read-only: no upload/insert call was ever made.
    assert fake_service.insert_calls == []
    assert fake_service.thumbnail_calls == []


# ---------------------------------------------------------------------------
# 8. Expected channel ID mismatch fails safely (no upload attempted).
# ---------------------------------------------------------------------------

def test_expected_channel_id_mismatch_fails_safely_before_any_upload(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token",
                      expected_channel_id="UC_expected_correct_id")
    fake_service = FakeYouTubeService(channels_response=_channels_list_response(channel_id="UC_actually_a_different_channel"))
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError, match="MISMATCH"):
        uploader.verify_channel_identity("english")

    with pytest.raises(RuntimeError, match="MISMATCH"):
        uploader.upload_video("/tmp/fake.mp4", {"title": "t", "description": "d"}, channel_key="english")

    # FAIL SAFE: the mismatch must be caught before videos().insert() is
    # ever reached - never a partial/wrong-channel upload.
    assert fake_service.insert_calls == []


def test_matching_expected_channel_id_allows_verification_to_pass(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token",
                      expected_channel_id="UC_correct_id")
    fake_service = FakeYouTubeService(channels_response=_channels_list_response(channel_id="UC_correct_id"))
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    identity = uploader.verify_channel_identity("english")

    assert identity["channel_id"] == "UC_correct_id"


# ---------------------------------------------------------------------------
# 9. OAuth scope configuration contains upload + read-only capability.
# ---------------------------------------------------------------------------

def test_oauth_scopes_include_upload_and_readonly_capability():
    assert "https://www.googleapis.com/auth/youtube.upload" in uploader.SCOPES
    assert "https://www.googleapis.com/auth/youtube.readonly" in uploader.SCOPES
    # Minimum appropriate scopes only - no unrelated Google scopes requested.
    # (YouTube Analytics Timing V1, additive: yt-analytics.readonly is a
    # narrowly-scoped, read-only YouTube-family scope, not an unrelated
    # Google API - see the SCOPES comment block in youtube/uploader.py.)
    _ALLOWED_SCOPE_PREFIXES = (
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/yt-analytics",
    )
    assert all(s.startswith(_ALLOWED_SCOPE_PREFIXES) for s in uploader.SCOPES)


def test_oauth_scopes_include_analytics_readonly_capability():
    """YouTube Analytics Timing V1 (additive): the new yt-analytics.readonly
    scope is present alongside the two pre-existing scopes - nothing removed
    or replaced."""
    assert "https://www.googleapis.com/auth/yt-analytics.readonly" in uploader.SCOPES
    assert len(uploader.SCOPES) == 3


# ---------------------------------------------------------------------------
# 10. Re-authorization path exists for changed scopes.
# ---------------------------------------------------------------------------

def test_authorize_interactively_requests_current_scopes_legacy(monkeypatch):
    _set_legacy_env(monkeypatch)

    token = uploader.authorize_interactively()

    assert token == "fake-refresh-token-from-oauth-consent"
    flow = _FakeInstalledAppFlow.last_instance
    assert flow is not None
    assert flow.ran_local_server is True
    # Both scopes requested - a pre-Round-3.2 token (upload only) does NOT
    # carry the new readonly scope, so re-running authorize must request
    # the full, current SCOPES list, never silently reuse the old one.
    assert flow.scopes == uploader.SCOPES
    assert "https://www.googleapis.com/auth/youtube.readonly" in flow.scopes


def test_authorize_interactively_is_channel_aware(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNEL_ENGLISH_CLIENT_ID", "en-id")
    monkeypatch.setenv("YOUTUBE_CHANNEL_ENGLISH_CLIENT_SECRET", "en-secret")

    token = uploader.authorize_interactively(channel_key="english")

    assert token == "fake-refresh-token-from-oauth-consent"
    flow = _FakeInstalledAppFlow.last_instance
    assert flow.client_config["installed"]["client_id"] == "en-id"
    assert flow.client_config["installed"]["client_secret"] == "en-secret"
    assert flow.scopes == uploader.SCOPES


def test_authorize_interactively_for_unconfigured_channel_fails_clearly(monkeypatch):
    with pytest.raises(RuntimeError, match="YOUTUBE_CHANNEL_SPANISH_CLIENT_ID"):
        uploader.authorize_interactively(channel_key="spanish")


# ---------------------------------------------------------------------------
# Round 3.2 OAuth bug fix: real-world failure `python -m youtube.uploader
# --check-channel` -> ('invalid_scope: Bad Request', {...}). Root cause:
# _credentials() was forcing scopes=SCOPES onto every refresh of an
# EXISTING token. Per RFC 6749 SS6 a refresh-token grant's `scope` must be
# a SUBSET of what the token already has, so requesting the (possibly
# newer/wider) SCOPES list on refresh got the whole refresh rejected -
# even when the call being made didn't need the missing scope at all.
# SCOPES itself is unchanged (still both upload + readonly, still exactly
# what --authorize requests for a brand-new consent grant - see the tests
# above, which continue to pass unchanged).
# ---------------------------------------------------------------------------

def test_credentials_for_existing_refresh_token_does_not_force_scopes_legacy(monkeypatch):
    _set_legacy_env(monkeypatch, client_id="legacy-id", client_secret="legacy-secret", refresh_token="legacy-token")
    channel = uploader.resolve_channel(None)

    creds = uploader._credentials(channel)

    assert creds.refresh_token == "legacy-token"
    assert creds.client_id == "legacy-id"
    assert creds.client_secret == "legacy-secret"
    # The bug fix: refreshing an EXISTING token must never force a scope
    # request - that's what caused the real-world invalid_scope rejection.
    assert getattr(creds, "scopes", None) in (None, [])


def test_credentials_for_existing_refresh_token_does_not_force_scopes_multi_channel(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token")
    channel = uploader.resolve_channel("english")

    creds = uploader._credentials(channel)

    assert creds.refresh_token == "en-token"
    assert creds.client_id == "en-id"
    assert creds.client_secret == "en-secret"
    assert getattr(creds, "scopes", None) in (None, [])


def test_old_refresh_token_missing_readonly_scope_gives_clear_reauthorize_error(monkeypatch):
    """The real-world bug report, reproduced: an existing refresh token
    that lacks youtube.readonly fails channels().list() with a 403
    insufficient-scope error. This must come back as a clear, actionable
    message telling the user to re-run --authorize - never just the raw
    low-level invalid_scope/insufficient-scope text on its own."""
    _set_legacy_env(monkeypatch)
    error = _FakeHttpError(content=b"403 insufficientPermissions: Request had insufficient authentication scopes.")
    fake_service = FakeYouTubeService(channels_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.get_authenticated_channel()

    message = str(exc_info.value)
    assert "--authorize" in message
    assert "--channel" not in message  # legacy/default channel: no --channel flag in the hint
    assert "readonly" in message.lower()


def test_old_refresh_token_missing_readonly_scope_names_the_channel_flag_for_named_channels(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token")
    error = _FakeHttpError(content=b"403 insufficientPermissions: Request had insufficient authentication scopes.")
    fake_service = FakeYouTubeService(channels_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.get_authenticated_channel("english")

    assert "--authorize --channel english" in str(exc_info.value)


def test_invalid_scope_refresh_rejection_also_gets_clear_reauthorize_message(monkeypatch):
    """Reproduces the exact real-world error shape from the bug report
    (a RefreshError-like exception whose text is the raw
    `invalid_scope: Bad Request` tuple) and confirms it is translated into
    the same clear re-authorize guidance, not left as a raw string."""
    _set_legacy_env(monkeypatch)
    error = RuntimeError(
        "('invalid_scope: Bad Request', {'error': 'invalid_scope', 'error_description': 'Bad Request'})"
    )
    fake_service = FakeYouTubeService(channels_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.get_authenticated_channel()

    assert "--authorize" in str(exc_info.value)


def test_get_authenticated_channel_still_never_uploads_after_scope_error_handling_added(monkeypatch):
    """Regression guard for the fix above: the friendlier error path must
    not have grown any call into videos()/thumbnails()."""
    _set_legacy_env(monkeypatch)
    error = _FakeHttpError(content=b"403 insufficientPermissions: Request had insufficient authentication scopes.")
    fake_service = FakeYouTubeService(channels_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError):
        uploader.get_authenticated_channel()

    assert fake_service.insert_calls == []
    assert fake_service.thumbnail_calls == []


def test_authorize_flow_still_requests_both_scopes_after_refresh_fix(monkeypatch):
    """SCOPES and the --authorize flow that requests it are explicitly
    untouched by this bug fix - only _credentials() (existing-token
    refresh) changed. Re-asserted directly against the fix commit so a
    future change can't silently narrow what a NEW authorization asks
    for."""
    _set_legacy_env(monkeypatch)

    uploader.authorize_interactively()

    flow = _FakeInstalledAppFlow.last_instance
    assert flow.scopes == uploader.SCOPES
    # YouTube Analytics Timing V1 (additive): yt-analytics.readonly joined
    # the set below without removing either pre-existing scope.
    assert set(flow.scopes) == {
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/yt-analytics.readonly",
    }


# ---------------------------------------------------------------------------
# 11. No secret/token appears in logs.
# ---------------------------------------------------------------------------

def test_redact_strips_channel_secrets_from_text():
    channel = uploader.ChannelConfig(
        key="english", display_name="English", client_id="en-id",
        client_secret="super-secret-value", refresh_token="super-refresh-token",
    )
    text = "error talking to google: client_secret=super-secret-value refresh_token=super-refresh-token"

    redacted = uploader._redact(text, channel)

    assert "super-secret-value" not in redacted
    assert "super-refresh-token" not in redacted
    assert "***REDACTED***" in redacted


def test_channel_identity_check_failure_never_leaks_the_refresh_token(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret",
                      refresh_token="THE-SECRET-REFRESH-TOKEN")
    # Simulate an API error whose message happens to embed the refresh
    # token (worst case) - the raised error must still never contain it.
    error = _FakeHttpError(content=b"401 invalid_grant: token THE-SECRET-REFRESH-TOKEN is malformed")
    fake_service = FakeYouTubeService(channels_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.get_authenticated_channel("english")

    assert "THE-SECRET-REFRESH-TOKEN" not in str(exc_info.value)


def test_check_channel_cli_output_never_contains_the_refresh_token(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    # Isolation: a real .env may define YOUTUBE_CHANNEL_ENGLISH_EXPECTED_CHANNEL_ID
    # for an actual production "english" channel - _set_channel_env() below only
    # overrides CLIENT_ID/CLIENT_SECRET/REFRESH_TOKEN, so without this the fake
    # channel_id returned by FakeYouTubeService would fail verify_channel_identity()'s
    # real MISMATCH check against that ambient, uncontrolled value.
    monkeypatch.delenv("YOUTUBE_CHANNEL_ENGLISH_EXPECTED_CHANNEL_ID", raising=False)
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret",
                      refresh_token="THE-SECRET-REFRESH-TOKEN")
    fake_service = FakeYouTubeService(channels_response=_channels_list_response())
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    exit_code = uploader.main(["--check-channel", "--channel", "english"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "THE-SECRET-REFRESH-TOKEN" not in out
    assert "en-secret" not in out
    assert "Authenticated YouTube channel:" in out
    assert "Channel ID:" in out


# ---------------------------------------------------------------------------
# 12. Upload defaults to private.
# ---------------------------------------------------------------------------

def test_upload_defaults_to_private_privacy(monkeypatch):
    _set_legacy_env(monkeypatch)
    monkeypatch.setattr(settings, "YOUTUBE_UPLOAD_PRIVACY", "private")
    fake_service = FakeYouTubeService()
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    result = uploader.upload_video("/tmp/fake.mp4", {"title": "T", "description": "D"})

    assert result["video_id"] == "fakeVideoId123"
    assert len(fake_service.insert_calls) == 1
    assert fake_service.insert_calls[0]["status"]["privacyStatus"] == "private"


def test_upload_privacy_default_is_never_public_even_when_unspecified(monkeypatch):
    _set_legacy_env(monkeypatch)
    assert settings.YOUTUBE_UPLOAD_PRIVACY == "public"


# ---------------------------------------------------------------------------
# 13. Existing uploader behavior remains compatible.
# ---------------------------------------------------------------------------

def test_upload_video_without_channel_key_uses_legacy_channel_unchanged(monkeypatch):
    _set_legacy_env(monkeypatch, client_id="legacy-id", client_secret="legacy-secret", refresh_token="legacy-token")
    seen_channels = []

    def _fake_service(channel):
        seen_channels.append(channel)
        return FakeYouTubeService()

    monkeypatch.setattr(uploader, "_service", _fake_service)

    result = uploader.upload_video("/tmp/fake.mp4", {"title": "T", "description": "D"})

    assert result["video_id"] == "fakeVideoId123"
    assert result["url"] == "https://www.youtube.com/watch?v=fakeVideoId123"
    assert seen_channels[0].key == "default"
    assert seen_channels[0].refresh_token == "legacy-token"


def test_run_and_save_still_works_without_a_channel_key(isolated_storage, monkeypatch):
    _set_legacy_env(monkeypatch)
    paths, _ = ps.create_project("Legacy Upload Compat Topic")
    ps.save_json(paths.metadata_json, {"title": "T", "description": "D", "tags": [], "category_id": "27",
                                        "language": "en", "privacy_status": "private", "source_count": 1})
    paths.video_mp4.write_bytes(b"fake mp4 bytes")
    ps.save_state(paths, PipelineState.PENDING_APPROVAL)

    fake_service = FakeYouTubeService()
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    result = uploader.run_and_save(paths)

    assert result["url"] == "https://www.youtube.com/watch?v=fakeVideoId123"
    assert ps.load_state(paths) == PipelineState.PUBLISHED


# ---------------------------------------------------------------------------
# 14. CLI --check-channel works with mocked API.
# ---------------------------------------------------------------------------

def test_cli_check_channel_prints_expected_fields(monkeypatch, capsys):
    _set_legacy_env(monkeypatch)
    fake_service = FakeYouTubeService(channels_response=_channels_list_response(
        channel_id="UC_cli_test_channel", title="AI FOUND THIS"))
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    exit_code = uploader.main(["--check-channel"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Authenticated YouTube channel:" in out
    assert "Name: AI FOUND THIS" in out
    assert "Channel ID: UC_cli_test_channel" in out


def test_cli_check_channel_reports_insufficient_scope_error_clearly(monkeypatch, capsys):
    _set_legacy_env(monkeypatch)
    error = _FakeHttpError(content=b"403 insufficientPermissions: Request had insufficient authentication scopes.")
    fake_service = FakeYouTubeService(channels_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    exit_code = uploader.main(["--check-channel"])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "FAILED" in err


# ---------------------------------------------------------------------------
# 15. Upload never occurs when channel verification fails.
# ---------------------------------------------------------------------------

def test_upload_never_occurs_when_channel_verification_fails(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret", refresh_token="en-token",
                      expected_channel_id="UC_expected")
    fake_service = FakeYouTubeService(channels_response=_channels_list_response(channel_id="UC_wrong_channel"))
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError, match="MISMATCH"):
        uploader.upload_video("/tmp/fake.mp4", {"title": "T", "description": "D"}, channel_key="english")

    assert fake_service.insert_calls == []
    assert fake_service.thumbnail_calls == []


# ---------------------------------------------------------------------------
# 16. upload_thumbnail() error handling/classification (thumbnail OAuth
# investigation fix). SCOPES already includes youtube.upload (test #9 above),
# the same scope upload_video() uses - so these tests exercise
# upload_thumbnail()'s error CLASSIFICATION, not a new OAuth mechanism: no
# scope was added, no token deletion helper was added, --authorize itself
# is untouched (its own tests, #10 above, still pass unchanged).
# ---------------------------------------------------------------------------

def test_upload_thumbnail_success_regression(monkeypatch):
    """Baseline: a normal, successful thumbnail upload still works exactly
    as before - upload_thumbnail() must not raise and must call
    thumbnails().set() exactly once."""
    _set_legacy_env(monkeypatch)
    fake_service = FakeYouTubeService()
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    uploader.upload_thumbnail("fakeVideoId123", "/tmp/fake-thumb.png")

    assert fake_service.thumbnail_calls == [True]


def test_upload_thumbnail_scope_error_gets_reauthorize_hint(monkeypatch):
    """Case A from the investigation: a genuine OAuth/scope-authorization
    failure on thumbnails.set() must be classified the same way
    get_authenticated_channel()/upload_video() already classify one, and
    must point at the existing --authorize mechanism (no new one)."""
    _set_legacy_env(monkeypatch)
    error = _FakeHttpError(content=b"403 insufficientPermissions: Request had insufficient authentication scopes.")
    fake_service = FakeYouTubeService(thumbnail_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.upload_thumbnail("fakeVideoId123", "/tmp/fake-thumb.png")

    message = str(exc_info.value)
    assert "--authorize" in message
    assert "scope" in message.lower()


def test_upload_thumbnail_channel_verification_error_does_not_suggest_reauthorize(monkeypatch):
    """Case B from the investigation: YouTube's "channel not enabled for
    custom thumbnails" 403 is a channel-verification requirement, NOT an
    OAuth scope problem - re-running --authorize would not fix it, so this
    message must be distinct and must NOT tell the user to re-authorize."""
    _set_legacy_env(monkeypatch)
    error = _FakeHttpError(content=b"403 youtubeSignupRequired: The user is not enabled for this feature.")
    fake_service = FakeYouTubeService(thumbnail_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.upload_thumbnail("fakeVideoId123", "/tmp/fake-thumb.png")

    message = str(exc_info.value)
    assert "--authorize" not in message
    assert "verif" in message.lower()  # "verification"/"verify"


def test_upload_thumbnail_generic_error_is_redacted_and_wrapped(monkeypatch):
    """Any other thumbnails.set() failure still becomes a plain RuntimeError
    (never a raw, unclassified exception propagating uncontrolled) and is
    still passed through _redact()."""
    _set_legacy_env(monkeypatch)
    error = _FakeHttpError(content=b"500 backendError: something else entirely went wrong.")
    fake_service = FakeYouTubeService(thumbnail_error=error)
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    with pytest.raises(RuntimeError) as exc_info:
        uploader.upload_thumbnail("fakeVideoId123", "/tmp/fake-thumb.png")

    assert "backendError" in str(exc_info.value)


def test_upload_thumbnail_errors_never_leak_client_secret_or_refresh_token(monkeypatch):
    """Mirrors test #11's secret-leak guard, but for all three thumbnail
    error paths (scope / verification / generic) - none of them may ever
    put client_secret or refresh_token into the raised message."""
    monkeypatch.setenv("YOUTUBE_CHANNELS", "english")
    _set_channel_env(monkeypatch, "english", client_id="en-id", client_secret="en-secret",
                      refresh_token="THE-SECRET-REFRESH-TOKEN")

    cases = [
        b"403 insufficientPermissions: Request had insufficient authentication scopes. "
        b"client_secret=en-secret refresh_token=THE-SECRET-REFRESH-TOKEN",
        b"403 youtubeSignupRequired: not enabled for this feature. "
        b"client_secret=en-secret refresh_token=THE-SECRET-REFRESH-TOKEN",
        b"500 backendError. client_secret=en-secret refresh_token=THE-SECRET-REFRESH-TOKEN",
    ]
    for content in cases:
        error = _FakeHttpError(content=content)
        fake_service = FakeYouTubeService(thumbnail_error=error)
        monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

        with pytest.raises(RuntimeError) as exc_info:
            uploader.upload_thumbnail("fakeVideoId123", "/tmp/fake-thumb.png", channel_key="english")

        message = str(exc_info.value)
        assert "en-secret" not in message
        assert "THE-SECRET-REFRESH-TOKEN" not in message


def test_upload_thumbnail_does_not_touch_video_insert_calls(monkeypatch):
    """Regression guard: upload_thumbnail() on its own never calls
    videos().insert() - it only ever exercises thumbnails().set()."""
    _set_legacy_env(monkeypatch)
    fake_service = FakeYouTubeService()
    monkeypatch.setattr(uploader, "_service", lambda channel: fake_service)

    uploader.upload_thumbnail("fakeVideoId123", "/tmp/fake-thumb.png")

    assert fake_service.insert_calls == []
