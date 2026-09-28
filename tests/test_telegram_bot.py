"""
Telegram preview link logic (spec Round 2 section 3): a video too large for
Telegram's inline upload must fall back to a public URL (or a clear
"not configured" message) - and must NEVER leak a local filesystem path.

Also covers spec Round 2 section 15: the bot token must never be logged.
httpx embeds the full request URL - which for every Telegram Bot API call
contains the token (https://api.telegram.org/bot<TOKEN>/method) - in its
own exceptions' str()/repr(), so any place that logs an uncaught httpx
error from a Telegram call would otherwise leak the live token into
logs/app.log the moment Telegram returns an error (bad token, rate limit,
a brief outage).
"""
import httpx
import pytest

import telegram_bot as tb
from core.config import settings
from storage import project_store as ps


def test_relative_media_path_is_relative_not_absolute(isolated_storage):
    paths, _ = ps.create_project("Preview Path Topic")
    rel = tb._relative_media_path(paths)
    assert not rel.startswith("/")
    assert str(settings.STORAGE_DIR) not in rel


def test_public_preview_url_is_none_without_public_base_url(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
    paths, _ = ps.create_project("No Public URL Topic")
    assert tb._public_preview_url(paths) is None


def test_public_preview_url_built_from_base_url(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://videos.example.com/")
    paths, _ = ps.create_project("Public URL Topic")
    url = tb._public_preview_url(paths)
    assert url.startswith("https://videos.example.com/")
    assert url.endswith("/video.mp4")
    assert str(settings.STORAGE_DIR) not in url


def test_send_video_or_link_never_mentions_local_path_when_too_large(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
    monkeypatch.setattr(tb, "MAX_UPLOAD_BYTES", 10)  # force the "too large" branch
    paths, _ = ps.create_project("Oversized Video Topic")
    paths.video_mp4.write_bytes(b"x" * 100)  # 100 bytes > forced 10-byte "limit"

    sent = {}

    def fake_send_text(text, reply_markup=None, chat_id=None):
        sent["text"] = text
        return 123

    monkeypatch.setattr(tb, "send_text", fake_send_text)
    tb._send_video_or_link(paths, "NEW VIDEO READY")

    assert str(paths.root) not in sent["text"]
    assert str(settings.STORAGE_DIR) not in sent["text"]
    assert "PUBLIC_BASE_URL" in sent["text"]


def test_send_video_or_link_uses_public_url_when_configured_and_too_large(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://videos.example.com")
    monkeypatch.setattr(tb, "MAX_UPLOAD_BYTES", 10)
    paths, _ = ps.create_project("Oversized With Public URL Topic")
    paths.video_mp4.write_bytes(b"x" * 100)

    sent = {}

    def fake_send_text(text, reply_markup=None, chat_id=None):
        sent["text"] = text
        return 123

    monkeypatch.setattr(tb, "send_text", fake_send_text)
    tb._send_video_or_link(paths, "NEW VIDEO READY")

    assert "https://videos.example.com" in sent["text"]
    assert str(paths.root) not in sent["text"]


def test_keyboard_has_all_four_actions():
    kb = tb._keyboard(42)
    callback_data = [btn["callback_data"] for row in kb["inline_keyboard"] for btn in row]
    assert callback_data == ["preview:42", "approve:42", "regenerate:42", "reject:42"]


def test_callback_error_handler_never_forwards_raw_exception_text(monkeypatch):
    """A failed approve/reject/regenerate must not leak the raw exception
    message to Telegram - many of this project's RuntimeErrors embed local
    filesystem paths (e.g. "no metadata.json found at /home/.../storage/
    ..."), which is exactly the kind of server-side detail spec Round 2
    section 15 says must never leave this machine. The full exception is
    still logged server-side via logger.exception (not asserted here -
    that's stdlib logging, not something this test needs to intercept)."""
    import pipeline as pipeline_module

    secret_path = "/home/claude/work/YOUTUBE_CONTENT_FACTORY/storage/2026/09/some-slug/metadata.json"

    def fake_handle_approve(video_id):
        raise RuntimeError(f"[YOUTUBE] no metadata.json found at {secret_path} - run metadata.py first")

    monkeypatch.setattr(pipeline_module, "handle_approve", fake_handle_approve)
    monkeypatch.setattr(tb, "_answer_callback", lambda *a, **k: None)

    edited = {}
    monkeypatch.setattr(tb, "edit_text", lambda message_id, text, reply_markup=None: edited.update(text=text))

    tb._handle_callback({"id": "cbq1", "data": "approve:7", "message": {"message_id": 99}})

    assert "text" in edited
    assert secret_path not in edited["text"]
    assert "/home/" not in edited["text"]
    assert "metadata.json" not in edited["text"]
    assert "failed" in edited["text"].lower()
    assert "video_id=7" in edited["text"]


def test_handle_callback_silently_ignores_approval_v1_callback_data(monkeypatch):
    """telegram_approval_listener.py's Telegram Approval V1 uses its own
    callback namespace: "approval:<approve|reject>:<slug>" (see
    youtube/telegram_approval_listener.py's process_update()). This legacy,
    video_id-based listener's own _handle_callback() must never try to
    parse that: its `action, _, video_id_str = data.partition(":")` would
    read action="approval" and video_id_str="approve:some-slug", then fail
    int(video_id_str) and answer the button "Invalid button data" - a
    false, misleading response to a press meant for the other listener.
    The fix is a guard at the top of _handle_callback() that returns
    immediately for any "approval:"-prefixed callback_data, with no
    answerCallbackQuery call and no edit_text call - leaving the update
    entirely unanswered here so the real Telegram Approval V1 listener
    (polling the same token) can process it normally, exactly as if this
    legacy listener were not running at all."""
    answered = {"called": False}
    edited = {"called": False}
    monkeypatch.setattr(tb, "_answer_callback", lambda *a, **k: answered.update(called=True))
    monkeypatch.setattr(tb, "edit_text", lambda *a, **k: edited.update(called=True))

    result = tb._handle_callback({
        "id": "cbq1",
        "data": "approval:approve:some-slug",
        "message": {"message_id": 99, "chat": {"id": 1}},
    })

    assert result is None
    assert answered["called"] is False
    assert edited["called"] is False


def test_redact_token_strips_token_from_a_url_string(monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:SECRET-TOKEN-VALUE")
    text = "Client error '401 Unauthorized' for url 'https://api.telegram.org/bot123456:SECRET-TOKEN-VALUE/sendMessage'"

    redacted = tb._redact_token(text)

    assert "123456:SECRET-TOKEN-VALUE" not in redacted
    assert "REDACTED" in redacted


def test_redact_token_is_a_no_op_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    assert tb._redact_token("some text with no secret in it") == "some text with no secret in it"


def _http_status_error_response(monkeypatch, secret_token, status_code=401):
    """Build a fake httpx.Client whose .post()/.get() raises a real
    httpx.HTTPStatusError carrying a request URL with the token embedded -
    reproducing exactly what a real 401/429/5xx from Telegram looks like."""
    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def _raise(self, url, **kwargs):
            request = httpx.Request("POST", url)
            response = httpx.Response(status_code, request=request, text="Unauthorized")
            raise httpx.HTTPStatusError(f"error {status_code}", request=request, response=response)

        def post(self, url, **kwargs):
            return self._raise(url)

        def get(self, url, **kwargs):
            return self._raise(url)

    return _FakeClient


def test_send_text_never_leaks_token_in_its_raised_error(monkeypatch):
    secret_token = "123456:SECRET-TOKEN-VALUE"
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", secret_token)
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tb.httpx, "Client", _http_status_error_response(monkeypatch, secret_token))

    with pytest.raises(RuntimeError) as exc_info:
        tb.send_text("hello")

    assert secret_token not in str(exc_info.value)
    assert exc_info.value.__cause__ is None  # the raw, token-bearing httpx exception is not chained


def test_answer_callback_never_leaks_token_when_logging(monkeypatch, caplog):
    secret_token = "123456:SECRET-TOKEN-VALUE"
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", secret_token)
    monkeypatch.setattr(tb.httpx, "Client", _http_status_error_response(monkeypatch, secret_token))

    import logging
    with caplog.at_level(logging.WARNING, logger="ycf.telegram"):
        tb._answer_callback("cbq1")  # must not raise

    assert secret_token not in caplog.text
