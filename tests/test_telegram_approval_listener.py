"""
youtube/telegram_approval_listener.py tests.

Covers all 14 required scenarios:
    1  callback update parsing
    2  approve dispatch
    3  reject dispatch
    4  invalid callback
    5  unknown slug
    6  duplicate approve
    7  update offset
    8  normal message ignored
    9  Telegram API error
   10  callback answer
   11  graceful shutdown
   12  listener dispatches existing handler (does not reimplement it)
   13  YouTube upload remains private
   14  upload failure -> upload_failed

Telegram and YouTube are always mocked - no real network call is ever made
by this file. get_updates() itself is monkeypatched in every poll_forever()
test so nothing here ever performs an actual HTTP request.
"""
from __future__ import annotations

import ast
import json

import httpx
import pytest

from core.config import settings
from youtube import telegram_approval as tap
from youtube import telegram_approval_listener as lst


# ---------------------------------------------------------------------------
# fixtures / helpers (self-contained, mirrors tests/test_telegram_approval.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def approval_env(isolated_storage, tmp_path, monkeypatch):
    video_root = tmp_path / "debug_audit" / "final_video_v2"
    thumb_root = tmp_path / "debug_audit" / "thumbnail_v1"
    video_root.mkdir(parents=True)
    thumb_root.mkdir(parents=True)
    monkeypatch.setattr(tap, "FINAL_VIDEO_ROOT", video_root)
    monkeypatch.setattr(tap, "THUMBNAIL_ROOT", thumb_root)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)
    return tap


def _write_project(storage_dir, slug, title="The Wow! Signal: Astronomy's Greatest Unsolved Mystery", scenes=33):
    project_dir = storage_dir / "2026" / "09" / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata_en.json").write_text(json.dumps({"title": title, "description": "d", "tags": []}))
    (project_dir / "scenes.json").write_text(json.dumps([{"scene_id": i} for i in range(1, scenes + 1)]))
    return project_dir


def _write_video(tap_module, slug, size_bytes=1000):
    path = tap_module.video_path_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)


def _write_thumbnail(tap_module, slug):
    path = tap_module.thumbnail_path_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)


def _full_fixture(approval_env, isolated_storage, slug="the-wow-signal-6"):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, slug)
    _write_video(approval_env, slug)
    _write_thumbnail(approval_env, slug)
    return project_dir


class _FakeYouTube:
    def __init__(self, video_id="abc123", url="https://www.youtube.com/watch?v=abc123", fail=False):
        self.video_calls = []
        self.thumbnail_calls = []
        self.video_id = video_id
        self.url = url
        self.fail = fail

    def upload_video(self, video_path, metadata, channel_key=None):
        self.video_calls.append((video_path, dict(metadata), channel_key))
        if self.fail:
            raise RuntimeError("[YOUTUBE] simulated upload failure")
        return {"video_id": self.video_id, "url": self.url, "channel_key": channel_key or "default"}

    def upload_thumbnail(self, video_id, thumbnail_path, channel_key=None):
        self.thumbnail_calls.append((video_id, thumbnail_path, channel_key))


@pytest.fixture
def fake_youtube(monkeypatch):
    from youtube import uploader
    fake = _FakeYouTube()
    monkeypatch.setattr(uploader, "upload_video", fake.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", fake.upload_thumbnail)
    return fake


@pytest.fixture
def fake_telegram_send(monkeypatch):
    sent = {"photos": [], "videos": []}

    def fake_photo(image_path, caption, keyboard, chat_id=None):
        message_id = 1000 + len(sent["photos"])
        sent["photos"].append({"message_id": message_id})
        return message_id

    def fake_video(video_path, caption="", chat_id=None):
        message_id = 2000 + len(sent["videos"])
        sent["videos"].append({"message_id": message_id})
        return message_id

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_photo)
    monkeypatch.setattr(tap, "send_video_preview", fake_video)
    return sent


@pytest.fixture
def fake_edit_text(monkeypatch):
    edits = []
    monkeypatch.setattr(tap.tb, "edit_text", lambda message_id, text, reply_markup=None: edits.append((message_id, text)))
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": edits.append(("ANSWER", cbq_id, text)))
    return edits


def _send(approval_env, isolated_storage, fake_telegram_send, slug="the-wow-signal-6"):
    project_dir = _full_fixture(approval_env, isolated_storage, slug)
    payload = tap.build_approval_payload(slug, project_dir=project_dir)
    result = tap.send_new_approval(payload)
    message_id = fake_telegram_send["photos"][-1]["message_id"]
    return result, message_id


def _update(update_id, action, slug, chat_id=999, message_id=None, cbq_id="cbq1"):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cbq_id,
            "data": f"approval:{action}:{slug}",
            "message": {"message_id": message_id, "chat": {"id": chat_id}},
        },
    }


def _batched_get_updates(batches):
    remaining = list(batches)

    def fake(offset, timeout=30):
        return remaining.pop(0) if remaining else []

    return fake


# ---------------------------------------------------------------------------
# offset persistence
# ---------------------------------------------------------------------------

def test_load_offset_defaults_to_zero_when_no_file(isolated_storage):
    assert lst.load_offset() == 0


def test_save_and_load_offset_roundtrip(isolated_storage):
    lst.save_offset(42)
    assert lst.load_offset() == 42


def test_load_offset_tolerates_corrupt_file(isolated_storage):
    lst.offset_file().parent.mkdir(parents=True, exist_ok=True)
    lst.offset_file().write_text("not-a-number")
    assert lst.load_offset() == 0


def test_offset_file_is_separate_from_telegram_bot_offset_file(isolated_storage):
    assert lst.offset_file().name == ".telegram_approval_listener_offset"
    assert lst.offset_file().name != ".telegram_offset"  # telegram_bot.OFFSET_FILE's own name
    assert lst.offset_file().name != ".telegram_approval_offset"  # the old (now-removed) tap.poll_forever's name


# ---------------------------------------------------------------------------
# Single-instance lock (Telegram 409 Conflict fix): only one listener
# process may hold the getUpdates lock for a given storage root at a time.
# ---------------------------------------------------------------------------

def test_lock_file_is_separate_from_offset_file(isolated_storage):
    assert lst.lock_file_path().name == ".telegram_approval_listener.lock"
    assert lst.lock_file_path() != lst.offset_file()


def test_single_instance_lock_blocks_a_second_acquire_on_the_same_path(isolated_storage):
    first = lst._SingleInstanceLock(lst.lock_file_path())
    first.acquire()
    try:
        second = lst._SingleInstanceLock(lst.lock_file_path())
        with pytest.raises(lst.ListenerAlreadyRunningError, match="already holds the lock"):
            second.acquire()
    finally:
        first.release()


def test_single_instance_lock_can_be_reacquired_after_release(isolated_storage):
    first = lst._SingleInstanceLock(lst.lock_file_path())
    first.acquire()
    first.release()

    second = lst._SingleInstanceLock(lst.lock_file_path())
    second.acquire()  # must not raise - the first holder released cleanly
    second.release()


def test_poll_forever_raises_when_another_instance_already_holds_the_lock(isolated_storage, monkeypatch, caplog):
    holder = lst._SingleInstanceLock(lst.lock_file_path())
    holder.acquire()
    try:
        called = {"n": 0}

        def _must_not_be_called(offset, timeout=30):
            called["n"] += 1
            return []

        monkeypatch.setattr(lst, "get_updates", _must_not_be_called)

        import logging
        with caplog.at_level(logging.INFO, logger="ycf.telegram_approval_listener"):
            with pytest.raises(lst.ListenerAlreadyRunningError):
                lst.poll_forever(iterations=1, _sleep=lambda s: None)

        # Never even got as far as the normal startup log lines or a
        # getUpdates call - the lock is checked before anything else.
        assert called["n"] == 0
        assert "listener started" not in caplog.text
    finally:
        holder.release()


def test_poll_forever_releases_the_lock_on_normal_completion(isolated_storage, monkeypatch):
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([[]]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    # A second, independent call must succeed immediately - proves the
    # first call's lock was actually released, not leaked. This is exactly
    # what a real "listener restarted" cycle depends on, and mirrors
    # test_offset_persists_and_is_not_reprocessed_after_restart's own
    # two-calls-in-a-row pattern.
    lst.poll_forever(iterations=1, _sleep=lambda s: None)


def test_poll_forever_releases_the_lock_after_keyboard_interrupt(isolated_storage, monkeypatch):
    def raise_interrupt(offset, timeout=30):
        raise KeyboardInterrupt()

    monkeypatch.setattr(lst, "get_updates", raise_interrupt)
    lst.poll_forever(iterations=None, _sleep=lambda s: None)  # returns, doesn't raise

    # The crashed/interrupted process must not leave the system locked -
    # a fresh acquire right after must succeed.
    fresh = lst._SingleInstanceLock(lst.lock_file_path())
    fresh.acquire()
    fresh.release()


def test_main_reports_a_clear_error_and_nonzero_exit_when_lock_already_held(isolated_storage, capsys):
    holder = lst._SingleInstanceLock(lst.lock_file_path())
    holder.acquire()
    try:
        exit_code = lst.main([])  # real poll_forever() - no monkeypatch needed, lock check happens first
        err = capsys.readouterr().err
        assert exit_code == 1
        assert "FAILED" in err
        assert "already holds the lock" in err
        assert "Traceback" not in err  # a clean, expected error - not an unhandled crash
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# 1 - callback update parsing / process_update()
# ---------------------------------------------------------------------------

def test_process_update_dispatches_approval_callback(isolated_storage, monkeypatch):
    calls = []
    monkeypatch.setattr(tap, "handle_approval_callback", lambda cbq: calls.append(cbq))
    update = _update(1, "approve", "some-slug", message_id=5)

    handled = lst.process_update(update)

    assert handled is True
    assert calls == [update["callback_query"]]


# ---------------------------------------------------------------------------
# 4 - invalid callback (non-approval namespace is never dispatched)
# ---------------------------------------------------------------------------

def test_process_update_ignores_non_approval_callback_data(isolated_storage, monkeypatch):
    calls = []
    monkeypatch.setattr(tap, "handle_approval_callback", lambda cbq: calls.append(cbq))
    update = {"update_id": 2, "callback_query": {"id": "c", "data": "approve:42", "message": {"message_id": 1, "chat": {"id": 1}}}}

    handled = lst.process_update(update)

    assert handled is False
    assert calls == []


# ---------------------------------------------------------------------------
# Daily Topic Discovery / Telegram Topic Selector dispatch (Phase A/B) -
# process_update() extended with a new "topic_select:" callback namespace
# and a "/topics" text command, alongside (not replacing) the above.
# ---------------------------------------------------------------------------

def test_process_update_dispatches_topic_select_callback(isolated_storage, monkeypatch):
    from youtube import topic_discovery

    calls = []
    monkeypatch.setattr(topic_discovery, "handle_topic_select_callback", lambda cbq: calls.append(cbq))
    update = {
        "update_id": 10,
        "callback_query": {"id": "c", "data": "topic_select:5", "message": {"message_id": 1, "chat": {"id": 1}}},
    }

    handled = lst.process_update(update)

    assert handled is True
    assert calls == [update["callback_query"]]


def test_process_update_dispatches_slash_topics_command(isolated_storage, monkeypatch):
    from youtube import topic_discovery

    calls = []
    monkeypatch.setattr(topic_discovery, "handle_topics_command", lambda msg: calls.append(msg))
    update = {"update_id": 11, "message": {"text": "/topics", "chat": {"id": 999}}}

    handled = lst.process_update(update)

    assert handled is True
    assert calls == [update["message"]]


def test_process_update_ignores_non_topics_message_text(isolated_storage, monkeypatch):
    from youtube import topic_discovery

    calls = []
    monkeypatch.setattr(topic_discovery, "handle_topics_command", lambda msg: calls.append(msg))
    for update in (
        {"update_id": 12, "message": {"text": "/start", "chat": {"id": 1}}},
        {"update_id": 13, "message": {"text": "hello", "chat": {"id": 1}}},
        {"update_id": 14, "message": {"chat": {"id": 1}}},  # no "text" key at all
    ):
        assert lst.process_update(update) is False
    assert calls == []


def test_get_updates_now_allows_message_updates_alongside_callback_query(isolated_storage, monkeypatch):
    """The "/topics" command can only reach this listener if Telegram is
    told to deliver "message" updates too - assert the getUpdates request
    itself now asks for both, not just "callback_query"."""
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": []}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None, **kwargs):
            captured["params"] = params
            return _FakeResponse()

    monkeypatch.setattr(lst.httpx, "Client", _FakeClient)

    lst.get_updates(0)

    # allowed_updates must be ONE JSON-serialized array string (Telegram
    # Bot API's documented format for this parameter) - never a raw Python
    # list handed to httpx, which encodes a list as the same query key
    # repeated once per item instead (see the two regression tests below,
    # which reproduce the exact malformed URL that caused a real 409
    # Conflict in production).
    allowed_updates_param = captured["params"]["allowed_updates"]
    assert isinstance(allowed_updates_param, str)
    assert json.loads(allowed_updates_param) == ["callback_query", "message"]


def test_get_updates_sends_allowed_updates_as_a_single_query_parameter_not_repeated(isolated_storage, monkeypatch):
    """Reproduces, at the actual httpx.Request/URL level, the exact
    production incident: a real Telegram getUpdates call came back 409
    Conflict with the request URL ending
    "...&allowed_updates=callback_query&allowed_updates=message" - two
    repeated query keys, which is what httpx produces when `params=` is
    given a Python list instead of a single JSON string for this
    parameter. This asserts the actual outgoing request URL contains
    "allowed_updates=" exactly once, regardless of what Telegram itself
    does with a malformed request - the request this code sends must
    match Telegram's own documented format."""
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    captured_urls = []

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": []}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None, **kwargs):
            request = httpx.Request("GET", url, params=params)
            captured_urls.append(str(request.url))
            return _FakeResponse()

    monkeypatch.setattr(lst.httpx, "Client", _FakeClient)

    lst.get_updates(645237280)

    assert captured_urls[0].count("allowed_updates=") == 1
    assert "&allowed_updates=callback_query&allowed_updates=message" not in captured_urls[0]


# ---------------------------------------------------------------------------
# 8 - normal message ignored
# ---------------------------------------------------------------------------

def test_process_update_ignores_non_callback_updates(isolated_storage, monkeypatch):
    calls = []
    monkeypatch.setattr(tap, "handle_approval_callback", lambda cbq: calls.append(cbq))
    for update in (
        {"update_id": 3, "message": {"text": "/start"}},
        {"update_id": 4, "message": {"text": "test"}},
        {"update_id": 5},
    ):
        assert lst.process_update(update) is False
    assert calls == []


# ---------------------------------------------------------------------------
# 2 / 3 / 10 - approve dispatch / reject dispatch / callback answer
# ---------------------------------------------------------------------------

def test_poll_forever_dispatches_approve_end_to_end(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(10, "approve", "the-wow-signal-6", message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 1
    # callback answer + message edits both went through the real handler
    assert any(e[0] == "ANSWER" for e in fake_edit_text)


def test_poll_forever_dispatches_reject_end_to_end(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(20, "reject", "the-wow-signal-6", message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_REJECTED
    assert fake_youtube.video_calls == []


# ---------------------------------------------------------------------------
# 5 - unknown slug
# ---------------------------------------------------------------------------

def test_poll_forever_unknown_slug_is_safely_ignored(isolated_storage, monkeypatch):
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(30, "approve", "never-created-slug", message_id=1)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)  # must not raise

    assert answered == ["Unknown project"]


# ---------------------------------------------------------------------------
# 6 - duplicate approve (never a second YouTube upload)
# ---------------------------------------------------------------------------

def test_poll_forever_duplicate_approve_across_two_polls_never_double_uploads(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    # Telegram itself can redeliver the SAME update_id if our answer was
    # slow/lost - simulate that by sending the identical callback twice
    # across two separate getUpdates batches (two different update_ids,
    # as a real double-tap would also produce).
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(40, "approve", "the-wow-signal-6", message_id=message_id, cbq_id="cbq-a")],
        [_update(41, "approve", "the-wow-signal-6", message_id=message_id, cbq_id="cbq-b")],
    ]))

    lst.poll_forever(iterations=2, _sleep=lambda s: None)

    assert len(fake_youtube.video_calls) == 1
    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOADED


# ---------------------------------------------------------------------------
# 7 - update offset persists across a "restart" (fresh poll_forever call)
# ---------------------------------------------------------------------------

def test_offset_persists_and_is_not_reprocessed_after_restart(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)

    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(50, "approve", "the-wow-signal-6", message_id=message_id)],
    ]))
    lst.poll_forever(iterations=1, _sleep=lambda s: None)
    assert lst.load_offset() == 51

    # "Restart": a fresh poll_forever() call loads the offset from disk and
    # passes it to get_updates - a server that correctly honors `offset`
    # would never redeliver update_id=50 again, so simulate that server
    # behavior by asserting the offset we'd send matches, then feeding an
    # empty batch (nothing new).
    seen_offsets = []

    def fake_get_updates(offset, timeout=30):
        seen_offsets.append(offset)
        return []

    monkeypatch.setattr(lst, "get_updates", fake_get_updates)
    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    assert seen_offsets == [51]
    assert len(fake_youtube.video_calls) == 1  # still only the one upload from before


# ---------------------------------------------------------------------------
# 9 - Telegram API error handling
# ---------------------------------------------------------------------------

def test_poll_forever_survives_get_updates_error_and_keeps_polling(isolated_storage, monkeypatch):
    call_count = {"n": 0}

    def flaky_get_updates(offset, timeout=30):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("[TELEGRAM] getUpdates failed: simulated network error")
        return []

    monkeypatch.setattr(lst, "get_updates", flaky_get_updates)
    slept = []

    lst.poll_forever(iterations=2, _sleep=lambda s: slept.append(s))  # must not raise

    assert call_count["n"] == 2
    assert slept == [2.0]


def test_get_updates_error_never_leaks_telegram_token(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:SECRET-TOKEN-VALUE")

    import httpx as httpx_module

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kwargs):
            request = httpx_module.Request("GET", url)
            response = httpx_module.Response(401, request=request, text="Unauthorized")
            raise httpx_module.HTTPStatusError("401", request=request, response=response)

    monkeypatch.setattr(lst.httpx, "Client", _FakeClient)

    with pytest.raises(RuntimeError) as exc_info:
        lst.get_updates(0)

    assert "SECRET-TOKEN-VALUE" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 11 - graceful shutdown
# ---------------------------------------------------------------------------

def test_poll_forever_handles_keyboard_interrupt_gracefully(isolated_storage, monkeypatch, caplog):
    def raise_interrupt(offset, timeout=30):
        raise KeyboardInterrupt()

    monkeypatch.setattr(lst, "get_updates", raise_interrupt)

    import logging
    with caplog.at_level(logging.INFO, logger="ycf.telegram_approval_listener"):
        lst.poll_forever(iterations=None, _sleep=lambda s: None)  # must return, not raise

    assert "listener stopped" in caplog.text
    assert "listener started" in caplog.text


def test_keyboard_interrupt_during_sleep_also_stops_cleanly(isolated_storage, monkeypatch, caplog):
    def failing_get_updates(offset, timeout=30):
        raise RuntimeError("[TELEGRAM] getUpdates failed: down")

    def interrupting_sleep(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(lst, "get_updates", failing_get_updates)

    import logging
    with caplog.at_level(logging.INFO, logger="ycf.telegram_approval_listener"):
        lst.poll_forever(iterations=None, _sleep=interrupting_sleep)

    assert "listener stopped" in caplog.text


# ---------------------------------------------------------------------------
# 12 - listener dispatches the EXISTING handler, does not reimplement it
# ---------------------------------------------------------------------------

def test_listener_module_does_not_import_storage_telegram_approvals_directly():
    """The listener must never touch the approvals table itself - all
    state transitions go exclusively through youtube.telegram_approval's
    already-tested functions."""
    source = open(lst.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "storage.telegram_approvals" not in imported
    assert "storage import telegram_approvals" not in source


def test_listener_module_calls_no_state_mutating_function_of_its_own():
    """Static proof there is no second/parallel state-machine
    implementation in this file - only orchestration (get_updates/
    process_update/poll_forever/main)."""
    source = open(lst.__file__, encoding="utf-8").read()
    for forbidden in ("ALLOWED_TRANSITIONS", "_transition(", "def approve(", "def reject(", "upload_video(", "upload_thumbnail("):
        assert forbidden not in source, f"listener must not reimplement approval logic: found {forbidden!r}"


def test_listener_imports_handle_approval_callback_from_telegram_approval():
    source = open(lst.__file__, encoding="utf-8").read()
    assert "from youtube import telegram_approval" in source
    assert "tap.handle_approval_callback" in source


def test_process_update_calls_the_real_unmodified_handle_approval_callback(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube):
    """End-to-end proof (no monkeypatching of handle_approval_callback
    itself) that dispatch reaches the real function and it behaves
    exactly as its own dedicated test suite (tests/test_telegram_approval.py)
    already verifies."""
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    update = _update(60, "approve", "the-wow-signal-6", message_id=message_id)

    handled = lst.process_update(update)

    assert handled is True
    assert tap.ta.get_approval(result["approval_id"])["status"] == tap.STATUS_UPLOADED


# ---------------------------------------------------------------------------
# 2026-09 follow-up incident: a production report said a real button press
# on approval_id=3 (sent via `--resend-approval-id 3`, after the
# BUTTON_DATA_INVALID fix) came back "unknown identifier=3" even though
# that approval genuinely existed. Direct review of
# storage/telegram_approvals.py's schema (id INTEGER PRIMARY KEY, a plain
# "WHERE id = ?" SELECT) and youtube/telegram_approval.py's parsing
# (int(identifier) -> ta.get_approval()) found no defect - the tests below
# dispatch through the REAL production keyboard-generation functions
# (build_keyboard_by_id() / resend_existing_approval(), not a hand-typed
# guess) and confirm the identifier resolves correctly end-to-end through
# this actual listener, in a single shared database. The most likely
# real-world explanation of the report is therefore a STORAGE_DIR/DB_PATH
# mismatch between the process that ran --resend-approval-id and the
# process running this listener (each resolves them independently from its
# own environment - see core/config.py) - see the new diagnostic log lines
# added to handle_approval_callback() and poll_forever() for that case.
# ---------------------------------------------------------------------------

def _id_update(update_id, action, approval_id, chat_id=999, message_id=None, cbq_id="cbq-id"):
    """Hand-built ID-based callback_data, for the "well-formed but
    genuinely unknown numeric id" negative case below."""
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cbq_id,
            "data": f"approval:{action}:{approval_id}",
            "message": {"message_id": message_id, "chat": {"id": chat_id}},
        },
    }


def _real_id_update(update_id, action, approval_id, chat_id=999, message_id=None, cbq_id="cbq-id"):
    """Builds callback_data via the ACTUAL production keyboard-generation
    function (tap.build_keyboard_by_id) instead of assuming a string
    format - if that function's format ever changes, this test changes
    with it, unlike the module-level _update() helper above (still
    slug-based, so it only ever exercises the legacy fallback path - not
    what a real send transmits today)."""
    keyboard = tap.build_keyboard_by_id(approval_id)
    button = keyboard["inline_keyboard"][0][0 if action == "approve" else 1]
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cbq_id,
            "data": button["callback_data"],
            "message": {"message_id": message_id, "chat": {"id": chat_id}},
        },
    }


def test_ta_get_approval_looks_up_an_existing_approval_by_its_real_id(approval_env, isolated_storage, fake_telegram_send):
    """Direct storage-layer lookup, mirroring exactly what
    handle_approval_callback() does for a numeric identifier - isolates
    the lookup itself (proven correct here) from any process/environment
    DB-path question."""
    result, _ = _send(approval_env, isolated_storage, fake_telegram_send)

    row = tap.ta.get_approval(result["approval_id"])

    assert row is not None
    assert row["id"] == result["approval_id"]
    assert row["slug"] == "the-wow-signal-6"


def test_poll_forever_dispatches_id_based_approve_end_to_end(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    """Reproduces the reported production scenario end-to-end: the real
    ID-based keyboard a send actually transmits, approved through the real
    listener, correctly resolves back to the SAME approval_id and drives
    the state machine through to upload."""
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_real_id_update(90, "approve", approval_id, message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 1


def test_poll_forever_dispatches_id_based_reject_end_to_end(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    approval_id = result["approval_id"]
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_real_id_update(91, "reject", approval_id, message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_REJECTED
    assert fake_youtube.video_calls == []


def test_poll_forever_unknown_numeric_identifier_is_safely_rejected(isolated_storage, monkeypatch):
    """Companion to test_poll_forever_unknown_slug_is_safely_ignored above -
    a well-formed but nonexistent NUMERIC identifier (the new ID-based
    format) must be rejected exactly as cleanly as an unknown slug, never
    raising and never accidentally falling through to the legacy slug
    lookup (int() succeeds first, so ta.get_approval() alone decides)."""
    answered = []
    monkeypatch.setattr(tap.tb, "_answer_callback", lambda cbq_id, text="": answered.append(text))
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_id_update(92, "approve", 999999, message_id=1)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)  # must not raise

    assert answered == ["Unknown project"]


def test_resend_approval_dispatch_through_listener_never_creates_new_approval_or_video(
    approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch,
):
    """Full reproduction of the exact reported incident: an approval stuck
    in draft (exactly how the real approval_id=3 was left after the
    original sendPhoto HTTP 400), repaired via resend_existing_approval()
    (never ta.create_approval(), never a pipeline/video run), then
    approved through the real listener using the ID-based keyboard resend
    actually transmits - and never creates a second approval row or a new
    video/project at any point in the flow."""
    from storage import db as _db

    _full_fixture(approval_env, isolated_storage, "the-wow-signal-6")
    approval_id = tap.ta.create_approval("the-wow-signal-6", "Some Title", status=tap.STATUS_DRAFT)
    before_approvals = len(tap.ta.list_approvals())
    before_videos = len(_db.list_videos())

    resend_result = tap.resend_existing_approval(approval_id)
    message_id = fake_telegram_send["photos"][-1]["message_id"]

    assert resend_result["approval_id"] == approval_id
    assert len(tap.ta.list_approvals()) == before_approvals  # resend created no new approval row
    assert len(_db.list_videos()) == before_videos  # resend created no new video/project

    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_real_id_update(93, "approve", approval_id, message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_UPLOADED
    assert len(fake_youtube.video_calls) == 1
    assert len(tap.ta.list_approvals()) == before_approvals  # still no new approval row after approve
    assert len(_db.list_videos()) == before_videos  # still no new video/project after upload trigger


# ---------------------------------------------------------------------------
# 13 - YouTube upload remains private
# ---------------------------------------------------------------------------

def test_listener_dispatched_upload_is_always_private(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, fake_youtube, monkeypatch):
    monkeypatch.setattr(settings, "YOUTUBE_UPLOAD_PRIVACY", "public")
    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(70, "approve", "the-wow-signal-6", message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)

    _, metadata, _ = fake_youtube.video_calls[0]
    assert metadata["privacy_status"] == "private"


# ---------------------------------------------------------------------------
# 14 - upload failure -> upload_failed
# ---------------------------------------------------------------------------

def test_listener_dispatched_upload_failure_marks_upload_failed(approval_env, isolated_storage, fake_telegram_send, fake_edit_text, monkeypatch):
    from youtube import uploader
    failing = _FakeYouTube(fail=True)
    monkeypatch.setattr(uploader, "upload_video", failing.upload_video)
    monkeypatch.setattr(uploader, "upload_thumbnail", failing.upload_thumbnail)

    result, message_id = _send(approval_env, isolated_storage, fake_telegram_send)
    monkeypatch.setattr(lst, "get_updates", _batched_get_updates([
        [_update(80, "approve", "the-wow-signal-6", message_id=message_id)],
    ]))

    lst.poll_forever(iterations=1, _sleep=lambda s: None)  # must not raise/crash the listener

    row = tap.ta.get_approval(result["approval_id"])
    assert row["status"] == tap.STATUS_UPLOAD_FAILED
    assert "simulated upload failure" in row["upload_error"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_invokes_poll_forever_with_parsed_poll_interval(isolated_storage, monkeypatch):
    captured = {}

    def fake_poll_forever(poll_interval=2.0):
        captured["poll_interval"] = poll_interval

    monkeypatch.setattr(lst, "poll_forever", fake_poll_forever)
    exit_code = lst.main(["--poll-interval", "5"])

    assert exit_code == 0
    assert captured["poll_interval"] == 5.0


def test_cli_module_never_imports_llm_client_researcher_or_tts():
    source = open(lst.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"llm_client", "researcher", "tts"}
    assert not (imported & banned), f"unexpected import(s): {imported & banned}"
