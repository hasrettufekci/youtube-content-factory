"""
Telegram Approval Callback Listener (spec: "TELEGRAM APPROVAL CALLBACK
LISTENER FIX").

The gap this fixes: youtube/telegram_approval_demo.py sends the approval
message and exits - nothing was left running to receive the APPROVE/REJECT
button press. This module is that missing long-running process. It does
NOT reimplement approval logic - its only job is:

    getUpdates (long polling, callback_query only)
        -> callback_query present?
        -> youtube.telegram_approval.handle_approval_callback()  (UNCHANGED)

handle_approval_callback() already does everything the spec asks for here
(callback data parsing, slug lookup, chat_id/message_id validation, state
transition, YouTube upload trigger, answerCallbackQuery, message
editing) - see youtube/telegram_approval.py, not modified by this file
except for the removal of its own dead/untested poll_forever(), which this
module supersedes. This file adds no new callback-handling architecture.

REUSE: `tb._url()` / `tb._redact_token()` (telegram_bot.py) for the raw
Telegram HTTP call - no new HTTP plumbing beyond the getUpdates GET itself,
which telegram_bot.py's own poll_forever() also does the same way.

POLLING: uses Telegram's own long-poll `timeout` parameter (default 30s)
so this never busy-loops - each getUpdates call blocks server-side until
either an update arrives or the timeout elapses. `allowed_updates=
["callback_query", "message"]` is passed so Telegram itself filters out
every OTHER update type (edited messages, channel posts, etc.) before they
ever reach this process. "message" was added for Daily Topic Discovery's
"/topics" manual trigger (spec: "YOUTUBE CONTENT FACTORY - DAILY TOPIC
DISCOVERY + TELEGRAM TOPIC SELECTOR") - process_update() below still only
ever acts on a callback_query, or a message whose text is exactly
"/topics"; every other message (a stray /start, small talk, telegram_bot.py's
own old keyboard's callbacks) is safely ignored, same as before. As a
second layer, a callback_query is only ever dispatched if its callback_data
starts with "approval:" (Telegram Approval V1) or "topic_select:" (Daily
Topic Discovery's Telegram Topic Selector) - see youtube/topic_discovery.py
- so a stray callback from telegram_bot.py's own OLD "approve:<video_id>"
keyboard (if that bot is ever polled by the same token at the same time)
is safely ignored here rather than misrouted into either newer handler.

OFFSET / DUPLICATE PROTECTION: the update offset is persisted to disk
(its own file, separate from telegram_bot.OFFSET_FILE) exactly like
telegram_bot.py's own poll_forever() does, so a restarted listener resumes
from the next unprocessed update rather than replaying already-handled
ones. This is defense-in-depth only - the DB-backed approval state
machine's allowed-transitions table in youtube/telegram_approval.py is
what actually guarantees a second APPROVE can never trigger a second
YouTube upload, per spec section 4/7: "Mevcut DB state machine ikinci
koruma olarak kalmalı."

Usage:
    python -m youtube.telegram_approval_listener

SINGLE-INSTANCE LOCK (Telegram 409 Conflict fix): Telegram's Bot API allows
only one outstanding getUpdates long-poll connection per bot token at a
time - a second one gets rejected with 409 Conflict. poll_forever() below
acquires an OS-level advisory file lock (see _SingleInstanceLock) before
doing anything else, so a second listener process started against the
same storage root fails immediately with a clear
ListenerAlreadyRunningError instead of silently fighting the first one
over getUpdates. This guards against a second instance of THIS listener;
it cannot stop an entirely different program (e.g. telegram_bot.py
--listen) from also polling the same token - see README.md's now-updated
listener section for that.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

import telegram_bot as tb
from core.config import settings
from core.logging_setup import get_logger
from youtube import telegram_approval as tap

logger = get_logger("telegram_approval_listener")

OFFSET_FILE_NAME = ".telegram_approval_listener_offset"
LOCK_FILE_NAME = ".telegram_approval_listener.lock"
GETUPDATES_TIMEOUT_SECONDS = 30  # Telegram-side long-poll wait
HTTP_CLIENT_TIMEOUT_SECONDS = GETUPDATES_TIMEOUT_SECONDS + 5  # headroom over the long-poll wait itself


class ListenerAlreadyRunningError(RuntimeError):
    """Raised by _SingleInstanceLock.acquire() when another
    telegram_approval_listener process already holds the lock - see
    poll_forever() below."""


def lock_file_path() -> Path:
    return settings.STORAGE_DIR / LOCK_FILE_NAME


class _SingleInstanceLock:
    """OS-level advisory file lock (fcntl.flock on POSIX, msvcrt.locking on
    Windows) - deliberately NOT a PID file.

    A PID-file lock needs a "is this PID still alive, and is it actually
    still our listener" liveness check, and without one, a crashed process
    leaves a stale lock that blocks the system forever (exactly the
    failure mode to avoid here). An OS-level lock has no such state to go
    stale: it is held against this process's own open file handle, and the
    operating system itself releases it the instant that handle closes -
    on a normal return, sys.exit(), Ctrl+C, an uncaught exception, or the
    process being killed outright (task manager, crash, power loss). There
    is nothing to detect or clean up.

    acquire() raises ListenerAlreadyRunningError immediately (non-blocking)
    if another process already holds the lock - it never waits."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    raise ListenerAlreadyRunningError(
                        f"another telegram_approval_listener process already holds the lock at "
                        f"{self.path} - only one instance may poll a given Telegram bot token at a "
                        f"time (this is exactly what causes Telegram's own 409 Conflict error). "
                        f"Stop the other process first, then retry."
                    ) from None
            else:
                import fcntl
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise ListenerAlreadyRunningError(
                        f"another telegram_approval_listener process already holds the lock at "
                        f"{self.path} - only one instance may poll a given Telegram bot token at a "
                        f"time (this is exactly what causes Telegram's own 409 Conflict error). "
                        f"Stop the other process first, then retry."
                    ) from None
        except ListenerAlreadyRunningError:
            fh.close()
            logger.error(
                "[TELEGRAM] refusing to start: another approval listener process is already running "
                "(lock file: %s)", self.path,
            )
            raise
        # Record our own PID purely so a human inspecting the lock file can
        # see who holds it - never read back or relied on for locking logic
        # itself (the OS-level lock above is the actual guard).
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def offset_file() -> Path:
    return settings.STORAGE_DIR / OFFSET_FILE_NAME


def load_offset() -> int:
    path = offset_file()
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return 0


def save_offset(offset: int) -> None:
    offset_file().write_text(str(offset))


def get_updates(offset: int, timeout: int = GETUPDATES_TIMEOUT_SECONDS) -> list[dict]:
    """One getUpdates long-poll call. Raises RuntimeError (token-redacted,
    mirrors telegram_bot.send_text()'s own rule) on an HTTP-level failure -
    callers decide how to handle that (poll_forever() below logs and backs
    off rather than crashing the whole listener).

    allowed_updates includes "message" (added for Daily Topic Discovery's
    "/topics" manual trigger - see process_update() below) alongside the
    original "callback_query" - Telegram only ever sends update types
    listed here, so without "message" a "/topics" text would never reach
    this process at all. process_update() still only ever acts on a
    callback_query or a message whose text is exactly "/topics" - every
    other message type is received but silently ignored, same as before.

    allowed_updates MUST be sent as a single JSON-serialized array string
    (Telegram Bot API's own documented format for this parameter), never
    as a raw Python list handed straight to httpx's `params=` - httpx (like
    requests) encodes a list value as the SAME query key repeated once per
    item (e.g. `allowed_updates=callback_query&allowed_updates=message`),
    which is a malformed request Telegram does not recognize as the
    documented format. `json.dumps([...])` produces the one correct
    `allowed_updates=%5B%22callback_query%22%2C...%5D` parameter instead -
    see the 409 Conflict regression this caused, fixed here."""
    params = {"offset": offset, "timeout": timeout, "allowed_updates": json.dumps(["callback_query", "message"])}
    try:
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT_SECONDS) as client:
            resp = client.get(tb._url("getUpdates"), params=params)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"[TELEGRAM] getUpdates failed: {tb._redact_token(str(e))}") from None
    return resp.json().get("result", [])


def process_update(update: dict) -> bool:
    """Dispatch exactly one update. Returns True if it was handled (an
    "approval:" callback, a "topic_select:" callback, or a "/topics" text
    message), False if it was ignored (no callback_query/message, a
    callback_query outside both known namespaces - e.g. a stray /start
    message, or telegram_bot.py's own legacy keyboard - or a message whose
    text isn't "/topics"). Never raises: handle_approval_callback() and
    topic_discovery's own handlers already catch and safely report every
    error case themselves (spec section 13's "listener crash olduğunda
    mümkün olduğunca kontrollü şekilde logla" is honored by not needing to
    - neither callback handler ever lets an exception escape)."""
    callback_query = update.get("callback_query")
    if callback_query:
        data = str(callback_query.get("data", ""))
        if data.startswith("approval:"):
            tap.handle_approval_callback(callback_query)
            return True
        from youtube import topic_discovery
        if data.startswith(topic_discovery.CALLBACK_PREFIX):
            topic_discovery.handle_topic_select_callback(callback_query)
            return True
        logger.debug("[TELEGRAM] ignoring non-approval/non-topic-select callback_data=%r", data)
        return False

    message = update.get("message")
    if message and str(message.get("text", "")).strip() == "/topics":
        from youtube import topic_discovery
        topic_discovery.handle_topics_command(message)
        return True

    logger.debug("[TELEGRAM] ignoring non-callback update_id=%s", update.get("update_id"))
    return False


def poll_forever(poll_interval: float = 2.0, iterations: int | None = None, _sleep=time.sleep) -> None:
    """The long-running loop. `iterations` bounds the number of getUpdates
    calls made before returning - production code never passes it (None =
    forever); tests pass a small integer so the loop terminates on its
    own. `_sleep` is injectable for the same reason (tests never want a
    real sleep on the error-retry path).

    Acquires the single-instance lock (see _SingleInstanceLock) BEFORE any
    logging or network call - if another instance already holds it, this
    raises ListenerAlreadyRunningError immediately and nothing below ever
    runs (no "listener started" log, no getUpdates call, no offset load).
    The lock is released in a `finally` that wraps the entire loop, so it
    comes back exactly once per acquire regardless of how the loop below
    exits - normal completion, KeyboardInterrupt, or an unexpected
    exception - matching the same "always clean up" contract the offset
    file's own on-disk state already relies on."""
    lock = _SingleInstanceLock(lock_file_path())
    lock.acquire()
    try:
        offset = load_offset()
        logger.info("[TELEGRAM] approval listener started")
        # 2026-09 production incident: a genuinely-existing approval was
        # reported "unknown identifier" by handle_approval_callback() - see
        # that function's own comment for the full analysis. STORAGE_DIR/
        # DB_PATH are resolved independently by every process from its own
        # environment (core/config.py), so logging them here - once, at
        # startup, before any callback - makes it immediate to confirm this
        # listener is pointed at the SAME database as whatever created or
        # resent the approval it's about to receive a callback for.
        logger.info("[TELEGRAM] storage_dir=%s db_path=%s", settings.STORAGE_DIR, settings.DB_PATH)
        logger.info("[TELEGRAM] waiting for callbacks... (offset=%d)", offset)

        count = 0
        try:
            while iterations is None or count < iterations:
                count += 1
                try:
                    updates = get_updates(offset)
                except RuntimeError as e:
                    logger.warning(str(e))
                    _sleep(poll_interval)
                    continue

                for update in updates:
                    offset = update["update_id"] + 1
                    save_offset(offset)
                    try:
                        process_update(update)
                    except Exception:
                        # Defense in depth only - handle_approval_callback()
                        # already swallows and safely reports every error case
                        # it can reach. If something still escapes (a bug in
                        # dispatch itself), never let it kill the listener
                        # process outright (spec section 13).
                        logger.exception(
                            "[TELEGRAM] unexpected error processing update_id=%s - continuing",
                            update.get("update_id"),
                        )
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("[TELEGRAM] listener stopped")
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Long-poll Telegram for Approval V1 APPROVE/REJECT button presses and dispatch them "
                    "to the existing youtube.telegram_approval callback handler."
    )
    parser.add_argument("--poll-interval", type=float, default=2.0,
                         help="Seconds to wait before retrying after a getUpdates error (default: 2.0)")
    args = parser.parse_args(argv)

    try:
        poll_forever(poll_interval=args.poll_interval)
    except ListenerAlreadyRunningError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
