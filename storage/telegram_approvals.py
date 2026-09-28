"""
Persistence for Telegram Approval V1 (slug-based video+thumbnail approval,
separate from the legacy video_id-based `videos`/telegram_bot.py flow).

Deliberately mirrors storage/db.py's own style (plain sqlite3, no ORM, one
small SCHEMA string, a `connect()` contextmanager) but lives in its own
module with its own table (`telegram_approvals`) rather than editing
storage/db.py - this is purely additive: nothing in storage/db.py or its
existing `videos`/`topics`/`research_sources`/`pipeline_runs` tables is
touched, so the legacy pipeline/telegram_bot.py flow is unaffected. Uses
the SAME sqlite file (settings.DB_PATH) as storage/db.py - sqlite happily
hosts multiple independent tables in one file - so no new DB file/config
is introduced ("gereksiz DB karmaşası oluşturma").

State machine (see youtube/telegram_approval.py for the transition rules -
this module is intentionally a dumb CRUD layer, not where the state
machine logic lives, mirroring storage/db.py's own division of
responsibility with pipeline.py):

    draft -> ready_for_approval -> approved -> uploading -> uploaded
                                 \\-> rejected            \\-> upload_failed

The original release kept only the minimum fields the V1 spec asked for
(id, slug, title, status, telegram_chat_id, telegram_message_id,
created_at, updated_at, youtube_video_id, youtube_url, rejection_reason,
upload_error). COLUMN_MIGRATIONS below has since added thumbnail_status/
thumbnail_error, and - for the production-pipeline integration - the
approved artifact/record itself (video_path, thumbnail_path,
duration_label, scene_count, language, privacy, channel_key), always via
the same idempotent ALTER TABLE mechanism, never a schema rewrite.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS telegram_approvals (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                    TEXT NOT NULL,
    title                   TEXT NOT NULL,
    status                  TEXT NOT NULL,
    telegram_chat_id        TEXT,
    telegram_message_id     INTEGER,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    youtube_video_id        TEXT,
    youtube_url             TEXT,
    rejection_reason        TEXT,
    upload_error            TEXT
);

CREATE INDEX IF NOT EXISTS idx_telegram_approvals_slug ON telegram_approvals(slug);
CREATE INDEX IF NOT EXISTS idx_telegram_approvals_status ON telegram_approvals(status);
"""

# Columns added after the table's initial release. CREATE TABLE IF NOT
# EXISTS is a no-op on an already-existing table, so a brand new column
# needs an explicit, idempotent ALTER TABLE - this list is what
# _migrate_columns() below applies. Appending here (never editing/removing
# an existing entry) is how this table gains new columns going forward
# without ever touching a real production DB's existing rows (a real
# database may already hold rows, such as approval_id=1 from the pilot -
# a migration must add columns without disturbing them).
COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("thumbnail_status", "TEXT"),  # 'ok' | 'failed' | 'skipped' | NULL (not yet attempted)
    ("thumbnail_error", "TEXT"),
    # Production-pipeline <-> Telegram Approval V1 integration: the exact
    # artifact/record that was shown for approval, persisted reliably so
    # process_approval() reads these back at upload time instead of
    # recomputing them from any fixed path convention (see
    # youtube/telegram_approval.py's send_new_approval()/process_approval()).
    ("video_path", "TEXT"),
    ("thumbnail_path", "TEXT"),
    ("duration_label", "TEXT"),
    ("scene_count", "INTEGER"),
    ("language", "TEXT"),
    ("privacy", "TEXT"),
    # NULL = the default/legacy YouTube channel (same meaning channel_key=
    # None already has everywhere else in this codebase) - never a magic
    # string like "default".
    ("channel_key", "TEXT"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(telegram_approvals)").fetchall()}
    for column_name, column_type in COLUMN_MIGRATIONS:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE telegram_approvals ADD COLUMN {column_name} {column_type}")


def init_db(db_path: Path | None = None) -> None:
    path = db_path or settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA)
        _migrate_columns(conn)
        conn.commit()
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    init_db()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_approval(slug: str, title: str, status: str = "draft") -> int:
    """Insert a new approval row (status defaults to 'draft'). Returns the
    new row's id - this id is the durable "approval_id" referenced
    elsewhere (spec section 11's chat_id/message_id/approval_id/slug
    validation combination)."""
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO telegram_approvals
               (slug, title, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (slug, title, status, now(), now()),
        )
        return cur.lastrowid


def get_approval(approval_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM telegram_approvals WHERE id = ?", (approval_id,)
        ).fetchone()


def get_latest_approval_by_slug(slug: str) -> sqlite3.Row | None:
    """Most recent approval row for this slug (a slug may legitimately have
    more than one approval row over time - e.g. re-sent after a restart -
    callback validation always resolves against the latest one, and an
    older/stale message's buttons are rejected via the message_id check -
    see youtube/telegram_approval.py)."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM telegram_approvals WHERE slug = ? ORDER BY id DESC LIMIT 1",
            (slug,),
        ).fetchone()


def update_fields(approval_id: int, **fields) -> None:
    """Generic column update (always bumps updated_at). Only ever called
    from youtube/telegram_approval.py's state-machine functions - never
    accepts an arbitrary caller-supplied status transition without that
    layer's validation."""
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = now()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [approval_id]
    with connect() as conn:
        conn.execute(f"UPDATE telegram_approvals SET {columns} WHERE id = ?", values)


def list_approvals(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM telegram_approvals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
