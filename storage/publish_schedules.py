"""
Persistence for Smart Publish Scheduler V1 (youtube/smart_publish.py's
PublishDecision, recorded per approval).

Mirrors storage/telegram_approvals.py's own style and reasoning exactly:
plain sqlite3, no ORM, one small SCHEMA string, a `connect()`
contextmanager, an idempotent COLUMN_MIGRATIONS list - lives in its own
table (`publish_schedules`) rather than editing storage/db.py or
storage/telegram_approvals.py, so this is purely additive. Uses the SAME
sqlite file (settings.DB_PATH) - no new DB file/config introduced.

IDEMPOTENCY (spec Faz 7/8, non-negotiable): "approval_id=3 icin scheduler
iki kere calistirilirsa: 1 schedule olmali, 2 schedule olmamali."
create_schedule() below enforces this itself (check-then-reuse, backed by a
UNIQUE index on approval_id as a defense-in-depth against a race) - callers
never need their own duplicate-guard logic, and calling it any number of
times for the same approval_id always returns the SAME row's id without
inserting a second one.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.config import settings

# Status lifecycle (spec Faz 7): a row starts "planned" the moment a publish
# decision is computed (before any YouTube call), moves to "uploaded" once
# the YouTube upload API call itself succeeds, "scheduled" once that upload
# carried a real publishAt (YouTube itself flips it to public later - V1 has
# no polling mechanism to detect that transition, so "published" is a
# status this schema supports for a future poller but nothing here sets it
# yet), "failed" if the upload errors, or "cancelled" for a future manual
# cancel path (also not yet driven by any code here).
STATUS_PLANNED = "planned"
STATUS_UPLOADED = "uploaded"
STATUS_SCHEDULED = "scheduled"
STATUS_PUBLISHED = "published"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

SCHEMA = """
CREATE TABLE IF NOT EXISTS publish_schedules (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            INTEGER,
    approval_id         INTEGER NOT NULL,
    scheduled_at        TEXT NOT NULL,
    timezone            TEXT NOT NULL,
    status              TEXT NOT NULL,
    source              TEXT NOT NULL,
    confidence          TEXT NOT NULL,
    reason              TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    youtube_video_id    TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_publish_schedules_approval_id ON publish_schedules(approval_id);
CREATE INDEX IF NOT EXISTS idx_publish_schedules_video_id ON publish_schedules(video_id);
CREATE INDEX IF NOT EXISTS idx_publish_schedules_status ON publish_schedules(status);
"""

# Columns added after this table's initial release - see
# storage/telegram_approvals.py's own COLUMN_MIGRATIONS for why this
# pattern (append-only, idempotent ALTER TABLE) is required: it must add
# columns to a real production DB's existing table without disturbing any
# row already in it (e.g. a real approval_id=3/video_id=5 row). Empty for
# this table's initial release - present so a future column never needs a
# schema rewrite, only an appended entry here.
COLUMN_MIGRATIONS: list[tuple[str, str]] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(publish_schedules)").fetchall()}
    for column_name, column_type in COLUMN_MIGRATIONS:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE publish_schedules ADD COLUMN {column_name} {column_type}")


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


def get_schedule_by_approval_id(approval_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM publish_schedules WHERE approval_id = ?", (approval_id,)
        ).fetchone()


def create_schedule(
    *,
    approval_id: int,
    scheduled_at: str,
    timezone: str,
    source: str,
    confidence: str,
    reason: str | None = None,
    video_id: int | None = None,
    status: str = STATUS_PLANNED,
) -> int:
    """Idempotent by approval_id: if a schedule already exists for this
    approval, its existing id is returned UNCHANGED (no new row, no field
    update) - calling this any number of times for the same approval_id
    never produces a second row. The UNIQUE index on approval_id is a
    second, DB-level guarantee against a race between the check and the
    insert (two near-simultaneous callers): if both reach the INSERT, the
    loser's IntegrityError is caught and it simply re-reads and returns the
    winner's row instead of raising."""
    existing = get_schedule_by_approval_id(approval_id)
    if existing is not None:
        return existing["id"]

    with connect() as conn:
        try:
            cur = conn.execute(
                """INSERT INTO publish_schedules
                   (video_id, approval_id, scheduled_at, timezone, status, source, confidence,
                    reason, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (video_id, approval_id, scheduled_at, timezone, status, source, confidence,
                 reason, now(), now()),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            conn.rollback()
            row = conn.execute(
                "SELECT * FROM publish_schedules WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise
            return row["id"]


def get_schedule(schedule_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM publish_schedules WHERE id = ?", (schedule_id,)
        ).fetchone()


def get_schedule_by_video_id(video_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM publish_schedules WHERE video_id = ? ORDER BY id DESC LIMIT 1", (video_id,)
        ).fetchone()


def update_fields(schedule_id: int, **fields) -> None:
    """Generic column update (always bumps updated_at) - mirrors
    storage/telegram_approvals.py's update_fields() exactly."""
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = now()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [schedule_id]
    with connect() as conn:
        conn.execute(f"UPDATE publish_schedules SET {columns} WHERE id = ?", values)


def list_schedules(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM publish_schedules ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
