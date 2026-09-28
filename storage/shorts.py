"""
Persistence for Shorts Engine V1 (AI FOUND THIS): one row per Short,
child of a parent Long `videos` row (storage/db.py, unchanged).

Mirrors storage/publish_schedules.py's / storage/telegram_approvals.py's
own style exactly: plain sqlite3, no ORM, one small SCHEMA string, a
`connect()` contextmanager, an idempotent COLUMN_MIGRATIONS list - lives in
its own table (`shorts`) rather than editing storage/db.py, so this is
purely additive. Uses the SAME sqlite file (settings.DB_PATH) - no new DB
file/config introduced.

RELATIONSHIP (spec: "DB must allow knowing this Short was generated from
this Long. Do not duplicate the entire Long record."): `parent_video_id`
is the Long's `videos.id` (storage/db.py), `short_number` is 1/2/3 -
UNIQUE(parent_video_id, short_number) both prevents a duplicate Short slot
and makes create_short() idempotent across a resumed/retried production
run (mirrors publish_schedules.create_schedule()'s own idempotent-by-key
pattern).

STATUS LIFECYCLE (spec: "clear statuses... follow existing status
conventions where available"): mirrors telegram_approvals.py's own
draft-like lifecycle but is intentionally its OWN, simpler, independently
re-driveable state machine - never gated by core/state.py's PipelineState
(that enum is a strict linear order for the Long's own `videos.state`
only) and never gated by telegram_approvals.ALLOWED_TRANSITIONS (whose
mark_uploading() only ever succeeds ONCE for the Long, which would block
retrying a partially-failed Shorts batch - see
youtube/telegram_approval.py's process_shorts_for_approval()):

    building -> ready -> approved -> uploading -> uploaded
             \\-> failed (QC)     \\-> rejected           \\-> upload_failed

A Short can move from failed/upload_failed back toward uploading on a
retry (process_shorts_for_approval() is safely re-callable any number of
times - see that function's own docstring for the exact idempotency
argument), which is why this is deliberately NOT gated by a strict
ALLOWED_TRANSITIONS table the way telegram_approvals.py's Long-side
approval is.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.config import settings

# --- status lifecycle -------------------------------------------------------
STATUS_BUILDING = "building"        # production in progress (planning/visual/tts/assembly/qc)
STATUS_READY = "ready"              # produced, QC passed, included in the sent Telegram package
STATUS_FAILED = "failed"            # production or QC failed - never sent as approve-ready
STATUS_APPROVED = "approved"        # package APPROVE received, eligible for upload
STATUS_UPLOADING = "uploading"
STATUS_UPLOADED = "uploaded"
STATUS_UPLOAD_FAILED = "upload_failed"
STATUS_REJECTED = "rejected"        # package REJECT received

# Statuses process_shorts_for_approval() (youtube/telegram_approval.py) will
# (re)attempt an upload for - i.e. every status except the one truly-done
# terminal state. FAILED/UPLOAD_FAILED are deliberately included so a
# transient failure (FAL/TTS/YouTube hiccup) is retryable simply by calling
# that function again, with no separate "retry" code path.
UPLOAD_ELIGIBLE_STATUSES = {STATUS_READY, STATUS_APPROVED, STATUS_FAILED, STATUS_UPLOAD_FAILED}

SCHEMA = """
CREATE TABLE IF NOT EXISTS shorts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_video_id     INTEGER NOT NULL,
    short_number        INTEGER NOT NULL,
    slug                TEXT,
    title               TEXT,
    angle               TEXT,
    script              TEXT,
    narration_path      TEXT,
    video_path          TEXT,
    thumbnail_path      TEXT,
    duration_seconds    REAL,
    status              TEXT NOT NULL,
    approval_id         INTEGER,
    channel_key         TEXT,
    youtube_video_id    TEXT,
    youtube_url         TEXT,
    scheduled_at        TEXT,
    timezone            TEXT,
    publish_source      TEXT,
    publish_confidence  TEXT,
    publish_reason      TEXT,
    qc_passed           INTEGER,
    qc_errors           TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_shorts_parent_number ON shorts(parent_video_id, short_number);
CREATE INDEX IF NOT EXISTS idx_shorts_approval_id ON shorts(approval_id);
CREATE INDEX IF NOT EXISTS idx_shorts_status ON shorts(status);
"""

# Append-only, idempotent ALTER TABLE list (see storage/publish_schedules.py's
# own COLUMN_MIGRATIONS for why this pattern is required against a real
# production DB). Empty for this table's initial release.
COLUMN_MIGRATIONS: list[tuple[str, str]] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(shorts)").fetchall()}
    for column_name, column_type in COLUMN_MIGRATIONS:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE shorts ADD COLUMN {column_name} {column_type}")


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


def get_short_by_parent_and_number(parent_video_id: int, short_number: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM shorts WHERE parent_video_id = ? AND short_number = ?",
            (parent_video_id, short_number),
        ).fetchone()


def create_short(
    *,
    parent_video_id: int,
    short_number: int,
    status: str = STATUS_BUILDING,
    **fields,
) -> int:
    """Idempotent by (parent_video_id, short_number): if a row already
    exists for this slot, its id is returned UNCHANGED and no field is
    touched here (callers update fields explicitly via update_fields()) -
    calling this any number of times for the same slot (e.g. a resumed
    production run) never produces a second row. The UNIQUE index above is
    a second, DB-level guarantee against a race, mirroring
    storage/publish_schedules.py's create_schedule() exactly."""
    existing = get_short_by_parent_and_number(parent_video_id, short_number)
    if existing is not None:
        return existing["id"]

    columns = ["parent_video_id", "short_number", "status", "created_at", "updated_at"]
    values = [parent_video_id, short_number, status, now(), now()]
    for k, v in fields.items():
        columns.append(k)
        values.append(v)
    placeholders = ", ".join("?" for _ in columns)
    with connect() as conn:
        try:
            cur = conn.execute(
                f"INSERT INTO shorts ({', '.join(columns)}) VALUES ({placeholders})", values,
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            conn.rollback()
            row = conn.execute(
                "SELECT * FROM shorts WHERE parent_video_id = ? AND short_number = ?",
                (parent_video_id, short_number),
            ).fetchone()
            if row is None:
                raise
            return row["id"]


def get_short(short_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM shorts WHERE id = ?", (short_id,)).fetchone()


def list_shorts_for_parent(parent_video_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM shorts WHERE parent_video_id = ? ORDER BY short_number ASC",
            (parent_video_id,),
        ).fetchall()


def list_shorts_for_approval(approval_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM shorts WHERE approval_id = ? ORDER BY short_number ASC",
            (approval_id,),
        ).fetchall()


def update_fields(short_id: int, **fields) -> None:
    """Generic column update (always bumps updated_at) - mirrors
    storage/telegram_approvals.py's / storage/publish_schedules.py's own
    update_fields() exactly."""
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = now()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [short_id]
    with connect() as conn:
        conn.execute(f"UPDATE shorts SET {columns} WHERE id = ?", values)


def set_approval_id_for_parent(parent_video_id: int, approval_id: int) -> None:
    """Attaches every Short currently belonging to this Long to the ONE
    package approval row just created (spec CRITICAL: "ONLY ONE approval
    for the whole content package") - never a per-Short approval id."""
    with connect() as conn:
        conn.execute(
            "UPDATE shorts SET approval_id = ?, updated_at = ? WHERE parent_video_id = ?",
            (approval_id, now(), parent_video_id),
        )


def list_shorts(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM shorts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
