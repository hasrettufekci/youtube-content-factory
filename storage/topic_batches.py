"""
Persistence for Daily Topic Discovery + Telegram Topic Selector (Phase A +
Phase B - "YOUTUBE CONTENT FACTORY - DAILY TOPIC DISCOVERY + TELEGRAM TOPIC
SELECTOR").

Deliberately mirrors storage/telegram_approvals.py's own pattern: plain
sqlite3, no ORM, its own SCHEMA string, its own `connect()` contextmanager,
living in its own module with its own tables (`topic_batches` /
`topic_candidates`) rather than editing storage/db.py - purely additive.
Nothing in storage/db.py or its existing `videos`/`topics`/
`research_sources`/`pipeline_runs` tables is touched, so topic_scout.py's
own simple candidate list (the `topics` table, used by
scheduler.run_once()) is completely unaffected. Uses the SAME sqlite file
(settings.DB_PATH) as storage/db.py - sqlite happily hosts multiple
independent tables in one file.

Why a NEW table instead of reusing `topics`: the existing `topics` table
has no concept of a daily "batch" of candidates presented together, no
why_now/trend_signal/sources/short_description fields, and no
discovered/presented/selected/rejected/expired lifecycle - it's a flat,
single-candidate list for manual `python -m youtube.topic_scout --topic`
entries. Reusing it would mean bolting an unrelated batch/selection state
machine onto a table another module already depends on for a simpler
purpose - a real duplication risk in the other direction. Two small,
purpose-built tables here are the minimal safe addition.

STATE MACHINE (see youtube/topic_discovery.py for the transition logic -
this module is intentionally a dumb CRUD + atomic-selection layer, not
where the business logic lives, mirroring storage/db.py's / storage/
telegram_approvals.py's own division of responsibility with their
respective callers):

    topic_batches.status:    presented -> selected
    topic_candidates.status: discovered -> presented -> selected
                                                      \\-> rejected
                                                      \\-> expired

`try_select_candidate()` below is the ONE atomic operation that makes a
double-click (or two near-simultaneous callbacks for the same candidate)
safe: a single `UPDATE ... WHERE status='presented'` can only ever affect
one row across any number of concurrent callers, so only the caller that
actually flipped rowcount 0->1 goes on to start production - see
youtube/topic_discovery.py's handle_topic_select_callback().
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_batches (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    status                  TEXT NOT NULL DEFAULT 'presented',
    triggered_by            TEXT,
    telegram_chat_id        TEXT,
    telegram_message_id     INTEGER,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_candidates (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id                INTEGER NOT NULL REFERENCES topic_batches(id),
    position                INTEGER NOT NULL,
    title                   TEXT NOT NULL,
    topic                   TEXT NOT NULL,
    normalized_topic        TEXT NOT NULL,
    short_description       TEXT,
    why_now                 TEXT,
    category                TEXT,
    trend_signal            TEXT,
    score                   REAL,
    score_json              TEXT,
    sources_json            TEXT,
    discovered_at           TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'discovered',
    video_id                INTEGER,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topic_candidates_batch ON topic_candidates(batch_id);
CREATE INDEX IF NOT EXISTS idx_topic_candidates_normalized_topic ON topic_candidates(normalized_topic);
"""

# Columns added after this table's initial release (Turkish Telegram
# Management UI patch). CREATE TABLE IF NOT EXISTS is a no-op on an
# already-existing table, so these need an explicit, idempotent ALTER
# TABLE - mirrors storage/telegram_approvals.py's own COLUMN_MIGRATIONS/
# _migrate_columns() pattern exactly. This is not a cosmetic choice: the
# Daily Topic Discovery + Telegram Topic Selector is already live in
# production (per the user's own report), so a real database may already
# have topic_batches/topic_candidates rows from before these columns
# existed - a migration must add them without disturbing those rows.
#
# These are DISPLAY-ONLY Turkish translations for the Telegram owner
# interface. The existing `title`/`topic`/`why_now`/`category`/
# `trend_signal` columns are UNCHANGED and remain English - they are what
# pipeline.produce_video() is called with (see youtube/topic_discovery.py's
# handle_topic_select_callback()). Never read these _tr columns for
# production; never write English text into them.
COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("title_tr", "TEXT"),
    ("why_now_tr", "TEXT"),
    ("category_tr", "TEXT"),
    ("trend_signal_tr", "TEXT"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(topic_candidates)").fetchall()}
    for column_name, column_type in COLUMN_MIGRATIONS:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE topic_candidates ADD COLUMN {column_name} {column_type}")


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


# --- batches ---------------------------------------------------------------

def create_batch(triggered_by: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO topic_batches (status, triggered_by, created_at, updated_at) VALUES ('presented', ?, ?, ?)",
            (triggered_by, now(), now()),
        )
        return cur.lastrowid


def get_batch(batch_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM topic_batches WHERE id = ?", (batch_id,)).fetchone()


def update_batch_fields(batch_id: int, **fields) -> None:
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = now()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [batch_id]
    with connect() as conn:
        conn.execute(f"UPDATE topic_batches SET {columns} WHERE id = ?", values)


# --- candidates --------------------------------------------------------------

def add_candidate(
    batch_id: int, position: int, title: str, topic: str, normalized_topic: str,
    short_description: str | None = None, why_now: str | None = None,
    category: str | None = None, trend_signal: str | None = None,
    score: float | None = None, score_breakdown: dict | None = None,
    sources: list[dict] | None = None, status: str = "discovered",
    title_tr: str | None = None, why_now_tr: str | None = None,
    category_tr: str | None = None, trend_signal_tr: str | None = None,
) -> int:
    """title/topic/why_now/category/trend_signal stay ENGLISH - they are
    what pipeline.produce_video() is called with. title_tr/why_now_tr/
    category_tr/trend_signal_tr are the Turkish text shown in the Telegram
    owner interface only (see youtube/topic_discovery.py's
    build_topic_batch_message()) - never used for production."""
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO topic_candidates
               (batch_id, position, title, topic, normalized_topic, short_description,
                why_now, category, trend_signal, score, score_json, sources_json,
                discovered_at, status, created_at, updated_at,
                title_tr, why_now_tr, category_tr, trend_signal_tr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id, position, title, topic, normalized_topic, short_description,
                why_now, category, trend_signal, score,
                json.dumps(score_breakdown) if score_breakdown is not None else None,
                json.dumps(sources) if sources is not None else None,
                now(), status, now(), now(),
                title_tr, why_now_tr, category_tr, trend_signal_tr,
            ),
        )
        return cur.lastrowid


def get_candidate(candidate_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM topic_candidates WHERE id = ?", (candidate_id,)).fetchone()


def list_candidates_for_batch(batch_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM topic_candidates WHERE batch_id = ? ORDER BY position", (batch_id,)
        ).fetchall()


def update_candidate_fields(candidate_id: int, **fields) -> None:
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = now()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [candidate_id]
    with connect() as conn:
        conn.execute(f"UPDATE topic_candidates SET {columns} WHERE id = ?", values)


def try_select_candidate(candidate_id: int) -> sqlite3.Row | None:
    """Atomically transitions candidate_id from 'presented' to 'selected',
    marks every OTHER 'presented' candidate in the same batch 'rejected',
    and marks the batch itself 'selected' - all in one transaction.

    Returns the updated candidate row on success, or None if it was not in
    a selectable state (unknown id, already selected/rejected/expired, or
    a concurrent caller already won the race for this exact candidate).
    None is the signal callers use to treat a callback as a safe no-op
    (duplicate/late click) instead of starting a second production run -
    see youtube/topic_discovery.py's handle_topic_select_callback()."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE topic_candidates SET status = 'selected', updated_at = ? WHERE id = ? AND status = 'presented'",
            (now(), candidate_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM topic_candidates WHERE id = ?", (candidate_id,)).fetchone()
        conn.execute(
            "UPDATE topic_candidates SET status = 'rejected', updated_at = ? "
            "WHERE batch_id = ? AND id != ? AND status = 'presented'",
            (now(), row["batch_id"], candidate_id),
        )
        conn.execute(
            "UPDATE topic_batches SET status = 'selected', updated_at = ? WHERE id = ?",
            (now(), row["batch_id"]),
        )
        return conn.execute("SELECT * FROM topic_candidates WHERE id = ?", (candidate_id,)).fetchone()
