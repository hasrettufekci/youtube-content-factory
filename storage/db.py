"""
SQLite persistence (spec section 29). Deliberately simple: four tables,
no ORM. Used for the things that need to be *queried* (idempotency /
duplicate-topic checks, history, run logs) - the actual per-video content
(research, script, scenes, media) lives on disk under storage/<video>/
(see project_store.py), not in the DB, per spec section 25/26.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slug            TEXT UNIQUE NOT NULL,
    topic           TEXT NOT NULL,
    normalized_topic TEXT NOT NULL,
    category        TEXT,
    project_id      TEXT NOT NULL,
    state           TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    youtube_video_id TEXT,
    youtube_url     TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_videos_normalized_topic ON videos(normalized_topic);
CREATE INDEX IF NOT EXISTS idx_videos_state ON videos(state);

CREATE TABLE IF NOT EXISTS topics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT NOT NULL,
    normalized_topic TEXT UNIQUE NOT NULL,
    category        TEXT,
    score           REAL,
    score_json      TEXT,
    status          TEXT NOT NULL DEFAULT 'candidate',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        INTEGER NOT NULL REFERENCES videos(id),
    title           TEXT,
    url             TEXT,
    publisher       TEXT,
    published_at    TEXT,
    relevance       TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        INTEGER NOT NULL REFERENCES videos(id),
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    error           TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    path = db_path or settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    init_db()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- videos -----------------------------------------------------------

def insert_video(
    slug: str,
    topic: str,
    normalized_topic: str,
    category: str | None,
    project_id: str,
    state: str,
    storage_path: str,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO videos
               (slug, topic, normalized_topic, category, project_id, state,
                storage_path, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (slug, topic, normalized_topic, category, project_id, state,
             storage_path, now(), now()),
        )
        return cur.lastrowid


def find_video_by_normalized_topic(normalized_topic: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE normalized_topic = ? ORDER BY id DESC LIMIT 1",
            (normalized_topic,),
        ).fetchone()


def get_video(video_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()


def get_video_by_slug(slug: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM videos WHERE slug = ?", (slug,)).fetchone()


def update_video_state(video_id: int, state: str, error_message: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE videos SET state = ?, error_message = ?, updated_at = ? WHERE id = ?",
            (state, error_message, now(), video_id),
        )


def update_video_youtube(video_id: int, youtube_video_id: str, youtube_url: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE videos SET youtube_video_id = ?, youtube_url = ?, updated_at = ? WHERE id = ?",
            (youtube_video_id, youtube_url, now(), video_id),
        )


def list_videos(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM videos ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


# --- topics (topic scout candidates) -----------------------------------

def upsert_topic_candidate(
    topic: str, normalized_topic: str, category: str | None,
    score: float | None, score_json: str | None,
) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO topics (topic, normalized_topic, category, score, score_json, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'candidate', ?)
               ON CONFLICT(normalized_topic) DO UPDATE SET
                   score = excluded.score, score_json = excluded.score_json""",
            (topic, normalized_topic, category, score, score_json, now()),
        )


def find_topic_candidate(normalized_topic: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM topics WHERE normalized_topic = ?", (normalized_topic,)
        ).fetchone()


def top_topic_candidates(limit: int = 20) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM topics WHERE status = 'candidate' ORDER BY score DESC LIMIT ?",
            (limit,),
        ).fetchall()


def mark_topic_used(normalized_topic: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE topics SET status = 'used' WHERE normalized_topic = ?",
            (normalized_topic,),
        )


# --- research sources ----------------------------------------------------

def insert_research_sources(video_id: int, sources: list[dict]) -> None:
    if not sources:
        return
    with connect() as conn:
        conn.executemany(
            """INSERT INTO research_sources (video_id, title, url, publisher, published_at, relevance)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    video_id,
                    s.get("title"),
                    s.get("url"),
                    s.get("publisher"),
                    s.get("published_at"),
                    s.get("relevance"),
                )
                for s in sources
            ],
        )


def get_research_sources(video_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM research_sources WHERE video_id = ?", (video_id,)
        ).fetchall()


# --- pipeline runs (per-stage log) --------------------------------------

def start_run(video_id: int, stage: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO pipeline_runs (video_id, stage, status, started_at) VALUES (?, ?, 'running', ?)",
            (video_id, stage, now()),
        )
        return cur.lastrowid


def finish_run(run_id: int, status: str, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
            (status, now(), error, run_id),
        )


def get_runs(video_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM pipeline_runs WHERE video_id = ? ORDER BY id", (video_id,)
        ).fetchall()
