"""
Persistence for YouTube Analytics Timing V1's simple TTL cache
(youtube/analytics_timing.py's CompositeAudienceTimingProvider) - avoids
re-querying the real YouTube Data/Analytics APIs on every single Telegram
APPROVE within YOUTUBE_ANALYTICS_CACHE_TTL_HOURS of the last successful (or
unavailable) lookup.

Mirrors storage/publish_schedules.py's own style exactly: plain sqlite3, no
ORM, one small SCHEMA string, a `connect()` contextmanager, an idempotent
COLUMN_MIGRATIONS list, the SAME sqlite file (settings.DB_PATH) - purely
additive, no new DB file/config introduced.

Caches the WHOLE outcome of a lookup, including "unavailable" (empty
slots) - this is deliberate: an unavailable result (e.g. insufficient
scope, or a channel with too few videos) is just as expensive to keep
re-deriving as a successful one, and re-trying it on every approval within
the TTL window would defeat the point of caching. A cache row's
`slots_json` is simply `"[]"` in that case.

Nothing here ever fabricates a cached value - set_cache() always persists
exactly what CompositeAudienceTimingProvider actually computed; this module
never invents or defaults a score/sample_size for anything not passed in.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS analytics_timing_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key       TEXT NOT NULL,
    channel_key     TEXT,
    timezone        TEXT NOT NULL,
    lookback_days   INTEGER NOT NULL,
    source          TEXT NOT NULL,
    sample_size     INTEGER NOT NULL,
    confidence      TEXT NOT NULL,
    slots_json      TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_timing_cache_key ON analytics_timing_cache(cache_key);
"""

# Append-only, idempotent ALTER TABLE list for any column added after this
# table's initial release - see storage/publish_schedules.py's own
# COLUMN_MIGRATIONS for why this pattern is required. Empty at initial
# release.
COLUMN_MIGRATIONS: list[tuple[str, str]] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cache_key(channel_key: str | None, timezone_name: str, lookback_days: int) -> str:
    """Deterministic cache key for one (channel, timezone, lookback window)
    combination - the three inputs that actually change what a lookup would
    return."""
    return f"{channel_key or 'default'}:{timezone_name}:{lookback_days}"


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(analytics_timing_cache)").fetchall()}
    for column_name, column_type in COLUMN_MIGRATIONS:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE analytics_timing_cache ADD COLUMN {column_name} {column_type}")


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


def get_fresh(cache_key: str, *, now_dt: datetime | None = None) -> dict | None:
    """Returns the cached result dict if a row exists for cache_key AND it
    has not yet expired; otherwise None (missing OR expired - the caller
    can't tell which, and doesn't need to: both mean "go fetch again").
    `now_dt` is injectable for tests; defaults to the real current time."""
    now_dt = now_dt or datetime.now(timezone.utc)
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM analytics_timing_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
    if row is None:
        return None
    expires_at = datetime.fromisoformat(row["expires_at"])
    if now_dt >= expires_at:
        return None
    return _row_to_dict(row)


def set_cache(
    *,
    cache_key: str,
    channel_key: str | None,
    timezone_name: str,
    lookback_days: int,
    source: str,
    sample_size: int,
    confidence: str,
    slots: list[dict],
    ttl_hours: int,
    fetched_at: datetime | None = None,
) -> None:
    """Upsert (replace-by-cache_key, backed by the UNIQUE index above) -
    every fresh lookup fully replaces whatever was cached before for this
    exact (channel, timezone, lookback) combination, never merges."""
    fetched_at = fetched_at or datetime.now(timezone.utc)
    expires_at = fetched_at + timedelta(hours=ttl_hours)
    slots_json = json.dumps(slots)

    with connect() as conn:
        conn.execute("DELETE FROM analytics_timing_cache WHERE cache_key = ?", (cache_key,))
        conn.execute(
            """INSERT INTO analytics_timing_cache
               (cache_key, channel_key, timezone, lookback_days, source, sample_size,
                confidence, slots_json, fetched_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cache_key, channel_key, timezone_name, lookback_days, source, sample_size,
             confidence, slots_json, fetched_at.isoformat(), expires_at.isoformat()),
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "cache_key": row["cache_key"],
        "channel_key": row["channel_key"],
        "timezone": row["timezone"],
        "lookback_days": row["lookback_days"],
        "source": row["source"],
        "sample_size": row["sample_size"],
        "confidence": row["confidence"],
        "slots": json.loads(row["slots_json"]),
        "fetched_at": row["fetched_at"],
        "expires_at": row["expires_at"],
    }


def clear_cache() -> None:
    """Test/diagnostic helper only - nothing in the approval/upload flow
    calls this."""
    with connect() as conn:
        conn.execute("DELETE FROM analytics_timing_cache")
