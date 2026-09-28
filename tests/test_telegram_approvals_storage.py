"""
storage/telegram_approvals.py tests - the dumb CRUD layer under Telegram
Approval V1 (see youtube/telegram_approval.py for the state machine that
sits on top of this).

Covers (of the spec's 18 scenarios):
    1  approval creation
   18  approval persistence after restart (a fresh connection sees the
       same row - sqlite-backed, not RAM-only)
plus generic CRUD correctness this module's own callers depend on, and
the idempotent thumbnail_status/thumbnail_error column migration (thumbnail
OAuth investigation fix) added below.
"""
from __future__ import annotations

import sqlite3

from storage import telegram_approvals as ta


def test_create_approval_defaults_status(isolated_storage):
    approval_id = ta.create_approval("the-wow-signal-6", "The Wow! Signal")
    row = ta.get_approval(approval_id)
    assert row["status"] == "draft"
    assert row["slug"] == "the-wow-signal-6"
    assert row["title"] == "The Wow! Signal"
    assert row["created_at"] and row["updated_at"]
    assert row["telegram_chat_id"] is None
    assert row["telegram_message_id"] is None
    assert row["youtube_video_id"] is None
    assert row["youtube_url"] is None
    assert row["rejection_reason"] is None
    assert row["upload_error"] is None


def test_create_approval_accepts_explicit_status(isolated_storage):
    approval_id = ta.create_approval("some-slug", "Some Title", status="ready_for_approval")
    assert ta.get_approval(approval_id)["status"] == "ready_for_approval"


def test_get_approval_missing_returns_none(isolated_storage):
    assert ta.get_approval(999999) is None


def test_get_latest_approval_by_slug_returns_none_when_unknown(isolated_storage):
    assert ta.get_latest_approval_by_slug("no-such-slug") is None


def test_get_latest_approval_by_slug_returns_most_recent(isolated_storage):
    first_id = ta.create_approval("repeat-slug", "First Title")
    second_id = ta.create_approval("repeat-slug", "Second Title")
    latest = ta.get_latest_approval_by_slug("repeat-slug")
    assert latest["id"] == second_id
    assert latest["id"] != first_id


def test_update_fields_updates_status_and_bumps_updated_at(isolated_storage):
    approval_id = ta.create_approval("update-slug", "Title")
    before = ta.get_approval(approval_id)
    ta.update_fields(approval_id, status="ready_for_approval", telegram_chat_id="42", telegram_message_id=7)
    after = ta.get_approval(approval_id)
    assert after["status"] == "ready_for_approval"
    assert after["telegram_chat_id"] == "42"
    assert after["telegram_message_id"] == 7
    assert after["updated_at"] >= before["updated_at"]


def test_update_fields_with_no_fields_is_a_no_op(isolated_storage):
    approval_id = ta.create_approval("noop-slug", "Title")
    before = dict(ta.get_approval(approval_id))
    ta.update_fields(approval_id)
    after = dict(ta.get_approval(approval_id))
    assert before == after


def test_list_approvals_orders_newest_first(isolated_storage):
    ta.create_approval("a-slug", "A")
    ta.create_approval("b-slug", "B")
    ta.create_approval("c-slug", "C")
    rows = ta.list_approvals(limit=10)
    slugs = [r["slug"] for r in rows]
    assert slugs == ["c-slug", "b-slug", "a-slug"]


def test_list_approvals_respects_limit(isolated_storage):
    for i in range(5):
        ta.create_approval(f"slug-{i}", f"Title {i}")
    rows = ta.list_approvals(limit=2)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# 18 - approval persistence after restart
# ---------------------------------------------------------------------------

def test_approval_persists_across_fresh_connections(isolated_storage):
    """Every ta.* call opens and closes its own sqlite3 connection (see
    connect()) - nothing is cached in Python-process memory - so this is
    exactly what "persists after restart" means for a sqlite-backed store:
    a brand new connection (simulating a fresh process) sees the row."""
    approval_id = ta.create_approval("restart-slug", "Restart Title", status="ready_for_approval")
    ta.update_fields(approval_id, telegram_chat_id="123", telegram_message_id=456)

    # Simulate "process restart": open an entirely fresh, independent
    # connection to the same DB file and read through it.
    with ta.connect() as conn:
        row = conn.execute("SELECT * FROM telegram_approvals WHERE id = ?", (approval_id,)).fetchone()
    assert row["slug"] == "restart-slug"
    assert row["status"] == "ready_for_approval"
    assert row["telegram_chat_id"] == "123"
    assert row["telegram_message_id"] == 456


def test_db_file_actually_exists_on_disk(isolated_storage):
    ta.create_approval("disk-slug", "Disk Title")
    assert isolated_storage.DB_PATH.exists()


# ---------------------------------------------------------------------------
# thumbnail_status/thumbnail_error column migration (thumbnail OAuth
# investigation fix). A real production DB may already hold rows (e.g. the
# real pilot's approval_id=1) from before these columns existed - the
# migration must add them without losing or corrupting anything already
# there. CREATE TABLE IF NOT EXISTS alone would silently do nothing for an
# existing table, which is exactly why _migrate_columns()/PRAGMA table_info
# exist - these tests pin that behavior down directly.
# ---------------------------------------------------------------------------

_PRE_MIGRATION_SCHEMA = """
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
"""


def _make_pre_migration_db(db_path):
    """Simulates a real, already-deployed DB from before thumbnail_status/
    thumbnail_error existed - built by hand (not via ta.init_db(), which
    already includes the migration) so the columns are genuinely absent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_PRE_MIGRATION_SCHEMA)
        conn.execute(
            """INSERT INTO telegram_approvals
               (slug, title, status, created_at, updated_at, youtube_video_id, youtube_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("the-wow-signal-6", "The Wow! Signal", "uploaded", "2026-09-01T00:00:00+00:00",
             "2026-09-01T00:00:00+00:00", "O0p4zf_JXyo", "https://www.youtube.com/watch?v=O0p4zf_JXyo"),
        )
        conn.commit()
    finally:
        conn.close()


def test_migration_adds_missing_thumbnail_columns(tmp_path):
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    ta.init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(telegram_approvals)").fetchall()}
    finally:
        conn.close()
    assert "thumbnail_status" in columns
    assert "thumbnail_error" in columns


def test_migration_adds_production_integration_columns(tmp_path):
    """video_path/thumbnail_path/duration_label/scene_count/language/
    privacy/channel_key - added for the pipeline.py <-> Telegram Approval
    V1 integration - migrate onto an existing (even pre-thumbnail_status)
    DB exactly like thumbnail_status/thumbnail_error did, without
    disturbing a real, already-uploaded row such as the production
    the-wow-signal-6 pilot's approval_id=1."""
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    ta.init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(telegram_approvals)").fetchall()}
        row = conn.execute("SELECT * FROM telegram_approvals WHERE slug = ?", ("the-wow-signal-6",)).fetchone()
    finally:
        conn.close()

    for col in ("video_path", "thumbnail_path", "duration_label", "scene_count", "language", "privacy", "channel_key"):
        assert col in columns
    # The already-uploaded pilot row survives with these new columns NULL,
    # never a placeholder value, and its existing fields untouched.
    assert row["status"] == "uploaded"
    assert row["youtube_video_id"] == "O0p4zf_JXyo"
    assert row["video_path"] is None
    assert row["thumbnail_path"] is None
    assert row["channel_key"] is None


def test_migration_preserves_existing_row_data(tmp_path):
    """The exact scenario the real production DB is in: an existing row
    (like the real pilot's approval_id=1) must survive the migration with
    every existing field intact, and the new columns simply come back NULL
    for it (never a default/placeholder value that would look like a real
    result)."""
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    ta.init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM telegram_approvals WHERE slug = ?", ("the-wow-signal-6",)).fetchone()
    finally:
        conn.close()
    assert row["title"] == "The Wow! Signal"
    assert row["status"] == "uploaded"
    assert row["youtube_video_id"] == "O0p4zf_JXyo"
    assert row["youtube_url"] == "https://www.youtube.com/watch?v=O0p4zf_JXyo"
    assert row["thumbnail_status"] is None
    assert row["thumbnail_error"] is None


def test_migration_is_idempotent(tmp_path):
    """Calling init_db() repeatedly (as connect() does on every single CRUD
    call) must never fail with a "duplicate column" error once the columns
    already exist."""
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    ta.init_db(db_path)
    ta.init_db(db_path)  # second call: columns already exist
    ta.init_db(db_path)  # third call, for good measure

    conn = sqlite3.connect(str(db_path))
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(telegram_approvals)").fetchall()]
    finally:
        conn.close()
    # Exactly one of each - never duplicated by a repeated ALTER TABLE.
    assert columns.count("thumbnail_status") == 1
    assert columns.count("thumbnail_error") == 1


def test_migration_on_a_brand_new_db_is_a_no_op(isolated_storage):
    """A DB created fresh via the current SCHEMA already includes both
    columns from CREATE TABLE itself - _migrate_columns() must recognize
    that and add nothing (no error, no duplicate column)."""
    approval_id = ta.create_approval("brand-new-slug", "Brand New")
    row = ta.get_approval(approval_id)
    assert row["thumbnail_status"] is None
    assert row["thumbnail_error"] is None


def test_thumbnail_fields_can_be_set_via_update_fields(isolated_storage):
    """update_fields() is generic (storage/telegram_approvals.py's own doc
    comment says so) - confirms the new columns work through it exactly
    like every other column, with no dedicated setter needed."""
    approval_id = ta.create_approval("thumb-slug", "Thumb Title")
    ta.update_fields(approval_id, thumbnail_status="success")
    assert ta.get_approval(approval_id)["thumbnail_status"] == "success"

    ta.update_fields(approval_id, thumbnail_status="failed", thumbnail_error="channel not verified")
    row = ta.get_approval(approval_id)
    assert row["thumbnail_status"] == "failed"
    assert row["thumbnail_error"] == "channel not verified"
