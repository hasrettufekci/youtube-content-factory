"""
storage/topic_batches.py tests - the dumb CRUD + atomic-selection layer
under the Daily Topic Discovery + Telegram Topic Selector (see
youtube/topic_discovery.py for the business logic that sits on top of
this).

Covers generic CRUD correctness this module's own callers depend on, the
atomic try_select_candidate() idempotency guarantee, and the idempotent
title_tr/why_now_tr/category_tr/trend_signal_tr column migration added by
the Turkish Telegram Management UI patch - mirrors
tests/test_telegram_approvals_storage.py's own migration-test pattern
exactly, since the Daily Topic Discovery feature is already live in
production (per the user's own report) and a real database may already
hold topic_batches/topic_candidates rows from before these columns
existed.
"""
from __future__ import annotations

import sqlite3

from storage import topic_batches as tbatch


def test_create_batch_defaults_status(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    row = tbatch.get_batch(batch_id)
    assert row["status"] == "presented"
    assert row["triggered_by"] == "scheduler"
    assert row["created_at"] and row["updated_at"]
    assert row["telegram_chat_id"] is None
    assert row["telegram_message_id"] is None


def test_get_batch_missing_returns_none(isolated_storage):
    assert tbatch.get_batch(999999) is None


def test_update_batch_fields_updates_and_bumps_updated_at(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    before = tbatch.get_batch(batch_id)
    tbatch.update_batch_fields(batch_id, telegram_chat_id="42", telegram_message_id=7)
    after = tbatch.get_batch(batch_id)
    assert after["telegram_chat_id"] == "42"
    assert after["telegram_message_id"] == 7
    assert after["updated_at"] >= before["updated_at"]


def test_update_batch_fields_with_no_fields_is_a_no_op(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    before = dict(tbatch.get_batch(batch_id))
    tbatch.update_batch_fields(batch_id)
    after = dict(tbatch.get_batch(batch_id))
    assert before == after


def test_add_candidate_stores_english_and_turkish_fields_separately(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    candidate_id = tbatch.add_candidate(
        batch_id=batch_id, position=1, title="The Wow Signal", topic="the wow signal",
        normalized_topic="the wow signal", category="Science / Mystery",
        why_now="Recently discussed widely.", trend_signal="Growing interest.",
        title_tr="Wow Sinyali", why_now_tr="Son zamanlarda yaygın konuşuluyor.",
        category_tr="Bilim / Gizem", trend_signal_tr="Artan ilgi.",
        status="presented",
    )
    row = tbatch.get_candidate(candidate_id)
    assert row["title"] == "The Wow Signal"
    assert row["title_tr"] == "Wow Sinyali"
    assert row["why_now"] == "Recently discussed widely."
    assert row["why_now_tr"] == "Son zamanlarda yaygın konuşuluyor."
    assert row["category"] == "Science / Mystery"
    assert row["category_tr"] == "Bilim / Gizem"
    assert row["trend_signal"] == "Growing interest."
    assert row["trend_signal_tr"] == "Artan ilgi."


def test_add_candidate_tr_fields_default_to_none(isolated_storage):
    """Confirms the four _tr parameters are optional - existing callers
    that don't pass them (or old code paths) must not break."""
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    candidate_id = tbatch.add_candidate(
        batch_id=batch_id, position=1, title="No Translation Yet",
        topic="no translation yet", normalized_topic="no translation yet",
    )
    row = tbatch.get_candidate(candidate_id)
    assert row["title_tr"] is None
    assert row["why_now_tr"] is None
    assert row["category_tr"] is None
    assert row["trend_signal_tr"] is None


def test_list_candidates_for_batch_ordered_by_position(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    for i in (3, 1, 2):
        tbatch.add_candidate(
            batch_id=batch_id, position=i, title=f"T{i}", topic=f"t{i}", normalized_topic=f"t{i}",
        )
    rows = tbatch.list_candidates_for_batch(batch_id)
    assert [r["position"] for r in rows] == [1, 2, 3]


# ---------------------------------------------------------------------------
# try_select_candidate() atomic idempotency guarantee
# ---------------------------------------------------------------------------

def test_try_select_candidate_selects_and_rejects_siblings(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    ids = [
        tbatch.add_candidate(
            batch_id=batch_id, position=i, title=f"T{i}", topic=f"t{i}",
            normalized_topic=f"t{i}", status="presented",
        )
        for i in range(1, 4)
    ]
    row = tbatch.try_select_candidate(ids[1])
    assert row["status"] == "selected"
    assert tbatch.get_candidate(ids[0])["status"] == "rejected"
    assert tbatch.get_candidate(ids[2])["status"] == "rejected"
    assert tbatch.get_batch(batch_id)["status"] == "selected"


def test_try_select_candidate_second_call_is_a_no_op(isolated_storage):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    cid = tbatch.add_candidate(
        batch_id=batch_id, position=1, title="T1", topic="t1", normalized_topic="t1", status="presented",
    )
    first = tbatch.try_select_candidate(cid)
    second = tbatch.try_select_candidate(cid)
    assert first is not None
    assert second is None


def test_try_select_candidate_unknown_id_returns_none(isolated_storage):
    assert tbatch.try_select_candidate(999999) is None


# ---------------------------------------------------------------------------
# title_tr/why_now_tr/category_tr/trend_signal_tr column migration
# (Turkish Telegram Management UI patch)
# ---------------------------------------------------------------------------

_PRE_MIGRATION_SCHEMA = """
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
"""


def _make_pre_migration_db(db_path):
    """Simulates a real, already-deployed DB from before title_tr/
    why_now_tr/category_tr/trend_signal_tr existed - built by hand (not via
    tbatch.init_db(), which already includes the migration) so the columns
    are genuinely absent, with one real presented candidate row like a
    production batch that was sent before this patch shipped."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_PRE_MIGRATION_SCHEMA)
        conn.execute(
            """INSERT INTO topic_batches (status, triggered_by, created_at, updated_at)
               VALUES ('presented', 'scheduler', '2026-09-01T09:00:00+00:00', '2026-09-01T09:00:00+00:00')"""
        )
        conn.execute(
            """INSERT INTO topic_candidates
               (batch_id, position, title, topic, normalized_topic, category,
                why_now, trend_signal, discovered_at, status, created_at, updated_at)
               VALUES (1, 1, 'The Wow Signal', 'the wow signal', 'the wow signal', 'Science / Mystery',
                       'Recently discussed widely.', 'Growing interest.',
                       '2026-09-01T09:00:00+00:00', 'presented',
                       '2026-09-01T09:00:00+00:00', '2026-09-01T09:00:00+00:00')"""
        )
        conn.commit()
    finally:
        conn.close()


def test_migration_adds_missing_turkish_columns(tmp_path):
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    tbatch.init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(topic_candidates)").fetchall()}
    finally:
        conn.close()
    for col in ("title_tr", "why_now_tr", "category_tr", "trend_signal_tr"):
        assert col in columns


def test_migration_preserves_existing_row_data(tmp_path):
    """The exact scenario the real production DB is in: an existing
    'presented' candidate row (from a batch sent before this patch shipped)
    must survive the migration with every existing field intact, and the
    new _tr columns simply come back NULL for it (never a placeholder
    value that would look like a real translation)."""
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    tbatch.init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM topic_candidates WHERE topic = ?", ("the wow signal",)).fetchone()
    finally:
        conn.close()
    assert row["title"] == "The Wow Signal"
    assert row["category"] == "Science / Mystery"
    assert row["status"] == "presented"
    assert row["title_tr"] is None
    assert row["why_now_tr"] is None
    assert row["category_tr"] is None
    assert row["trend_signal_tr"] is None


def test_migration_is_idempotent(tmp_path):
    """Calling init_db() repeatedly (as connect() does on every single CRUD
    call) must never fail with a "duplicate column" error once the columns
    already exist."""
    db_path = tmp_path / "pre_migration.db"
    _make_pre_migration_db(db_path)

    tbatch.init_db(db_path)
    tbatch.init_db(db_path)  # second call: columns already exist
    tbatch.init_db(db_path)  # third call, for good measure

    conn = sqlite3.connect(str(db_path))
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(topic_candidates)").fetchall()]
    finally:
        conn.close()
    for col in ("title_tr", "why_now_tr", "category_tr", "trend_signal_tr"):
        assert columns.count(col) == 1


def test_migration_on_a_brand_new_db_is_a_no_op(isolated_storage):
    """A DB created fresh via the current SCHEMA + add_candidate() already
    supports the _tr columns from CREATE TABLE / the initial migration run
    - _migrate_columns() must recognize that and add nothing (no error, no
    duplicate column)."""
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    candidate_id = tbatch.add_candidate(
        batch_id=batch_id, position=1, title="Brand New", topic="brand new", normalized_topic="brand new",
    )
    row = tbatch.get_candidate(candidate_id)
    assert row["title_tr"] is None
    assert row["why_now_tr"] is None


def test_batch_and_candidates_persist_across_fresh_connections(isolated_storage):
    """Every tbatch.* call opens and closes its own sqlite3 connection (see
    connect()) - nothing is cached in Python-process memory - so this is
    exactly what "persists after restart" means for a sqlite-backed store:
    a brand new connection (simulating a fresh process) sees the row."""
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    candidate_id = tbatch.add_candidate(
        batch_id=batch_id, position=1, title="Restart Title", topic="restart topic",
        normalized_topic="restart topic", title_tr="Yeniden Başlatma Başlığı", status="presented",
    )

    with tbatch.connect() as conn:
        row = conn.execute("SELECT * FROM topic_candidates WHERE id = ?", (candidate_id,)).fetchone()
    assert row["title"] == "Restart Title"
    assert row["title_tr"] == "Yeniden Başlatma Başlığı"
    assert row["status"] == "presented"
