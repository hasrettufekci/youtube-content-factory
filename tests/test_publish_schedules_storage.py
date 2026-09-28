"""
storage/publish_schedules.py tests - Smart Publish Scheduler V1's
persistence layer (Faz 7).

Mirrors tests/test_telegram_approvals_storage.py's own CRUD + migration
test pattern. The one property this table is EXPLICITLY required to
guarantee (spec, verbatim): "approval_id=3 icin scheduler iki kere
calistirilirsa: 1 schedule olmali, 2 schedule olmamali" - see the
idempotency tests below, which are the core of this file.
"""
from __future__ import annotations

import sqlite3

import pytest

from storage import publish_schedules as ps


def test_create_schedule_returns_a_new_id(isolated_storage):
    schedule_id = ps.create_schedule(
        approval_id=3, video_id=5, scheduled_at="2026-09-28T20:00:00+03:00",
        timezone="Europe/Istanbul", source="bootstrap", confidence="bootstrap",
        reason="Channel does not yet have sufficient audience timing data.",
    )
    assert isinstance(schedule_id, int)
    row = ps.get_schedule(schedule_id)
    assert row["approval_id"] == 3
    assert row["video_id"] == 5
    assert row["scheduled_at"] == "2026-09-28T20:00:00+03:00"
    assert row["timezone"] == "Europe/Istanbul"
    assert row["status"] == ps.STATUS_PLANNED
    assert row["source"] == "bootstrap"
    assert row["confidence"] == "bootstrap"
    assert row["youtube_video_id"] is None
    assert row["created_at"] and row["updated_at"]


def test_get_schedule_missing_returns_none(isolated_storage):
    assert ps.get_schedule(999999) is None


def test_get_schedule_by_approval_id_missing_returns_none(isolated_storage):
    assert ps.get_schedule_by_approval_id(999999) is None


# ---------------------------------------------------------------------------
# Idempotency (Faz 7/8 - the core requirement of this whole table)
# ---------------------------------------------------------------------------

def test_create_schedule_called_twice_for_same_approval_id_yields_one_schedule(isolated_storage):
    """The exact scenario from the spec: approval_id=3, scheduler run twice
    -> exactly 1 row, never 2."""
    id1 = ps.create_schedule(
        approval_id=3, video_id=5, scheduled_at="2026-09-28T20:00:00+03:00",
        timezone="Europe/Istanbul", source="bootstrap", confidence="bootstrap", reason="r1",
    )
    id2 = ps.create_schedule(
        approval_id=3, video_id=5, scheduled_at="2026-09-29T20:00:00+03:00",
        timezone="Europe/Istanbul", source="bootstrap", confidence="bootstrap", reason="r2",
    )

    assert id1 == id2
    all_rows = [r for r in ps.list_schedules() if r["approval_id"] == 3]
    assert len(all_rows) == 1
    # The FIRST computed decision wins - a second call never overwrites the
    # already-persisted scheduled_at with a newly (and differently)
    # recomputed one.
    assert all_rows[0]["scheduled_at"] == "2026-09-28T20:00:00+03:00"
    assert all_rows[0]["reason"] == "r1"


def test_create_schedule_called_three_times_still_yields_one_schedule(isolated_storage):
    ids = [
        ps.create_schedule(
            approval_id=3, scheduled_at=f"2026-09-2{n}T20:00:00+03:00", timezone="Europe/Istanbul",
            source="bootstrap", confidence="bootstrap", reason=f"attempt {n}",
        )
        for n in (7, 8, 9)
    ]
    assert len(set(ids)) == 1
    assert len(ps.list_schedules()) == 1


def test_create_schedule_for_different_approvals_yields_separate_schedules(isolated_storage):
    id_a = ps.create_schedule(
        approval_id=3, scheduled_at="2026-09-28T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    id_b = ps.create_schedule(
        approval_id=4, scheduled_at="2026-09-29T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    assert id_a != id_b
    assert len(ps.list_schedules()) == 2


def test_unique_index_on_approval_id_rejects_a_direct_duplicate_insert(isolated_storage):
    """Defense-in-depth check: the schema-level UNIQUE constraint itself (not
    just create_schedule()'s check-then-insert) refuses a second row for the
    same approval_id, even bypassing create_schedule()'s own guard."""
    ps.create_schedule(
        approval_id=3, scheduled_at="2026-09-28T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    with ps.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO publish_schedules
                   (video_id, approval_id, scheduled_at, timezone, status, source, confidence,
                    reason, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (None, 3, "2026-09-29T20:00:00+03:00", "Europe/Istanbul", ps.STATUS_PLANNED,
                 "bootstrap", "bootstrap", "r2", ps.now(), ps.now()),
            )


# ---------------------------------------------------------------------------
# update_fields
# ---------------------------------------------------------------------------

def test_update_fields_updates_and_bumps_updated_at(isolated_storage):
    schedule_id = ps.create_schedule(
        approval_id=3, scheduled_at="2026-09-28T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    before = ps.get_schedule(schedule_id)
    ps.update_fields(schedule_id, status=ps.STATUS_UPLOADED, youtube_video_id="abc123")
    after = ps.get_schedule(schedule_id)
    assert after["status"] == ps.STATUS_UPLOADED
    assert after["youtube_video_id"] == "abc123"
    assert after["updated_at"] >= before["updated_at"]


def test_update_fields_with_no_fields_is_a_no_op(isolated_storage):
    schedule_id = ps.create_schedule(
        approval_id=3, scheduled_at="2026-09-28T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    before = dict(ps.get_schedule(schedule_id))
    ps.update_fields(schedule_id)
    after = dict(ps.get_schedule(schedule_id))
    assert before == after


# ---------------------------------------------------------------------------
# get_schedule_by_video_id / list_schedules
# ---------------------------------------------------------------------------

def test_get_schedule_by_video_id_returns_the_latest(isolated_storage):
    ps.create_schedule(
        approval_id=1, video_id=5, scheduled_at="2026-09-27T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    ps.create_schedule(
        approval_id=2, video_id=5, scheduled_at="2026-09-28T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    latest = ps.get_schedule_by_video_id(5)
    assert latest["approval_id"] == 2


def test_list_schedules_orders_newest_first(isolated_storage):
    id_a = ps.create_schedule(
        approval_id=1, scheduled_at="2026-09-27T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    id_b = ps.create_schedule(
        approval_id=2, scheduled_at="2026-09-28T20:00:00+03:00", timezone="Europe/Istanbul",
        source="bootstrap", confidence="bootstrap", reason="r",
    )
    rows = ps.list_schedules()
    assert rows[0]["id"] == id_b
    assert rows[1]["id"] == id_a


# ---------------------------------------------------------------------------
# status enum values used by the actual approve-flow wiring
# ---------------------------------------------------------------------------

def test_status_constants_match_the_spec_enum():
    assert {ps.STATUS_PLANNED, ps.STATUS_UPLOADED, ps.STATUS_SCHEDULED,
            ps.STATUS_PUBLISHED, ps.STATUS_FAILED, ps.STATUS_CANCELLED} == {
        "planned", "uploaded", "scheduled", "published", "failed", "cancelled",
    }


# ---------------------------------------------------------------------------
# Migration idempotency (mirrors tests/test_telegram_approvals_storage.py's
# own pattern) - this table's COLUMN_MIGRATIONS list is empty at initial
# release, so this is a forward-looking safety net: init_db() must remain
# callable any number of times (exactly as connect() does on every CRUD
# call) without ever erroring, on a brand-new DB or an existing one.
# ---------------------------------------------------------------------------

def test_init_db_is_idempotent_on_a_brand_new_db(tmp_path):
    db_path = tmp_path / "fresh.db"
    ps.init_db(db_path)
    ps.init_db(db_path)
    ps.init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(publish_schedules)").fetchall()]
    finally:
        conn.close()
    assert columns.count("approval_id") == 1


def test_migration_preserves_existing_row_data(tmp_path):
    """A real production row (e.g. the actual approval_id=3/video_id=5
    schedule) must survive any future column migration with every existing
    field intact - mirrors telegram_approvals' own guarantee."""
    db_path = tmp_path / "existing.db"
    ps.init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """INSERT INTO publish_schedules
               (video_id, approval_id, scheduled_at, timezone, status, source, confidence,
                reason, created_at, updated_at, youtube_video_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (5, 3, "2026-09-28T20:00:00+03:00", "Europe/Istanbul", "uploaded", "bootstrap",
             "bootstrap", "r", "2026-09-27T00:00:00+00:00", "2026-09-27T00:00:00+00:00", "abc123"),
        )
        conn.commit()
    finally:
        conn.close()

    ps.init_db(db_path)  # re-run the migration path

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM publish_schedules WHERE approval_id = ?", (3,)).fetchone()
    finally:
        conn.close()
    assert row["video_id"] == 5
    assert row["status"] == "uploaded"
    assert row["youtube_video_id"] == "abc123"
