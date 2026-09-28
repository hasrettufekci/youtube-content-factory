"""
storage/shorts.py tests - the Shorts Engine V1 persistence layer.

Covers spec scenarios 2 (same parent video id), 3 (exactly 3 Shorts), 4
(Short numbering 1/2/3), plus create_short()'s own idempotency (mirrors
storage/publish_schedules.py's create_schedule() contract) and the
approval-linking helper used by the single-package-approval flow.
"""
from __future__ import annotations

from storage import shorts as shorts_store


def test_create_short_persists_parent_and_number(isolated_storage):
    short_id = shorts_store.create_short(parent_video_id=7, short_number=1, angle="hook")
    row = shorts_store.get_short(short_id)
    assert row["parent_video_id"] == 7
    assert row["short_number"] == 1
    assert row["angle"] == "hook"
    assert row["status"] == shorts_store.STATUS_BUILDING


def test_create_short_is_idempotent_by_parent_and_number(isolated_storage):
    """Calling create_short() twice for the same (parent_video_id,
    short_number) must never create a second row - mirrors
    storage/publish_schedules.py's create_schedule() idempotency contract,
    and is what makes a resumed pipeline run safe."""
    id1 = shorts_store.create_short(parent_video_id=1, short_number=1)
    id2 = shorts_store.create_short(parent_video_id=1, short_number=1)
    assert id1 == id2
    assert len(shorts_store.list_shorts_for_parent(1)) == 1


def test_three_shorts_share_the_same_parent_video_id(isolated_storage):
    for n in (1, 2, 3):
        shorts_store.create_short(parent_video_id=42, short_number=n)
    rows = shorts_store.list_shorts_for_parent(42)
    assert len(rows) == 3
    assert {r["short_number"] for r in rows} == {1, 2, 3}
    assert all(r["parent_video_id"] == 42 for r in rows)


def test_shorts_for_different_parents_are_independent(isolated_storage):
    shorts_store.create_short(parent_video_id=1, short_number=1)
    shorts_store.create_short(parent_video_id=2, short_number=1)
    assert len(shorts_store.list_shorts_for_parent(1)) == 1
    assert len(shorts_store.list_shorts_for_parent(2)) == 1


def test_update_fields_persists_and_bumps_updated_at(isolated_storage):
    short_id = shorts_store.create_short(parent_video_id=1, short_number=1)
    before = shorts_store.get_short(short_id)
    shorts_store.update_fields(short_id, status=shorts_store.STATUS_READY, title="A Short Title")
    after = shorts_store.get_short(short_id)
    assert after["status"] == shorts_store.STATUS_READY
    assert after["title"] == "A Short Title"
    assert after["updated_at"] >= before["updated_at"]


def test_set_approval_id_for_parent_attaches_every_short_to_one_approval(isolated_storage):
    """CRITICAL (spec): ONLY ONE approval for the whole package - every
    Short belonging to a Long gets the SAME approval_id, never its own."""
    for n in (1, 2, 3):
        shorts_store.create_short(parent_video_id=9, short_number=n)
    shorts_store.set_approval_id_for_parent(9, approval_id=555)
    rows = shorts_store.list_shorts_for_approval(555)
    assert len(rows) == 3
    assert all(r["approval_id"] == 555 for r in rows)


def test_list_shorts_for_approval_is_ordered_by_short_number(isolated_storage):
    for n in (3, 1, 2):
        shorts_store.create_short(parent_video_id=1, short_number=n)
    shorts_store.set_approval_id_for_parent(1, approval_id=1)
    rows = shorts_store.list_shorts_for_approval(1)
    assert [r["short_number"] for r in rows] == [1, 2, 3]


def test_upload_eligible_statuses_excludes_only_terminal_uploaded(isolated_storage):
    assert shorts_store.STATUS_UPLOADED not in shorts_store.UPLOAD_ELIGIBLE_STATUSES
    for status in (shorts_store.STATUS_READY, shorts_store.STATUS_APPROVED,
                   shorts_store.STATUS_FAILED, shorts_store.STATUS_UPLOAD_FAILED):
        assert status in shorts_store.UPLOAD_ELIGIBLE_STATUSES
