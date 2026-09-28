"""
youtube/telegram_approval_demo.py tests.

Covers:
   18  full CLI run (dry-run and mocked real-send)
   16  no modification of storage/
       output directory / DB isolation
plus precondition failures and the CLI's --dry-run no-network guarantee.
"""
from __future__ import annotations

import json

import pytest

from core.config import settings
from youtube import telegram_approval as tap
from youtube import telegram_approval_demo as demo


@pytest.fixture
def approval_env(isolated_storage, tmp_path, monkeypatch):
    video_root = tmp_path / "debug_audit" / "final_video_v2"
    thumb_root = tmp_path / "debug_audit" / "thumbnail_v1"
    video_root.mkdir(parents=True)
    thumb_root.mkdir(parents=True)
    monkeypatch.setattr(tap, "FINAL_VIDEO_ROOT", video_root)
    monkeypatch.setattr(tap, "THUMBNAIL_ROOT", thumb_root)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)
    return tap


def _write_project(storage_dir, slug, title="The Wow! Signal: Astronomy's Greatest Unsolved Mystery", scenes=33):
    project_dir = storage_dir / "2026" / "09" / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata_en.json").write_text(json.dumps({"title": title, "description": "d", "tags": []}))
    (project_dir / "scenes.json").write_text(json.dumps([{"scene_id": i} for i in range(1, scenes + 1)]))
    return project_dir


def _write_video(tap_module, slug, size_bytes=1000):
    path = tap_module.video_path_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)


def _write_thumbnail(tap_module, slug):
    path = tap_module.thumbnail_path_for(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)


def _full_fixture(approval_env, isolated_storage, slug="the-wow-signal-6"):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, slug)
    _write_video(approval_env, slug)
    _write_thumbnail(approval_env, slug)
    return project_dir


@pytest.fixture
def fake_telegram_send(monkeypatch):
    sent = {"photos": [], "videos": []}

    def fake_photo(image_path, caption, keyboard, chat_id=None):
        message_id = 1000 + len(sent["photos"])
        sent["photos"].append({"image_path": image_path, "caption": caption, "keyboard": keyboard, "chat_id": chat_id})
        return message_id

    def fake_video(video_path, caption="", chat_id=None):
        message_id = 2000 + len(sent["videos"])
        sent["videos"].append({"video_path": video_path, "chat_id": chat_id})
        return message_id

    monkeypatch.setattr(tap, "send_photo_with_keyboard", fake_photo)
    monkeypatch.setattr(tap, "send_video_preview", fake_video)
    return sent


# ---------------------------------------------------------------------------
# --dry-run: zero network, zero DB writes
# ---------------------------------------------------------------------------

def test_dry_run_makes_no_telegram_call_and_creates_no_approval_row(approval_env, isolated_storage, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("send_photo_with_keyboard must never be called in --dry-run")
    monkeypatch.setattr(tap, "send_photo_with_keyboard", boom)
    monkeypatch.setattr(tap, "send_video_preview", boom)

    project_dir = _full_fixture(approval_env, isolated_storage, "dry-run-slug")

    result = demo.run("dry-run-slug", project_dir=project_dir, dry_run=True)

    assert result["dry_run"] is True
    assert "WAITING FOR APPROVAL" in result["caption"]
    assert tap.ta.get_latest_approval_by_slug("dry-run-slug") is None


def test_dry_run_reports_resolved_paths(approval_env, isolated_storage):
    project_dir = _full_fixture(approval_env, isolated_storage, "paths-slug")
    result = demo.run("paths-slug", project_dir=project_dir, dry_run=True)
    assert result["video_path"] == str(tap.video_path_for("paths-slug"))
    assert result["thumbnail_path"] == str(tap.thumbnail_path_for("paths-slug"))
    assert result["video_sendable"] is True


def test_dry_run_works_even_without_telegram_config(approval_env, isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "")
    project_dir = _full_fixture(approval_env, isolated_storage, "no-config-slug")
    result = demo.run("no-config-slug", project_dir=project_dir, dry_run=True)
    assert result["dry_run"] is True


# ---------------------------------------------------------------------------
# 18 - full CLI run
# ---------------------------------------------------------------------------

def test_cli_dry_run_exits_zero_and_prints_caption(approval_env, isolated_storage, capsys):
    project_dir = _full_fixture(approval_env, isolated_storage, "cli-dry-slug")
    exit_code = demo.main(["--project-dir", str(project_dir), "--dry-run"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "WAITING FOR APPROVAL" in captured.out
    assert "DRY RUN" in captured.out


def test_cli_real_send_creates_approval_and_prints_ids(approval_env, isolated_storage, fake_telegram_send, capsys):
    project_dir = _full_fixture(approval_env, isolated_storage, "cli-real-slug")
    exit_code = demo.main(["--project-dir", str(project_dir)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "approval_id:" in captured.out
    assert "message_id:" in captured.out

    row = tap.ta.get_latest_approval_by_slug("cli-real-slug")
    assert row["status"] == tap.STATUS_READY


def test_cli_resolves_project_by_slug_under_storage(approval_env, isolated_storage, fake_telegram_send):
    _full_fixture(approval_env, isolated_storage, "the-wow-signal-6")
    exit_code = demo.main(["--slug", "the-wow-signal-6"])
    assert exit_code == 0
    assert tap.ta.get_latest_approval_by_slug("the-wow-signal-6") is not None


# ---------------------------------------------------------------------------
# preconditions / failures
# ---------------------------------------------------------------------------

def test_cli_requires_slug_or_project_dir():
    exit_code = demo.main([])
    assert exit_code != 0


def test_cli_reports_failure_for_unknown_slug(capsys):
    exit_code = demo.main(["--slug", "no-such-project-at-all"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "FAILED" in captured.err


def test_cli_reports_failure_for_missing_video(approval_env, isolated_storage, capsys):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, "missing-video-slug")
    _write_thumbnail(approval_env, "missing-video-slug")
    exit_code = demo.main(["--project-dir", str(project_dir)])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "final video not found" in captured.err


def test_cli_reports_failure_for_missing_thumbnail(approval_env, isolated_storage, capsys):
    project_dir = _write_project(isolated_storage.STORAGE_DIR, "missing-thumb-slug")
    _write_video(approval_env, "missing-thumb-slug")
    exit_code = demo.main(["--project-dir", str(project_dir)])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "thumbnail not found" in captured.err


# ---------------------------------------------------------------------------
# 16 - no modification of storage/
# ---------------------------------------------------------------------------

def test_full_run_never_modifies_source_project_directory(approval_env, isolated_storage, fake_telegram_send):
    project_dir = _full_fixture(approval_env, isolated_storage, "readonly-slug")
    before = {p: p.read_bytes() for p in sorted(project_dir.iterdir())}

    demo.main(["--project-dir", str(project_dir)])

    after = {p: p.read_bytes() for p in sorted(project_dir.iterdir())}
    assert after == before


def test_dry_run_never_modifies_source_project_directory(approval_env, isolated_storage):
    project_dir = _full_fixture(approval_env, isolated_storage, "readonly-dry-slug")
    before = {p: p.read_bytes() for p in sorted(project_dir.iterdir())}

    demo.main(["--project-dir", str(project_dir), "--dry-run"])

    after = {p: p.read_bytes() for p in sorted(project_dir.iterdir())}
    assert after == before


# ---------------------------------------------------------------------------
# cost/API safety: never Claude, OpenAI/TTS, or research
# ---------------------------------------------------------------------------

def test_demo_module_never_imports_llm_client_researcher_or_tts():
    import ast
    source = open(demo.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"llm_client", "researcher", "tts"}
    assert not (imported & banned), f"unexpected import(s): {imported & banned}"


# ---------------------------------------------------------------------------
# 2026-09 production incident: --slug resolution relied solely on
# _find_project_dir()'s filesystem glob and never found a real, DB-backed
# pipeline project (video_id=5, slug=medieval-norman-castle-discovered-
# hidden-inside-3) - it looked for metadata_en.json at the wrong path
# entirely. These tests cover the fix: --video-id/--slug now resolve
# against the REAL `videos` DB row first (storage/project_store.py, the
# same source of truth pipeline.py itself writes to), using the real
# storage/<slug>/video.mp4 + thumbnail.png artifacts - falling back to the
# legacy debug_audit/ convention only when no DB row exists (covered
# above by test_cli_resolves_project_by_slug_under_storage, unchanged).
# Plus --resend-approval-id: recovers an approval stuck in status=draft
# after a failed send, without ever creating a new video or approval row.
# ---------------------------------------------------------------------------

def _write_real_pipeline_video(paths, size_bytes=1000):
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.video_mp4.write_bytes(b"x" * size_bytes)


def _write_real_pipeline_thumbnail(paths):
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.thumbnail_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)


@pytest.fixture
def real_pipeline_project(isolated_storage, monkeypatch):
    """A real, DB-backed already-produced project - storage/<year>/<month>/
    <slug>/video.mp4 + thumbnail.png + metadata_en.json + scenes.json,
    with a genuine `videos` table row (via storage/project_store.
    create_project(), exactly like the real production pipeline), mirroring
    the actual video_id=5 / slug=medieval-norman-castle-discovered-hidden-
    inside-3 incident this fix addresses."""
    from storage import project_store as ps

    paths, _created = ps.create_project(
        "Medieval Norman Castle Discovered Hidden Inside", category="History",
    )
    _write_real_pipeline_video(paths)
    _write_real_pipeline_thumbnail(paths)
    paths.metadata_en_json.write_text(json.dumps({
        "title": "Medieval Norman Castle Discovered Hidden Inside a Mountain",
        "description": "A short description.", "tags": ["a", "b"],
    }))
    paths.scenes_json.write_text(json.dumps([{"scene_id": i} for i in range(1, 19)]))
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)
    return paths


def test_resolves_real_db_backed_project_by_video_id(real_pipeline_project, fake_telegram_send):
    result = demo.run(video_id=real_pipeline_project.video_id)
    assert result["slug"] == real_pipeline_project.slug
    row = tap.ta.get_latest_approval_by_slug(real_pipeline_project.slug)
    assert row is not None
    assert row["video_path"] == str(real_pipeline_project.video_mp4)
    assert row["thumbnail_path"] == str(real_pipeline_project.thumbnail_png)


def test_resolves_real_db_backed_project_by_slug(real_pipeline_project, fake_telegram_send):
    result = demo.run(slug=real_pipeline_project.slug)
    assert result["slug"] == real_pipeline_project.slug
    row = tap.ta.get_latest_approval_by_slug(real_pipeline_project.slug)
    assert row["video_path"] == str(real_pipeline_project.video_mp4)


def test_cli_resolves_by_video_id_end_to_end(real_pipeline_project, fake_telegram_send, capsys):
    exit_code = demo.main(["--video-id", str(real_pipeline_project.video_id)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "approval_id:" in captured.out
    row = tap.ta.get_latest_approval_by_slug(real_pipeline_project.slug)
    assert row["status"] == tap.STATUS_READY


def test_dry_run_by_video_id_makes_no_network_call_and_no_approval_row(real_pipeline_project, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("send_photo_with_keyboard must never be called in --dry-run")
    monkeypatch.setattr(tap, "send_photo_with_keyboard", boom)
    monkeypatch.setattr(tap, "send_video_preview", boom)

    result = demo.run(video_id=real_pipeline_project.video_id, dry_run=True)

    assert result["dry_run"] is True
    assert result["video_path"] == str(real_pipeline_project.video_mp4)
    assert result["thumbnail_path"] == str(real_pipeline_project.thumbnail_png)
    assert tap.ta.get_latest_approval_by_slug(real_pipeline_project.slug) is None


def test_unknown_video_id_raises_clear_error(isolated_storage):
    with pytest.raises(tap.TelegramApprovalError, match="video_id=999999"):
        demo.run(video_id=999999)


def test_cli_reports_failure_for_unknown_video_id(capsys, isolated_storage):
    exit_code = demo.main(["--video-id", "999999"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "video_id=999999" in captured.err


# --- --resend-approval-id: the real recovery path for approval_id=3 -------

@pytest.fixture
def stuck_draft_approval(real_pipeline_project):
    """Reproduces the exact 2026-09 incident: ta.create_approval() ran (the
    approval row exists) but send_photo_with_keyboard() raised before
    ta.update_fields() could mark it ready_for_approval - so the row is
    stuck in status=draft with no telegram_message_id/video_path/
    thumbnail_path ever persisted on it (see send_new_approval()'s own
    ordering in youtube/telegram_approval.py)."""
    approval_id = tap.ta.create_approval(real_pipeline_project.slug, "Some Title", status=tap.STATUS_DRAFT)
    return real_pipeline_project, approval_id


def test_resend_uses_the_existing_approval_id(stuck_draft_approval, fake_telegram_send):
    paths, approval_id = stuck_draft_approval

    result = demo.resend(approval_id)

    assert result["approval_id"] == approval_id
    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_READY
    assert row["video_path"] == str(paths.video_mp4)
    assert row["thumbnail_path"] == str(paths.thumbnail_png)


def test_resend_creates_no_new_approval_row(stuck_draft_approval, fake_telegram_send):
    paths, approval_id = stuck_draft_approval
    before_count = len(tap.ta.list_approvals(limit=1000))

    demo.resend(approval_id)

    after_count = len(tap.ta.list_approvals(limit=1000))
    assert after_count == before_count
    # Still exactly one approval row for this slug - not a second one.
    assert tap.ta.get_latest_approval_by_slug(paths.slug)["id"] == approval_id


def test_resend_creates_no_new_video_or_project(stuck_draft_approval, fake_telegram_send, isolated_storage):
    paths, approval_id = stuck_draft_approval
    project_dirs_before = list(isolated_storage.STORAGE_DIR.glob("*/*/*"))

    demo.resend(approval_id)

    project_dirs_after = list(isolated_storage.STORAGE_DIR.glob("*/*/*"))
    assert project_dirs_after == project_dirs_before
    assert len(project_dirs_after) == 1


def test_resend_never_calls_produce_video(stuck_draft_approval, fake_telegram_send, monkeypatch):
    import pipeline as pipeline_module

    def boom(*a, **k):
        raise AssertionError("resend must never call pipeline.produce_video()")
    monkeypatch.setattr(pipeline_module, "produce_video", boom)
    _paths, approval_id = stuck_draft_approval

    demo.resend(approval_id)  # must not raise


def test_resend_dry_run_makes_no_network_call_and_does_not_change_status(stuck_draft_approval, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("send_photo_with_keyboard must never be called in --resend --dry-run")
    monkeypatch.setattr(tap, "send_photo_with_keyboard", boom)
    monkeypatch.setattr(tap, "send_video_preview", boom)
    _paths, approval_id = stuck_draft_approval

    result = demo.resend(approval_id, dry_run=True)

    assert result["dry_run"] is True
    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_DRAFT  # untouched


def test_cli_resend_end_to_end_prints_no_new_approval_note(stuck_draft_approval, fake_telegram_send, capsys):
    _paths, approval_id = stuck_draft_approval

    exit_code = demo.main(["--resend-approval-id", str(approval_id)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert f"approval_id: {approval_id}" in captured.out
    assert "no new approval row created" in captured.out


def test_resend_refuses_an_already_approved_video(real_pipeline_project):
    approval_id = tap.ta.create_approval(real_pipeline_project.slug, "Some Title", status=tap.STATUS_UPLOADED)
    with pytest.raises(tap.TelegramApprovalError, match="already been decided"):
        demo.resend(approval_id)


def test_resend_unknown_approval_id_raises(isolated_storage):
    with pytest.raises(tap.ApprovalNotFoundError):
        demo.resend(999999)


def test_cli_resend_cannot_be_combined_with_slug(capsys):
    exit_code = demo.main(["--resend-approval-id", "3", "--slug", "some-slug"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "cannot be combined" in captured.err


# ---------------------------------------------------------------------------
# 2026-09 BUTTON_DATA_INVALID incident: after the sendPhoto-resolution fix,
# `--resend-approval-id 3` reached Telegram but sendPhoto itself was
# rejected with HTTP 400 BUTTON_DATA_INVALID, because the keyboard's
# callback_data embedded the full (49-char) real slug, pushing it past
# Telegram's 64-byte limit. This test reproduces that exact slug through
# the real resend path and confirms the transmitted keyboard is now the
# short, ID-based one.
# ---------------------------------------------------------------------------

_REAL_INCIDENT_SLUG = "medieval-norman-castle-discovered-hidden-inside-3"


@pytest.fixture
def stuck_draft_approval_with_real_incident_slug(isolated_storage, monkeypatch):
    from storage import db, project_store as ps

    root = isolated_storage.STORAGE_DIR / "2026" / "09" / _REAL_INCIDENT_SLUG
    video_id = db.insert_video(
        slug=_REAL_INCIDENT_SLUG, topic="Medieval Norman Castle Discovered Hidden Inside",
        normalized_topic="medieval norman castle discovered hidden inside",
        category="History", project_id="p", state="pending_approval", storage_path=str(root),
    )
    paths = ps._paths_for(video_id, _REAL_INCIDENT_SLUG, str(root))
    _write_real_pipeline_video(paths)
    _write_real_pipeline_thumbnail(paths)
    paths.metadata_en_json.write_text(json.dumps({
        "title": "Medieval Norman Castle Discovered Hidden Inside a Mountain",
        "description": "A short description.", "tags": ["a", "b"],
    }))
    paths.scenes_json.write_text(json.dumps([{"scene_id": i} for i in range(1, 19)]))
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr(tap.tb, "_probe_duration_seconds", lambda path: 579.0)

    approval_id = tap.ta.create_approval(_REAL_INCIDENT_SLUG, "Some Title", status=tap.STATUS_DRAFT)
    return paths, approval_id


def test_resend_transmits_short_id_based_callback_data_for_the_real_incident_slug(
    stuck_draft_approval_with_real_incident_slug, fake_telegram_send,
):
    _paths, approval_id = stuck_draft_approval_with_real_incident_slug

    result = demo.resend(approval_id)

    assert result["approval_id"] == approval_id
    sent_keyboard = fake_telegram_send["photos"][-1]["keyboard"]
    buttons = sent_keyboard["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == f"approval:approve:{approval_id}"
    assert buttons[1]["callback_data"] == f"approval:reject:{approval_id}"
    for b in buttons:
        assert len(b["callback_data"].encode("utf-8")) <= 64
        assert _REAL_INCIDENT_SLUG not in b["callback_data"]
    row = tap.ta.get_approval(approval_id)
    assert row["status"] == tap.STATUS_READY
