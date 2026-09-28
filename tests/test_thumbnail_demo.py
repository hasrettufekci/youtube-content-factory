"""
youtube/thumbnail_demo.py tests.

Covers:
   14  FAL provider called exactly once (through the full pipeline)
   15  no FAL call in fixture mode (through the full pipeline)
   16  no modification of storage/
   17  output directory isolation
   18  full CLI fixture run
   21  (existing project tests remain green - see the delivery report;
       not itself a unit test here)
plus never overwriting an existing production thumbnail.png, precondition
failures, and a static "never calls Claude/OpenAI/TTS/research" check.
"""
from __future__ import annotations

import ast
import json

import pytest
from PIL import Image

from core.config import settings
from youtube import image_gen
from youtube import thumbnail_demo as demo

WIDTH, HEIGHT = settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT


def _write_project(project_dir, title="The Wow! Signal: An Alien Message?", with_research=True):
    project_dir.mkdir(parents=True, exist_ok=True)
    script = {"title": title, "hook": "hook text", "sections": [], "full_narration": ""}
    (project_dir / "script_en.json").write_text(json.dumps(script), encoding="utf-8")
    if with_research:
        research = {"topic": title.split(":")[0].strip(), "summary": "A short neutral summary."}
        (project_dir / "research.json").write_text(json.dumps(research), encoding="utf-8")
    return project_dir


# ---------------------------------------------------------------------------
# 18 - full CLI fixture run
# ---------------------------------------------------------------------------

def test_full_cli_fixture_run_creates_all_three_expected_files(tmp_path):
    project_dir = _write_project(tmp_path / "project")
    out_dir = tmp_path / "out"

    exit_code = demo.main([
        "--project-dir", str(project_dir),
        "--out-dir", str(out_dir),
        "--fixture",
    ])

    assert exit_code == 0
    assert (out_dir / "strategy.json").exists()
    assert (out_dir / "background.png").exists()
    assert (out_dir / "thumbnail.png").exists()

    strategy = json.loads((out_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["headline"] == "THE WOW! SIGNAL"

    img = Image.open(out_dir / "thumbnail.png")
    assert img.size == (WIDTH, HEIGHT)


def test_run_thumbnail_pipeline_returns_full_result(tmp_path):
    project_dir = _write_project(tmp_path / "project")
    out_dir = tmp_path / "out"

    result = demo.run_thumbnail_pipeline(project_dir, out_dir, fixture=True)

    assert result.qc.ok is True
    assert result.deterministic is True
    assert result.thumbnail_path == out_dir / "thumbnail.png"
    assert result.background_path == out_dir / "background.png"
    assert result.strategy_path == out_dir / "strategy.json"


# ---------------------------------------------------------------------------
# 16 - no modification of storage/
# ---------------------------------------------------------------------------

def test_full_run_never_modifies_source_project_directory(tmp_path):
    project_dir = _write_project(tmp_path / "project")
    out_dir = tmp_path / "out"

    before = {p.name: p.read_bytes() for p in sorted(project_dir.iterdir())}
    demo.run_thumbnail_pipeline(project_dir, out_dir, fixture=True)
    after = {p.name: p.read_bytes() for p in sorted(project_dir.iterdir())}

    assert after == before, "source project directory must never be modified by this pass"


def test_out_dir_cannot_be_inside_storage_dir(tmp_path, monkeypatch):
    fake_storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "STORAGE_DIR", fake_storage)
    project_dir = _write_project(tmp_path / "elsewhere" / "project")
    bad_out_dir = fake_storage / "2026" / "09" / "the-wow-signal-6" / "thumbnail_demo_output"

    with pytest.raises(demo.ThumbnailDemoError, match="storage"):
        demo.run_thumbnail_pipeline(project_dir, bad_out_dir, fixture=True)


def test_never_overwrites_existing_production_thumbnail(tmp_path):
    """An existing project may already have a real production
    thumbnail.png (from the older youtube/thumbnail.py system) sitting in
    its own directory - this pass must never touch it, because its output
    directory is never the project directory itself."""
    project_dir = _write_project(tmp_path / "project")
    existing_thumbnail = project_dir / "thumbnail.png"
    existing_thumbnail.write_bytes(b"EXISTING PRODUCTION THUMBNAIL - MUST NOT BE TOUCHED")
    before_bytes = existing_thumbnail.read_bytes()

    demo.run_thumbnail_pipeline(project_dir, tmp_path / "out", fixture=True)

    assert existing_thumbnail.read_bytes() == before_bytes


def test_out_dir_cannot_equal_project_dir(tmp_path):
    project_dir = _write_project(tmp_path / "project")
    with pytest.raises(demo.ThumbnailDemoError, match="out_dir"):
        demo.run_thumbnail_pipeline(project_dir, project_dir, fixture=True)


# ---------------------------------------------------------------------------
# 17 - output directory isolation
# ---------------------------------------------------------------------------

def test_output_only_written_under_the_given_out_dir(tmp_path):
    project_dir = _write_project(tmp_path / "project")
    out_dir = tmp_path / "isolated_output"

    before_tmp_entries = set(tmp_path.iterdir())
    demo.run_thumbnail_pipeline(project_dir, out_dir, fixture=True)
    after_tmp_entries = set(tmp_path.iterdir())

    # the only new top-level entry under tmp_path is out_dir itself - no
    # stray files/directories written anywhere else
    assert after_tmp_entries - before_tmp_entries == {out_dir}
    assert set(p.name for p in out_dir.iterdir()) == {"strategy.json", "background.png", "thumbnail.png"}


def test_default_out_dir_matches_expected_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project_dir = _write_project(tmp_path / "src_project")

    exit_code = demo.main(["--project-dir", str(project_dir), "--slug", "the-wow-signal-6", "--fixture"])

    assert exit_code == 0
    expected = tmp_path / "debug_audit" / "thumbnail_v1" / "the-wow-signal-6"
    assert (expected / "thumbnail.png").exists()


# ---------------------------------------------------------------------------
# 14 / 15 - provider call count through the full pipeline
# ---------------------------------------------------------------------------

def test_fixture_run_never_calls_image_gen_get_provider(tmp_path, monkeypatch):
    def _fail_if_called():
        raise AssertionError("image_gen.get_provider() must never be called in --fixture mode")
    monkeypatch.setattr(image_gen, "get_provider", _fail_if_called)

    project_dir = _write_project(tmp_path / "project")
    result = demo.run_thumbnail_pipeline(project_dir, tmp_path / "out", fixture=True)
    assert result.qc.ok is True


def test_live_mode_path_delegates_to_image_gen_get_provider_exactly_once(tmp_path, monkeypatch):
    calls = {"n": 0}

    class _CountingProvider(image_gen.ImageProvider):
        def generate(self, prompt, width, height):
            calls["n"] += 1
            from youtube.thumbnail_generator import FixtureImageProvider
            return FixtureImageProvider().generate(prompt, width, height)

    monkeypatch.setattr(image_gen, "get_provider", lambda: _CountingProvider())
    project_dir = _write_project(tmp_path / "project")

    result = demo.run_thumbnail_pipeline(project_dir, tmp_path / "out", fixture=False)

    assert calls["n"] == 1
    assert result.qc.ok is True


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def test_missing_script_json_fails_clearly(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    with pytest.raises(demo.ThumbnailDemoError, match="script_en.json"):
        demo.run_thumbnail_pipeline(project_dir, tmp_path / "out", fixture=True)


def test_main_cli_requires_slug_or_project_dir():
    exit_code = demo.main(["--fixture"])
    assert exit_code != 0


def test_main_cli_reports_failure_for_missing_project(capsys):
    exit_code = demo.main(["--slug", "no-such-project-at-all", "--fixture"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "FAILED" in captured.err


def test_main_cli_reports_success_report(tmp_path, capsys):
    project_dir = _write_project(tmp_path / "project")
    out_dir = tmp_path / "out"

    exit_code = demo.main([
        "--project-dir", str(project_dir), "--out-dir", str(out_dir), "--fixture",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "QC: ok=True" in captured.out


# ---------------------------------------------------------------------------
# Cost/API safety: never Claude, OpenAI/TTS, or research
# ---------------------------------------------------------------------------

def test_demo_module_never_imports_llm_client_researcher_or_tts():
    source = open(demo.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    banned = {"llm_client", "researcher", "tts", "httpx", "requests"}
    assert not (imported_modules & banned), f"unexpected import(s): {imported_modules & banned}"


def test_metadata_and_research_are_used_when_present_but_never_regenerated(tmp_path):
    """metadata_en.json, when present, is read (not required) - and this
    pipeline never calls researcher.research_topic() to produce a NEW
    research.json, even when one is missing."""
    project_dir = _write_project(tmp_path / "project", with_research=False)
    assert not (project_dir / "research.json").exists()

    result = demo.run_thumbnail_pipeline(project_dir, tmp_path / "out", fixture=True)

    assert result.qc.ok is True
    assert not (project_dir / "research.json").exists()  # still never created
