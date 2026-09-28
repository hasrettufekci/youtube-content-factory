"""
Video assembly failure handling (spec Round 2 section 8):

    - a missing scene image must be caught with a clear error, before ever
      shelling out to ffmpeg
    - an ffmpeg failure (non-zero exit code) must transition the pipeline
      to FAILED, never be silently swallowed or reported as success
"""
import subprocess

import pytest

from core.state import PipelineState
from storage import project_store as ps
from youtube import assemble


def _scenes(n=3):
    return [
        {"scene_id": i, "visual_prompt": f"p{i}", "narration": f"n{i}", "duration_hint": 5}
        for i in range(1, n + 1)
    ]


def test_missing_scene_image_raises_before_invoking_ffmpeg(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Missing Scene Image Topic")
    scenes = _scenes(3)
    # Only write scene 1 and 3 - scene 2's image is missing.
    paths.scene_image(1).write_bytes(b"fake png")
    paths.scene_image(3).write_bytes(b"fake png")
    paths.narration_mp3.write_bytes(b"fake mp3")

    def _fail_if_called(*a, **k):
        raise AssertionError("ffmpeg/ffprobe must not be invoked when a scene image is missing")

    monkeypatch.setattr(assemble.subprocess, "run", _fail_if_called)

    with pytest.raises(RuntimeError, match="missing"):
        assemble.assemble_video(paths, scenes)


def test_missing_narration_raises_before_invoking_ffmpeg(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Missing Narration Topic")
    scenes = _scenes(2)
    for s in scenes:
        paths.scene_image(s["scene_id"]).write_bytes(b"fake png")
    # narration.mp3 deliberately not written

    def _fail_if_called(*a, **k):
        raise AssertionError("ffmpeg/ffprobe must not be invoked when narration audio is missing")

    monkeypatch.setattr(assemble.subprocess, "run", _fail_if_called)

    with pytest.raises(RuntimeError, match="narration audio not found"):
        assemble.assemble_video(paths, scenes)


def test_run_and_save_transitions_to_failed_state_on_missing_images(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Run And Save Missing Images Topic")
    scenes = _scenes(2)
    ps.save_json(paths.scenes_json, scenes)
    paths.narration_mp3.write_bytes(b"fake mp3")
    # no scene images written at all

    monkeypatch.setattr(assemble.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not reach ffmpeg")))

    with pytest.raises(RuntimeError):
        assemble.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "video_assembly"
    assert "missing" in data["error"]


def test_ffmpeg_nonzero_exit_transitions_pipeline_to_failed_state(isolated_storage, monkeypatch):
    """A real ffmpeg failure (bad filter graph, corrupt input, disk full,
    etc. - simulated here as an arbitrary non-zero exit) must surface as a
    FAILED pipeline stage, never be treated as a successful assembly."""
    paths, _ = ps.create_project("FFmpeg Failure Topic")
    scenes = _scenes(2)
    ps.save_json(paths.scenes_json, scenes)
    for s in scenes:
        paths.scene_image(s["scene_id"]).write_bytes(b"fake png")
    paths.narration_mp3.write_bytes(b"fake mp3")

    # Bypass the real ffprobe duration probe (no real media files here) but
    # exercise the real _run_ffmpeg -> subprocess.run failure path.
    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 12.0)

    class _FakeCompletedProcess:
        returncode = 1
        stderr = "Error: no such filter 'zoompan' (simulated ffmpeg failure)"
        stdout = ""

    monkeypatch.setattr(assemble.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())

    with pytest.raises(RuntimeError):
        assemble.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "video_assembly"
    assert "FFMPEG" in data["error"] or "ffmpeg" in data["error"].lower()
    assert not paths.video_mp4.exists()


def test_ffmpeg_timeout_transitions_pipeline_to_failed_state(isolated_storage, monkeypatch):
    """A hung ffmpeg process (subprocess.run raising TimeoutExpired) must
    also be treated as a failure, not left to crash the whole pipeline
    process uncaught."""
    paths, _ = ps.create_project("FFmpeg Timeout Topic")
    scenes = _scenes(2)
    ps.save_json(paths.scenes_json, scenes)
    for s in scenes:
        paths.scene_image(s["scene_id"]).write_bytes(b"fake png")
    paths.narration_mp3.write_bytes(b"fake mp3")

    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 12.0)

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1800)

    monkeypatch.setattr(assemble.subprocess, "run", _timeout)

    with pytest.raises(Exception):
        assemble.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED


def test_run_and_save_records_failed_state_when_scenes_json_missing(isolated_storage):
    """Regression: this precondition check used to raise BEFORE
    db.start_run()/the try block, so it never went through
    save_failure() - the video was left stuck with no FAILED state
    recorded and no diagnostic trail in state.json."""
    paths, _ = ps.create_project("Assemble Missing Scenes JSON Topic")
    # scenes.json deliberately not written

    with pytest.raises(RuntimeError, match="no scenes.json found"):
        assemble.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "video_assembly"
