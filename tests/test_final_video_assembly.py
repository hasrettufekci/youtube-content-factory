"""
YOUTUBE_CONTENT_FACTORY - youtube/final_video_assembly_demo.py tests.

Deliberately never calls Claude, FAL, or OpenAI - this module's whole job
is to re-assemble ALREADY-generated/approved scene images with the
EXISTING narration and scene timing via ffmpeg only. Covers:

  - every precondition failure (wrong scene count, missing/wrong-sized/
    malformed scene image, missing narration) raises BEFORE ffmpeg is
    ever invoked (mirrors tests/test_assemble.py's own pattern for the
    production assembler)
  - the frame-quantization helper (_frame_counts_for) is deterministic
    and always covers at least the narration's length
  - one real end-to-end ffmpeg render against a 33-scene fixture: output
    is exactly 1920x1080/H.264/AAC/30fps, the reported audio/video
    duration difference is small and the narration is never truncated,
    and re-running produces a byte-identical file (deterministic)
  - the source project directory is never modified
  - the CLI (main()) reports failure/success correctly
"""
from __future__ import annotations

import json
import math
import subprocess

import pytest
from PIL import Image

from youtube import final_video_assembly_demo as fva

WIDTH, HEIGHT, FPS = 1920, 1080, 30
EXPECTED_SCENE_COUNT = fva.EXPECTED_SCENE_COUNT


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _scenes(n: int, duration_hint: float = 1.2) -> list[dict]:
    return [{"scene_id": i, "duration_hint": duration_hint} for i in range(1, n + 1)]


def _write_scene_image(path, width=WIDTH, height=HEIGHT, color=(10, 20, 30)):
    Image.new("RGB", (width, height), color).save(path)


def _build_project(tmp_path, scenes: list[dict], with_narration: bool = True):
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    if with_narration:
        (project_dir / "narration_en.mp3").write_bytes(b"fake mp3 - never opened by validation-only tests")
    return project_dir


def _build_rendered_dir(tmp_path, scenes: list[dict], skip_for: set[int] | None = None,
                         wrong_size_for: set[int] | None = None):
    skip_for = skip_for or set()
    wrong_size_for = wrong_size_for or set()
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        sid = scene["scene_id"]
        if sid in skip_for:
            continue
        if sid in wrong_size_for:
            _write_scene_image(rendered_dir / f"scene_{sid:03d}.png", width=1920, height=1072)
        else:
            _write_scene_image(rendered_dir / f"scene_{sid:03d}.png")
    return rendered_dir


def _make_silent_narration(path, duration_seconds: float):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", f"{duration_seconds}", "-q:a", "9", str(path)],
        check=True, capture_output=True,
    )


def _fail_if_ffmpeg_invoked(monkeypatch):
    """Same pattern as tests/test_assemble.py: proves a precondition
    failure is caught BEFORE ever shelling out."""
    def _fail(*a, **k):
        raise AssertionError("ffmpeg/ffprobe must not be invoked when a precondition check should fail first")
    monkeypatch.setattr(subprocess, "run", _fail)


# ---------------------------------------------------------------------------
# Preconditions - all must fail loudly, before ffmpeg is ever invoked
# ---------------------------------------------------------------------------

def test_wrong_scene_count_fails_before_ffmpeg(tmp_path, monkeypatch):
    scenes = _scenes(32)  # not 33
    project_dir = _build_project(tmp_path, scenes)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="expected exactly 33"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


def test_too_many_scenes_also_fails(tmp_path, monkeypatch):
    scenes = _scenes(34)  # not 33
    project_dir = _build_project(tmp_path, scenes)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="expected exactly 33"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


def test_missing_scene_image_fails_before_ffmpeg(tmp_path, monkeypatch):
    scenes = _scenes(EXPECTED_SCENE_COUNT)
    project_dir = _build_project(tmp_path, scenes)
    rendered_dir = _build_rendered_dir(tmp_path, scenes, skip_for={17})
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="missing 1 rendered scene image"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


def test_wrong_sized_scene_image_fails_before_ffmpeg(tmp_path, monkeypatch):
    """A scene image that exists but isn't exactly 1920x1080 (e.g. the
    final-visual-render pass was never run / an old 1920x1072 source
    slipped through) must fail clearly - this pass trusts but verifies
    the upstream contract rather than silently re-normalizing."""
    scenes = _scenes(EXPECTED_SCENE_COUNT)
    project_dir = _build_project(tmp_path, scenes)
    rendered_dir = _build_rendered_dir(tmp_path, scenes, wrong_size_for={5})
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="not 1920x1080"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


def test_malformed_scene_image_fails_clearly(tmp_path, monkeypatch):
    scenes = _scenes(EXPECTED_SCENE_COUNT)
    project_dir = _build_project(tmp_path, scenes)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)
    (rendered_dir / "scene_009.png").write_bytes(b"not a real png at all")
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="could not be opened"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


def test_missing_narration_fails_before_ffmpeg(tmp_path, monkeypatch):
    scenes = _scenes(EXPECTED_SCENE_COUNT)
    project_dir = _build_project(tmp_path, scenes, with_narration=False)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="narration audio not found"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


def test_missing_scenes_json_fails_clearly(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()
    _fail_if_ffmpeg_invoked(monkeypatch)

    with pytest.raises(fva.VideoAssemblyError, match="no scenes.json found"):
        fva.render_final_video(project_dir, rendered_dir, tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# _frame_counts_for - deterministic frame quantization
# ---------------------------------------------------------------------------

def test_frame_counts_are_deterministic_across_repeated_calls():
    durations = [1.234, 0.9, 2.1, 1.5]
    a = fva._frame_counts_for(durations, narration_duration=5.734)
    b = fva._frame_counts_for(durations, narration_duration=5.734)
    assert a == b


def test_frame_counts_total_covers_narration_never_shorter():
    """Total frames must always be enough that the video is at least as
    long as the narration - this is what guarantees the narration is
    never truncated (no -shortest trimming is used at all)."""
    durations = [1.234, 0.9, 2.1]
    narration_duration = 4.234
    frames = fva._frame_counts_for(durations, narration_duration)
    total_seconds = sum(frames) / FPS
    assert total_seconds >= narration_duration


def test_frame_counts_last_scene_absorbs_the_residual():
    durations = [1.0, 1.0, 1.0]
    frames = fva._frame_counts_for(durations, narration_duration=3.0)
    assert frames[0] == round(1.0 * FPS)
    assert frames[1] == round(1.0 * FPS)
    assert sum(frames) == max(1, math.ceil(3.0 * FPS))


def test_frame_counts_never_goes_below_one_frame():
    frames = fva._frame_counts_for([0.001, 0.001], narration_duration=0.001)
    assert all(f >= 1 for f in frames)


# ---------------------------------------------------------------------------
# Real end-to-end ffmpeg render against a 33-scene fixture
# ---------------------------------------------------------------------------

def test_real_render_end_to_end(tmp_path):
    """One comprehensive real-ffmpeg test (kept singular - a real 33-scene
    encode has an inherent floor of ~33s of 1080p30 video regardless of
    narration length, since _compute_scene_durations' MIN_SCENE_SECONDS
    floor is reused unchanged from production - so this covers output
    correctness, determinism, and source-directory safety together rather
    than paying that encode cost three separate times)."""
    scenes = _scenes(EXPECTED_SCENE_COUNT, duration_hint=1.2)  # 33 * 1.2 = 39.6s target
    project_dir = _build_project(tmp_path, scenes, with_narration=False)
    _make_silent_narration(project_dir / "narration_en.mp3", 39.6)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)

    project_before = {p.name: p.read_bytes() for p in sorted(project_dir.iterdir())}
    rendered_before = {p.name: p.read_bytes() for p in sorted(rendered_dir.iterdir())}

    out1 = tmp_path / "out1.mp4"
    out2 = tmp_path / "out2.mp4"

    result1 = fva.render_final_video(project_dir, rendered_dir, out1)
    result2 = fva.render_final_video(project_dir, rendered_dir, out2)

    assert result1.scene_count == EXPECTED_SCENE_COUNT
    assert out1.exists()

    # deterministic: identical inputs -> identical computed timing AND an
    # identical encoded file, not just "looks the same"
    assert result1.scene_durations == result2.scene_durations
    assert result1.scene_frame_counts == result2.scene_frame_counts
    assert out1.read_bytes() == out2.read_bytes()

    # neither the source project directory nor the rendered scene images
    # were modified by either render
    project_after = {p.name: p.read_bytes() for p in sorted(project_dir.iterdir())}
    rendered_after = {p.name: p.read_bytes() for p in sorted(rendered_dir.iterdir())}
    assert project_after == project_before, "source project directory must never be modified"
    assert rendered_after == rendered_before, "rendered scene images must never be modified"

    report = fva.probe_output(out1)
    assert report["resolution"] == "1920x1080"
    assert report["width"] == 1920
    assert report["height"] == 1080
    assert report["video_codec"] == "h264"
    assert report["audio_codec"] == "aac"
    assert round(report["fps"]) == FPS
    assert report["file_size_bytes"] == out1.stat().st_size
    assert report["file_size_bytes"] > 0

    # narration must never be truncated: video is always >= narration
    assert report["video_duration"] >= report["audio_duration"] - 0.01
    # and the drift is small - at most a couple of frames, never seconds
    assert report["audio_video_duration_diff"] < (2 / FPS) + 0.01


# ---------------------------------------------------------------------------
# probe_output - ffprobe report
# ---------------------------------------------------------------------------

def test_probe_output_fails_clearly_when_ffprobe_cannot_read_file(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"definitely not an mp4")

    with pytest.raises(fva.VideoAssemblyError, match="could not inspect|no video stream|no audio stream"):
        fva.probe_output(bogus)


# ---------------------------------------------------------------------------
# CLI (main())
# ---------------------------------------------------------------------------

def test_main_cli_reports_failure_and_exits_nonzero_for_wrong_scene_count(tmp_path, capsys, monkeypatch):
    scenes = _scenes(10)  # not 33
    project_dir = _build_project(tmp_path, scenes)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)
    _fail_if_ffmpeg_invoked(monkeypatch)

    exit_code = fva.main([
        "--project-dir", str(project_dir),
        "--rendered-dir", str(rendered_dir),
        "--out", str(tmp_path / "out.mp4"),
    ])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "33" in captured.err


def test_main_cli_reports_success_and_prints_full_report(tmp_path, capsys):
    scenes = _scenes(EXPECTED_SCENE_COUNT, duration_hint=1.05)
    project_dir = _build_project(tmp_path, scenes, with_narration=False)
    _make_silent_narration(project_dir / "narration_en.mp3", 34.0)
    rendered_dir = _build_rendered_dir(tmp_path, scenes)
    out_path = tmp_path / "out.mp4"

    exit_code = fva.main([
        "--project-dir", str(project_dir),
        "--rendered-dir", str(rendered_dir),
        "--out", str(out_path),
    ])

    assert exit_code == 0
    assert out_path.exists()
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "resolution=1920x1080" in captured.out
    assert "video_codec=h264" in captured.out
    assert "audio_codec=aac" in captured.out


def test_main_cli_requires_slug_or_project_dir(capsys):
    exit_code = fva.main([])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "FAILED" in captured.err


def test_main_cli_slug_lookup_reports_failure_for_unknown_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(fva.settings, "STORAGE_DIR", tmp_path / "empty_storage")
    exit_code = fva.main(["--slug", "no-such-project"])
    assert exit_code != 0
