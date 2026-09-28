"""
Post-assembly quality control (spec section 36; extended for multilingual
production in Round 3 section 18).

Runs after video assembly + thumbnail + metadata and before the video is
sent for Telegram approval. Any failure here moves the video to FAILED
(handled by pipeline.py, which then retries only the responsible stage).

Multilingual checks (Round 3): for the master language and every
configured localization (core.config.settings.LANGUAGE_MASTER /
LANGUAGE_LOCALIZATIONS) this additionally verifies that language's script,
narration audio (existence + a real, readable duration), and metadata all
exist - named script_<lang>_exists / narration_<lang>_exists /
narration_<lang>_valid / metadata_<lang>_exists, so adding a language to
config automatically extends what's checked here with zero code changes.
Per-language narration durations are collected separately on
`QualityReport.audio_durations` (spec: "report language-specific audio
durations separately") rather than folded into the pass/fail `checks` dict.

Standalone test:
    python -m youtube.quality_check --video-id 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field

from core.config import settings
from core.logging_setup import get_logger
from storage import project_store as ps

logger = get_logger("qc")

DURATION_TOLERANCE_SECONDS = 3.0


@dataclass
class QualityReport:
    passed: bool
    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    audio_durations: dict = field(default_factory=dict)  # {language: seconds}


def _ffprobe_json(path) -> dict:
    import json
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"[FFMPEG] ffprobe failed on {path}: {result.stderr}")
    return json.loads(result.stdout)


def _qc_languages(languages: list[str] | None = None) -> list[str]:
    """Master language plus whichever localizations are actually in scope
    for this QC run.

    `languages`, when given, is the SAME per-run override
    pipeline.py's produce_video()/_stage_plan() already thread through
    every other stage (e.g. `languages=[]` for an --english-only run,
    via pipeline._localization_languages() - this mirrors that function
    exactly so QC and the stage plan always agree on what was actually
    requested/generated for this run). `None` (the default - every
    existing caller, and the standalone `python -m youtube.quality_check
    --video-id 1` CLI, which has no per-run language concept) preserves
    the original behavior exactly: every language configured in
    settings.LANGUAGE_LOCALIZATIONS is required, unchanged.

    Deliberately reads only the `languages` value passed in - never CLI
    args, env vars, or pipeline state - so QC stays a pure function of
    "what was this run actually asked to produce", the same answer every
    other stage already uses."""
    localizations = languages if languages is not None else settings.LANGUAGE_LOCALIZATIONS
    return [settings.LANGUAGE_MASTER] + [
        lang for lang in localizations if lang != settings.LANGUAGE_MASTER
    ]


def run_quality_check(paths: ps.ProjectPaths, languages: list[str] | None = None) -> QualityReport:
    report = QualityReport(passed=True)

    def check(name: str, ok: bool, detail: str = ""):
        report.checks[name] = {"passed": ok, "detail": detail}
        if not ok:
            report.passed = False
            report.errors.append(f"{name}: {detail}")

    # 1. video file exists
    video_exists = paths.video_mp4.exists() and paths.video_mp4.stat().st_size > 0
    check("video_file_exists", video_exists, str(paths.video_mp4))
    if not video_exists:
        return report  # nothing further can be checked meaningfully

    # 2. it's actually a valid, ffprobe-readable MP4 - not just a file that
    # happens to exist. A corrupt/truncated/non-video file at video.mp4
    # must show up here as a reported, named check failure (so the caller
    # gets a clear quality-check report and the video is correctly marked
    # FAILED) rather than as an uncaught RuntimeError from _ffprobe_json
    # blowing past every check below it.
    try:
        probe = _ffprobe_json(paths.video_mp4)
        probe_error = ""
    except RuntimeError as e:
        probe = None
        probe_error = str(e)
    check("valid_mp4_container", probe is not None, probe_error)
    if probe is None:
        return report  # nothing further can be meaningfully checked on an unreadable file

    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    video_duration = float(probe.get("format", {}).get("duration", 0))

    # 3. video stream present + resolution + codec
    check("video_stream_present", len(video_streams) > 0, f"{len(video_streams)} video stream(s)")
    if video_streams:
        w, h = video_streams[0].get("width"), video_streams[0].get("height")
        check("resolution_1920x1080", (w == settings.VIDEO_WIDTH and h == settings.VIDEO_HEIGHT), f"got {w}x{h}")
        vcodec = video_streams[0].get("codec_name")
        check("video_codec_h264", vcodec == "h264", f"got codec_name={vcodec}")
    else:
        check("resolution_1920x1080", False, "no video stream found")
        check("video_codec_h264", False, "no video stream found")

    # 4. audio stream present + codec
    check("audio_present", len(audio_streams) > 0, f"{len(audio_streams)} audio stream(s)")
    if audio_streams:
        acodec = audio_streams[0].get("codec_name")
        check("audio_codec_aac", acodec == "aac", f"got codec_name={acodec}")
    else:
        check("audio_codec_aac", False, "no audio stream found")

    # 5. audio duration matches video duration (narration should not be cut off)
    if paths.narration_mp3.exists():
        try:
            narration_probe = _ffprobe_json(paths.narration_mp3)
            narration_duration = float(narration_probe.get("format", {}).get("duration", 0))
            diff = abs(video_duration - narration_duration)
            check("audio_duration_matches_video", diff <= DURATION_TOLERANCE_SECONDS,
                  f"video={video_duration:.1f}s narration={narration_duration:.1f}s diff={diff:.1f}s")
        except RuntimeError as e:
            check("audio_duration_matches_video", False, f"narration.mp3 unreadable: {e}")
    else:
        check("audio_duration_matches_video", False, "narration.mp3 missing")

    # 6. scene image count matches scenes.json
    scenes = ps.load_json(paths.scenes_json) or []
    scene_images = paths.scene_images()
    check("scene_image_count_matches_plan", len(scene_images) == len(scenes),
          f"plan={len(scenes)} images={len(scene_images)}")

    # 7. thumbnail exists and is the correct 1280x720 size
    thumbnail_exists = paths.thumbnail_png.exists() and paths.thumbnail_png.stat().st_size > 0
    check("thumbnail_exists", thumbnail_exists, str(paths.thumbnail_png))
    if thumbnail_exists:
        try:
            from PIL import Image
            with Image.open(paths.thumbnail_png) as im:
                tw, th = im.size
            check("thumbnail_dimensions_1280x720",
                  (tw == settings.THUMBNAIL_WIDTH and th == settings.THUMBNAIL_HEIGHT), f"got {tw}x{th}")
        except Exception as e:
            check("thumbnail_dimensions_1280x720", False, f"could not read thumbnail: {e}")
    else:
        check("thumbnail_dimensions_1280x720", False, "thumbnail.png missing")

    # 8. metadata exists
    metadata = ps.load_json(paths.metadata_json)
    check("metadata_exists", bool(metadata), str(paths.metadata_json))

    # 9. research sources present
    research = ps.load_json(paths.research_json) or {}
    sources = research.get("sources", []) if isinstance(research, dict) else []
    check("research_sources_present", len(sources) > 0, f"{len(sources)} source(s)")

    # duration sanity range: an 8-20 minute documentary should land roughly
    # 6-22 minutes; well outside that means something upstream went wrong.
    check("duration_in_documentary_range", 360 <= video_duration <= 1320, f"{video_duration:.1f}s")

    # 10. multilingual: script / narration / metadata exist and are valid
    # for the master language and every language actually in scope for
    # this run (Round 3 section 18; English-only integration fix) -
    # generic over _qc_languages(), so a future language added to config
    # is checked here automatically, and a per-run override (e.g.
    # --english-only) is respected exactly like every other stage.
    languages_to_check = _qc_languages(languages)
    for lang in languages_to_check:
        script_path = paths.script_en_json if lang == settings.LANGUAGE_MASTER else paths.script_localized_json(lang)
        script_data = ps.load_json(script_path)
        check(f"script_{lang}_exists", bool(script_data), str(script_path))

        narration_path = paths.narration_mp3_for(lang)
        narration_exists = narration_path.exists() and narration_path.stat().st_size > 0
        check(f"narration_{lang}_exists", narration_exists, str(narration_path))
        if narration_exists:
            try:
                narration_probe = _ffprobe_json(narration_path)
                duration = float(narration_probe.get("format", {}).get("duration", 0))
                report.audio_durations[lang] = duration
                check(f"narration_{lang}_valid", duration > 0, f"duration={duration:.1f}s")
            except RuntimeError as e:
                check(f"narration_{lang}_valid", False, f"unreadable: {e}")
        else:
            check(f"narration_{lang}_valid", False, "audio file missing")

        metadata_path = paths.metadata_en_json if lang == settings.LANGUAGE_MASTER else paths.metadata_json_for(lang)
        metadata_data = ps.load_json(metadata_path)
        check(f"metadata_{lang}_exists", bool(metadata_data), str(metadata_path))

    logger.info("[QC] %s: %d/%d checks passed", paths.slug,
                sum(1 for c in report.checks.values() if c["passed"]), len(report.checks))
    return report


def run_and_enforce(paths: ps.ProjectPaths, languages: list[str] | None = None) -> QualityReport:
    report = run_quality_check(paths, languages=languages)
    if not report.passed:
        ps.save_failure(paths, "quality_check", "; ".join(report.errors))
        logger.error("[QC] FAILED for %s: %s", paths.slug, report.errors)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run post-assembly quality checks (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    report = run_and_enforce(paths)
    for name, result in report.checks.items():
        status = "OK" if result["passed"] else "FAIL"
        print(f"[{status}] {name}: {result['detail']}")
    print("PASSED" if report.passed else "FAILED")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
