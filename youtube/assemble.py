"""
Video assembly (spec sections 18-20, 36).

New engine, independent of any short-form/vertical video system. Builds a
16:9, 1920x1080, 30fps documentary video from scene images + narration:

    scale/crop each scene to frame -> per-scene Ken Burns motion (zoom +
    pan, movement varies scene to scene) -> crossfade transitions between
    scenes -> mux with narration (+ optional background music, ducked
    under the narration) -> H.264/AAC MP4 with a fast-start moov atom.

Built as a SINGLE ffmpeg invocation with one filter_complex graph (all
scaling/zoompan/xfade steps chained in one pass) rather than N sequential
merge calls, so cost stays roughly linear in scene count even at the
spec'd 25-40 scenes instead of the quadratic re-encoding cost repeated
pairwise merges would incur.

Scene timing: each scene's on-screen duration is derived from its
`duration_hint` (word-count based, from scene_planner.py) but rescaled so
the *actual* narration audio duration (measured with ffprobe, since TTS
timing never matches a naive words/second estimate exactly) is matched -
crossfade overlap is compensated for the same way, so the final video
length equals the narration length (not shorter/cut off, not padded).

Standalone test:
    python -m youtube.assemble --video-id 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from core.config import settings
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps

logger = get_logger("assemble")

FPS = settings.VIDEO_FPS
WIDTH = settings.VIDEO_WIDTH
HEIGHT = settings.VIDEO_HEIGHT
ZOOM_MAX = 1.15
MIN_SCENE_SECONDS = 1.0
NARRATION_VOLUME = 1.0
MUSIC_VOLUME = 0.12

# Movement presets cycle scene to scene so no two consecutive scenes move
# identically (spec section 19). All are zoom-in variants (most reliable
# with ffmpeg's zoompan) combined with different pan directions.
_MOVEMENTS = ["zoom_center", "zoom_pan_right", "zoom_pan_left"]
_TRANSITIONS = ["fade", "fadeblack"]


def _run_ffmpeg(args: list[str], timeout: int = 1800) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"[FFMPEG] command failed (code {result.returncode}):\n{result.stderr[-3000:]}")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"[FFMPEG] ffprobe could not read duration of {path}: {result.stderr}")
    return float(result.stdout.strip())


def _zoompan_expr(movement: str, frames: int, zoom_step: float) -> tuple[str, str, str]:
    z = f"min(zoom+{zoom_step:.6f},{ZOOM_MAX})"
    if movement == "zoom_pan_right":
        x = f"(iw-iw/zoom)*(on/{max(frames - 1, 1)})"
        y = "ih/2-(ih/zoom/2)"
    elif movement == "zoom_pan_left":
        x = f"(iw-iw/zoom)*(1-on/{max(frames - 1, 1)})"
        y = "ih/2-(ih/zoom/2)"
    else:  # zoom_center
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    return z, x, y


def _compute_scene_durations(scenes: list[dict], narration_duration: float, xfade_duration: float) -> list[float]:
    raw = [max(s.get("duration_hint", 0) or 0, 0.1) for s in scenes]
    total_raw = sum(raw)
    n = len(scenes)
    target_total = narration_duration + (n - 1) * xfade_duration
    scaled = [max(r * target_total / total_raw, MIN_SCENE_SECONDS) for r in raw]
    return scaled


def _pick_music_file(music_filename: str | None) -> Path | None:
    if music_filename:
        candidate = settings.MEDIA_MUSIC_DIR / music_filename
        return candidate if candidate.exists() else None
    files = sorted(settings.MEDIA_MUSIC_DIR.glob("*.mp3"))
    return files[0] if files else None


def assemble_video(
    paths: ps.ProjectPaths,
    scenes: list[dict],
    background_music: bool = False,
    music_filename: str | None = None,
    language: str | None = None,
    width: int | None = None,
    height: int | None = None,
    scene_image_paths: list[Path] | None = None,
    narration_path: Path | None = None,
    dest_path: Path | None = None,
) -> Path:
    # Round 3: the primary pipeline only ever calls this with the default
    # (master/English) language - ONE shared visual production + the
    # master narration track, per spec section 2/14. `language` exists so
    # a future/optional per-language render is a one-line call away
    # (paths.video_mp4_for(language)) without ever regenerating the scene
    # images passed in here - it is not invoked for any other language by
    # any stage in pipeline.py today.
    #
    # width/height/scene_image_paths/narration_path/dest_path (Shorts
    # Engine V1, additive): when given, they override the module-level
    # WIDTH/HEIGHT constants and the paths.*_for(language) conventions
    # below - e.g. width=1080/height=1920/scene_image_paths=[...vertical
    # visuals...]/narration_path=paths.short_narration_path(n)/dest_path=
    # paths.short_video_path(n) - reusing this exact same ffmpeg
    # filter_complex/Ken-Burns/crossfade pipeline for a vertical Short
    # without duplicating it. None (every pre-existing caller) is 100%
    # unchanged behavior byte-for-byte.
    language = language or settings.LANGUAGE_MASTER
    width = width or WIDTH
    height = height or HEIGHT
    narration_path = narration_path or paths.narration_mp3_for(language)
    dest = dest_path or paths.video_mp4_for(language)

    scene_images = scene_image_paths if scene_image_paths is not None else [paths.scene_image(s["scene_id"]) for s in scenes]
    missing = [p for p in scene_images if not p.exists()]
    if missing:
        raise RuntimeError(f"[FFMPEG] missing {len(missing)} scene image(s), e.g. {missing[0]}")
    if not narration_path.exists():
        raise RuntimeError(f"[FFMPEG] narration audio not found at {narration_path}")

    narration_duration = _probe_duration(narration_path)
    n = len(scenes)

    # First pass to pick a sane crossfade length, second pass to compute the
    # crossfade-compensated per-scene durations (same two-step logic as the
    # original short-form engine, generalized to per-scene weighted hints).
    rough = _compute_scene_durations(scenes, narration_duration, xfade_duration=0)
    xfade_duration = max(0.35, min(0.9, min(rough) / 3))
    durations = _compute_scene_durations(scenes, narration_duration, xfade_duration)

    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, (img_path, dur) in enumerate(zip(scene_images, durations)):
        frames = max(int(round(dur * FPS)), 1)
        zoom_step = (ZOOM_MAX - 1) / frames
        movement = _MOVEMENTS[i % len(_MOVEMENTS)]
        z, x, y = _zoompan_expr(movement, frames, zoom_step)

        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur + 0.5:.3f}", "-i", str(img_path)]
        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={FPS},"
            f"setsar=1[v{i}]"
        )

    # Chain crossfades in one pass: v0,v1 -> vx1 ; vx1,v2 -> vx2 ; ...
    cur = "v0"
    offset = durations[0] - xfade_duration
    for i in range(1, n):
        out_label = f"vx{i}"
        transition = _TRANSITIONS[i % len(_TRANSITIONS)]
        filter_parts.append(
            f"[{cur}][v{i}]xfade=transition={transition}:duration={xfade_duration:.3f}:offset={offset:.3f}[{out_label}]"
        )
        cur = out_label
        if i < n - 1:
            offset += durations[i] - xfade_duration

    narration_index = n
    inputs += ["-i", str(narration_path)]

    audio_label = f"{narration_index}:a"
    music_path = _pick_music_file(music_filename) if background_music else None
    if music_path is not None:
        music_index = narration_index + 1
        inputs += ["-stream_loop", "-1", "-i", str(music_path)]
        filter_parts.append(
            f"[{music_index}:a]volume={MUSIC_VOLUME}[bgm]"
        )
        filter_parts.append(
            f"[{narration_index}:a]volume={NARRATION_VOLUME}[narr]"
        )
        filter_parts.append(
            "[narr][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        audio_label = "aout"
        logger.info("[FFMPEG] mixing background music: %s", music_path.name)
    elif background_music:
        logger.warning("[FFMPEG] background_music requested but no music file found in %s - continuing without it", settings.MEDIA_MUSIC_DIR)

    filter_complex = ";".join(filter_parts)
    audio_map = f"[{audio_label}]" if audio_label == "aout" else audio_label  # "aout" is a filter label; "N:a" is a raw stream specifier

    args = [
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{cur}]",
        "-map", audio_map,
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(dest),
    ]

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[FFMPEG] assembling %d scenes -> %s (language=%s, narration=%.1fs, xfade=%.2fs)",
                n, dest.name, language, narration_duration, xfade_duration)
    _run_ffmpeg(args)
    logger.info("[FFMPEG] wrote %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def run_and_save(
    paths: ps.ProjectPaths,
    background_music: bool = False,
    music_filename: str | None = None,
    language: str | None = None,
) -> Path:
    # The precondition check (missing scenes.json) lives inside the try
    # block below, not before it: every failure here must go through the
    # same save_failure() path as a real ffmpeg error, or the video is
    # left stuck with no FAILED state recorded and no diagnostic trail.
    language = language or settings.LANGUAGE_MASTER
    is_master = language == settings.LANGUAGE_MASTER
    stage_label = "video_assembly" if is_master else f"video_assembly_{language}"
    run_id = db.start_run(paths.video_id, stage_label)
    try:
        scenes = ps.load_json(paths.scenes_json)
        if not scenes:
            raise RuntimeError(f"[FFMPEG] no scenes.json found at {paths.scenes_json} - run scene_planner.py first")

        result = assemble_video(paths, scenes, background_music=background_music, music_filename=music_filename, language=language)
        if is_master:
            ps.save_state(paths, PipelineState.VIDEO_ASSEMBLED)
        db.finish_run(run_id, "success")
        return result
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, stage_label, str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the final video (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--music", action="store_true", help="Mix in background music if a file exists under storage/music/")
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        result = run_and_save(paths, background_music=args.music)
        print(f"OK - video saved to {result}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
