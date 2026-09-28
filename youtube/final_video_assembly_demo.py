"""
YOUTUBE_CONTENT_FACTORY - SAFE FINAL VIDEO REASSEMBLY PASS.

Takes the 33 APPROVED final scene images produced by the safe final visual
render pass (youtube/final_visual_render_demo.py ->
debug_audit/final_visual_render_v2/<slug>/scene_001.png ...
scene_033.png) and reassembles them into a new video, muxed against the
EXISTING production narration track and paced using the EXISTING per-scene
timing already stored in scenes.json - it never calls Claude, FAL, or
OpenAI, never regenerates images or TTS, and never writes anything under
the source project's own directory (storage/2026/09/<slug>/) - only under
a brand-new output directory this module creates itself. It never opens
storage/2026/09/<slug>/video.mp4 at all, let alone overwrites it.

WHY A SEPARATE MODULE, NOT youtube/assemble.py ITSELF:
youtube/assemble.py is the general-purpose PRODUCTION assembler - it reads
its scene images from the project's own storage/ directory
(paths.scene_image()), applies Ken Burns motion + crossfade transitions,
and writes to storage/.../video.mp4. This pass is deliberately narrower
and safer: its whole job is to re-render ONE specific, already-approved
33-scene project from the NEW rendered-scenes directory
(debug_audit/final_visual_render_v2/<slug>/) rather than the live
storage/ scene images, write only under debug_audit/, and validate the
33-scene contract explicitly (see EXPECTED_SCENE_COUNT). It reuses
assemble.py's own scene-duration math and ffmpeg/ffprobe helpers directly
(_compute_scene_durations, _probe_duration, _run_ffmpeg) rather than
reimplementing them, so "use existing scene durations/timing" and
"preserve existing narration synchronization" mean EXACTLY what they mean
in production - there is no second, divergent copy of that math.

TIMING / SYNCHRONIZATION: each scene's on-screen duration is derived from
scenes.json's existing duration_hint (word-count based), rescaled so the
per-scene durations sum to the *measured* narration_en.mp3 duration - the
identical algorithm youtube/assemble.py already uses for its own
crossfade-compensated timing, called here with xfade_duration=0 because
this pass does hard cuts (no Ken Burns motion, no crossfade transitions -
a re-grade of the original production render is a separate concern from
"reassemble the 33 approved final images with correct pacing"). Each
scene's raw duration is then frame-quantized (round(duration * fps)) -
exactly like assemble.py's own zoompan frame-count math - and the LAST
scene absorbs whatever small residual is left so the total encoded video
length always lands on ceil(narration_duration * fps): at least as long as
the narration, never shorter, so the narration audio is NEVER truncated
(no -shortest trimming is used - the full narration track is always
mapped and always plays in full; any drift is a few hundredths of a
second of trailing picture with no new narration, not lost audio). This
is deterministic: the same scenes.json + the same narration_en.mp3 always
produce the same per-scene frame counts and the same output.

VALIDATION (fails loudly, never silently produces a partial/wrong video -
render_final_video() checks every precondition BEFORE ever invoking
ffmpeg, so a bad run never wastes an encode or leaves a partial output at
the destination path):
  - exactly 33 scenes must be declared in scenes.json (EXPECTED_SCENE_COUNT
    - this pass is scoped to the-wow-signal-6-style 33-scene projects
    specifically, unlike assemble.py which handles any scene count)
  - every one of those 33 scene_NNN.png files must exist under the
    rendered-scenes directory, AND must already be exactly
    settings.VIDEO_WIDTH x settings.VIDEO_HEIGHT - this pass trusts but
    verifies the upstream final-visual-render contract rather than
    re-normalizing/re-padding images itself (that logic already has its
    one source of truth in youtube/image_normalize.py and is not
    duplicated here)
  - narration_en.mp3 must exist under the source project directory

Usage (against the real production project + its approved final renders -
e.g. on the Windows machine, once this file is copied over):

    python -m youtube.final_video_assembly_demo --slug the-wow-signal-6

Output: debug_audit/final_video_v2/<slug>-v2.mp4 - NEVER
storage/2026/09/<slug>/video.mp4 (that file is never read, written, or
even opened by this module).

After a successful render, main() prints an ffprobe-derived report of the
output (duration, resolution, video codec, audio codec, fps, file size,
and the audio/video duration difference) - see probe_output().
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from core.config import settings
from core.logging_setup import get_logger
from youtube.assemble import _compute_scene_durations, _probe_duration, _run_ffmpeg

logger = get_logger("final_video_assembly_demo")

WIDTH = settings.VIDEO_WIDTH
HEIGHT = settings.VIDEO_HEIGHT
FPS = settings.VIDEO_FPS

# This pass is deliberately scoped to the specific 33-scene approved
# project this task is about - unlike assemble.py (which assembles
# whatever scene count a project has), a scene count that doesn't match
# what was actually visually approved is treated as a hard, loud failure
# rather than "assemble whatever's there".
EXPECTED_SCENE_COUNT = 33

RENDERED_SCENES_ROOT = Path("debug_audit") / "final_visual_render_v2"
OUTPUT_ROOT = Path("debug_audit") / "final_video_v2"

NARRATION_FILENAME = "narration_en.mp3"


class VideoAssemblyError(RuntimeError):
    """Raised for any precondition failure (wrong scene count, a missing or
    wrong-sized scene image, missing narration audio) or an underlying
    ffmpeg/ffprobe failure. Always carries a clear, human-readable reason -
    never raised for anything this module silently papers over."""


@dataclass
class AssemblyResult:
    output_path: Path
    scene_count: int
    narration_duration: float
    scene_durations: list[float] = field(default_factory=list)
    scene_frame_counts: list[int] = field(default_factory=list)


def _find_project_dir(slug: str) -> Path | None:
    """Read-only search under storage/<year>/<month>/<slug> - mirrors
    youtube/final_visual_render_demo.py's own lookup exactly (no DB
    dependency, never writes or modifies anything it finds)."""
    storage_root = settings.STORAGE_DIR
    if not storage_root.exists():
        return None
    matches = sorted(storage_root.glob(f"*/*/{slug}"))
    return matches[0] if matches else None


def _load_scenes_json(project_dir: Path) -> list[dict]:
    scenes_path = project_dir / "scenes.json"
    if not scenes_path.exists():
        raise VideoAssemblyError(f"no scenes.json found at {scenes_path}")
    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        raise VideoAssemblyError(f"{scenes_path} does not contain a JSON list of scenes")
    return sorted(scenes, key=lambda s: s["scene_id"])


def _validate_inputs(scenes: list[dict], rendered_dir: Path, narration_path: Path) -> None:
    """Every precondition this pass requires, checked up front and
    collected into ONE clear error each - never a bare/late KeyError or
    ffmpeg stderr dump standing in for "scene 014's image is missing"."""
    if len(scenes) != EXPECTED_SCENE_COUNT:
        raise VideoAssemblyError(
            f"expected exactly {EXPECTED_SCENE_COUNT} scenes, found {len(scenes)} in scenes.json"
        )

    missing: list[Path] = []
    wrong_size: list[tuple[Path, tuple[int, int]]] = []
    for scene in scenes:
        sid = scene["scene_id"]
        img_path = rendered_dir / f"scene_{sid:03d}.png"
        if not img_path.exists():
            missing.append(img_path)
            continue
        try:
            with Image.open(img_path) as img:
                size = img.size
        except Exception as e:
            raise VideoAssemblyError(f"rendered scene image at {img_path} could not be opened: {e}") from e
        if size != (WIDTH, HEIGHT):
            wrong_size.append((img_path, size))

    if missing:
        raise VideoAssemblyError(
            f"missing {len(missing)} rendered scene image(s) under {rendered_dir}, e.g. {missing[0]}"
        )
    if wrong_size:
        path, size = wrong_size[0]
        raise VideoAssemblyError(
            f"{len(wrong_size)} rendered scene image(s) are not {WIDTH}x{HEIGHT}, "
            f"e.g. {path} is {size[0]}x{size[1]} - run the final visual render pass first"
        )
    if not narration_path.exists():
        raise VideoAssemblyError(f"narration audio not found at {narration_path}")


def _frame_counts_for(durations: list[float], narration_duration: float) -> list[int]:
    """Frame-quantizes each scene's target duration (round(d * FPS),
    minimum 1 frame - identical rounding convention to assemble.py's own
    zoompan frame-count math), then adjusts ONLY the last scene so the
    total always lands on ceil(narration_duration * FPS) - at least as
    long as the narration, never shorter, so mapping the full narration
    track later never truncates it. Deterministic: a pure function of
    `durations` and `narration_duration`, no randomness, no wall-clock
    time."""
    frames = [max(1, round(d * FPS)) for d in durations]
    target_total = max(1, math.ceil(narration_duration * FPS))
    residual = target_total - sum(frames[:-1])
    frames[-1] = max(1, residual)
    return frames


def render_final_video(project_dir: Path, rendered_dir: Path, output_path: Path) -> AssemblyResult:
    """Renders the new reassembled video from `rendered_dir`'s 33 approved
    scene images + `project_dir`'s existing narration_en.mp3/scenes.json.
    `project_dir` (the source project, normally under storage/) is only
    ever READ - nothing is written there, and its video.mp4 (if any) is
    never opened. Writes exclusively to `output_path`. Raises
    VideoAssemblyError on any precondition failure or ffmpeg/ffprobe
    failure - never leaves a partial/incorrect file at output_path (the
    write only happens via ffmpeg's own atomic-on-success file creation;
    a failed ffmpeg run never produces the destination file)."""
    scenes = _load_scenes_json(project_dir)
    narration_path = project_dir / NARRATION_FILENAME
    _validate_inputs(scenes, rendered_dir, narration_path)

    narration_duration = _probe_duration(narration_path)

    # Identical algorithm to production's own crossfade-compensated
    # timing (youtube/assemble.py), called with xfade_duration=0 because
    # this pass concatenates hard cuts - see this module's docstring.
    durations = _compute_scene_durations(scenes, narration_duration, xfade_duration=0)
    frame_counts = _frame_counts_for(durations, narration_duration)

    scene_ids = [s["scene_id"] for s in scenes]
    image_paths = [rendered_dir / f"scene_{sid:03d}.png" for sid in scene_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, (img_path, frames) in enumerate(zip(image_paths, frame_counts)):
        seconds = frames / FPS
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds:.6f}", "-i", str(img_path)]
        # setsar=1 only - no scale/pad here: the 1920x1080 contract is
        # already validated above and owned upstream by
        # youtube/image_normalize.py, not re-applied here.
        filter_parts.append(f"[{i}:v]setsar=1,fps={FPS}[v{i}]")

    concat_labels = "".join(f"[v{i}]" for i in range(len(image_paths)))
    filter_parts.append(f"{concat_labels}concat=n={len(image_paths)}:v=1:a=0[vout]")

    narration_index = len(image_paths)
    inputs += ["-i", str(narration_path)]

    filter_complex = ";".join(filter_parts)

    args = [
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{narration_index}:a",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(
        "[FINAL_VIDEO] assembling %d scenes -> %s (narration=%.2fs, total_frames=%d)",
        len(scenes), output_path.name, narration_duration, sum(frame_counts),
    )
    try:
        _run_ffmpeg(args)
    except RuntimeError as e:
        raise VideoAssemblyError(f"ffmpeg assembly failed: {e}") from e

    if not output_path.exists():
        raise VideoAssemblyError(f"ffmpeg reported success but no output file was written at {output_path}")

    logger.info("[FINAL_VIDEO] wrote %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return AssemblyResult(
        output_path=output_path,
        scene_count=len(scenes),
        narration_duration=narration_duration,
        scene_durations=durations,
        scene_frame_counts=frame_counts,
    )


def probe_output(path: Path) -> dict:
    """ffprobe-based inspection of a rendered output: duration,
    resolution, video codec, audio codec, fps, file size, and the
    audio/video stream duration difference - exactly the report this
    module's CLI prints after a successful render. Raises
    VideoAssemblyError if ffprobe can't read the file or it has no
    video/audio stream (never returns a partial/guessed report)."""
    import subprocess

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise VideoAssemblyError(f"ffprobe could not inspect {path}: {result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video_stream is None:
        raise VideoAssemblyError(f"{path} has no video stream")
    if audio_stream is None:
        raise VideoAssemblyError(f"{path} has no audio stream")

    def _stream_duration(stream: dict) -> float:
        if stream.get("duration") is not None:
            return float(stream["duration"])
        return float(fmt["duration"])  # fallback for containers that only report format-level duration

    video_duration = _stream_duration(video_stream)
    audio_duration = _stream_duration(audio_stream)

    num_str, _, den_str = str(video_stream.get("r_frame_rate", "0/1")).partition("/")
    num, den = float(num_str or 0), float(den_str or 1)
    fps = num / den if den else 0.0

    return {
        "duration": round(float(fmt["duration"]), 3),
        "resolution": f"{video_stream['width']}x{video_stream['height']}",
        "width": video_stream["width"],
        "height": video_stream["height"],
        "video_codec": video_stream["codec_name"],
        "audio_codec": audio_stream["codec_name"],
        "fps": round(fps, 3),
        "file_size_bytes": path.stat().st_size,
        "video_duration": round(video_duration, 3),
        "audio_duration": round(audio_duration, 3),
        "audio_video_duration_diff": round(abs(video_duration - audio_duration), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SAFE FINAL VIDEO REASSEMBLY PASS: reassemble the 33 approved final scene images "
                     "(debug_audit/final_visual_render_v2/<slug>/) with the existing narration and scene "
                     "timing into a new video under debug_audit/final_video_v2/. Never calls Claude/FAL/"
                     "OpenAI, never regenerates images/TTS, never touches storage/ or the existing video.mp4.",
    )
    parser.add_argument("--slug", help="Project slug under storage/<year>/<month>/<slug> (e.g. the-wow-signal-6)")
    parser.add_argument("--project-dir", help="Explicit source project directory (overrides --slug)")
    parser.add_argument("--rendered-dir", default=None,
                         help="Directory with the 33 approved scene_NNN.png files "
                              "(default: debug_audit/final_visual_render_v2/<slug>)")
    parser.add_argument("--out", default=None,
                         help="Output video path (default: debug_audit/final_video_v2/<slug>-v2.mp4) - "
                              "never storage/.../video.mp4")
    args = parser.parse_args(argv)

    if args.project_dir:
        project_dir = Path(args.project_dir)
        slug_name = args.slug or project_dir.name
    elif args.slug:
        project_dir = _find_project_dir(args.slug)
        if project_dir is None:
            print(f"FAILED: no project found for slug={args.slug!r} under {settings.STORAGE_DIR}", file=sys.stderr)
            return 1
        slug_name = args.slug
    else:
        print("FAILED: one of --slug or --project-dir is required", file=sys.stderr)
        return 1

    rendered_dir = Path(args.rendered_dir) if args.rendered_dir else RENDERED_SCENES_ROOT / slug_name
    output_path = Path(args.out) if args.out else OUTPUT_ROOT / f"{slug_name}-v2.mp4"

    logger.info("[FINAL_VIDEO] project=%s rendered=%s -> %s", project_dir, rendered_dir, output_path)
    try:
        result = render_final_video(project_dir, rendered_dir, output_path)
        report = probe_output(result.output_path)
    except VideoAssemblyError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print(
        f"OK - assembled {result.scene_count} scene(s), narration={result.narration_duration:.2f}s "
        f"-> {result.output_path}"
    )
    print(
        "duration={duration}s resolution={resolution} video_codec={video_codec} audio_codec={audio_codec} "
        "fps={fps} file_size_bytes={file_size_bytes} audio_video_duration_diff={audio_video_duration_diff}s".format(
            **report
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
