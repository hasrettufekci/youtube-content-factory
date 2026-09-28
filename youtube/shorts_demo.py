"""
Shorts Engine V1 - safe local/demo CLI.

    python -m youtube.shorts_demo --video-id 1              # plan only (no LLM/FAL/TTS calls, no cost)
    python -m youtube.shorts_demo --video-id 1 --produce     # full production

--video-id alone is READ-ONLY and requires no paid API at all: it only
calls youtube.shorts_planner.plan_short_angles() (a pure function - no
LLM call, no network) against an existing project's already-produced
script_en.json, and prints which Long section feeds each Short - useful
for sanity-checking the angle-selection logic against a real script with
zero API cost.

--produce actually runs youtube.shorts_engine.produce_shorts_for_video()
(LLM condensation + image reuse/generation + TTS + ffmpeg assembly + QC) -
this DOES call whatever providers are configured
(IMAGE_PROVIDER/TTS_PROVIDER/ANTHROPIC_API_KEY), so it is only free of
paid-API cost when those are set to mock/unset in the environment it runs
in (e.g. IMAGE_PROVIDER=mock TTS_PROVIDER=mock). It NEVER touches Telegram
or YouTube - it stops after production + QC and prints each Short's own
result; sending the package for approval and uploading remain
pipeline.py's/youtube.telegram_approval.py's job.
"""
from __future__ import annotations

import argparse
import sys

from storage import project_store as ps
from youtube import shorts_planner


def plan_only(video_id: int) -> list[dict]:
    paths = ps.get_project(video_id)
    if not paths:
        raise RuntimeError(f"no project with video_id={video_id}")
    script = ps.load_json(paths.script_en_json)
    if not script:
        raise RuntimeError(f"no script_en.json found for video_id={video_id} - run the Long pipeline first")
    return shorts_planner.plan_short_angles(script)


def produce(video_id: int) -> list[dict]:
    from youtube import shorts_engine

    paths = ps.get_project(video_id)
    if not paths:
        raise RuntimeError(f"no project with video_id={video_id}")
    script = ps.load_json(paths.script_en_json)
    scenes = ps.load_json(paths.scenes_json)
    if not script or not scenes:
        raise RuntimeError(f"no script_en.json/scenes.json found for video_id={video_id} - run the Long pipeline first")
    return shorts_engine.produce_shorts_for_video(paths, script, scenes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Shorts Engine V1 - safe local/demo CLI (plan-only is read-only and free)."
    )
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument(
        "--produce", action="store_true",
        help="Actually run full Shorts production (uses configured providers) instead of just planning angles",
    )
    args = parser.parse_args(argv)

    try:
        if args.produce:
            results = produce(args.video_id)
            for r in results:
                print(f"short_number={r['short_number']} status={r['status']} title={r.get('title')!r}")
        else:
            plans = plan_only(args.video_id)
            for p in plans:
                print(f"short_number={p['short_number']} angle={p['angle']} <- section {p['section_type']!r}: {p['section_title']!r}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
