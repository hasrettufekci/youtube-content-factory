"""
Shorts Engine V1 - production orchestration.

For a Long video that already has a script (script_en.json) and a shared
scene plan (scenes.json) - i.e. it has already run the normal pipeline.py
stages up through "scenes"/"images" - this module builds 1 Short per
configured angle (SHORTS_PER_LONG, default 3): a condensed narration
(youtube.shorts_planner), a vertical (9:16) visual + video assembly
(youtube.shorts_visual + youtube.assemble, reused with vertical overrides),
narration audio (youtube.tts, reused with a dest_path override), a vertical
thumbnail (youtube.thumbnail, reused with width/height overrides), and its
own QC pass (youtube.shorts_qc) - each persisted as its own row in
storage/shorts.py, linked to the Long via parent_video_id.

RESILIENCE (spec: "do not allow one Short failure to leave an inconsistent
approval package"): each Short's production is wrapped independently - one
Short raising (a FAL/TTS/ffmpeg hiccup) never stops the other Shorts from
being attempted, and is recorded as storage.shorts.STATUS_FAILED with the
error, not a crash. The CALLER (pipeline.py's _notify_telegram()) is what
decides that ANY failed/non-ready Short blocks the whole package from being
sent as approve-ready - this module only reports each Short's own outcome.

Called only when settings.SHORTS_ENABLED (see pipeline.py) - when disabled,
nothing in this module is ever imported or invoked, so the Long-only
pipeline is 100% unaffected.
"""
from __future__ import annotations

from core.config import settings
from core.logging_setup import get_logger
from storage import project_store as ps
from storage import shorts as shorts_store
from youtube import assemble, shorts_planner, shorts_qc, shorts_visual, thumbnail, tts

logger = get_logger("shorts_engine")


def _short_duration_seconds(video_path) -> float | None:
    from youtube.quality_check import _ffprobe_json
    try:
        probe = _ffprobe_json(video_path)
        return float(probe.get("format", {}).get("duration", 0)) or None
    except Exception:
        return None


def _produce_one_short(paths: ps.ProjectPaths, script: dict, scenes: list[dict], plan_entry: dict) -> dict:
    """Builds and QCs ONE Short end to end, persists its result to
    storage/shorts.py, and returns a small summary dict. Never raises -
    any failure is caught, recorded as STATUS_FAILED with the error, and
    reported back in the summary (see module docstring - RESILIENCE)."""
    n = plan_entry["short_number"]
    width, height = settings.SHORTS_WIDTH, settings.SHORTS_HEIGHT

    short_id = shorts_store.create_short(
        parent_video_id=paths.video_id, short_number=n,
        status=shorts_store.STATUS_BUILDING, angle=plan_entry["angle"],
    )

    # Resumability: a pipeline run may be re-invoked for the same video_id
    # (pipeline.py's own resume-by-video_id path) after a Short has already
    # progressed past production (approved/uploading/uploaded) - never
    # silently reproduce/overwrite its artifacts or waste FAL/TTS cost in
    # that case. BUILDING/FAILED/REJECTED are all still (re)produced -
    # BUILDING/FAILED are the normal "not done yet or needs a retry" cases,
    # and a REJECTED row only exists after a real human decision on an
    # already-sent package (force_new production paths always get a fresh
    # video_id/short rows, so this is a defensive edge case, not a normal
    # flow).
    existing = shorts_store.get_short(short_id)
    if existing is not None and existing["status"] in (
        shorts_store.STATUS_READY, shorts_store.STATUS_APPROVED,
        shorts_store.STATUS_UPLOADING, shorts_store.STATUS_UPLOADED,
    ):
        logger.info("[SHORTS_ENGINE] short_number=%d already %s - skipping re-production", n, existing["status"])
        return {
            "short_id": short_id, "short_number": n, "status": existing["status"],
            "title": existing["title"], "qc_passed": bool(existing["qc_passed"]), "qc_errors": [],
        }

    try:
        title = plan_entry["title"]
        narration_text = plan_entry["narration"]
        slug = ps.slugify(title)

        # 1. visuals - reuse-or-regenerate per the crop-suitability heuristic
        selected_scenes = shorts_visual.select_scenes_for_section(scenes, script, plan_entry["section_type"])
        visual_paths = []
        for i, scene in enumerate(selected_scenes):
            dest = paths.short_dir() / f"short_{n:02d}_visual_{i:02d}.png"
            shorts_visual.build_vertical_visual(paths, scene, dest, width=width, height=height)
            visual_paths.append(dest)
        scene_dicts_for_assembly = [{"duration_hint": s.get("duration_hint") or 3.0} for s in selected_scenes]

        # 2. narration audio - existing TTS architecture/voice, redirected via dest_path
        narration_path = tts.generate_narration(
            paths, narration_text, language=settings.LANGUAGE_MASTER,
            dest_path=paths.short_narration_path(n),
        )

        # 3. vertical video assembly - existing ffmpeg pipeline, vertical overrides
        video_path = assemble.assemble_video(
            paths, scene_dicts_for_assembly, background_music=False, language=settings.LANGUAGE_MASTER,
            width=width, height=height, scene_image_paths=visual_paths,
            narration_path=narration_path, dest_path=paths.short_video_path(n),
        )

        # 4. vertical thumbnail - existing LLM text + Pillow composer, vertical overrides
        hook_text = narration_text.split(". ")[0][:200]
        thumb_text = thumbnail.generate_thumbnail_text(title, hook_text, language=settings.LANGUAGE_MASTER)
        thumbnail_path = thumbnail.compose_thumbnail(
            visual_paths[0], thumb_text, paths.short_thumbnail_path(n), width=width, height=height,
        )

        duration_seconds = _short_duration_seconds(video_path)

        # 5. own QC pass - blocks package approval if it fails (caller's job)
        qc_report = shorts_qc.run_short_quality_check(
            video_path, thumbnail_path, narration_text, title,
            source_text=plan_entry.get("source_text", plan_entry["narration"]),
            width=width, height=height,
        )

        if qc_report.passed:
            shorts_store.update_fields(
                short_id, status=shorts_store.STATUS_READY, title=title, slug=slug, script=narration_text,
                video_path=str(video_path), thumbnail_path=str(thumbnail_path),
                narration_path=str(narration_path), duration_seconds=duration_seconds,
                qc_passed=1, qc_errors=None, error=None,
            )
            logger.info("[SHORTS_ENGINE] short_number=%d READY (%s)", n, title)
        else:
            error_summary = "; ".join(qc_report.errors)[:2000]
            shorts_store.update_fields(
                short_id, status=shorts_store.STATUS_FAILED, title=title, slug=slug, script=narration_text,
                video_path=str(video_path), thumbnail_path=str(thumbnail_path),
                narration_path=str(narration_path), duration_seconds=duration_seconds,
                qc_passed=0, qc_errors=error_summary, error=error_summary,
            )
            logger.warning("[SHORTS_ENGINE] short_number=%d FAILED QC: %s", n, error_summary)

        return {
            "short_id": short_id, "short_number": n, "status": shorts_store.get_short(short_id)["status"],
            "title": title, "qc_passed": qc_report.passed, "qc_errors": qc_report.errors,
        }
    except Exception as e:
        logger.exception("[SHORTS_ENGINE] short_number=%d production FAILED: %s", n, e)
        shorts_store.update_fields(short_id, status=shorts_store.STATUS_FAILED, error=str(e)[:2000])
        return {
            "short_id": short_id, "short_number": n, "status": shorts_store.STATUS_FAILED,
            "title": plan_entry.get("title"), "qc_passed": False, "qc_errors": [str(e)[:500]],
        }


def produce_shorts_for_video(paths: ps.ProjectPaths, script: dict, scenes: list[dict],
                              shorts_per_long: int | None = None) -> list[dict]:
    """Produces every configured Short for this Long (angle selection ->
    per-short script -> visuals -> narration -> assembly -> thumbnail ->
    QC), persisting each as its own storage/shorts.py row. Returns a
    summary list, one dict per Short, in short_number order. Never raises -
    see _produce_one_short()'s own docstring."""
    plans = shorts_planner.plan_and_generate_shorts(script, shorts_per_long=shorts_per_long)
    results = [_produce_one_short(paths, script, scenes, plan) for plan in plans]
    logger.info("[SHORTS_ENGINE] video_id=%d produced %d short(s): %s",
                paths.video_id, len(results), [(r["short_number"], r["status"]) for r in results])
    return results


def all_shorts_ready(results: list[dict]) -> bool:
    """True only if every Short in `results` reached STATUS_READY - the
    gate pipeline.py's _notify_telegram() uses to decide whether the
    package may be sent as approve-ready (spec: "If ANY Short fails QC: DO
    NOT send the package as APPROVE-ready")."""
    return bool(results) and all(r["status"] == shorts_store.STATUS_READY for r in results)
