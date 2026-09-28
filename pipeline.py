"""
YOUTUBE_CONTENT_FACTORY - main pipeline orchestrator (spec sections 26, 43;
extended for multilingual production in Round 3 section 12).

    TOPIC -> RESEARCH -> ENGLISH MASTER SCRIPT -> [LOCALIZATION per
    configured language] -> SHARED SCENE PLAN -> SHARED AI VISUALS ->
    [NARRATION per configured language] -> VIDEO -> THUMBNAIL ->
    [METADATA per configured language] -> QUALITY CHECK -> TELEGRAM
    APPROVAL -> (human approves) -> YOUTUBE UPLOAD

ONE research, ONE master script, ONE shared scene plan, ONE shared set of
AI-generated scene images (never regenerated or duplicated per language -
spec Round 3 section 2/21); localization/narration/metadata each run once
per language in core.config.settings.LANGUAGE_LOCALIZATIONS, in addition to
the master language. Adding a future localization language is a one-line
config change (LANGUAGE_LOCALIZATIONS in .env) - it automatically adds that
language's localization/narration/metadata stages below with zero code
changes, and never re-triggers research or visual generation.

Resumable: progress is derived from which artifact files already exist on
disk (storage/<slug>/research.json, script_en.json, ...), not just the
state label, so re-running the same video after a mid-pipeline failure
re-does ONLY the stage that failed (spec section 25/26) - nothing upstream
is recomputed, nothing downstream is skipped. This applies per-language
too: a video with script_en.json/scenes.json/visuals/narration_en.mp3
already done but script_tr.json missing resumes straight into the Turkish
localization stage, never re-researching, never rebuilding the script or
scenes, never regenerating a single image. That resume also covers a
narrower edge case (spec Round 3.1): if scenes.json/visuals were already
built BEFORE a language's script_<lang>.json existed, a scene_sync_<lang>
stage (scene_planner.sync_localized_narrations - a small, deterministic
enrichment of the EXISTING scenes.json, never a replan) fills in that
language's narration_<lang> on the shared scenes without touching
scene_id, visual_prompt, or a single image file.

Usage:
    python pipeline.py --topic "The Dyatlov Pass Incident" --category Mysteries
    python pipeline.py --video-id 3                  # resume/retry a video
    python pipeline.py --approve 3                    # approve + upload (normally done via Telegram button)
    python pipeline.py --reject 3
    python pipeline.py --regenerate 3
"""
from __future__ import annotations

import argparse
import sys

from core.config import settings
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import researcher, script_generator, localizer, scene_planner, image_gen, tts, assemble, thumbnail, metadata, quality_check

logger = get_logger("pipeline")


class PipelineFailed(RuntimeError):
    pass


def _localization_languages(languages: list[str] | None = None) -> list[str]:
    """Configured localization languages, excluding the master language
    itself in case of a config mistake (LANGUAGE_MASTER accidentally also
    listed in LANGUAGE_LOCALIZATIONS).

    `languages`, when given, overrides settings.LANGUAGE_LOCALIZATIONS for
    THIS call only - it never mutates the global config (settings stays
    exactly as the user's .env configured it, so every other caller/test
    in the process keeps seeing the real default). This is how
    produce_video(..., languages=[]) below runs a single English-only
    production without touching LANGUAGE_LOCALIZATIONS globally or adding
    any new localization architecture - it's the exact same
    _stage_plan()/per-language-loop mechanism the multilingual pipeline
    already uses, just fed an explicit (possibly empty) list instead of
    reading settings directly."""
    source = languages if languages is not None else settings.LANGUAGE_LOCALIZATIONS
    return [lang for lang in source if lang != settings.LANGUAGE_MASTER]


def _stage_plan(paths: ps.ProjectPaths, topic: str, category: str | None, languages: list[str] | None = None):
    """Ordered (name, is_done, run) triples. `is_done` is checked against
    what's actually on disk, which is what makes resume/retry safe.

    The English-master stages (research/script/scenes/images/thumbnail) are
    fixed; a localization/narration/metadata triple is appended once per
    language in `languages` (settings.LANGUAGE_LOCALIZATIONS by default,
    unless produce_video() was given an explicit override - see
    _localization_languages() above), so the plan's length and contents
    adapt to config without any code change here."""
    localization_langs = _localization_languages(languages)
    stages = [
        ("research", lambda: paths.research_json.exists(),
         lambda: researcher.run_and_save(paths, topic, category=category)),
        ("script_en", lambda: paths.script_en_json.exists(),
         lambda: script_generator.run_and_save(paths)),
    ]
    for lang in localization_langs:
        stages.append((
            f"localization_{lang}",
            (lambda lang=lang: paths.script_localized_json(lang).exists()),
            (lambda lang=lang: localizer.run_and_save(paths, language=lang)),
        ))
    stages += [
        ("scenes", lambda: paths.scenes_json.exists(),
         lambda: scene_planner.run_and_save_v2(paths)),
        ("images", lambda: _images_done(paths),
         lambda: image_gen.run_and_save(paths)),
    ]
    for lang in localization_langs:
        # Resume edge case (spec Round 3.1): scenes.json/visuals may already
        # have existed BEFORE this language's script_<lang>.json did (e.g.
        # an earlier run stopped before that localization finished), in
        # which case plan_scenes() never had that language's text to
        # attach narration_<lang> with. This stage is a small, deterministic
        # enrichment of the EXISTING scenes.json only - it never replans
        # scenes, never touches scene_id/visual_prompt, and never
        # regenerates a single image. In the normal forward-flow case
        # (localization already ran before "scenes" above), scene_planner
        # already attached narration_<lang> at planning time, so
        # _scenes_synced_for() is already true here and this is a no-op.
        stages.append((
            f"scene_sync_{lang}",
            (lambda lang=lang: _scenes_synced_for(paths, lang)),
            (lambda lang=lang: scene_planner.sync_localized_narrations(paths, language=lang)),
        ))
    stages.append((
        f"narration_{settings.LANGUAGE_MASTER}", lambda: paths.narration_mp3_for(settings.LANGUAGE_MASTER).exists(),
        lambda: tts.run_and_save(paths, language=settings.LANGUAGE_MASTER),
    ))
    for lang in localization_langs:
        stages.append((
            f"narration_{lang}",
            (lambda lang=lang: paths.narration_mp3_for(lang).exists()),
            (lambda lang=lang: tts.run_and_save(paths, language=lang)),
        ))
    stages += [
        ("video_assembly", lambda: paths.video_mp4.exists(),
         lambda: assemble.run_and_save(paths)),
        ("thumbnail", lambda: paths.thumbnail_png.exists(),
         lambda: thumbnail.run_and_save(paths)),
        (f"metadata_{settings.LANGUAGE_MASTER}", lambda: paths.metadata_json_for(settings.LANGUAGE_MASTER).exists(),
         lambda: metadata.run_and_save(paths, category=category, language=settings.LANGUAGE_MASTER)),
    ]
    for lang in localization_langs:
        stages.append((
            f"metadata_{lang}",
            (lambda lang=lang: paths.metadata_json_for(lang).exists()),
            (lambda lang=lang: metadata.run_and_save(paths, category=category, language=lang)),
        ))
    return stages


def _images_done(paths: ps.ProjectPaths) -> bool:
    scenes = ps.load_json(paths.scenes_json)
    if not scenes:
        return False
    return len(paths.scene_images()) == len(scenes)


def _scenes_synced_for(paths: ps.ProjectPaths, language: str) -> bool:
    """Whether the shared scenes.json already carries narration_<language>
    on every scene - true either because plan_scenes() attached it at
    planning time (the normal forward-flow case) or because a prior
    scene_sync_<language> run already enriched it (spec Round 3.1)."""
    scenes = ps.load_json(paths.scenes_json)
    if not scenes:
        return False
    key = f"narration_{language}"
    return all(key in scene for scene in scenes)


def produce_video(
    topic: str,
    category: str | None = None,
    project_id: str | None = None,
    video_id: int | None = None,
    force_new: bool = False,
    background_music: bool = False,
    languages: list[str] | None = None,
) -> ps.ProjectPaths:
    """`languages`: per-call localization override (see
    _localization_languages() above) - e.g. `languages=[]` for an
    English-only run. `None` (the default) means "use
    settings.LANGUAGE_LOCALIZATIONS exactly as configured", so every
    existing caller is unaffected."""
    if video_id is not None:
        paths = ps.get_project(video_id)
        if paths is None:
            raise PipelineFailed(f"[PIPELINE] no project with video_id={video_id}")
        topic = db.get_video(video_id)["topic"]
    else:
        paths, created = ps.create_project(topic, category=category, project_id=project_id, force_new=force_new)
        logger.info("[PIPELINE] video_id=%d slug=%s (%s)", paths.video_id, paths.slug,
                    "new" if created else "reusing existing video for this topic (idempotency)")

    for name, is_done, run in _stage_plan(paths, topic, category, languages=languages):
        if is_done():
            logger.info("[PIPELINE] [%s] already done, skipping (resume)", name)
            continue
        logger.info("[PIPELINE] [%s] running...", name)
        try:
            run()
        except Exception as e:
            logger.error("[PIPELINE] [%s] FAILED: %s", name, e)
            raise PipelineFailed(f"[{name}] {e}") from e

    # Unlike every stage in _stage_plan() above, quality_check isn't run
    # through that loop's try/except - so without its own handling here, an
    # unexpected crash inside run_and_enforce (as opposed to a normal
    # "some checks failed" report) would propagate straight out of
    # produce_video() uncaught: never recorded as FAILED in state.json,
    # never reported to Telegram, and - since every upstream artifact file
    # already exists on disk - silently un-resumable (every future retry
    # would hit the exact same uncaught crash forever). Treat a
    # quality_check crash exactly like any other stage failure.
    try:
        # Same `languages` override _stage_plan() above was given (e.g.
        # languages=[] for --english-only) - QC must check exactly the
        # artifacts this run actually requested/generated, never more.
        report = quality_check.run_and_enforce(paths, languages=languages)
    except Exception as e:
        logger.error("[PIPELINE] [quality_check] FAILED: %s", e)
        ps.save_failure(paths, "quality_check", str(e))
        raise PipelineFailed(f"[quality_check] {e}") from e
    if not report.passed:
        raise PipelineFailed(f"[quality_check] {'; '.join(report.errors)}")

    ps.save_state(paths, PipelineState.PENDING_APPROVAL)
    logger.info("[PIPELINE] video_id=%d ready for approval: %s", paths.video_id, paths.video_mp4)

    # Shorts Engine V1 (additive, opt-in via settings.SHORTS_ENABLED): for
    # every selected Long topic, also produce SHORTS_PER_LONG Shorts from
    # the SAME research/script - no second research call, no separate
    # Shorts topic selection. When disabled (the default), _produce_shorts()
    # is never called and _notify_telegram() sends the exact original
    # single-video message, byte-for-byte, as before this feature existed.
    shorts_results = _produce_shorts(paths) if settings.SHORTS_ENABLED else None

    _notify_telegram(paths, shorts_results=shorts_results)
    return paths


def _produce_shorts(paths: ps.ProjectPaths) -> list[dict] | None:
    """Runs youtube.shorts_engine.produce_shorts_for_video() against this
    Long's own already-produced script_en.json/scenes.json. Any failure
    here (including a missing script/scenes, which should never happen at
    this point in produce_video()) is logged and swallowed, never allowed
    to fail the Long's own already-successful production - the Long is
    already fully produced and QC-passed by the time this runs (spec: a
    Shorts failure must never lose the Long's own approval). Returns None
    on any failure (including "SHORTS_ENABLED but nothing could be
    produced") - _notify_telegram() below treats None as "no Shorts info
    available" and falls back to the original single-video notification,
    same as before this feature existed, rather than silently sending
    a package with no Shorts in it."""
    try:
        from youtube import shorts_engine
        script = ps.load_json(paths.script_en_json)
        scenes = ps.load_json(paths.scenes_json)
        if not script or not scenes:
            logger.warning("[PIPELINE] [shorts] skipped - script_en.json/scenes.json not found for video_id=%d",
                            paths.video_id)
            return None
        return shorts_engine.produce_shorts_for_video(paths, script, scenes)
    except Exception:
        logger.exception("[PIPELINE] [shorts] production failed unexpectedly for video_id=%d", paths.video_id)
        return None


def _notify_telegram(paths: ps.ProjectPaths, shorts_results: list[dict] | None = None) -> None:
    """The production pipeline's ONE approval notification - Telegram
    Approval V1 (youtube/telegram_approval.py), never the legacy
    telegram_bot.py video_id-based flow. Passes the pipeline's own real,
    just-produced artifacts (paths.video_mp4 / paths.thumbnail_png -
    storage/<slug>/video.mp4 + thumbnail.png, the actual production
    output) straight through to build_approval_payload()'s explicit
    video_path/thumbnail_path override, so Approval V1 never falls back to
    its debug_audit/ demo-tool convention for a pipeline-produced video.
    telegram_bot.py itself is completely untouched - it is simply no
    longer called from here, so a video is never notified twice.

    Same safety contract as the function this replaced: any failure here
    (missing Telegram config, a missing artifact, a network error) is
    logged and swallowed, never allowed to fail the pipeline run itself -
    the video is already fully produced and sitting in PENDING_APPROVAL by
    the time this runs, so a notification failure just means "send the
    approval manually later" (youtube/telegram_approval_demo.py), not
    "the whole run failed".

    Shorts Engine V1 (additive): `shorts_results` is None for every
    pre-existing caller (SHORTS_ENABLED disabled, or nothing to report) -
    in that case this sends the EXACT original single-video message,
    byte-for-byte, exactly as before this feature existed. When Shorts
    were produced, this sends ONE combined package approval (spec
    CRITICAL: only one approval, thumbnails only, no video sent to
    Telegram) if every Short passed QC, or a plain failure notice with NO
    approve/reject button at all if any did not (spec: "Do not silently
    approve a partial package")."""
    try:
        from youtube import telegram_approval as tap

        if shorts_results is not None:
            from youtube import shorts_engine
            from storage import shorts as shorts_store

            if shorts_engine.all_shorts_ready(shorts_results):
                shorts_rows = shorts_store.list_shorts_for_parent(paths.video_id)
                metadata = ps.load_json(paths.metadata_en_json) or {}
                long_title = metadata.get("title") or paths.slug
                payload = tap.build_package_approval_payload(
                    paths.slug, long_title, paths.video_id,
                    long_video_path=paths.video_mp4, long_thumbnail_path=paths.thumbnail_png,
                    shorts_rows=shorts_rows,
                )
                tap.send_new_package_approval(payload)
            else:
                metadata = ps.load_json(paths.metadata_en_json) or {}
                long_title = metadata.get("title") or paths.slug
                tap.send_shorts_production_failed_notice(long_title, paths.slug, shorts_results)
            return

        payload = tap.build_approval_payload(
            paths.slug, project_dir=paths.root,
            video_path=paths.video_mp4, thumbnail_path=paths.thumbnail_png,
        )
        tap.send_new_approval(payload)
    except RuntimeError as e:
        logger.warning("[TELEGRAM] approval notification skipped: %s", e)
    except Exception:
        logger.exception("[TELEGRAM] failed to send approval notification")


def handle_approve(video_id: int) -> dict:
    from youtube import uploader

    paths = ps.get_project(video_id)
    if not paths:
        raise PipelineFailed(f"[PIPELINE] no project with video_id={video_id}")
    state = ps.load_state(paths)

    # Checked BEFORE touching state: if we're already published, don't even
    # move the state to APPROVED - that would overwrite a correct PUBLISHED
    # state.json with a stale one while still refusing (via uploader.py's own
    # guard) to actually re-upload, corrupting the recorded history for no
    # reason. Same video_id approved twice (double tap, duplicate Telegram
    # callback, a stale button on an old message) must never cause a second
    # YouTube upload.
    if state == PipelineState.PUBLISHED:
        row = db.get_video(video_id)
        logger.warning("[PIPELINE] video_id=%d already PUBLISHED (%s) - ignoring duplicate approval",
                        video_id, row["youtube_url"] if row else "unknown url")
        return {
            "video_id": row["youtube_video_id"] if row else None,
            "url": row["youtube_url"] if row else None,
            "already_published": True,
        }
    if state == PipelineState.UPLOADING:
        raise PipelineFailed(f"[PIPELINE] video_id={video_id} is already uploading - wait for it to finish")

    if state not in (PipelineState.PENDING_APPROVAL,):
        logger.warning("[PIPELINE] video_id=%d not in PENDING_APPROVAL (state=%s) - approving anyway", video_id, state)
    ps.save_state(paths, PipelineState.APPROVED)
    return uploader.run_and_save(paths)


def handle_reject(video_id: int, reason: str = "") -> None:
    paths = ps.get_project(video_id)
    if not paths:
        raise PipelineFailed(f"[PIPELINE] no project with video_id={video_id}")
    ps.save_failure(paths, "telegram_approval", f"rejected by reviewer{': ' + reason if reason else ''}")
    logger.info("[PIPELINE] video_id=%d rejected", video_id)


def handle_regenerate(video_id: int) -> ps.ProjectPaths:
    row = db.get_video(video_id)
    if not row:
        raise PipelineFailed(f"[PIPELINE] no project with video_id={video_id}")
    handle_reject(video_id, reason="regenerate requested")
    return produce_video(row["topic"], category=row["category"], project_id=row["project_id"], force_new=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Pure CLI-surface construction, deliberately split out of main() so it
    can be exercised on its own (e.g. parser.parse_args([...])) without ever
    reaching the try/except execution block below - no pipeline stage, no
    produce_video call, no side effect of any kind. main() is the only
    caller in production code; tests are free to call this directly."""
    parser = argparse.ArgumentParser(description="Run the YOUTUBE_CONTENT_FACTORY pipeline.")
    parser.add_argument("--topic", help="New topic to produce")
    parser.add_argument("--category", default=None)
    parser.add_argument("--project", default=None, help="Project/brand config id (defaults to core.config.DEFAULT_PROJECT_ID)")
    parser.add_argument("--video-id", type=int, help="Resume/retry an existing video")
    parser.add_argument("--force-new", action="store_true", help="Create a new video even if this topic was already produced")
    parser.add_argument("--english-only", action="store_true",
                         help="Skip every configured localization language for THIS run only - "
                              "does not change LANGUAGE_LOCALIZATIONS globally")
    parser.add_argument("--music", action="store_true")
    parser.add_argument("--approve", type=int, metavar="VIDEO_ID")
    parser.add_argument("--reject", type=int, metavar="VIDEO_ID")
    parser.add_argument("--regenerate", type=int, metavar="VIDEO_ID")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.approve:
            result = handle_approve(args.approve)
            print(f"OK - published: {result['url']}")
        elif args.reject:
            handle_reject(args.reject)
            print("OK - rejected")
        elif args.regenerate:
            paths = handle_regenerate(args.regenerate)
            print(f"OK - regenerated as video_id={paths.video_id}")
        elif args.video_id:
            languages = [] if args.english_only else None
            paths = produce_video("", video_id=args.video_id, background_music=args.music, languages=languages)
            print(f"OK - video_id={paths.video_id} pending approval: {paths.video_mp4}")
        elif args.topic:
            languages = [] if args.english_only else None
            paths = produce_video(
                args.topic, category=args.category, project_id=args.project,
                force_new=args.force_new, background_music=args.music, languages=languages,
            )
            print(f"OK - video_id={paths.video_id} pending approval: {paths.video_mp4}")
        else:
            parser.print_help()
            return 1
        return 0
    except PipelineFailed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
