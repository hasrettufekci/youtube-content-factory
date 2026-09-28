"""
Scene planning (spec section 15).

Splits a finished script's narration into 25-40 timed scenes and asks
Claude for a cinematic, non-repeating visual prompt per scene (in a single
batched call, not one call per scene - cheaper and keeps narration/timing
fully deterministic since that part is computed in code, not by the LLM).

Scene shape (Round 3: ONE shared scene plan for every language - see
module-level note below):
    {
      "scene_id": 1,
      "duration_hint": 12.4,
      "narration_en": "...",
      "narration_tr": "...",        # present once that language's script is available
      "visual_prompt": "...",        # language-independent - shared by every language
      "visual_type": "cinematic"
    }

Round 3 multilingual note: scenes/visuals are planned ONCE, from the
English master narration only (chunk boundaries + visual prompts are
derived purely from script_en.json) - this is what "ONE shared visual
production" (spec Round 3 section 2) means concretely: image_gen.py never
even knows a second language exists. Each configured localization
language's own full_narration (if its script_<lang>.json already exists at
planning time) is independently split into the SAME number of chunks using
the same chunking algorithm, and attached to each scene as narration_<lang>
- so scene N's narration_tr corresponds to roughly the same point in the
story as scene N's narration_en (both preserve chronological/story-
structure order, per the localization stage's own rules), without
requiring any per-sentence translation alignment. A language whose script
isn't available yet simply has no narration_<lang> on any scene until a
resume run adds it - no visual replanning, no new images. That "a resume
run adds it" step is `sync_localized_narrations()` below (Round 3.1): a
small, deterministic enrichment of the EXISTING scenes.json for exactly
this case - scenes.json/visuals already existed when that language's
script_<lang>.json later became available, so plan_scenes() never had a
chance to attach it. It reuses the same chunking helper, never replans,
never touches scene_id/visual_prompt, and never regenerates an image.

Scene count is never a hard-coded constant: it's derived from narration word
count via `_target_scene_count()`, and MIN_SCENES/MAX_SCENES/
WORDS_PER_SCENE_TARGET are read from core.config.settings, which is itself
env-driven (SCENE_MIN_COUNT, SCENE_MAX_COUNT, SCENE_WORDS_PER_SCENE_TARGET) -
see core/config.py. Chunking splits at sentence boundaries first; if the
script's sentences are too few/long to reach the target on their own (a
narration written in long, complex sentences has fewer natural cut points
than one written in short spoken-style sentences - see
`_expand_with_clause_breaks`), long sentences are further broken at clause
boundaries (commas/semicolons/colons) so the target is still reachable
without ever splitting a chunk mid-clause.

Standalone test:
    python -m youtube.scene_planner --video-id 1

Scene Planner V2 Phase 2 (deterministic prompt builder - see the "Phase 2"
section far below, after plan_scenes_v2()): takes Phase 1's classification
output (claim/scene_type/evidence_level/data_points/visual_relevance_reason/
prompt_segments) and deterministically builds visual_prompt/avoid_strategy/
accuracy_hints/overlay_required/render_hash in pure Python - no LLM call, no
FAL call, no image generation. Not wired into run_and_save()/pipeline.py yet
(that remains later-phase scope).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.config import PROJECT_ROOT, settings
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import llm_client

logger = get_logger("scenes")

# Sourced from config (env-overridable) rather than hard-coded, per spec
# section 15/37 - kept as plain module attributes (not read inline from
# `settings` everywhere) so existing tests can monkeypatch them directly.
MIN_SCENES = settings.SCENE_MIN_COUNT
MAX_SCENES = settings.SCENE_MAX_COUNT
WORDS_PER_SCENE_TARGET = settings.SCENE_WORDS_PER_SCENE_TARGET
WORDS_PER_SECOND = settings.SCENE_WORDS_PER_SECOND

VISUAL_TYPES = ["cinematic", "wide", "closeup", "archival", "map", "portrait", "abstract"]

SYSTEM_PROMPT = """You are a documentary cinematographer/storyboard artist. You will be given
a numbered list of narration chunks, in order, from a finished documentary script, plus the
video's overall visual style. For EACH numbered chunk, write one AI-image-generation visual
prompt that illustrates that exact moment of narration.

Rules:
- Each visual_prompt must be a self-contained, detailed English prompt (composition, subject,
  setting, lighting, mood) - the image model sees each scene independently, with no memory of
  other scenes, so do not write things like "same as before".
- Maintain one consistent visual language (art direction, lighting, color grade) across ALL
  scenes, as described in "visual_style" below, so the finished video feels like one coherent
  film rather than unrelated images.
- Consecutive scenes must not be visually repetitive - vary composition, camera distance,
  angle, or subject focus even when the narration stays on the same subject.
- Never include on-screen text, letters, numbers, captions, subtitles, watermarks or logos
  in the image itself.
- 16:9 cinematic documentary framing.
- Pick a visual_type for each scene from exactly this list: cinematic, wide, closeup,
  archival, map, portrait, abstract - archival for anything about verifiable historical
  documents/photos/records, map for geography/location scenes, portrait for scenes centered
  on a specific person, abstract only for genuinely unknown/conceptual/mysterious moments.

Return ONLY a JSON array (no markdown, no commentary) with exactly one object per input
chunk, in the same order, each shaped:
{"scene_id": <int>, "visual_prompt": "...", "visual_type": "<one of the allowed types>"}"""


def _split_sentences(full_narration: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", full_narration.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_clauses(sentence: str) -> list[str]:
    """Break one sentence at clause boundaries (after a comma/semicolon/colon
    followed by whitespace). Purely whitespace-based - never drops or
    reorders a character, so re-joining clauses always reconstructs the
    original sentence exactly."""
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    return [p.strip() for p in parts if p.strip()]


def _expand_with_clause_breaks(sentences: list[str], target_count: int) -> list[str]:
    """Used only when there aren't enough sentences to reach target_count on
    their own (root cause of scene counts landing below MIN_SCENES for
    narration written in long, complex sentences - see module docstring):
    break sentences into clauses to create more natural cut points, without
    ever cutting mid-clause or losing/reordering words."""
    units: list[str] = []
    for sentence in sentences:
        clauses = _split_clauses(sentence)
        units.extend(clauses if len(clauses) > 1 else [sentence])
    return units


def _group_units_into_chunks(units: list[str], target_count: int) -> list[str]:
    """Greedily group narration units (sentences, or finer clause-level units
    when needed) into `target_count` chunks, picking each chunk boundary at
    whichever point is numerically CLOSEST to target_words_per_chunk - not
    just the first point that reaches it. A naive "stop as soon as we hit
    the target" rule systematically overshoots whenever units are smaller
    than the target (e.g. many ~20-word clauses against a ~45-word target),
    which silently collapses back down toward one chunk per original
    sentence and undermines the whole point of clause-level splitting.

    A second, more severe failure this also guards against: simply
    requiring "enough units remain to still fill out the target chunk
    count" as a precondition for closing (and otherwise falling back to
    the fit heuristic) is not enough on its own. At the exact tight
    boundary - where the units remaining equals the chunks still needed,
    so every single remaining unit MUST become its own chunk from here
    on - the fit heuristic can still say "keep accumulating" (because the
    current chunk is numerically closer to target_words_per_chunk that
    way). Deferring even once at that boundary permanently breaks
    feasibility for every unit after it (one fewer unit remains each step,
    while the chunk count needed does not shrink), which silently
    collapses the entire remainder of the narration into one oversized
    final chunk. So closing is forced - regardless of fit - whenever
    deferring would already make target_count unreachable."""
    total_words = sum(len(u.split()) for u in units)
    target_words_per_chunk = max(total_words / target_count, 1)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for i, unit in enumerate(units):
        unit_words = len(unit.split())

        if current_words > 0 and len(chunks) < target_count - 1:
            overshoot_if_added = (current_words + unit_words) - target_words_per_chunk
            undershoot_if_closed_now = target_words_per_chunk - current_words
            units_left_including_this = len(units) - i
            chunks_still_needed_after_this_one = target_count - len(chunks) - 1
            # Close now if that's numerically the better fit...
            fits_better_closed_now = overshoot_if_added > undershoot_if_closed_now
            # ...OR if deferring this unit into the current chunk would
            # leave fewer units remaining than chunks still needed - i.e.
            # this is the last possible moment to close without making
            # target_count unreachable. This must be a forcing condition,
            # not just an extra AND-gate on the fit check above, or the
            # tight-boundary case described in the docstring silently
            # dumps the rest of the narration into the final chunk.
            must_close_to_stay_reachable = units_left_including_this <= chunks_still_needed_after_this_one
            if fits_better_closed_now or must_close_to_stay_reachable:
                chunks.append(" ".join(current))
                current = []
                current_words = 0

        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_into_chunks(full_narration: str, target_count: int) -> list[str]:
    sentences = _split_sentences(full_narration)
    if not sentences:
        return []

    units = sentences
    if len(sentences) < target_count:
        # Sentence-level breaks alone can't reach the target (e.g. a script
        # written in long, complex sentences rather than short spoken-style
        # ones) - fall back to clause-level breaks for finer granularity.
        units = _expand_with_clause_breaks(sentences, target_count)
        logger.info(
            "[SCENES] only %d sentence(s) for a target of %d scenes - "
            "splitting at clause boundaries too (%d units available)",
            len(sentences), target_count, len(units),
        )

    return _group_units_into_chunks(units, target_count)


def _target_scene_count(total_words: int) -> int:
    estimate = round(total_words / WORDS_PER_SCENE_TARGET)
    return max(MIN_SCENES, min(MAX_SCENES, estimate))


def _chunk_localized_narrations(localized_scripts: dict[str, dict], target_count: int) -> dict[str, list[str]]:
    """Independently chunk each already-available localization's
    full_narration into `target_count` pieces (same algorithm as the
    English master), so scene N gets a narration_<lang> covering roughly
    the same point in the story as scene N's narration_en. If a
    localization's narration doesn't have quite enough sentence/clause
    breaks to reach target_count exactly (rare - see
    _group_units_into_chunks' docstring), the shortfall scenes simply have
    no narration_<lang> rather than guessing/duplicating content."""
    result: dict[str, list[str]] = {}
    for lang, loc_script in (localized_scripts or {}).items():
        loc_full = (loc_script or {}).get("full_narration", "").strip()
        if not loc_full:
            continue
        loc_chunks = _split_into_chunks(loc_full, target_count)
        if len(loc_chunks) != target_count:
            logger.warning(
                "[SCENES] %s localization produced %d chunk(s), expected %d (matching the "
                "shared scene count) - %d scene(s) will have no narration_%s until this is re-run",
                lang, len(loc_chunks), target_count, target_count - len(loc_chunks), lang,
            )
        result[lang] = loc_chunks
    return result


def plan_scenes(
    script: dict,
    localized_scripts: dict[str, dict] | None = None,
    visual_style: str = "cinematic documentary, moody natural lighting, photorealistic",
) -> list[dict]:
    full_narration = script["full_narration"]
    total_words = len(full_narration.split())
    target_count = _target_scene_count(total_words)

    chunks = _split_into_chunks(full_narration, target_count)
    if len(chunks) < MIN_SCENES:
        logger.warning(
            "[SCENES] only %d scene(s) reachable (target was %d, minimum is %d) - the narration "
            "doesn't have enough sentence/clause breaks for more without splitting mid-clause; "
            "proceeding anyway rather than fabricating cuts that would land mid-thought",
            len(chunks), target_count, MIN_SCENES,
        )

    logger.info("[SCENES] planning %d scenes for %d words of narration", len(chunks), total_words)

    numbered = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(chunks))
    user_prompt = f"visual_style: {visual_style}\n\nNarration chunks:\n{numbered}"

    visuals, _ = llm_client.chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=6000)
    if not isinstance(visuals, list):
        raise RuntimeError("[SCENES] expected a JSON array of visual prompts")
    if len(visuals) != len(chunks):
        raise RuntimeError(f"[SCENES] expected {len(chunks)} visual prompts, got {len(visuals)}")

    localized_chunks = _chunk_localized_narrations(localized_scripts or {}, len(chunks))

    scenes = []
    for i, (chunk, visual) in enumerate(zip(chunks, visuals), start=1):
        words = len(chunk.split())
        duration_hint = round(words / WORDS_PER_SECOND, 1)
        visual_type = visual.get("visual_type", "cinematic")
        if visual_type not in VISUAL_TYPES:
            visual_type = "cinematic"
        scene = {
            "scene_id": i,
            "duration_hint": duration_hint,
            "narration_en": chunk,
            "visual_prompt": visual.get("visual_prompt", "").strip(),
            "visual_type": visual_type,
        }
        for lang, loc_chunks in localized_chunks.items():
            if i - 1 < len(loc_chunks):
                scene[f"narration_{lang}"] = loc_chunks[i - 1]
        scenes.append(scene)
        if not scene["visual_prompt"]:
            raise RuntimeError(f"[SCENES] scene {i} has an empty visual_prompt")

    return scenes


def run_and_save(paths: ps.ProjectPaths, visual_style: str | None = None) -> list[dict]:
    # The precondition check (missing script.json) lives inside the try
    # block below, not before it: every failure here must go through the
    # same save_failure() path as a real planning error, or the video is
    # left stuck with no FAILED state recorded and no diagnostic trail.
    run_id = db.start_run(paths.video_id, "scene_plan")
    try:
        script = ps.load_json(paths.script_json)
        if not script:
            raise RuntimeError(f"[SCENES] no script.json found at {paths.script_json} - run script_generator.py first")

        # Opportunistic: in the normal forward pipeline flow, localization
        # already ran before scene planning, so every configured
        # language's script_<lang>.json exists here and its narration gets
        # attached to the shared scene plan in this same run. If a
        # language's localization hasn't happened yet (e.g. this scene
        # plan predates that language being added to config), scenes are
        # still built from English alone - narration_<lang> is simply
        # absent, and a later resume run can fill it in without ever
        # replanning scenes or regenerating visuals (spec Round 3 section 11).
        localized_scripts = {}
        for lang in settings.LANGUAGE_LOCALIZATIONS:
            loc_script = ps.load_json(paths.script_localized_json(lang))
            if loc_script:
                localized_scripts[lang] = loc_script

        kwargs = {"visual_style": visual_style} if visual_style else {}
        scenes = plan_scenes(script, localized_scripts=localized_scripts, **kwargs)
        ps.save_json(paths.scenes_json, scenes)
        ps.save_state(paths, PipelineState.SCENES_PLANNED, extra={"scene_count": len(scenes)})
        db.finish_run(run_id, "success")
        return scenes
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "scene_plan", str(e))
        raise


def run_and_save_v2(paths: ps.ProjectPaths) -> list[dict]:
    """Visual Generator V2 integration - explicit V2 pipeline entry point,
    run_and_save()'s sibling (that function and plan_scenes() above are
    completely untouched and remain independently callable - full backward
    compatibility, not a replacement).

    Runs Phase 1 (plan_scenes_v2: LLM classification + deterministic
    strategy resolution + validate_scenes_v2) then Phase 2 (build_prompts_v2:
    deterministic visual_prompt/avoid_strategy/accuracy_hints/
    overlay_required/render_hash + validate_prompts_v2), and writes the
    result through the EXACT SAME storage primitives v1 uses - ps.save_json
    to the same paths.scenes_json path, ps.save_state to the same
    PipelineState.SCENES_PLANNED state, the same "scene_plan" db stage name.
    Downstream code (image_gen.py, assemble.py, pipeline.py's
    _images_done()/_scenes_synced_for()) reads scenes.json/state.json the
    same way regardless of which function produced them - no parallel
    storage system, no new file format.
    """
    run_id = db.start_run(paths.video_id, "scene_plan")
    try:
        script = ps.load_json(paths.script_json)
        if not script:
            raise RuntimeError(f"[SCENES][V2] no script.json found at {paths.script_json} - run script_generator.py first")

        # Same opportunistic localized-narration attachment v1's
        # run_and_save() does - see that function's own comment for why.
        localized_scripts = {}
        for lang in settings.LANGUAGE_LOCALIZATIONS:
            loc_script = ps.load_json(paths.script_localized_json(lang))
            if loc_script:
                localized_scripts[lang] = loc_script

        scenes = plan_scenes_v2(script, localized_scripts=localized_scripts)
        scenes = build_prompts_v2(scenes)
        ps.save_json(paths.scenes_json, scenes)
        ps.save_state(paths, PipelineState.SCENES_PLANNED, extra={"scene_count": len(scenes)})
        db.finish_run(run_id, "success")
        return scenes
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "scene_plan", str(e))
        raise


def sync_localized_narrations(paths: ps.ProjectPaths, language: str) -> list[dict]:
    """Round 3.1 fix: a resume edge case that plan_scenes()'s own
    opportunistic attachment (see module docstring) doesn't cover - if
    scenes.json/visuals were already produced BEFORE script_<language>.json
    existed (e.g. that localization language was added, or simply hadn't
    finished, on an earlier interrupted run), the shared scene plan never
    replans and never gets narration_<language> attached at planning time,
    even once the localized script later becomes available. Without this,
    that gap is permanent - re-running the pipeline resumes straight past
    the already-existing scenes.json (see pipeline.py's is_done() checks)
    and the localization is simply never reflected in scenes.json.

    This is a small, deterministic, additive enrichment of the EXISTING
    scenes.json - never a replan:
        - loads the existing scenes.json (never recreates it)
        - loads script_<language>.json
        - re-chunks that language's full_narration into exactly
          len(scenes) pieces via `_chunk_localized_narrations()` - the
          SAME chunking algorithm plan_scenes() itself uses, not a
          separate/divergent one
        - attaches narration_<language> onto each existing scene dict by
          position - every other field (scene_id, visual_prompt,
          visual_type, duration_hint, any other language's narration_*)
          is left completely untouched
        - saves scenes.json atomically (temp file + rename)
        - never touches image files, never calls plan_scenes(), never
          calls the visual-prompt LLM or the image generation API again

    Identical for every configured localization language (tr, es, de, fr,
    it, pt, id, ...) - there is no language-specific branching here beyond
    the `language` argument itself, so a future language enriches this
    exact same scenes.json the same way.

    Idempotent: if scenes.json already has narration_<language> on every
    scene (the normal forward-flow case, where plan_scenes() already
    attached it), this is a cheap no-op read - it does not rewrite the
    file or touch its mtime.
    """
    stage_label = f"scene_sync_{language}"
    run_id = db.start_run(paths.video_id, stage_label)
    try:
        scenes = ps.load_json(paths.scenes_json)
        if not scenes:
            raise RuntimeError(f"[SCENES] no scenes.json found at {paths.scenes_json} - run scene_planner.py first")

        key = f"narration_{language}"
        if all(key in scene for scene in scenes):
            logger.info("[SCENES] scenes.json already has %s on every scene - nothing to sync", key)
            db.finish_run(run_id, "success")
            return scenes

        loc_script = ps.load_json(paths.script_localized_json(language))
        if not loc_script:
            raise RuntimeError(
                f"[SCENES] no localized script found at {paths.script_localized_json(language)} "
                f"for language={language!r} - run the localization stage first"
            )

        loc_chunks = _chunk_localized_narrations({language: loc_script}, len(scenes)).get(language, [])
        enriched = 0
        for i, scene in enumerate(scenes):
            if i < len(loc_chunks):
                scene[key] = loc_chunks[i]
                enriched += 1

        ps.save_json_atomic(paths.scenes_json, scenes)
        logger.info(
            "[SCENES] synced %s onto existing scenes.json (%d/%d scenes enriched; "
            "scene_id/visual_prompt/images untouched)",
            key, enriched, len(scenes),
        )
        db.finish_run(run_id, "success")
        return scenes
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, stage_label, str(e))
        raise


# ---------------------------------------------------------------------------
# Scene Planner V2 - Phase 1 ONLY (V2 scene classification + deterministic
# strategy + validation). Everything below this line is purely ADDITIVE:
# plan_scenes()/run_and_save()/sync_localized_narrations() above are
# completely untouched, remain the only thing pipeline.py calls, and this
# section does not change their behavior in any way. V2 is not wired into
# the pipeline yet (Phase 6 scope), produces no visual_prompt (Phase 2),
# calls no image API and writes no overlay/scientific-graphics output
# (Phase 3-5 scope).
#
# Three-axis design (per the approved V2 spec): the LLM proposes scene_type
# (semantic classification), evidence_level (certainty) and data_points (a
# typed data inventory) for each scene. resolve_visual_strategy() below is
# the ONLY thing that ever decides visual_strategy (the technical rendering
# method) and the ONLY thing allowed to override scene_type (forcing a
# hypothesis into a diagram, or an archival_source - which V2.0 has no real
# source-asset pipeline for - into a recreated document). The LLM never
# sees, proposes, or produces a visual_strategy field.
# ---------------------------------------------------------------------------

SCENE_TYPES_V2 = [
    "archival_source", "document_recreation", "scientific_visualization",
    "scientific_graphic", "astronomical_map", "historical_reenactment",
    "real_location", "evidence_closeup", "atmospheric_establishing",
    "hypothesis_diagram", "research_evidence",
]

EVIDENCE_LEVELS_V2 = [
    "primary_source", "documented_fact", "scientific_interpretation",
    "hypothesis", "atmospheric",
]

DATA_TYPES_V2 = [
    "scalar_fact", "measurement", "exact_text", "coordinate", "time_series",
    "curve", "spectrum", "timeline", "relationship", "location", "none",
]

# data_type values whose only faithful representation is a drawn chart/plot,
# never an AI-generated photo-style image - resolve_visual_strategy() rule 3.
_PROGRAMMATIC_GRAPHIC_DATA_TYPES = {"time_series", "curve", "spectrum", "timeline"}

# scene_type values a "hypothesis" evidence_level is allowed to keep as-is
# (resolve_visual_strategy() rule 1) - anything else is forced to
# hypothesis_diagram, because a speculative idea must never be visualized in
# a way that reads as documented fact.
_HYPOTHESIS_ALLOWED_SCENE_TYPES = ("hypothesis_diagram", "scientific_visualization")

# Phrases that read as filler rather than a real, scene-specific
# justification - visual_relevance_reason is hard-rejected by
# validate_scenes_v2() if it contains one of these (case-insensitive), and
# the LLM is told not to write them in SYSTEM_PROMPT_V2_CLASSIFICATION.
_GENERIC_RELEVANCE_PHRASES = {
    "sets the mood", "establishes the mood", "establishes atmosphere",
    "establishes the atmosphere", "creates atmosphere", "creates the atmosphere",
    "general illustration", "generic illustration", "visual interest",
    "adds visual interest", "matches the tone", "matches the mood",
    "supports the narration", "fits the narration", "just for context",
}

# scene_type -> legacy v1 visual_type, so anything downstream that still
# reads the old 7-value field (image_gen.py, quality_check.py's structural
# checks, etc.) keeps working on a V2 scene until those are migrated in a
# later phase. Deliberately never "archival" for archival_source itself -
# resolve_visual_strategy() always normalizes that scene_type away before it
# would reach here (see rule ordering below), but the mapping is kept for
# defensiveness.
_SCENE_TYPE_TO_LEGACY_VISUAL_TYPE = {
    "archival_source": "archival",
    "document_recreation": "archival",
    "scientific_visualization": "abstract",
    "scientific_graphic": "abstract",
    "astronomical_map": "map",
    "historical_reenactment": "cinematic",
    "real_location": "wide",
    "evidence_closeup": "closeup",
    "atmospheric_establishing": "wide",
    "hypothesis_diagram": "abstract",
    "research_evidence": "closeup",
}

SYSTEM_PROMPT_V2_CLASSIFICATION = f"""You are a documentary research analyst, not a stock-photo captioner. You
will be given a numbered list of narration chunks, in order, from a finished documentary script. For EACH
numbered chunk, classify what kind of visual evidence/illustration it calls for - you are NOT writing an
image prompt and NOT choosing how it will be rendered; a separate, deterministic system makes that decision
from your classification.

For EACH chunk, return exactly these fields (no others - in particular, do NOT include a "visual_strategy"
or "visual_type" field; that decision is made in code, never by you):

- "input_chunk_index": the integer number (1-based) of the input narration chunk this classification is
  for. It MUST exactly equal the number prefixing that chunk in the numbered list below (e.g. the
  classification object for the chunk shown as "3. <text>" MUST have "input_chunk_index": 3) - regardless
  of what position this object occupies in your returned array. This lets a deterministic system verify
  and realign every classification against its exact source chunk without relying on array order.
- "claim": one sentence stating, in your own words, the specific factual claim or narrative beat this chunk
  of narration makes - the underlying claim being illustrated, not a paraphrase of the narration itself.
- "scene_type": exactly one of {SCENE_TYPES_V2}.
    - archival_source: a specific, real historical document/photograph/recording is being described.
    - document_recreation: no real archival asset is being pointed to, but a document/printout/chart
      artifact should still be depicted (a recreation, not the real thing). RESTRICTED: only use this when
      the narration is actually pointing at a specific document, printout, report, log, letter, or
      published record - "a study found..." or "records show..." is not enough on its own. Never use
      document_recreation just because the narration states a historical fact; a fact being true is not
      the same as a document being depicted. If in doubt between document_recreation and research_evidence
      or real_location, prefer whichever one names an actual physical document vs. an actual physical
      place/finding.
    - scientific_visualization: an abstract/diagrammatic depiction of a real scientific concept or process.
    - scientific_graphic: a chart, plot, or graph of real data - prefer this (or research_evidence, see
      below) whenever the claim centers on a NUMBER: a measurement, frequency, signal strength,
      signal-to-noise ratio, spectrum, duration, count, or comparison of values. Put the actual number(s)
      in data_points (see below), never describe them in prose in prompt_segments.
    - astronomical_map: sky maps, telescope pointing, coordinates, star fields.
    - historical_reenactment: a real past human action/event being dramatized.
    - real_location: a real, named, physical place - a specific facility, building, site, or region that
      exists or existed, described as a place (not as a document about that place).
    - evidence_closeup: a closeup on one specific physical piece of evidence.
    - atmospheric_establishing: mood/tone/transition shots with no specific factual content.
    - hypothesis_diagram: a speculative/unproven idea, diagrammed explicitly AS speculation. RESTRICTED:
      only for narration that is actually uncertain/speculative in its own words - phrases like "could be",
      "one possibility", "another explanation", "if this is correct", "remains unconfirmed", "competing
      hypothesis", or naming a proposed-but-unproven mechanism. A narrated research finding, reanalysis, or
      published result is NOT automatically a hypothesis just because it's scientific - see research_evidence.
    - research_evidence: findings, analysis, or data from a specific piece of research that doesn't fit the
      categories above - use this (not hypothesis_diagram, and not document_recreation just because a paper
      is mentioned) for narration describing what researchers did or found: "researchers reanalyzed...",
      "a new analysis showed...", "published in...", "the [year] study concluded...", "archival data
      revealed...", "a preprint suggested...", "observational data indicated...". This is presenting a
      finding as a finding, not the same as depicting the literal document it appeared in
      (document_recreation) or presenting it as still-unproven speculation (hypothesis_diagram, which is
      only for genuinely open/competing explanations).
- "evidence_level": exactly one of {EVIDENCE_LEVELS_V2} - your honest assessment of how certain this claim
  is. Use "hypothesis" for ANY idea that is speculative, unproven, disputed, or one of several competing
  explanations - never classify a hypothesis as documented_fact or scientific_interpretation just because
  it sounds plausible or is stated confidently in the narration. As a general guide to keep evidence_level
  and scene_type consistent with each other (not a hard rule the system enforces beyond the
  hypothesis/hypothesis_diagram pairing above, but your own reasoning should follow it): primary_source
  claims usually pair with a document/evidence-oriented scene_type (archival_source/document_recreation/
  evidence_closeup); documented_fact usually pairs with a factual visualization (real_location,
  astronomical_map, a timeline/graphic, historical_reenactment); research_evidence-level findings usually
  pair with scene_type=research_evidence, scientific_graphic, or an archival source; hypothesis always
  pairs with hypothesis_diagram (enforced); atmospheric always pairs with atmospheric_establishing.
- "data_points": a JSON array (may be empty) of the specific concrete data this scene needs to convey
  exactly, each shaped {{"data_type": <one of {DATA_TYPES_V2}>, "value": "<the exact text/number/coordinate
  string, verbatim from the narration>", "exact_render": <true if this must appear as literal readable
  text/numbers on screen, false if it only needs to inform the visual conceptually>}}. Use
  data_type="exact_text" for any verbatim string (a code, a quote, a label, a name) that must be rendered
  character-for-character. Use the closest specific data_type (measurement, coordinate, time_series, curve,
  spectrum, timeline, ...) for anything numeric or scientific - a frequency, a coordinate pair, a signal
  strength or signal-to-noise ratio, a date/timeline, a measured value, a scientific label - so it is
  represented as structured data, never only as prose in prompt_segments (an image model cannot reliably
  render exact numbers, so this is the ONLY place they should live). Use data_type="none" (and an empty
  array) only when there is genuinely no specific data in this chunk.
- "visual_relevance_reason": one specific sentence explaining why THIS visual choice illustrates THIS
  scene's claim - never a generic/filler reason. Phrases like "sets the mood", "establishes atmosphere",
  "general illustration of the topic", "matches the tone of the narration", or "visual interest" are
  REJECTED by the system that reads this - be concrete about what is depicted and why it maps to the claim.
- "prompt_segments": an object with these six string fields - subject, setting, lighting, mood,
  composition, style - each a short descriptive phrase (not a full prompt; a later stage assembles the
  final prompt from these). NEVER put a literal number, code, coordinate, or quoted exact string in
  prompt_segments - those belong ONLY in data_points, because an image model cannot reliably render exact
  text/numbers; they are burned onto the image separately.

INPUT/OUTPUT CARDINALITY - ABSOLUTE RULE: if the input contains N numbered narration chunks, you MUST
return EXACTLY N classification objects. One input chunk = exactly one output object. NEVER split one
chunk into multiple objects, merge two chunks into one object, omit a chunk, invent an extra chunk, or
return an object without a corresponding input chunk. Output array positions MUST correspond exactly:
output[0] -> chunk 1, output[1] -> chunk 2, ... output[N-1] -> chunk N. Before returning JSON, internally
count the input chunks and verify that your output array contains exactly that same number of objects.

Return ONLY a JSON array (no markdown, no commentary) with exactly one object per input chunk, in the same
order."""


def resolve_visual_strategy(
    scene_type: str, evidence_level: str, data_points: list[dict] | None,
) -> tuple[str, str]:
    """The ONLY place visual_strategy is decided (requirement #3/#4) - pure,
    deterministic, and Python-only; the LLM never sees or produces this
    field. Also the only place scene_type is ever overridden after the LLM
    proposes it. First-match-wins ordered rule chain:

      0. (normalization, not a "rule" the spec numbered, but must run before
         every rule below) V2.0 has no real source_asset_ref pipeline yet -
         archival_source can never actually be honored, so it is always
         normalized to document_recreation first. This runs unconditionally
         so archival_source can never leak into the output down ANY path
         (not just the "falls through to default" path) - see
         validate_scenes_v2()'s self-consistency check, which relies on
         this to catch an illegally-stored archival_source scene_type.
      1. evidence_level == "hypothesis" -> force scene_type into
         _HYPOTHESIS_ALLOWED_SCENE_TYPES (hypothesis_diagram if it isn't
         already one of them), strategy = "hypothesis_visualization". A
         speculative idea must never be visualized as if it were documented
         fact, so this takes priority over every other signal.
      2. any data_point with exact_render truthy -> strategy =
         "ai_generated_plus_overlay" (a generated base image plus a
         burned-in text/number overlay - never trust an image model to
         render exact characters).
      3. any data_point whose data_type is time_series/curve/spectrum/
         timeline -> strategy = "programmatic_graphic" (drawn, not
         photographed/generated).
      4. default -> strategy = "ai_generated".

    Idempotent by construction (resolve_visual_strategy(*resolve_visual_strategy(...))
    == resolve_visual_strategy(...)) - this is what lets
    validate_scenes_v2() recompute from a scene's OWN stored
    scene_type/evidence_level/data_points and compare against its OWN
    stored scene_type/visual_strategy as a single unified consistency
    check, instead of re-deriving and duplicating every rule a second time.
    """
    data_points = data_points or []
    resolved_scene_type = scene_type

    if resolved_scene_type == "archival_source":
        resolved_scene_type = "document_recreation"

    if evidence_level == "hypothesis":
        if resolved_scene_type not in _HYPOTHESIS_ALLOWED_SCENE_TYPES:
            resolved_scene_type = "hypothesis_diagram"
        return resolved_scene_type, "hypothesis_visualization"

    if any((dp or {}).get("exact_render") for dp in data_points):
        return resolved_scene_type, "ai_generated_plus_overlay"

    if any((dp or {}).get("data_type") in _PROGRAMMATIC_GRAPHIC_DATA_TYPES for dp in data_points):
        return resolved_scene_type, "programmatic_graphic"

    return resolved_scene_type, "ai_generated"


def _legacy_visual_type_for(scene_type: str) -> str:
    return _SCENE_TYPE_TO_LEGACY_VISUAL_TYPE.get(scene_type, "cinematic")


def _derive_exact_text(data_points: list[dict] | None) -> list[str]:
    """Exact on-screen text is DERIVED here from data_points, never
    independently authored by the LLM (requirement #7) - this is the single
    source of truth for which literal characters/numbers must render
    exactly, for a later overlay stage (Phase 3) to burn onto the image.
    Order-preserving, de-duplicated."""
    seen: set[str] = set()
    items: list[str] = []
    for dp in data_points or []:
        dp = dp or {}
        if dp.get("data_type") == "exact_text" or dp.get("exact_render"):
            value = str(dp.get("value", "")).strip()
            if value and value not in seen:
                seen.add(value)
                items.append(value)
    return items


def _derive_text_render(exact_text: list[str], evidence_level: str) -> dict:
    """Deterministic overlay metadata derived from exact_text - not authored
    by the LLM. `hypothesis_badge` tells the (Phase 3, not-yet-written)
    overlay stage to visibly mark a hypothesis-visualization scene as
    speculative, independent of whatever exact_text it also carries."""
    return {
        "items": list(exact_text or []),
        "hypothesis_badge": evidence_level == "hypothesis",
    }


def validate_scenes_v2(scenes: list[dict]) -> None:
    """PRE-IMAGE quality gate (V2 spec's PRE-IMAGE/POST-IMAGE split;
    POST-IMAGE doesn't exist yet - no images are produced in Phase 1) -
    fails loud, before any image-generation cost is spent, by collecting
    EVERY violation across every scene rather than stopping at the first
    one, so a single fix-and-rerun cycle surfaces the whole list.

    Only scenes carrying schema_version == 2 are checked - a scenes.json
    that mixes legacy v1 scenes (no schema_version field at all) with new
    V2 scenes leaves the v1 ones completely untouched, so backward
    compatibility never depends on this function running (or not) against
    old data.

    Deliberately does NOT check whether exact_text appears inside
    visual_prompt - there is no visual_prompt field yet in Phase 1 (see
    plan_scenes_v2()'s docstring); that check is explicitly deferred to
    Phase 2's prompt builder.
    """
    errors: list[str] = []

    for scene in scenes:
        if scene.get("schema_version") != 2:
            continue
        scene_id = scene.get("scene_id", "?")
        prefix = f"scene {scene_id}"

        claim = scene.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            errors.append(f"{prefix}: missing/empty 'claim'")

        scene_type = scene.get("scene_type")
        if scene_type not in SCENE_TYPES_V2:
            errors.append(f"{prefix}: invalid scene_type {scene_type!r} (must be one of {SCENE_TYPES_V2})")

        evidence_level = scene.get("evidence_level")
        if evidence_level not in EVIDENCE_LEVELS_V2:
            errors.append(f"{prefix}: invalid evidence_level {evidence_level!r} (must be one of {EVIDENCE_LEVELS_V2})")

        data_points = scene.get("data_points")
        if not isinstance(data_points, list):
            errors.append(f"{prefix}: data_points must be a list (got {type(data_points).__name__})")
            data_points = []
        for j, dp in enumerate(data_points):
            dp = dp or {}
            data_type = dp.get("data_type")
            if data_type not in DATA_TYPES_V2:
                errors.append(f"{prefix}: data_points[{j}] has invalid data_type {data_type!r} (must be one of {DATA_TYPES_V2})")
            if data_type == "exact_text" and not str(dp.get("value", "")).strip():
                errors.append(f"{prefix}: data_points[{j}] declares data_type=exact_text but has no 'value' - exact_text cannot be derived")

        reason = scene.get("visual_relevance_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{prefix}: empty visual_relevance_reason")
        else:
            lowered = reason.strip().lower()
            hit = next((p for p in _GENERIC_RELEVANCE_PHRASES if p in lowered), None)
            if hit:
                errors.append(f"{prefix}: visual_relevance_reason reads as generic/filler ({reason!r}, matched {hit!r}) - must justify THIS scene's specific visual choice")

        if evidence_level == "hypothesis" and scene_type in SCENE_TYPES_V2 and scene_type not in _HYPOTHESIS_ALLOWED_SCENE_TYPES:
            errors.append(
                f"{prefix}: evidence_level=hypothesis but scene_type={scene_type!r} is not one of "
                f"{_HYPOTHESIS_ALLOWED_SCENE_TYPES} - a hypothesis must be visualized as a diagram, "
                f"never presented as if it were documented fact"
            )

        exact_text = scene.get("exact_text")
        if exact_text:
            has_source = any(
                (dp or {}).get("data_type") == "exact_text" or (dp or {}).get("exact_render")
                for dp in data_points
            )
            if not has_source:
                errors.append(f"{prefix}: exact_text is set ({exact_text!r}) but no data_points entry declares data_type=exact_text or exact_render=True to justify it")

        # Belt-and-suspenders: recompute the deterministic resolution from
        # this scene's OWN stored inputs and confirm it matches what's
        # actually stored - catches any other invalid
        # scene_type/evidence_level/data_points -> visual_strategy
        # combination, including an illegally-stored scene_type of
        # "archival_source" itself, since resolve_visual_strategy() never
        # returns that value (see its rule 0).
        if scene_type in SCENE_TYPES_V2 and evidence_level in EVIDENCE_LEVELS_V2:
            expected_scene_type, expected_strategy = resolve_visual_strategy(scene_type, evidence_level, data_points)
            stored_strategy = scene.get("visual_strategy")
            if (expected_scene_type, expected_strategy) != (scene_type, stored_strategy):
                errors.append(
                    f"{prefix}: stored scene_type/visual_strategy ({scene_type!r}/{stored_strategy!r}) does not "
                    f"match what resolve_visual_strategy() deterministically derives from this scene's own "
                    f"scene_type/evidence_level/data_points ({expected_scene_type!r}/{expected_strategy!r})"
                )

    if errors:
        raise RuntimeError(
            "[SCENES][V2] validate_scenes_v2 found {} issue(s):\n  - {}".format(len(errors), "\n  - ".join(errors))
        )


# ---------------------------------------------------------------------------
# TEMPORARY DIAGNOSTIC (not a permanent feature) - added to root-cause a
# real-world failure where a real Claude call returned more classification
# objects than input narration chunks (e.g. "expected 35 classifications,
# got 36") despite SYSTEM_PROMPT_V2_CLASSIFICATION's explicit cardinality
# rule. Captures exactly what was sent and what came back so the actual
# mismatch shape can be inspected (e.g. one input chunk answered with two
# objects vs. a genuine off-by-one) before any architectural fix is
# designed. Does NOT change plan_scenes_v2()'s pass/fail behavior - the
# existing hard RuntimeError below is unchanged, this only runs (and writes
# a file) immediately before it. Delete this block (and its call site) once
# the real cause is confirmed and a permanent fix lands.
# ---------------------------------------------------------------------------

_DEBUG_AUDIT_DIR = PROJECT_ROOT / "debug_audit" / "scene_planner_v2"


def _write_cardinality_diagnostic(
    chunks: list[str],
    classifications: list,
    output_dir: Path | None = None,
    extra: dict | None = None,
) -> Path:
    """Writes a JSON snapshot of an input-chunk/output-classification-count
    mismatch to a local debug file (never to logs, never to storage/) and
    returns the path written.

    Contains ONLY data already present in `chunks` (the narration text this
    process itself built) and `classifications` (the LLM's own JSON
    response body) - it never touches settings, headers, or the API key,
    so there is nothing here that could leak a secret; this function
    structurally cannot see one, not just "is careful not to log one".

    Captures, per requirement: total counts, every input chunk (so a human
    can see the exact text each numbered line actually contained), and
    every returned object's index/claim/scene_type/evidence_level plus
    data_points/prompt_segments/visual_relevance_reason - the fields most
    likely to reveal that two returned objects both derived from the same
    single input chunk (a split) or that a returned object doesn't match
    any input chunk at all (an invention).

    `extra` (additive, optional - never removes/renames an existing field
    above): the indexed-classification hardening pass's own findings -
    duplicate_indices/missing_indices/invalid_indices (see
    _analyze_classification_indices) plus repair_attempted/repair_indices/
    retry_stage (see _classify_chunks_v2) - merged directly into the
    payload so a human reading this file no longer has to manually diff
    chunks against classifications to find a split/merge, the way the real
    35->36 production failure had to be diagnosed by hand.

    Never raises: a diagnostic that itself crashes the pipeline would be
    worse than no diagnostic, so any failure while writing is logged as a
    warning and swallowed - the caller's real RuntimeError about the
    mismatch is what must still surface."""
    directory = output_dir or _DEBUG_AUDIT_DIR
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"cardinality_mismatch_{timestamp}.json"

    def _safe(cls, key):
        return cls.get(key) if isinstance(cls, dict) else None

    payload = {
        "diagnostic_type": "scene_planner_v2_cardinality_mismatch",
        "generated_at_utc": timestamp,
        "note": (
            "TEMPORARY diagnostic - captures the raw LLM classification array when its "
            "length does not match the number of input narration chunks. Contains only "
            "narration/classification content already local to this process; no API key, "
            "headers, or credentials are ever read or written here."
        ),
        "expected_object_count": len(chunks),
        "returned_object_count": len(classifications) if isinstance(classifications, list) else None,
        "input_chunks": [
            {"chunk_index": i, "word_count": len(chunk.split()), "text": chunk}
            for i, chunk in enumerate(chunks, start=1)
        ],
        "returned_classifications": [
            {
                "returned_index": i,
                "raw_keys": sorted(cls.keys()) if isinstance(cls, dict) else None,
                "claim": _safe(cls, "claim"),
                "scene_type": _safe(cls, "scene_type"),
                "evidence_level": _safe(cls, "evidence_level"),
                "data_points": _safe(cls, "data_points"),
                "visual_relevance_reason": _safe(cls, "visual_relevance_reason"),
                "prompt_segments": _safe(cls, "prompt_segments"),
            }
            for i, cls in enumerate(classifications if isinstance(classifications, list) else [])
        ],
    }
    if extra:
        payload.update(extra)

    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("[SCENES][V2][DIAGNOSTIC] failed to write cardinality diagnostic to %s: %s", path, e)
    return path


# ---------------------------------------------------------------------------
# CTA/end-screen separation (root-cause fix for the cardinality mismatch the
# diagnostic above captured: a real classification run returned 36 objects
# for 35 input chunks because the final chunk secretly mixed the
# documentary's closing beat with the script's separate call-to-action -
# full_narration is just every section's narration joined together with no
# boundary preserved, so the CTA (script_generator.py's required, always-
# present "cta" section type) ends up glued onto the tail of the last
# documentary chunk. The LLM wasn't violating the cardinality rule so much
# as correctly recognizing that input chunk actually contained two scenes.
#
# Fix: use the structural section/type boundary script_generator.py (and
# localizer.py, for every localized script - documented as "a compatible
# superset of script_en.json's own shape") ALREADY provides, deterministic
# and Python-only, instead of asking the LLM to somehow infer this from
# prose it never even needed to see. The CTA's narration never enters the
# chunking/classification pool at all - it cannot cause a cardinality
# mismatch because it never reaches the LLM. It is still represented, as
# one separate, fully deterministic scene (schema_version=2, is_cta=True)
# built entirely in Python with zero LLM involvement, for later video
# assembly to recognize and handle differently from documentary content
# (e.g. skip it for B-roll purposes, or route it to a dedicated end-screen
# template) - see _build_cta_scene() below.
# ---------------------------------------------------------------------------

def _split_documentary_and_cta(script: dict) -> tuple[str, str]:
    """Deterministically separates a script's documentary narration from its
    call-to-action narration using "sections"/"type" (never by guessing from
    the prose itself - no "subscribe"/"comment" keyword matching, which
    would be fragile and language-specific for every localization).

    Backward compatible by construction: if `script` has no non-empty
    "sections" list (e.g. an older or synthetic {"full_narration": ...}
    fixture/caller), the ENTIRE full_narration is returned as documentary
    text with an empty CTA string - byte-for-byte the same input
    plan_scenes_v2() chunked before this fix, so anything with that shape
    (including every existing test fixture) behaves exactly as before."""
    sections = script.get("sections")
    if not sections:
        return script.get("full_narration", "") or "", ""

    documentary = "\n\n".join(
        s["narration"].strip() for s in sections if s.get("type") != "cta" and s.get("narration")
    )
    cta = "\n\n".join(
        s["narration"].strip() for s in sections if s.get("type") == "cta" and s.get("narration")
    )
    return documentary, cta


def _chunk_localized_narrations_v2(
    localized_scripts: dict[str, dict], target_count: int,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """V2 sibling of _chunk_localized_narrations() (v1, completely
    unchanged) - the only difference is that each localization's own CTA-
    typed section(s) are excluded from the documentary-chunk pool before
    chunking (via _split_documentary_and_cta()), exactly mirroring what
    plan_scenes_v2() itself now does for the English master. This keeps
    scene N's narration_<lang> aligned with scene N's narration_en - without
    it, the CTA could become an extra, misaligned chunk in one language but
    not another, or shift chunk boundaries differently per language.

    Also returns each language's own CTA narration text, separately from
    the per-scene documentary chunks, for attaching to the single synthetic
    CTA scene alongside its narration_en (requirement: CTA text must not be
    silently dropped, in any available language)."""
    localized_chunks: dict[str, list[str]] = {}
    localized_cta: dict[str, str] = {}
    for lang, loc_script in (localized_scripts or {}).items():
        loc_documentary, loc_cta = _split_documentary_and_cta(loc_script or {})
        if loc_documentary.strip():
            loc_chunks = _split_into_chunks_v2(loc_documentary, target_count)
            if len(loc_chunks) != target_count:
                logger.warning(
                    "[SCENES][V2] %s localization produced %d chunk(s), expected %d (matching the "
                    "shared scene count) - %d scene(s) will have no narration_%s until this is re-run",
                    lang, len(loc_chunks), target_count, target_count - len(loc_chunks), lang,
                )
            localized_chunks[lang] = loc_chunks
        if loc_cta.strip():
            localized_cta[lang] = loc_cta
    return localized_chunks, localized_cta


# Deterministic prompt_segments for the synthetic CTA scene - not LLM-
# authored (there is no classification call for this scene at all), just a
# plausible, generic-but-specific-enough placeholder so a later
# build_visual_prompt() call (Phase 2) produces something sane for it too,
# without ever quoting the CTA's actual wording (image models don't need it
# - the on-screen subscribe/comment prompt is a post-production overlay
# concern, same principle as exact_text elsewhere in V2).
_CTA_PROMPT_SEGMENTS = {
    "subject": "the channel's closing outro moment, no on-screen text or logos",
    "setting": "a calm, minimal outro backdrop related to the documentary's subject",
    "lighting": "soft ambient lighting",
    "mood": "warm and inviting, low-key",
    "composition": "clean negative space reserved for on-screen subscribe/comment graphics",
    "style": "cinematic documentary, photorealistic, uncluttered",
}


def _build_cta_scene(cta_narration_en: str, scene_id: int, localized_cta: dict[str, str]) -> dict | None:
    """Builds the one synthetic, fully deterministic CTA/end-screen scene -
    zero LLM involvement, so it structurally cannot cause a cardinality
    mismatch. Returns None when there is no CTA text to represent (the
    script had no "sections"-based CTA at all - see
    _split_documentary_and_cta()'s backward-compat fallback), so nothing
    is appended for a script shaped like the pre-fix input.

    scene_type/evidence_level are fixed at "atmospheric_establishing"/
    "atmospheric" - both already-valid SCENE_TYPES_V2/EVIDENCE_LEVELS_V2
    values (no schema change needed) that accurately describe an outro/
    end-screen shot: mood/tone content with no specific factual claim to
    evidence. resolve_visual_strategy() still runs on these (same as every
    other V2 scene) so validate_scenes_v2()'s self-consistency check passes
    the normal way, not via a special case.

    is_cta=True is the marker later video-assembly code should key off of
    to treat this scene differently from documentary content (e.g. route it
    to a dedicated end-screen template instead of stock B-roll generation).
    Every OTHER V2 scene now also carries is_cta=False (added in
    plan_scenes_v2() below), so the field is always present - callers never
    need a .get("is_cta", False) default dance."""
    if not cta_narration_en.strip():
        return None

    words = len(cta_narration_en.split())
    duration_hint = round(words / WORDS_PER_SECOND, 1)
    scene_type = "atmospheric_establishing"
    evidence_level = "atmospheric"
    data_points: list[dict] = []

    resolved_scene_type, visual_strategy = resolve_visual_strategy(scene_type, evidence_level, data_points)
    exact_text = _derive_exact_text(data_points)
    text_render = _derive_text_render(exact_text, evidence_level)

    scene = {
        "scene_id": scene_id,
        "schema_version": 2,
        "duration_hint": duration_hint,
        "narration_en": cta_narration_en,
        "claim": "Closing call-to-action inviting the viewer to subscribe, comment, or otherwise engage with the channel.",
        "scene_type": resolved_scene_type,
        "visual_strategy": visual_strategy,
        "evidence_level": evidence_level,
        "data_points": data_points,
        "exact_text": exact_text,
        "text_render": text_render,
        "visual_relevance_reason": (
            "This is the video's closing call-to-action, not documentary evidence - a calm, "
            "non-competing atmospheric shot lets the on-screen subscribe/comment prompt and "
            "channel outro graphics take visual priority without competing documentary content."
        ),
        "prompt_segments": dict(_CTA_PROMPT_SEGMENTS),
        "visual_type": _legacy_visual_type_for(resolved_scene_type),
        "source_asset_ref": None,
        "preferred_source": None,
        "fallback_strategy": None,
        "avoid_strategy": None,
        "is_cta": True,
    }
    for lang, text in localized_cta.items():
        if text.strip():
            scene[f"narration_{lang}"] = text
    return scene


# ---------------------------------------------------------------------------
# Scene Planner V2 - dated-beat chunk rebalancing: a second, independent
# real-world cardinality mismatch (Wow! Signal audit: "expected 35 classifi-
# cations, got 36"), root-caused NOT to the CTA (that fix above was already
# confirmed working against the real API) but to the deterministic chunker
# itself.
#
# _group_units_into_chunks (shared with v1) picks chunk boundaries purely by
# numeric word-count fit - it has no notion of "documentary claim" at all.
# For the real script, that fit heuristic packed two COMPLETE, independently
# dated sentences into one chunk (input chunk 17):
#   "John Kraus published the first general description in Cosmic Search in
#   1979." + "The Ohio State SETI program operated from 1973 to 1997 and
#   detected no comparable Wow! signal during nearly 24 years."
# The classification LLM then - correctly, from a storytelling standpoint -
# recognized these as two separate visual beats (a 1979 publication vs. a
# 1973-1997 observing program) and returned two classifications for that one
# input chunk: the same shape of failure as the CTA case, and for the same
# underlying reason. An LLM instruction ("one input chunk = one output
# object") is necessary but not sufficient to override what the model sees
# as the semantically correct answer - so, as with the CTA fix, the reliable
# fix is deterministic Python, not a stronger prompt.
#
# _has_conflicting_dated_beats() is the structural (non-LLM) proxy for "this
# chunk bundles multiple independent documentary claims": two sentence-level
# units that each carry their OWN 4-digit year token with NO year in common
# are treated as two separate beats. A SHARED year (e.g. two sentences both
# about the same 1977 night) is deliberately NOT a conflict - that's one
# continuous beat, safe to keep together. This is a narrow, precise signal -
# documentary/history scripts in this pipeline anchor almost every
# individual claim to a date, exactly the shape of the real chunk-17 case.
#
# _split_into_chunks_v2() splits any chunk with a conflict at the sentence
# boundary between the conflicting beats, then re-merges back down to
# target_count using only SAFE adjacent merges (pairs that would not
# themselves create a new conflict), so the "34 documentary scenes" target
# is preserved whenever a safe merge is available. Only ADJACENT units are
# ever split or merged, so chronological/story order and every word of the
# narration are always preserved exactly - the same lossless guarantee
# _split_into_chunks already makes. If it runs out of safe merges before
# reaching target_count, it stops and returns the (slightly larger) result
# rather than merging two independent dated beats back together -
# correctness of the split takes priority over hitting the count exactly.
# ---------------------------------------------------------------------------

_YEAR_TOKEN_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def _dated_beat_years(unit: str) -> set[str]:
    """The set of distinct 4-digit year tokens (1600-2099) a sentence/clause
    contains - the structural proxy for "this unit anchors its own
    independently-citable documentary claim" (a publication date, an
    operating period, a discovery year, ...)."""
    return set(_YEAR_TOKEN_RE.findall(unit))


def _has_conflicting_dated_beats(units: list[str]) -> bool:
    """True if two or more of the given units each carry their own dated
    beat and share NO year - i.e. they anchor different points in time and
    are very likely two separate documentary claims. Units that share a
    year are NOT a conflict - that's one continuous beat."""
    year_sets = [ys for ys in (_dated_beat_years(u) for u in units) if ys]
    for i in range(len(year_sets)):
        for j in range(i + 1, len(year_sets)):
            if not (year_sets[i] & year_sets[j]):
                return True
    return False


def _split_chunk_at_dated_beats(chunk: str) -> list[str]:
    """Splits one already-built chunk at the sentence boundary between
    conflicting dated beats (see _has_conflicting_dated_beats), producing
    the fewest pieces that remove every conflict. A chunk with 0 or 1
    sentence, or with no conflicting beats, is returned unchanged (as a
    single-element list) - this only ever fires for the specific bug shape.
    Lossless: " ".join(result).split() == chunk.split() always, same
    guarantee _split_into_chunks/_split_clauses already make."""
    sentences = _split_sentences(chunk)
    if len(sentences) < 2:
        return [chunk]

    pieces: list[list[str]] = [[sentences[0]]]
    for sentence in sentences[1:]:
        if _has_conflicting_dated_beats(pieces[-1] + [sentence]):
            pieces.append([sentence])
        else:
            pieces[-1].append(sentence)
    if len(pieces) == 1:
        return [chunk]
    return [" ".join(p) for p in pieces]


def _rebalance_dated_beat_chunks(chunks: list[str], target_count: int) -> list[str]:
    """Post-processes the shared word-count-balanced chunker's output for
    V2: splits every chunk that bundles conflicting dated beats, then
    re-merges back down to target_count using only safe adjacent merges
    (never a merge that would itself create a new conflict), picking the
    smallest-combined-word-count safe pair each time to keep chunk sizes
    balanced. If it runs out of safe merges before reaching target_count, it
    stops and returns the (slightly larger) result. A no-op whenever no
    chunk has a conflict (e.g. every existing test fixture, which uses
    undated filler text) - byte-for-byte the same output _split_into_chunks
    already gave."""
    expanded: list[str] = []
    for chunk in chunks:
        expanded.extend(_split_chunk_at_dated_beats(chunk))

    while len(expanded) > target_count:
        safe_pairs = [
            i for i in range(len(expanded) - 1)
            if not _has_conflicting_dated_beats(
                _split_sentences(expanded[i]) + _split_sentences(expanded[i + 1])
            )
        ]
        if not safe_pairs:
            break
        best = min(
            safe_pairs,
            key=lambda i: len(expanded[i].split()) + len(expanded[i + 1].split()),
        )
        expanded[best:best + 2] = [(expanded[best] + " " + expanded[best + 1]).strip()]

    return expanded


def _split_into_chunks_v2(full_narration: str, target_count: int) -> list[str]:
    """V2 sibling of _split_into_chunks(): identical sentence/clause-level
    chunking (shared, unmodified), plus the dated-beat rebalancing pass
    above. v1's plan_scenes()/_split_into_chunks() call the original
    function directly and are completely unaffected by this."""
    chunks = _split_into_chunks(full_narration, target_count)
    return _rebalance_dated_beat_chunks(chunks, target_count)


# ---------------------------------------------------------------------------
# Indexed classification hardening (real production failure: "expected 35
# classifications, got 36" - a real Claude call split one input chunk into
# two classification objects, which desynced its own positional bookkeeping
# and caused the next chunk's content to be absorbed into the split's
# second half). Forensic analysis proved this is a genuine model
# instruction-following slip, not a chunker defect - the chunker (above)
# and SYSTEM_PROMPT_V2_CLASSIFICATION's existing ABSOLUTE RULE paragraph
# are both left unchanged by this fix, per the same project principle
# already applied to cardinality and exact_text leakage elsewhere in this
# file: an LLM instruction is necessary but never sufficient on its own -
# only deterministic Python code reliably enforces an invariant.
#
# The fix does NOT try to stop the model from ever splitting/merging again
# (nothing can guarantee that). Instead it makes every occurrence of this
# failure class precisely DETECTABLE and LOCALIZABLE (via the new required
# "input_chunk_index" field on every classification object - see the
# SYSTEM_PROMPT_V2_CLASSIFICATION field list above) and, for the common
# localized case, AUTOMATICALLY RECOVERABLE without a full re-run:
#
#   1. original full-batch call.
#   2. if the response is unclean (a duplicate/missing input_chunk_index)
#      AND every returned index is at least well-formed (no invalid
#      values to distrust): ONE targeted repair call containing ONLY the
#      affected chunks' own text, no surrounding context - splice the
#      repaired objects back in by index.
#   3. if still unclean (repair not attempted, or attempted and still
#      unclean): ONE full-batch retry, resending the complete, unmodified
#      original chunk set - the existing FAL_MAX_RETRIES-style "bounded,
#      then fail" pattern already used elsewhere in this codebase, applied
#      here to a content-repair retry rather than a transport retry.
#   4. if still unclean: the existing hard RuntimeError, unchanged in
#      spirit - never loops indefinitely, never silently accepts a
#      mismatch. _write_cardinality_diagnostic's payload is additively
#      enriched with exactly what was tried (see its own docstring).
#
# Every step re-validates through the SAME deterministic check
# (_analyze_classification_indices + _classification_indices_are_clean) -
# a repair or retry response is never trusted just because a model
# "should" have fixed it; it goes through the identical scrutiny the
# original response did.
# ---------------------------------------------------------------------------

_TARGETED_REPAIR_MAX_TOKENS = 4000


def _numbered_chunk_prompt(indexed_chunks: list[tuple[int, str]]) -> str:
    """Builds the same 'N. <text>' numbered-list user prompt the original
    classification call has always used - factored out so both the
    original full-batch call (every chunk, indices 1..N) and a targeted
    repair call (only the affected chunks, their REAL input_chunk_index
    numbers preserved, never renumbered 1..k) share one code path. Using
    each chunk's real index for a repair call means the model's existing
    'input_chunk_index must equal the number prefixing this chunk'
    instruction applies completely unchanged - no repair-specific prompt
    variant is needed."""
    numbered = "\n".join(f"{idx}. {text}" for idx, text in indexed_chunks)
    return f"Narration chunks:\n{numbered}"


def _request_classifications(user_prompt: str, max_tokens: int = 14000) -> list:
    """Thin wrapper around the one llm_client.chat_json call every stage of
    the recovery ladder shares (original call, targeted repair, full-batch
    retry) - only user_prompt/max_tokens ever differ between them; the
    system prompt and response-shape expectations are identical. Keeping
    this thin avoids duplicating the chat_json call site three times."""
    classifications, _ = llm_client.chat_json(SYSTEM_PROMPT_V2_CLASSIFICATION, user_prompt, max_tokens=max_tokens)
    return classifications


def _analyze_classification_indices(classifications, expected_indices: set[int]) -> dict:
    """Pure, deterministic analysis - never raises, never calls the LLM.
    Works for both the full-batch case (expected_indices = every chunk
    index 1..N) and the targeted-repair case (expected_indices = just the
    handful of chunks being repaired), which is why this single function
    backs every stage of the recovery ladder rather than duplicating the
    logic per stage.

    For each returned object, its OWN declared "input_chunk_index" - never
    its array position - is what determines which input chunk it's for
    (requirement D: never trust LLM array order). An object is counted as
    invalid (not duplicate/missing) when its input_chunk_index is absent,
    not an int (a bool is deliberately treated as not-an-int here too,
    since Python's bool is an int subclass and True/False could otherwise
    silently alias to chunk 1/0), or outside expected_indices.

    Returns:
        duplicate_indices - indices claimed by more than one object (the
            exact shape of the real chunk-18 failure: one input chunk
            answered with two classification objects).
        missing_indices - expected indices with zero valid objects.
        invalid_indices - [{"returned_index": <array position>, "value":
            <raw input_chunk_index value>}] for every object whose index
            couldn't be trusted at all.
        position_by_index - {valid index: [array positions]} - the lookup
            table callers use to pull the right object(s) out of
            `classifications` without re-scanning.
    """
    position_by_index: dict[int, list[int]] = {}
    invalid_indices: list[dict] = []

    for pos, cls in enumerate(classifications or []):
        cls = cls if isinstance(cls, dict) else None
        idx = cls.get("input_chunk_index") if cls is not None else None
        if isinstance(idx, bool) or not isinstance(idx, int) or idx not in expected_indices:
            invalid_indices.append({"returned_index": pos, "value": idx})
            continue
        position_by_index.setdefault(idx, []).append(pos)

    duplicate_indices = sorted(idx for idx, positions in position_by_index.items() if len(positions) > 1)
    missing_indices = sorted(expected_indices - set(position_by_index.keys()))

    return {
        "duplicate_indices": duplicate_indices,
        "missing_indices": missing_indices,
        "invalid_indices": invalid_indices,
        "position_by_index": position_by_index,
    }


def _classification_indices_are_clean(analysis: dict) -> bool:
    """True only when every expected index has exactly one valid object
    and nothing invalid slipped in - the single gate every stage of the
    recovery ladder checks before accepting a response as done."""
    return not analysis["duplicate_indices"] and not analysis["missing_indices"] and not analysis["invalid_indices"]


def _ordered_classifications_by_index(classifications: list, analysis: dict, expected_indices: set[int]) -> list:
    """Only called once `_classification_indices_are_clean` is True (so
    every index in expected_indices has EXACTLY one valid position) -
    returns the classifications re-ordered to match chunk order, sorted by
    input_chunk_index, never by whatever order the model's array happened
    to use (requirement D)."""
    position_by_index = analysis["position_by_index"]
    return [classifications[position_by_index[idx][0]] for idx in sorted(expected_indices)]


def _attempt_targeted_repair(chunks: list[str], repair_indices: list[int]) -> dict[int, dict] | None:
    """ONE narrow re-ask covering ONLY the chunks named in repair_indices -
    no surrounding chunks, on the theory (borne out by the forensic
    analysis of the real failure) that removing the batch context that
    likely invited a split/merge gives the model a much cleaner shot at
    classifying that one piece of text on its own. Reuses the exact same
    SYSTEM_PROMPT_V2_CLASSIFICATION and numbered-list convention as the
    original call, under each chunk's REAL input_chunk_index number.

    Returns {input_chunk_index: classification_object}, one entry per
    requested index, only when the repair response cleanly and completely
    covers exactly that set (same _analyze_classification_indices/
    _classification_indices_are_clean gate as everywhere else) - or None
    if it doesn't, so the caller falls through to a full-batch retry
    rather than splicing in a still-broken partial repair."""
    subset_prompt = _numbered_chunk_prompt([(idx, chunks[idx - 1]) for idx in repair_indices])
    repaired = _request_classifications(subset_prompt, max_tokens=_TARGETED_REPAIR_MAX_TOKENS)
    if not isinstance(repaired, list):
        return None

    expected = set(repair_indices)
    analysis = _analyze_classification_indices(repaired, expected)
    if not _classification_indices_are_clean(analysis):
        return None

    return {idx: repaired[analysis["position_by_index"][idx][0]] for idx in repair_indices}


def _splice_repaired_classifications(
    classifications: list,
    analysis: dict,
    repaired: dict[int, dict],
    expected_indices: set[int],
) -> list:
    """Rebuilds a full, index-ordered classifications list: the repaired
    objects for exactly the indices they cover, and the ORIGINAL
    response's own already-valid objects for every other index - never
    discards a part of the original response that was already correct
    just because a different part needed repair."""
    result = []
    for idx in sorted(expected_indices):
        if idx in repaired:
            result.append(repaired[idx])
        else:
            positions = analysis["position_by_index"][idx]
            result.append(classifications[positions[0]])
    return result


def _classify_chunks_v2(chunks: list[str]) -> list[dict]:
    """The classification call, now with the bounded recovery ladder
    described in this section's banner above. Returns the classifications
    list, guaranteed len(chunks)-long and sorted by input_chunk_index
    (never trusting the model's own array order), on success. Raises the
    same RuntimeError as before - message format unchanged - once the
    ladder is exhausted; the diagnostic file is written exactly once, only
    at that final point, additively enriched with what was tried."""
    expected_count = len(chunks)
    expected_indices = set(range(1, expected_count + 1))
    full_prompt = _numbered_chunk_prompt(list(enumerate(chunks, start=1)))

    classifications = _request_classifications(full_prompt)
    if not isinstance(classifications, list):
        raise RuntimeError("[SCENES][V2] expected a JSON array of scene classifications")

    analysis = _analyze_classification_indices(classifications, expected_indices)
    repair_attempted = False
    repair_indices: list[int] = []
    retry_stage = "initial"

    if _classification_indices_are_clean(analysis):
        return _ordered_classifications_by_index(classifications, analysis, expected_indices)

    # --- Stage 2: targeted repair - only when every offending object's own
    # index is trustworthy (no invalid_indices); an untrustworthy index
    # gives nothing reliable to anchor a narrow re-ask on, so that case
    # skips straight to the full-batch retry below.
    candidate_repair_indices = sorted(set(analysis["duplicate_indices"]) | set(analysis["missing_indices"]))
    if candidate_repair_indices and not analysis["invalid_indices"]:
        repair_attempted = True
        repair_indices = candidate_repair_indices
        repaired = _attempt_targeted_repair(chunks, repair_indices)
        if repaired is not None:
            retry_stage = "targeted_repair"
            classifications = _splice_repaired_classifications(classifications, analysis, repaired, expected_indices)
            analysis = _analyze_classification_indices(classifications, expected_indices)
            if _classification_indices_are_clean(analysis):
                return classifications  # _splice_repaired_classifications already returns index order

    # --- Stage 3: exactly one full-batch retry, unmodified original chunks.
    retry_classifications = _request_classifications(full_prompt)
    if isinstance(retry_classifications, list):
        retry_stage = "full_batch_retry"
        classifications = retry_classifications
        analysis = _analyze_classification_indices(classifications, expected_indices)
        if _classification_indices_are_clean(analysis):
            return _ordered_classifications_by_index(classifications, analysis, expected_indices)

    # --- Stage 4: recovery ladder exhausted - fail exactly as before,
    # with an additively-enriched diagnostic.
    diagnostic_path = _write_cardinality_diagnostic(
        chunks, classifications,
        extra={
            "duplicate_indices": analysis["duplicate_indices"],
            "missing_indices": analysis["missing_indices"],
            "invalid_indices": analysis["invalid_indices"],
            "repair_attempted": repair_attempted,
            "repair_indices": repair_indices,
            "retry_stage": retry_stage,
        },
    )
    returned_count = len(classifications)
    logger.error(
        "[SCENES][V2] cardinality mismatch: expected %d classifications, got %d - "
        "diagnostic written to %s",
        expected_count, returned_count, diagnostic_path,
    )
    raise RuntimeError(
        f"[SCENES][V2] expected {expected_count} classifications, got {returned_count} "
        f"(diagnostic written to {diagnostic_path})"
    )


def plan_scenes_v2(
    script: dict,
    localized_scripts: dict[str, dict] | None = None,
) -> list[dict]:
    """Scene Planner V2 - Phase 1 + CTA-separation fix + dated-beat chunk
    rebalancing: V2 classification + deterministic strategy resolution +
    validation, over DOCUMENTARY narration only. Chunking starts from the
    same deterministic pipeline as v1's plan_scenes()
    (_target_scene_count/_split_into_chunks), then runs one additional V2-
    only pass (_split_into_chunks_v2 - see its section banner above) that
    prevents two independently-dated documentary beats from being packed
    into the same chunk, while preserving the target chunk count whenever a
    safe rebalancing merge is available. v1's plan_scenes() is completely
    unaffected - it still calls _split_into_chunks() directly.

    The script's CTA/end-screen narration (script_generator.py's required
    "cta" section type - see _split_documentary_and_cta()) is deliberately
    excluded from chunking/classification entirely and never reaches the
    LLM: a real classification run returned 36 objects for 35 chunks
    because the final chunk secretly mixed the documentary's closing beat
    with the CTA, and the model - correctly - saw two scenes there. The CTA
    is instead represented as exactly one separate, fully deterministic
    scene (is_cta=True, zero LLM involvement) appended after the classified
    documentary scenes - see _build_cta_scene(). A script with no
    "sections"-based CTA (e.g. an older/synthetic {"full_narration": ...}
    caller) behaves exactly as before: nothing is appended.

    Does NOT produce a visual_prompt field (Phase 2's prompt builder is not
    written yet), does NOT call FAL/image_gen, and does NOT populate
    source_asset_ref/preferred_source/fallback_strategy/avoid_strategy/
    render_hash (Phase 3/5/6 scope - left as None/absent here). Every
    returned scene carries schema_version=2 so downstream code (and
    validate_scenes_v2()) can always tell a V2 scene from a legacy v1 one.
    Not called from run_and_save()/pipeline.py yet - that wiring is Phase 6.
    """
    documentary_narration, cta_narration_en = _split_documentary_and_cta(script)
    total_words = len(documentary_narration.split())
    target_count = _target_scene_count(total_words)

    chunks = _split_into_chunks_v2(documentary_narration, target_count)
    if len(chunks) < MIN_SCENES:
        logger.warning(
            "[SCENES][V2] only %d scene(s) reachable (target was %d, minimum is %d) - same "
            "shortfall condition v1's plan_scenes() logs, proceeding anyway",
            len(chunks), target_count, MIN_SCENES,
        )

    logger.info(
        "[SCENES][V2] classifying %d documentary scene(s) for %d word(s) of narration "
        "(CTA narration, if any, is handled separately - see _build_cta_scene)",
        len(chunks), total_words,
    )

    # Indexed classification + bounded repair/retry ladder - see the
    # "Indexed classification hardening" section banner above
    # _split_into_chunks_v2 for the full design. Returns a classifications
    # list that is guaranteed len(chunks)-long and sorted by
    # input_chunk_index (never trusting the model's own array order), or
    # raises the same RuntimeError as before once the ladder is exhausted.
    classifications = _classify_chunks_v2(chunks)

    localized_chunks, localized_cta = _chunk_localized_narrations_v2(localized_scripts or {}, len(chunks))

    scenes = []
    for i, (chunk, cls) in enumerate(zip(chunks, classifications), start=1):
        cls = cls or {}
        words = len(chunk.split())
        duration_hint = round(words / WORDS_PER_SECOND, 1)

        scene_type = cls.get("scene_type")
        evidence_level = cls.get("evidence_level")
        data_points = cls.get("data_points")
        if not isinstance(data_points, list):
            data_points = []

        resolved_scene_type, visual_strategy = resolve_visual_strategy(scene_type, evidence_level, data_points)
        exact_text = _derive_exact_text(data_points)
        text_render = _derive_text_render(exact_text, evidence_level)

        scene = {
            "scene_id": i,
            "schema_version": 2,
            "duration_hint": duration_hint,
            "narration_en": chunk,
            "claim": (cls.get("claim") or "").strip(),
            "scene_type": resolved_scene_type,
            "visual_strategy": visual_strategy,
            "evidence_level": evidence_level,
            "data_points": data_points,
            "exact_text": exact_text,
            "text_render": text_render,
            "visual_relevance_reason": (cls.get("visual_relevance_reason") or "").strip(),
            "prompt_segments": cls.get("prompt_segments") or {},
            "visual_type": _legacy_visual_type_for(resolved_scene_type),
            "source_asset_ref": None,
            "preferred_source": None,
            "fallback_strategy": None,
            "avoid_strategy": None,
            "is_cta": False,
        }
        for lang, loc_chunks in localized_chunks.items():
            if i - 1 < len(loc_chunks):
                scene[f"narration_{lang}"] = loc_chunks[i - 1]
        scenes.append(scene)

    cta_scene = _build_cta_scene(cta_narration_en, scene_id=len(chunks) + 1, localized_cta=localized_cta)
    if cta_scene is not None:
        scenes.append(cta_scene)

    validate_scenes_v2(scenes)
    return scenes


# ---------------------------------------------------------------------------
# Scene Planner V2 - Phase 2 ONLY (deterministic prompt builder). Purely
# additive on top of Phase 1: nothing above this line is touched.
# plan_scenes()/run_and_save()/sync_localized_narrations() (v1) and
# plan_scenes_v2()/resolve_visual_strategy()/validate_scenes_v2() (Phase 1)
# are completely unaffected - Phase 2 only ever READS the fields Phase 1
# produces, on a copy of each scene, and never re-invents or overrides a
# Phase 1 field. Not wired into run_and_save()/pipeline.py yet (later-phase
# scope). No FAL call, no image_gen change, no overlay renderer, no
# scientific-graphics renderer, no FFmpeg/thumbnail/uploader change.
#
# Design: visual_prompt is assembled from exactly two sources, both already
# guaranteed (by SYSTEM_PROMPT_V2_CLASSIFICATION and by construction here)
# to never contain a literal exact_text value - (1) the LLM's own
# prompt_segments (subject/setting/lighting/mood/composition/style - six
# segments), and (2) two further, purely deterministic segments this phase
# adds: a technical-requirements segment (base "no on-screen text" rule,
# strengthened for scenes with exact_text, plus accuracy hints for
# non-exact numeric data) and an avoid segment (resolve_avoid_strategy() -
# the first thing that gives the long-deferred `avoid_strategy` field real
# teeth: it actually shapes the prompt now, not just metadata). Eight
# segments in total. Neither deterministic segment - nor an LLM segment,
# which the classification prompt explicitly forbids from doing this - ever
# quotes an exact_text value, a coordinate, or a measurement's digits
# directly; scenes needing those on screen get them via post-production
# overlay (Phase 3, not built here), never via the AI image prompt. This is
# enforced twice: structurally by never constructing such text here, and
# defensively by validate_prompts_v2()'s hard leak check below.
# ---------------------------------------------------------------------------

_BASE_TECHNICAL_CONSTRAINT = (
    "16:9 cinematic documentary framing. Never include on-screen text, letters, "
    "numbers, captions, subtitles, watermarks, or logos in the image itself."
)

# Applied in addition to _BASE_TECHNICAL_CONSTRAINT whenever a scene has any
# exact_text - requirement: explicitly forbid the AI image model from
# rendering exact text, even though the scene concerns specific data: exact
# text/numbers are ALWAYS a post-production overlay concern (Phase 3), never
# an image-generation concern.
_NO_EXACT_TEXT_CONSTRAINT = (
    "This scene involves exact text/numbers that must be added in "
    "post-production, NEVER rendered by the image model - do not attempt to "
    "depict any specific characters, digits, codes, or coordinates anywhere "
    "in the image, even approximately. Leave clear, uncluttered space where "
    "such text can be overlaid afterward."
)

# scene_type -> deterministic avoid-phrases. Never LLM-authored (matches the
# same "Python-only, never LLM-decided" principle Phase 1 applied to
# visual_strategy) - resolve_avoid_strategy() is the only thing that
# produces this.
_AVOID_BY_SCENE_TYPE: dict[str, list[str]] = {
    "archival_source": [
        "presenting this as genuine, unaltered archival footage or a real historical photograph",
        "modern anachronistic elements (contemporary technology, clothing, or branding)",
    ],
    "document_recreation": [
        "presenting this as genuine, unaltered archival footage or a real historical photograph",
        "modern anachronistic elements (contemporary technology, clothing, or branding)",
    ],
    "scientific_visualization": [
        "generic stock 'science' cliches (glowing DNA helixes, random floating equations) unrelated to this specific concept",
    ],
    "scientific_graphic": [
        "chart or graph elements rendered as part of the photo - fabricated axes, labels, or data points",
    ],
    "astronomical_map": [
        "fictional constellations or star patterns inconsistent with real astronomy",
    ],
    "historical_reenactment": [
        "modern anachronistic elements (contemporary technology, clothing, or branding)",
        "presenting speculative reenactment details as documented certainty",
    ],
    "real_location": [
        "a generic, stock-photo-looking location instead of the specific real place described",
    ],
    "evidence_closeup": [
        "any legible text, numbers, or labels rendered directly in the image",
    ],
    "atmospheric_establishing": [
        "overly literal or on-the-nose imagery",
        "specific factual claims not supported by this scene's narration",
    ],
    "hypothesis_diagram": [
        "definitive, confirmed, or settled-fact visual framing",
        "presenting this hypothesis as proven or as the only possible explanation",
    ],
    "research_evidence": [
        "a generic 'scientist typing at a laptop' or generic stock-office visual cliche",
        "repeating the same visual setup used for other research_evidence scenes in this video - vary the setting, framing, or activity",
    ],
}

# Applied in addition to the scene_type table above, keyed on the
# (Python-resolved) visual_strategy rather than scene_type - catches the
# cases where the RENDERING METHOD itself, not the subject matter, implies
# an extra constraint.
_AVOID_FOR_HYPOTHESIS_VISUALIZATION = [
    "definitive or confirmed-looking visual framing",
    "imagery that reads as proof rather than a proposed idea",
]
_AVOID_FOR_OVERLAY_STRATEGY = [
    "a busy or cluttered composition that would leave no clear space for text to be overlaid in post-production",
]

# Soft (non-blocking) leakage heuristics - see _scan_suspicious_leakage().
_SUSPICIOUS_ALPHANUMERIC_CODE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b")
_SUSPICIOUS_COORDINATE_PATTERN = re.compile(
    r"\d+\s*(?:h|hr|hours)\s*\d+\s*m|\d+\s*°|right ascension|declination", re.IGNORECASE
)
_SUSPICIOUS_DIGIT_SEQUENCE = re.compile(r"\d{3,}")


def resolve_avoid_strategy(scene_type: str, visual_strategy: str) -> list[str]:
    """The ONLY place avoid_strategy is decided - pure, deterministic,
    Python-only (mirrors resolve_visual_strategy()'s own principle from
    Phase 1). Combines scene_type-specific avoid-phrases with
    visual_strategy-specific ones, de-duplicated, order preserved. This is
    what gives the previously-always-None `avoid_strategy` field (Phase 1
    left it unpopulated, deferred here) real effect: build_visual_prompt()
    below actually renders these phrases into the prompt's "Avoid:"
    segment, it does not just attach them as inert metadata."""
    avoid = list(_AVOID_BY_SCENE_TYPE.get(scene_type, []))
    if visual_strategy == "hypothesis_visualization":
        avoid.extend(_AVOID_FOR_HYPOTHESIS_VISUALIZATION)
    elif visual_strategy == "ai_generated_plus_overlay":
        avoid.extend(_AVOID_FOR_OVERLAY_STRATEGY)

    seen: set[str] = set()
    result: list[str] = []
    for phrase in avoid:
        if phrase not in seen:
            seen.add(phrase)
            result.append(phrase)
    return result


def _build_accuracy_hints(data_points: list[dict] | None) -> list[str]:
    """Requirement: numeric/scientific data_points that do NOT need exact
    on-screen rendering (exact_render is falsy) still deserve a hint so the
    image model conveys the right IDEA (precision, scale, a specific
    measurement being involved) without ever being told the actual digits -
    the literal value is deliberately never included here, only a generic,
    data_type-driven phrase, so this can never itself become an exact_text
    leak. A data_point with exact_render truthy is skipped entirely - it is
    exclusively a post-production overlay concern (_NO_EXACT_TEXT_CONSTRAINT
    already covers it), not an accuracy-hint concern."""
    hints: list[str] = []
    for dp in data_points or []:
        dp = dp or {}
        if dp.get("exact_render"):
            continue
        data_type = dp.get("data_type")
        if data_type in ("measurement", "scalar_fact"):
            hints.append(
                "This scene involves a specific numeric measurement - convey it "
                "conceptually (instrument readouts, scale, a sense of precision) "
                "without attempting to render exact digits or numbers anywhere in "
                "the image."
            )
        elif data_type == "coordinate":
            hints.append(
                "This scene involves specific sky coordinates - convey general "
                "position/direction conceptually without rendering exact "
                "coordinate figures in the image."
            )

    seen: set[str] = set()
    result: list[str] = []
    for hint in hints:
        if hint not in seen:
            seen.add(hint)
            result.append(hint)
    return result


# Root cause (real Phase 2.2 audit finding: scene 13, exact_text=
# ["Sagittarius"], leaked into visual_prompt): SYSTEM_PROMPT_V2_
# CLASSIFICATION instructs the LLM to "NEVER put a literal number, code,
# coordinate, or quoted exact string in prompt_segments" - but that is only
# an instruction, and the same lesson the cardinality rule already taught
# applies here too: an LLM instruction is not enforcement. The model can
# independently (and, from its own point of view, reasonably) describe a
# scene's setting/subject using the very same word it separately flagged as
# an exact_text data_point needing verbatim on-screen rendering - e.g.
# naming "Sagittarius" in `setting` ("the night sky toward the Sagittarius
# constellation") while ALSO declaring "Sagittarius" as an exact_text value
# elsewhere in the same classification. Both choices are individually
# sensible; their combination violates the invariant.
#
# Fixed deterministically, not by trusting the prompt harder: every one of
# the six LLM-authored prompt_segments strings is scrubbed of every
# exact_text value (case-insensitive, same substring semantics
# _exact_text_leak() itself uses) before being assembled into
# visual_prompt. This NEVER touches exact_text/data_points/text_render on
# the scene - those stay exactly as classified/derived, fully intact for
# the post-production overlay stage; only the AI-image-prompt text is
# scrubbed. validate_prompts_v2()'s hard leak check is completely
# unchanged and still runs on the result - this is what now normally keeps
# it from firing for this class of leak, not a replacement for it.
_EXACT_TEXT_REDACTION_PLACEHOLDER = "this specific detail"


def _scrub_exact_text_from_segments(prompt_segments: dict, exact_text: list[str] | None) -> dict:
    """Returns a NEW prompt_segments dict with every exact_text value
    removed (case-insensitive, whole-occurrence substring match - the same
    matching semantics _exact_text_leak() uses, so this scrubs exactly what
    that check would otherwise flag, no more and no less) from every
    segment string, replaced with a generic, data-type-agnostic
    placeholder. Pure and deterministic - never mutates the input dict, and
    a no-op copy when exact_text is empty (the overwhelming majority of
    scenes)."""
    segments = dict(prompt_segments or {})
    for value in exact_text or []:
        value = (value or "").strip()
        if not value:
            continue
        pattern = re.compile(re.escape(value), re.IGNORECASE)
        for key, text in list(segments.items()):
            if isinstance(text, str) and pattern.search(text):
                segments[key] = pattern.sub(_EXACT_TEXT_REDACTION_PLACEHOLDER, text)
    return segments


def build_visual_prompt(scene: dict) -> str | None:
    """Deterministic 8-segment prompt assembly - the ONLY function that
    produces visual_prompt in V2, and it is pure Python, never an LLM call.
    Six segments come verbatim from the LLM's own prompt_segments (subject/
    setting/lighting/mood/composition/style) - EXCEPT first scrubbed of any
    exact_text value via _scrub_exact_text_from_segments() (see its
    docstring/the comment above it for why this exists) - plus two more
    added here, purely deterministically: technical requirements (base
    constraint, strengthened for exact_text, plus accuracy hints) and avoid
    (from scene["avoid_strategy"], which must already be resolved on this
    scene - see build_scene_prompt()).

    Returns None for visual_strategy == "programmatic_graphic": that path
    is drawn by a chart/graphic renderer directly from data_points (Phase 4
    scope, not built here), never from an AI image prompt, so building one
    here would be dead output nobody consumes.

    Same input always produces the exact same output string - no
    randomness, no LLM, no I/O - which is what makes render_hash()
    (below) a meaningful identity for caching/resume purposes."""
    if scene.get("visual_strategy") == "programmatic_graphic":
        return None

    segments = _scrub_exact_text_from_segments(scene.get("prompt_segments") or {}, scene.get("exact_text"))
    subject = (segments.get("subject") or "").strip()
    setting = (segments.get("setting") or "").strip()
    lighting = (segments.get("lighting") or "").strip()
    mood = (segments.get("mood") or "").strip()
    composition = (segments.get("composition") or "").strip()
    style = (segments.get("style") or "").strip()

    technical = [_BASE_TECHNICAL_CONSTRAINT]
    if scene.get("exact_text"):
        technical.append(_NO_EXACT_TEXT_CONSTRAINT)
    technical.extend(_build_accuracy_hints(scene.get("data_points")))

    avoid = scene.get("avoid_strategy") or []

    parts = [
        f"Subject: {subject}",
        f"Setting: {setting}",
        f"Lighting: {lighting}",
        f"Mood: {mood}",
        f"Composition: {composition}",
        f"Style: {style}",
        f"Technical requirements: {' '.join(technical)}",
    ]
    if avoid:
        parts.append(f"Avoid: {'; '.join(avoid)}.")

    return "\n".join(parts)


def compute_render_hash(scene: dict) -> str:
    """Deterministic identity hash over RENDER-RELEVANT fields only
    (scene_type, visual_strategy, visual_prompt, data_points,
    avoid_strategy) - deliberately excludes scene_id/duration_hint/
    narration_*/claim/visual_relevance_reason, which can change (scene
    renumbering, narration rewording, a clearer relevance justification)
    without changing what should actually be rendered. sha256 over a
    canonical (sort_keys=True) JSON encoding, so the same rendering inputs
    always produce the same hash and any change to a render-relevant field
    changes it. Not wired into any caching/resume logic here - this only
    computes the hash; using it to skip re-rendering is later-phase scope."""
    payload = {
        "scene_type": scene.get("scene_type"),
        "visual_strategy": scene.get("visual_strategy"),
        "visual_prompt": scene.get("visual_prompt"),
        "data_points": scene.get("data_points"),
        "avoid_strategy": scene.get("avoid_strategy"),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_scene_prompt(scene: dict) -> dict:
    """Enriches ONE V2 scene with Phase 2's fields, returning a NEW dict (the
    input is never mutated) with every Phase 1 field preserved unchanged
    plus: avoid_strategy (resolved, replacing Phase 1's always-None
    placeholder), accuracy_hints, overlay_required, visual_prompt,
    render_hash. A legacy v1 scene (schema_version != 2) is returned
    completely untouched - Phase 2 never touches v1 scenes, same as every
    Phase 1 function.

    Requirement: visual_relevance_reason is mandatory before a prompt is
    built - Phase 1's validate_scenes_v2() already hard-fails on an empty
    one, but this is checked again here defensively, in case a caller ever
    hands scenes to this function without having run that gate first."""
    if scene.get("schema_version") != 2:
        return scene

    reason = scene.get("visual_relevance_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError(
            f"[SCENES][V2][PROMPT] scene {scene.get('scene_id', '?')}: visual_relevance_reason "
            f"is required before a visual_prompt can be built"
        )

    enriched = dict(scene)
    enriched["avoid_strategy"] = resolve_avoid_strategy(scene.get("scene_type"), scene.get("visual_strategy"))
    enriched["accuracy_hints"] = _build_accuracy_hints(scene.get("data_points"))
    enriched["overlay_required"] = scene.get("visual_strategy") == "ai_generated_plus_overlay"
    enriched["visual_prompt"] = build_visual_prompt(enriched)
    enriched["render_hash"] = compute_render_hash(enriched)
    return enriched


def _exact_text_leak(visual_prompt: str | None, exact_text: list[str] | None) -> list[str]:
    if not visual_prompt:
        return []
    lowered = visual_prompt.lower()
    return [value for value in (exact_text or []) if value and value.lower() in lowered]


def _scan_suspicious_leakage(visual_prompt: str) -> list[str]:
    """Soft (non-blocking) heuristic scan for numeric/alphanumeric-code-like
    text that may have leaked into visual_prompt despite
    SYSTEM_PROMPT_V2_CLASSIFICATION's instruction never to put exact
    strings/numbers into prompt_segments. Catches drift the hard
    exact_text-leak check (validate_prompts_v2, below) cannot: it looks for
    ANY suspicious pattern, not just values this scene itself declared as
    exact_text - e.g. a number the LLM restated despite instructions, or a
    coordinate-shaped string that was never formally captured as a
    data_point. Logged only (see validate_prompts_v2); never raises."""
    hits: list[str] = []
    for pattern, label in (
        (_SUSPICIOUS_ALPHANUMERIC_CODE, "alphanumeric code"),
        (_SUSPICIOUS_COORDINATE_PATTERN, "coordinate-like text"),
        (_SUSPICIOUS_DIGIT_SEQUENCE, "digit sequence"),
    ):
        for match in pattern.finditer(visual_prompt):
            hits.append(f"{label}: {match.group(0)!r}")
    return hits


def validate_prompts_v2(scenes: list[dict]) -> None:
    """POST-PROMPT-BUILD quality gate (Phase 2) - runs the exact_text-in-
    visual_prompt check that Phase 1's validate_scenes_v2() explicitly
    deferred here (Phase 1 never produced a visual_prompt to check against).
    Same collect-every-violation-then-raise-once pattern as
    validate_scenes_v2(); only checks schema_version==2 scenes, leaving
    legacy v1 scenes untouched. Also runs the soft suspicious-leakage scan
    per scene (logged via logger.warning, never raises - see
    _scan_suspicious_leakage)."""
    errors: list[str] = []

    for scene in scenes:
        if scene.get("schema_version") != 2:
            continue
        scene_id = scene.get("scene_id", "?")
        prefix = f"scene {scene_id}"
        visual_prompt = scene.get("visual_prompt")

        if scene.get("visual_strategy") == "programmatic_graphic":
            if visual_prompt is not None:
                errors.append(
                    f"{prefix}: visual_strategy=programmatic_graphic must not have a visual_prompt "
                    f"(drawn by a chart/graphic renderer from data_points, never an AI image model)"
                )
            continue

        if not visual_prompt:
            errors.append(f"{prefix}: missing visual_prompt for visual_strategy={scene.get('visual_strategy')!r}")
            continue

        leaked = _exact_text_leak(visual_prompt, scene.get("exact_text"))
        if leaked:
            errors.append(
                f"{prefix}: visual_prompt leaks exact_text value(s) {leaked} - the AI image model must "
                f"never be asked to render these; they belong to post-production overlay only"
            )

        suspicious = _scan_suspicious_leakage(visual_prompt)
        if suspicious:
            logger.warning(
                "[SCENES][V2][PROMPT] %s: suspicious numeric/text pattern(s) in visual_prompt "
                "(soft warning, not blocking): %s", prefix, suspicious,
            )

    if errors:
        raise RuntimeError(
            "[SCENES][V2][PROMPT] validate_prompts_v2 found {} issue(s):\n  - {}".format(
                len(errors), "\n  - ".join(errors)
            )
        )


def build_prompts_v2(scenes: list[dict]) -> list[dict]:
    """Scene Planner V2 - Phase 2 ONLY entry point: deterministic prompt
    building from Phase 1's classification output. Pure Python - no LLM
    call, no FAL call, no image generation, no overlay/scientific-graphics
    writing. Takes the scene list plan_scenes_v2() (Phase 1) produced and,
    for every schema_version==2 scene, builds visual_prompt/avoid_strategy/
    accuracy_hints/overlay_required/render_hash via build_scene_prompt() -
    never re-invents or overrides a Phase 1 field. Legacy v1 scenes pass
    through completely untouched, same as every Phase 1 function.

    Runs Phase 1's validate_scenes_v2() first (defensive precondition - a
    scene must already be structurally valid before a prompt is built from
    it), then builds every scene's prompt, then runs validate_prompts_v2()
    (the exact_text-in-visual_prompt check Phase 1 deferred to here) before
    returning.

    Not wired into run_and_save()/pipeline.py yet - later-phase scope."""
    validate_scenes_v2(scenes)
    built = [build_scene_prompt(scene) for scene in scenes]
    validate_prompts_v2(built)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan scenes from script.json (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        scenes = run_and_save(paths)
        print(f"OK - {len(scenes)} scenes saved to {paths.scenes_json}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
