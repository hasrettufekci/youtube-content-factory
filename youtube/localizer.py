"""
Script localization (Round 3 sections 3-4).

Turns the canonical English master script (script_en.json) into a natural,
professionally-narrated documentary script in another language - NOT a
literal/word-for-word translation. The localized script must preserve:

    facts | chronology | meaning | uncertainty | story structure | scene relationship

and must never introduce a new fact that wasn't in the English master.

Runs strictly after script_generator.py and strictly before scene_planner.py
in the pipeline (RESEARCH -> ENGLISH MASTER SCRIPT -> LOCALIZATION -> SHARED
SCENE PLAN -> ...), and produces one script_<lang>.json per configured
localization language (core.config.settings.LANGUAGE_LOCALIZATIONS) - never
touching research.json, images, or any other language's files, so adding a
future language never re-triggers research or visual generation (spec
section 21).

Output shape - a compatible superset of script_en.json's own shape (same
"title"/"hook"/"sections"/"full_narration"/"word_count"/"sources" keys, so
youtube/metadata.py's existing build_metadata() works unchanged for every
language), plus "language"/"source_language":

    {
      "language": "tr",
      "source_language": "en",
      "title": "...",             # localized title
      "hook": "...",              # = sections[0]["narration"], recomputed in code
      "sections": [{"type": "hook", "title": "...", "narration": "..."}, ...],
      "full_narration": "...",    # recomputed in code, never trusted from the LLM
      "word_count": 1180,
      "sources": [...]            # carried over from script_en.json unchanged - a
                                   # documentary's cited sources are not "translated"
    }

Standalone test:
    python -m youtube.localizer --video-id 1 --language tr
"""
from __future__ import annotations

import argparse
import sys

from core.logging_setup import get_logger
from storage import db, project_store as ps
from youtube import llm_client

logger = get_logger("localizer")

# Human-readable names for the languages this project currently knows how to
# prompt for. Extending to a new language (spec section 1/9: es, de, fr, it,
# pt, id, ...) is adding one line here plus the language's code to
# LANGUAGE_LOCALIZATIONS in .env - no other code changes.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "tr": "Turkish",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "id": "Indonesian",
}


def _system_prompt(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, language)
    return f"""You are a professional documentary localization writer, native in {name}. You will be
given a finished English documentary narration script, section by section, and must produce a
natural, professionally-narrated {name} version of it - the kind a real {name}-language
documentary channel would air, not a machine-translated or literal word-for-word rendering.

RULES (all mandatory):
- Preserve exactly: every fact, every date/number, the chronology, the overall meaning, which
  points are presented as UNCERTAIN or disputed (never upgrade an uncertain point to a stated
  fact, or downgrade a fact to uncertain), the story structure, and how each section relates to
  the scene/visual it will accompany.
- Do NOT invent, add, drop, or embellish any fact, quote, date, or claim that is not already in
  the supplied English text.
- Write natural, fluent, spoken-language {name} (this will be read aloud by a narrator) with the
  same calm, authoritative, cinematic documentary tone as the English original - not a stiff or
  unnatural translation, not literal English sentence structure carried over, and no unnecessary
  English loanwords when a natural {name} phrase exists.
- Keep the exact same section structure: the same number of sections, in the same order, with
  the same "type" values as given.
- Output ONLY a single valid JSON object, no markdown/code fences/commentary, in this exact
  shape:

{{
  "title": "a natural {name} documentary title (not a literal translation of the English title -
             a title that would genuinely work for a {name}-speaking audience, describing the
             same story)",
  "sections": [
    {{"type": "<same type as the matching input section>", "title": "...", "narration": "..."}},
    ...
  ]
}}"""


def _build_user_prompt(script_en: dict) -> str:
    parts = [f"English documentary title: {script_en.get('title', '')}", "", "Sections:"]
    for s in script_en.get("sections", []):
        parts.append(f"\n[{s.get('type')}] {s.get('title', '')}\n{s.get('narration', '')}")
    return "\n".join(parts)


def localize_script(script_en: dict, language: str = "tr") -> dict:
    en_sections = script_en.get("sections", [])
    en_types = [s.get("type") for s in en_sections]
    logger.info("[LOCALIZE] localizing script to language=%s (%d sections)", language, len(en_sections))

    data, _ = llm_client.chat_json(
        _system_prompt(language),
        _build_user_prompt(script_en),
        max_tokens=6000,
    )

    if "title" not in data or "sections" not in data:
        raise RuntimeError(f"[LOCALIZE] {language}: output missing 'title' or 'sections'")

    sections = data["sections"]
    got_types = [s.get("type") for s in sections]
    if got_types != en_types:
        # Scene relationship (spec section 3) depends on the localized
        # sections lining up 1:1, in order, with the English master's - a
        # dropped/reordered/added section would silently desync the two
        # languages' scene narration later, in scene_planner.py.
        raise RuntimeError(
            f"[LOCALIZE] {language}: section types {got_types} do not match the English "
            f"master's {en_types} (same count, same order, same 'type' values required)"
        )

    for s in sections:
        if not s.get("narration", "").strip():
            raise RuntimeError(f"[LOCALIZE] {language}: section {s.get('type')} has empty narration")

    full_narration = "\n\n".join(s["narration"].strip() for s in sections)
    word_count = len(full_narration.split())
    hook_section = next((s for s in sections if s["type"] == "hook"), sections[0])

    localized = {
        "language": language,
        "source_language": script_en.get("language", "en"),
        "title": data["title"],
        "hook": hook_section["narration"],
        "sections": sections,
        "full_narration": full_narration,
        "word_count": word_count,
        "sources": script_en.get("sources", []),
    }
    logger.info("[LOCALIZE] done language=%s title=%r words=%d", language, localized["title"], word_count)
    return localized


def run_and_save(paths: ps.ProjectPaths, language: str = "tr") -> dict:
    stage_label = f"localization_{language}"
    run_id = db.start_run(paths.video_id, stage_label)
    try:
        script_en = ps.load_json(paths.script_en_json)
        if not script_en:
            raise RuntimeError(
                f"[LOCALIZE] no script_en.json found at {paths.script_en_json} - run script_generator.py first"
            )

        localized = localize_script(script_en, language=language)
        ps.save_json(paths.script_localized_json(language), localized)
        db.finish_run(run_id, "success")
        return localized
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, stage_label, str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Localize script_en.json into another language (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--language", default="tr")
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        run_and_save(paths, language=args.language)
        print(f"OK - localized script saved to {paths.script_localized_json(args.language)}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
