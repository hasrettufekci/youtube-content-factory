"""
Script generation (spec sections 13-14).

Turns a research.json result into an 8-10 minute documentary script
(~1200-1600 English words) with the required structural beats:

    HOOK -> INTRODUCTION -> STORY DEVELOPMENT -> EVIDENCE ->
    MYSTERY/CONFLICT -> REVEAL -> CONCLUSION -> CTA

Output shape:
    {
      "title": "...",
      "hook": "...",
      "sections": [{"type": "hook", "title": "...", "narration": "..."}, ...],
      "full_narration": "...",   # = "\\n\\n".join(s["narration"] for s in sections), recomputed
                                   # in code so it's always consistent with "sections"
      "word_count": 1384,
      "sources": [...]           # carried over from research.json
    }

Standalone test:
    python -m youtube.script_generator --video-id 1
"""
from __future__ import annotations

import argparse
import sys

from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import llm_client

logger = get_logger("script")

REQUIRED_SECTION_TYPES = [
    "hook", "introduction", "story_development", "evidence",
    "mystery", "reveal", "conclusion", "cta",
]

TARGET_WORDS_MIN = 1200
TARGET_WORDS_MAX = 1600

SYSTEM_PROMPT = f"""You are a senior scriptwriter for a factual YouTube documentary channel.
Write an 8-10 minute narration script (target {TARGET_WORDS_MIN}-{TARGET_WORDS_MAX} English
words total) from the supplied research. Documentary tone: calm, authoritative, cinematic,
engaging - never tabloid, never robotic.

Structure (write ALL eight sections, in this order):
1. hook - first 15-30 seconds. A strong curiosity gap that earns the click and the watch.
   Never start with "Today we are going to..." or "In this video..." or similar cliches.
2. introduction - brief framing of the subject.
3. story_development - the main narrative.
4. evidence - documents, records, research, historical/scientific findings that support it.
5. mystery - what is known vs. what genuinely remains unknown or disputed.
6. reveal - the most important/surprising information, given real weight.
7. conclusion - a satisfying close, not abrupt.
8. cta - short, natural call-to-action (subscribe/comment), 1-2 sentences, not salesy.

RULES:
- Base every factual claim ONLY on the supplied research (summary/facts/timeline/sources).
  Do not invent facts, quotes, dates, or sources that are not in the research. If the
  research lists something under "uncertainties", present it as uncertain/disputed in the
  script too - never state it as settled fact.
- Avoid needless repetition of phrases or sentence structures across sections.
- Natural spoken-language sentences (this will be read aloud by a narrator), not written
  essay style.
- Output ONLY a single valid JSON object, no markdown/code fences/commentary, in this exact
  shape:

{{
  "title": "a strong, honest (not fabricated) documentary title",
  "sections": [
    {{"type": "hook", "title": "Hook", "narration": "..."}},
    {{"type": "introduction", "title": "...", "narration": "..."}},
    {{"type": "story_development", "title": "...", "narration": "..."}},
    {{"type": "evidence", "title": "...", "narration": "..."}},
    {{"type": "mystery", "title": "...", "narration": "..."}},
    {{"type": "reveal", "title": "...", "narration": "..."}},
    {{"type": "conclusion", "title": "...", "narration": "..."}},
    {{"type": "cta", "title": "...", "narration": "..."}}
  ]
}}"""


def _build_user_prompt(research: dict) -> str:
    facts = "\n".join(f"- {f}" for f in research.get("facts", []))
    timeline = "\n".join(f"- {t.get('date', '?')}: {t.get('event', '')}" for t in research.get("timeline", []))
    uncertainties = "\n".join(f"- {u}" for u in research.get("uncertainties", []))
    return f"""Topic: {research.get('topic')}

Summary:
{research.get('summary', '')}

Known facts:
{facts or '(none)'}

Timeline:
{timeline or '(none)'}

Uncertainties / disputed points (present these as uncertain, not fact):
{uncertainties or '(none)'}

Number of verified sources available: {len(research.get('sources', []))}
"""


def generate_script(research: dict) -> dict:
    topic = research.get("topic", "")
    logger.info("[SCRIPT] generating script for topic=%r", topic)

    data, _ = llm_client.chat_json(
        SYSTEM_PROMPT,
        _build_user_prompt(research),
        max_tokens=6000,
    )

    if "title" not in data or "sections" not in data:
        raise RuntimeError("[SCRIPT] Claude output missing 'title' or 'sections'")

    sections = data["sections"]
    got_types = [s.get("type") for s in sections]
    missing = [t for t in REQUIRED_SECTION_TYPES if t not in got_types]
    if missing:
        raise RuntimeError(f"[SCRIPT] script is missing required section(s): {missing}")

    for s in sections:
        if not s.get("narration", "").strip():
            raise RuntimeError(f"[SCRIPT] section {s.get('type')} has empty narration")

    full_narration = "\n\n".join(s["narration"].strip() for s in sections)
    word_count = len(full_narration.split())

    if word_count < TARGET_WORDS_MIN * 0.6 or word_count > TARGET_WORDS_MAX * 1.6:
        # Far enough off that this isn't a normal 8-10min script - fail loudly
        # rather than silently shipping a 2-minute or 25-minute "documentary".
        raise RuntimeError(
            f"[SCRIPT] word count {word_count} is far outside the "
            f"{TARGET_WORDS_MIN}-{TARGET_WORDS_MAX} target range"
        )
    if not (TARGET_WORDS_MIN <= word_count <= TARGET_WORDS_MAX):
        logger.warning(
            "[SCRIPT] word count %d is outside target %d-%d (proceeding anyway)",
            word_count, TARGET_WORDS_MIN, TARGET_WORDS_MAX,
        )

    hook_section = next((s for s in sections if s["type"] == "hook"), sections[0])

    script = {
        "title": data["title"],
        "hook": hook_section["narration"],
        "sections": sections,
        "full_narration": full_narration,
        "word_count": word_count,
        "sources": research.get("sources", []),
    }
    logger.info("[SCRIPT] done title=%r words=%d sections=%d", script["title"], word_count, len(sections))
    return script


def run_and_save(paths: ps.ProjectPaths) -> dict:
    # The precondition check (missing research.json) lives inside the try
    # block below, not before it: every failure here must go through the
    # same save_failure() path as a real generation error, or the video is
    # left stuck with no FAILED state recorded and no diagnostic trail.
    run_id = db.start_run(paths.video_id, "script")
    try:
        research = ps.load_json(paths.research_json)
        if not research:
            raise RuntimeError(f"[SCRIPT] no research.json found at {paths.research_json} - run researcher.py first")

        script = generate_script(research)
        ps.save_json(paths.script_json, script)
        ps.save_state(paths, PipelineState.SCRIPT_GENERATED)
        db.finish_run(run_id, "success")
        return script
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "script", str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a documentary script from research.json (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        run_and_save(paths)
        print(f"OK - script saved to {paths.script_json}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
