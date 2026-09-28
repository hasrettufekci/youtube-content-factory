"""
Shorts Engine V1 - Shorts Planner (angle selection + script condensation).

CONTENT MODEL (spec, non-negotiable): ONE topic -> research -> Long script
-> Long video -> Shorts Planner -> Short 1/2/3, all derived from the SAME
Long research/script. This module performs NO second research call and
invents NO facts - it only selects and condenses text that is already
present in the Long's own script_en.json.

ANGLE SELECTION (design decision): script_en.json's structured sections
(youtube/script_generator.py's REQUIRED_SECTION_TYPES) are already
semantically labeled, so the three required Short angles map directly onto
them instead of a fragile new heuristic or a second LLM classification
pass:

    SHORT 1 - strongest/most curiosity-driven HOOK      -> sections["hook"]
    SHORT 2 - most surprising/counterintuitive fact      -> sections["reveal"]
              (script_generator.SYSTEM_PROMPT's own words for "reveal":
              "the most important/surprising information, given real weight")
    SHORT 3 - best twist/mystery/unexpected conclusion    -> sections["mystery"]
              (script_generator.SYSTEM_PROMPT's own words for "mystery":
              "what is known vs. what genuinely remains unknown or disputed")

Each angle's own section narration is then condensed (never expanded with
new material) into a short's-own ~30-50s narration script via ONE
llm_client.chat_json() call per Short, strictly grounded in that section's
own text plus the Long's own title (for a natural CTA reference) - never
given the full research payload, so there is no way for it to introduce a
claim the Long script itself does not already make.
"""
from __future__ import annotations

from core.config import settings
from core.logging_setup import get_logger
from youtube import llm_client

logger = get_logger("shorts_planner")

# (short_number, angle_label, source_section_type) - the fixed, generic
# (not per-topic hardcoded) mapping described above. A short_number beyond
# this list (SHORTS_PER_LONG > 3) has no defined angle yet in V1 - see
# plan_short_angles() below, which raises rather than guessing one.
ANGLE_PLAN: list[tuple[int, str, str]] = [
    (1, "hook", "hook"),
    (2, "surprising_fact", "reveal"),
    (3, "twist_mystery", "mystery"),
]

# DURATION HARD-GATE PATCH (2026-09): word budget now derived from
# settings.SHORTS_WORDS_PER_SECOND - a conservative, real-narration-pace
# estimate kept separate from scene_planner.py's/tts.py's own 2.5 wps
# constants (see core/config.py's SHORTS_WORDS_PER_SECOND docstring for
# why). HARD_WORDS_MIN/MAX bound the non-negotiable QC gate band;
# PREFERRED_WORDS_MIN/MAX bound the narrower sweet spot the prompt below
# actually asks the model to aim for (spec: "Tercihen: 40-48 saniye").
WORDS_PER_SECOND = settings.SHORTS_WORDS_PER_SECOND
TARGET_WORDS_MIN = max(int(settings.SHORTS_TARGET_SECONDS_MIN * WORDS_PER_SECOND), 1)
TARGET_WORDS_MAX = int(settings.SHORTS_TARGET_SECONDS_MAX * WORDS_PER_SECOND)
PREFERRED_WORDS_MIN = max(int(settings.SHORTS_PREFERRED_SECONDS_MIN * WORDS_PER_SECOND), 1)
PREFERRED_WORDS_MAX = int(settings.SHORTS_PREFERRED_SECONDS_MAX * WORDS_PER_SECOND)

SYSTEM_PROMPT = f"""You write a YouTube Shorts narration script, condensed from ONE section of an
existing, already-researched documentary script. This is a condensation task, not a new writing task.

LENGTH (IMPORTANT - real spoken narration, not a word-count guess, is what gets checked afterward):
- Absolute hard limits: the narration must read aloud in no less than {settings.SHORTS_TARGET_SECONDS_MIN}
  and no more than {settings.SHORTS_TARGET_SECONDS_MAX} seconds. A Short outside this band is rejected
  automatically after narration is recorded, so treat these as limits you must never cross.
- Preferred target: aim for {settings.SHORTS_PREFERRED_SECONDS_MIN}-{settings.SHORTS_PREFERRED_SECONDS_MAX}
  seconds of natural, unhurried spoken narration - roughly {PREFERRED_WORDS_MIN}-{PREFERRED_WORDS_MAX} words
  at a calm, deliberate documentary pace (never {TARGET_WORDS_MAX} words - that pace reads far closer to the
  {settings.SHORTS_TARGET_SECONDS_MAX}s ceiling than to the preferred range).
- If you are unsure whether a draft is a little long or a little short, prefer the shorter side of the
  preferred range - it is far safer than drifting toward the maximum.
- Never sacrifice a complete sentence or a natural ending to hit a word count. Write the best complete,
  well-paced mini-narration for the preferred range; do not pad it out just to reach the maximum either.

RULES (STRICT):
- Base the narration ONLY on the "SOURCE TEXT" given below. Do not add any fact, name, date, number,
  or claim that is not already present in the source text. Do not invent anything.
- Hook the viewer in the first sentence - no "Hey guys", no channel intro, no "In this video...", no
  long setup.
- Maintain the same calm, authoritative, cinematic documentary tone as the source text.
- The Short must work as an independent mini-story - a viewer who has never seen the Long video must
  still understand it.
- End with a short, honest call-to-action that never overstates what the full video adds (e.g.
  "Watch the full story on AI FOUND THIS." or a close natural variant) - never a false claim like
  promising an ending that isn't in the full video.
- English only.
- Output ONLY a single valid JSON object, no markdown/commentary, in this exact shape:
{{"title": "a short, punchy title for THIS short (not identical to the Long's title)", "narration": "..."}}"""


def _build_user_prompt(long_title: str, angle_label: str, section_title: str, source_text: str) -> str:
    return f"""Long video title: {long_title}
This Short's angle: {angle_label}
Source section title: {section_title}

SOURCE TEXT (the only material you may draw facts from):
{source_text}
"""


def _section_by_type(script: dict, section_type: str) -> dict | None:
    for section in script.get("sections", []):
        if section.get("type") == section_type:
            return section
    return None


def plan_short_angles(script: dict, shorts_per_long: int | None = None) -> list[dict]:
    """Pure/no-LLM-call selection step: for each configured Short slot,
    picks its angle label and its SOURCE section from the Long's own
    script_en.json - never touches research.json, never calls the LLM.
    Returns [{"short_number", "angle", "section_type", "section_title",
    "source_text"}, ...]. Raises ValueError if the script is missing a
    required section (should never happen for a script produced by
    script_generator.py, which already enforces REQUIRED_SECTION_TYPES)."""
    shorts_per_long = shorts_per_long if shorts_per_long is not None else settings.SHORTS_PER_LONG
    plans = []
    for short_number, angle_label, section_type in ANGLE_PLAN[:shorts_per_long]:
        section = _section_by_type(script, section_type)
        if section is None:
            raise ValueError(
                f"[SHORTS_PLANNER] script has no {section_type!r} section - cannot plan short_number={short_number}"
            )
        narration = (section.get("narration") or "").strip()
        if not narration:
            raise ValueError(
                f"[SHORTS_PLANNER] section {section_type!r} has empty narration - cannot plan short_number={short_number}"
            )
        plans.append({
            "short_number": short_number,
            "angle": angle_label,
            "section_type": section_type,
            "section_title": section.get("title", section_type),
            "source_text": narration,
        })
    if shorts_per_long > len(ANGLE_PLAN):
        raise ValueError(
            f"[SHORTS_PLANNER] SHORTS_PER_LONG={shorts_per_long} exceeds the number of defined angles "
            f"({len(ANGLE_PLAN)}) - Shorts Engine V1 only defines angles for hook/reveal/mystery"
        )
    return plans


def generate_short_script(long_title: str, plan: dict) -> dict:
    """ONE llm_client.chat_json() call, strictly grounded in plan["source_text"]
    only (never the full research/script) - see module docstring. Returns
    {"title", "narration", "word_count"}."""
    data, _ = llm_client.chat_json(
        SYSTEM_PROMPT,
        _build_user_prompt(long_title, plan["angle"], plan["section_title"], plan["source_text"]),
        max_tokens=1200,
    )
    narration = str(data.get("narration", "")).strip()
    title = str(data.get("title", "")).strip()
    if not narration:
        raise RuntimeError(f"[SHORTS_PLANNER] empty narration returned for short_number={plan['short_number']}")
    if not title:
        title = plan["section_title"]
    word_count = len(narration.split())
    logger.info("[SHORTS_PLANNER] short_number=%d angle=%s title=%r words=%d",
                plan["short_number"], plan["angle"], title, word_count)
    # Visibility only (NEVER mechanically truncated/padded here - the real,
    # ffprobe-measured audio duration is what youtube/shorts_qc.py's hard
    # gate actually enforces; see that module). A word count outside the
    # hard band is logged loudly so a drifting planner/prompt is noticed
    # long before QC would catch it downstream.
    if not (TARGET_WORDS_MIN <= word_count <= TARGET_WORDS_MAX):
        logger.warning(
            "[SHORTS_PLANNER] short_number=%d words=%d outside hard target word band (%d-%d, ~%.1fs-%.1fs "
            "at %.2f words/sec) - narration NOT altered here; real duration will be hard-gated by shorts_qc.py",
            plan["short_number"], word_count, TARGET_WORDS_MIN, TARGET_WORDS_MAX,
            settings.SHORTS_TARGET_SECONDS_MIN, settings.SHORTS_TARGET_SECONDS_MAX, WORDS_PER_SECOND,
        )
    elif not (PREFERRED_WORDS_MIN <= word_count <= PREFERRED_WORDS_MAX):
        logger.info(
            "[SHORTS_PLANNER] short_number=%d words=%d inside the hard band but outside the preferred "
            "%d-%d word sweet spot (~%ds-%ds)", plan["short_number"], word_count,
            PREFERRED_WORDS_MIN, PREFERRED_WORDS_MAX,
            settings.SHORTS_PREFERRED_SECONDS_MIN, settings.SHORTS_PREFERRED_SECONDS_MAX,
        )
    return {"title": title, "narration": narration, "word_count": word_count}


def plan_and_generate_shorts(script: dict, shorts_per_long: int | None = None) -> list[dict]:
    """Convenience wrapper: plan_short_angles() + generate_short_script()
    for every configured slot. Returns
    [{"short_number", "angle", "section_type", "title", "narration",
    "word_count"}, ...]."""
    plans = plan_short_angles(script, shorts_per_long=shorts_per_long)
    results = []
    for plan in plans:
        generated = generate_short_script(script.get("title", ""), plan)
        results.append({
            "short_number": plan["short_number"],
            "angle": plan["angle"],
            "section_type": plan["section_type"],
            "title": generated["title"],
            "narration": generated["narration"],
            "word_count": generated["word_count"],
        })
    return results
