"""
Thumbnail Strategist (Thumbnail System V1 - stage 1 of 4:
    STRATEGIST -> FAL BACKGROUND GENERATION -> TEXT OVERLAY -> QC).

Consumes ONLY existing project artifacts - metadata_en.json, script_en.json,
research.json (whichever are available; script's "title" is the one
required field) - and DETERMINISTICALLY derives a structured "thumbnail
strategy" JSON that directs stage 2 (the FAL background prompt) and stage 3
(the exact headline/secondary text Pillow draws). It never makes a new
research/API call.

WHY NO LLM CALL:
This module makes NO call to youtube/llm_client.py (Claude) or any other
LLM, by design - it imports nothing from llm_client at all (see
tests/test_thumbnail_strategist.py's static-source check). Three reasons:
  1. The calling task's own cost/safety section is explicit that the
     thumbnail demo pipeline "should NOT call: Claude" - a strategist that
     silently reaches for an LLM would violate that on every run.
  2. youtube/metadata.py already establishes the precedent for this exact
     situation - "deliberately deterministic/templated (no extra LLM call)
     - the script already has a good title... this just assembles them"
     - a thumbnail headline is the same kind of derivation-from-existing-
       text problem metadata.py already solves without an LLM.
  3. A deterministic derivation is trivially safe against "the strategist
     MUST NOT invent factual claims": every word in the resulting headline/
     secondary_text is a substring of text a prior, already-fact-checked
     pipeline stage (script_generator.py, whose SYSTEM_PROMPT itself
     forbids fabrication) produced - this module only ever trims/selects,
     never generates new wording, so there is no new claim to invent.
If a future version genuinely needs LLM-quality headline writing, that
would follow llm_client.chat_json()'s existing pattern (see
youtube/thumbnail.py's older generate_thumbnail_text() for the shape) and
must log that an LLM call is being made explicitly - see this task's own
"If the strategist requires an LLM call, make that explicit in the log"
requirement. V1, as implemented here, requires none, and says so in its
own log line (see build_thumbnail_strategy()).

DERIVATION ALGORITHM (all deterministic - hashlib, never Python's salted
built-in hash(), so behavior is identical across processes/machines, not
just within one run):
  - headline/secondary_text: script["title"] is split on a colon/dash/pipe
    (a very common documentary-title pattern: "Punchy Hook: Explanatory
    Subtitle"). The SHORTER segment becomes the headline (capped to
    MAX_HEADLINE_WORDS_HARD whole words - never a mid-word cut); if a
    second, short (<= MAX_SECONDARY_WORDS words) segment exists, it
    becomes secondary_text. A single-segment (no punctuation split) title
    yields a headline with no secondary_text at all - this module never
    forces a secondary_text that would weaken the thumbnail, exactly as
    required. Nothing here is specific to any one video - the same
    algorithm derives a punchy headline + short subtitle from ANY
    "Hook Fragment: Explanatory Subtitle"-shaped project title, with no
    video-specific branch anywhere in this file (see
    tests/test_thumbnail_strategist.py's "no hardcoded production
    headline" check, which greps this file for the pilot project's own
    literal title/headline strings and requires they never appear here).
  - subject: research["topic"] (preferred - already a "concise version of
    the topic", per researcher.py's own schema) or script["title"],
    with a small set of generic scaffolding prefixes ("the mystery of ",
    "the story of ", ...) stripped - never anything invented.
  - emotion: a small deterministic keyword table over the topic/summary
    text, falling back to a fixed genre-appropriate default ("curiosity
    and wonder" - this channel's whole genre per metadata.py's own tag
    suffix table is mystery/documentary).
  - composition / text_style / avoid: fixed, channel-wide constants (art-
    direction guidance that is genre-appropriate for every video, not a
    per-video factual claim).
  - background_prompt: a template combining subject/emotion/composition
    with this task's own required negative-constraint list - always
    includes the literal string "NO TEXT" (see
    tests/test_thumbnail_strategist.py).
  - text_position: one of a curated, tested subset of the renderer's six
    valid semantic positions (see youtube/thumbnail_renderer.py -
    VALID_POSITIONS), chosen by a stable sha256 hash of the derived
    subject - deterministic (same project -> same position, every run,
    every process) while still varying project to project. Never an
    arbitrary pixel coordinate - the renderer only ever accepts one of
    these named positions and maps it to layout itself.
"""
from __future__ import annotations

import hashlib
import re

from core.logging_setup import get_logger

logger = get_logger("thumbnail_strategist")

MAX_HEADLINE_WORDS = 5          # "preferably 2-5 words"
MAX_HEADLINE_WORDS_HARD = 8     # never exceeded, whatever the source title looks like
MAX_SECONDARY_WORDS = 4

# The renderer (youtube/thumbnail_renderer.py) accepts all six semantic
# positions; the strategist only ever picks from this curated, tested
# subset - "top-*" is deliberately excluded here (a thumbnail's headline
# reads better lower/centered against a typical subject-in-upper-frame
# composition), without narrowing what the renderer itself supports.
_TEXT_POSITION_CHOICES = ("bottom-left", "bottom-right", "center-left", "center-right")

_TITLE_SPLIT_RE = re.compile(r"[:–—|]")
_TRAILING_PUNCT = ":-–— "

_FRAMING_PHRASES = (
    "the mystery of ", "the strange case of ", "the story of ", "the strange story of ",
    "what happened to ", "the truth about ", "the truth behind ", "inside ",
    "the secret of ", "the secret history of ",
)

_EMOTION_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("disaster", "tragedy", "crash", "died", "death", "killed", "vanished", "disappeared"),
     "somber, weighty tension"),
    (("hoax", "fraud", "scam", "conspiracy", "cover-up", "cover up", "lied", "lie"),
     "suspicion and skepticism"),
    (("discovery", "breakthrough", "found", "revealed", "uncovered"),
     "awe and revelation"),
    (("signal", "alien", "ufo", "unidentified", "unexplained", "mystery", "mysterious"),
     "curiosity and wonder"),
)
_DEFAULT_EMOTION = "curiosity and wonder"

_COMPOSITION = (
    "single dominant subject placed off-center per the rule of thirds, shallow depth of "
    "field, generous negative space on the opposite side of the frame reserved for text, "
    "no competing secondary subjects"
)

_NEGATIVE_CONSTRAINTS = (
    "no text, no letters, no numbers, no logos, no watermark, no UI, no typography"
)

_TEXT_STYLE = (
    "large bold sans-serif, high-contrast fill with a dark stroke outline and a soft drop "
    "shadow, tight controlled line spacing, no decorative or script typography"
)

_AVOID = (
    "clutter", "multiple competing subjects", "readable text or letters in the background",
    "logos or watermarks", "low contrast between subject and background",
)

STRATEGY_KEYS = (
    "headline", "secondary_text", "visual_concept", "subject", "emotion",
    "composition", "background_prompt", "text_position", "text_style", "avoid",
)


class ThumbnailStrategistError(RuntimeError):
    """Raised when the strategist cannot derive a valid, faithful strategy
    from the given project data - e.g. no usable title at all. Never
    raised for "the title was awkward to shorten"; the word-capping
    fallback below always produces SOMETHING from real title text."""


def _clean_words(text: str) -> list[str]:
    return [w for w in text.split() if w.strip()]


def _cap_words(text: str, max_words: int) -> str:
    words = _clean_words(text)
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


def _derive_headline_and_secondary(title: str) -> tuple[str, str | None]:
    title = title.strip()
    segments = [s.strip() for s in _TITLE_SPLIT_RE.split(title) if s.strip()]

    if len(segments) >= 2:
        headline_src = min(segments, key=lambda s: len(_clean_words(s)))
        secondary_src = next((s for s in segments if s != headline_src), None)
    else:
        headline_src = title
        secondary_src = None

    # Cap at the PREFERRED word count (5), not the hard ceiling (8) -
    # MAX_HEADLINE_WORDS_HARD exists only as an absolute validation
    # backstop (see _validate_strategy_schema()), never as the everyday
    # target: a single-segment title (no colon/dash to split on) should
    # still come out short and punchy, not merely "under 8 words".
    headline = _cap_words(headline_src, MAX_HEADLINE_WORDS)
    headline = headline.strip(_TRAILING_PUNCT).strip().upper()
    if not headline:
        raise ThumbnailStrategistError(f"derived headline from title {title!r} is empty")

    secondary_text = None
    if secondary_src:
        sec_words = _clean_words(secondary_src)
        if 0 < len(sec_words) <= MAX_SECONDARY_WORDS:
            candidate = secondary_src.strip(_TRAILING_PUNCT).strip().upper()
            if candidate:
                secondary_text = candidate

    return headline, secondary_text


def _derive_subject(research: dict | None, script: dict) -> str:
    topic = ""
    if research and str(research.get("topic") or "").strip():
        topic = research["topic"].strip()
    if not topic:
        topic = (script.get("title") or "").strip()
    if not topic:
        raise ThumbnailStrategistError("no research.topic or script.title available to derive a subject from")

    lowered = topic.lower()
    for phrase in _FRAMING_PHRASES:
        if lowered.startswith(phrase):
            topic = topic[len(phrase):]
            break
    return topic.strip().strip(".,!?").strip() or topic


def _derive_emotion(context_text: str) -> str:
    """Word-boundary keyword matching (never a bare substring `in` check):
    a naive substring match would false-positive on, e.g., "Alien"
    containing "lie" - \\b anchors every keyword to whole-word matches
    only, including multi-word phrases like "cover up"."""
    lowered = context_text.lower()
    for keywords, emotion in _EMOTION_KEYWORDS:
        if any(re.search(r"\b" + re.escape(k) + r"\b", lowered) for k in keywords):
            return emotion
    return _DEFAULT_EMOTION


def _select_text_position(seed_text: str) -> str:
    """Deterministic (hashlib, not Python's salted hash()) selection from
    the curated position subset - same seed_text always yields the same
    position, in every process, on every machine."""
    digest = hashlib.sha256(seed_text.strip().lower().encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(_TEXT_POSITION_CHOICES)
    return _TEXT_POSITION_CHOICES[index]


def _build_background_prompt(subject: str, emotion: str) -> str:
    return (
        f"Cinematic documentary photography of {subject}. Dramatic, high-contrast lighting, "
        f"{emotion} atmosphere. {_COMPOSITION}. Strong focal subject, negative space reserved "
        f"for a text overlay. Photorealistic, professional editorial quality. "
        f"NO TEXT. {_NEGATIVE_CONSTRAINTS}."
    )


def _validate_strategy_schema(strategy: dict) -> None:
    missing = [k for k in STRATEGY_KEYS if k not in strategy]
    if missing:
        raise ThumbnailStrategistError(f"strategy is missing required key(s): {missing}")

    headline = strategy["headline"]
    if not isinstance(headline, str) or not headline.strip():
        raise ThumbnailStrategistError("strategy.headline must be a non-empty string")
    if len(_clean_words(headline)) > MAX_HEADLINE_WORDS_HARD:
        raise ThumbnailStrategistError(
            f"strategy.headline has more than {MAX_HEADLINE_WORDS_HARD} words: {headline!r}"
        )

    secondary = strategy["secondary_text"]
    if secondary is not None and (not isinstance(secondary, str) or not secondary.strip()):
        raise ThumbnailStrategistError("strategy.secondary_text must be None or a non-empty string")

    from youtube.thumbnail_renderer import VALID_POSITIONS  # local import avoids a module-level cycle
    if strategy["text_position"] not in VALID_POSITIONS:
        raise ThumbnailStrategistError(
            f"strategy.text_position {strategy['text_position']!r} is not one of {VALID_POSITIONS}"
        )

    if "no text" not in strategy["background_prompt"].lower():
        raise ThumbnailStrategistError('strategy.background_prompt must include the "NO TEXT" instruction')

    if not isinstance(strategy["avoid"], list):
        raise ThumbnailStrategistError("strategy.avoid must be a list")


def build_thumbnail_strategy(
    script: dict, research: dict | None = None, metadata: dict | None = None,
) -> dict:
    """Pure function: derives and returns the structured thumbnail
    strategy JSON from already-loaded project data. Makes no I/O, no
    network call, no LLM call - see the module docstring's "WHY NO LLM
    CALL". Raises ThumbnailStrategistError (never a bare KeyError/TypeError)
    if `script` has no usable title."""
    if not script or not str(script.get("title") or "").strip():
        raise ThumbnailStrategistError("script has no usable title - cannot derive a thumbnail headline")

    title = script["title"].strip()
    headline, secondary_text = _derive_headline_and_secondary(title)

    subject = _derive_subject(research, script)
    context_text = " ".join(filter(None, [title, (research or {}).get("summary", "")]))
    emotion = _derive_emotion(context_text)
    background_prompt = _build_background_prompt(subject, emotion)
    text_position = _select_text_position(subject)

    strategy = {
        "headline": headline,
        "secondary_text": secondary_text,
        "visual_concept": f"{subject} - {emotion}, cinematic documentary framing",
        "subject": subject,
        "emotion": emotion,
        "composition": _COMPOSITION,
        "background_prompt": background_prompt,
        "text_position": text_position,
        "text_style": _TEXT_STYLE,
        "avoid": list(_AVOID),
    }
    _validate_strategy_schema(strategy)

    logger.info(
        "[THUMBNAIL_STRATEGIST] no LLM call made (deterministic derivation only) - "
        "title=%r headline=%r secondary_text=%r subject=%r position=%s "
        "(metadata_available=%s research_available=%s)",
        title, headline, secondary_text, subject, text_position,
        metadata is not None, research is not None,
    )
    return strategy
