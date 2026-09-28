"""
Topic scoring (spec section 12).

Scores a candidate topic on content-merit criteria only:
    curiosity, uniqueness, search_potential, documentary_potential,
    visual_potential, factual_reliability, competition, title_potential

IMPORTANT (spec section 12): this module intentionally has no concept of
political alignment, ideology, or any preference/ranking over political
topics - it only ever judges documentary-production merits (curiosity,
visuals, verifiability, etc.) of the approved content categories in
projects/*.json (mysteries, history, science, unexplained places...).
It must never be extended with a "political topic" scoring dimension.

Two backends:
- heuristic (default, no API key needed, deterministic, unit-testable)
- llm (optional, higher quality, requires ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import re

from core.logging_setup import get_logger
from youtube import llm_client

logger = get_logger("scorer")

CRITERIA = [
    "curiosity",
    "uniqueness",
    "search_potential",
    "documentary_potential",
    "visual_potential",
    "factual_reliability",
    "competition",
    "title_potential",
]

_CURIOSITY_WORDS = (
    "mystery", "mysterious", "unexplained", "secret", "lost", "disappeared",
    "vanished", "unsolved", "hidden", "forbidden", "ancient", "cursed",
    "strange", "no one knows", "nobody", "impossible", "undiscovered",
    "forgotten", "abandoned", "haunted",
)
_VISUAL_WORDS = (
    "place", "city", "island", "temple", "ruins", "ocean", "forest",
    "mountain", "cave", "desert", "ship", "tomb", "pyramid", "castle",
    "underground", "jungle",
)


def score_topic_heuristic(topic: str, category: str | None = None) -> dict:
    t = topic.lower()
    words = t.split()

    curiosity = min(10, 4 + 2 * sum(1 for w in _CURIOSITY_WORDS if w in t))
    visual_potential = min(10, 4 + 2 * sum(1 for w in _VISUAL_WORDS if w in t))
    uniqueness = 8 if len(words) >= 6 else 5  # specific titles tend to be less generic
    title_potential = min(10, max(3, 10 - abs(len(words) - 9)))  # ~9 words reads well as a title
    documentary_potential = min(10, round((curiosity + visual_potential) / 2))
    factual_reliability = 5  # unknown until researcher.py actually verifies it
    search_potential = 6 if any(w in t for w in _CURIOSITY_WORDS) else 5
    competition = 6  # neutral default; only research/real trend data could refine this

    scores = {
        "curiosity": curiosity,
        "uniqueness": uniqueness,
        "search_potential": search_potential,
        "documentary_potential": documentary_potential,
        "visual_potential": visual_potential,
        "factual_reliability": factual_reliability,
        "competition": competition,
        "title_potential": title_potential,
    }
    overall = round(sum(scores.values()) / len(scores), 2)
    return {
        "topic": topic,
        "category": category,
        "backend": "heuristic",
        "scores": scores,
        "overall": overall,
        "notes": "Heuristic score (keyword/length based). Re-score with an LLM or "
                 "after research.py for a more meaningful factual_reliability figure.",
    }


_LLM_SYSTEM_PROMPT = """You score candidate topics for a factual YouTube documentary channel
(mysteries, unexplained places, history, science, lost civilizations, extraordinary human
stories). Score ONLY documentary/content merit - never political alignment or ideology.

Return ONLY a JSON object:
{
  "scores": {
    "curiosity": 0-10, "uniqueness": 0-10, "search_potential": 0-10,
    "documentary_potential": 0-10, "visual_potential": 0-10,
    "factual_reliability": 0-10, "competition": 0-10, "title_potential": 0-10
  },
  "notes": "one or two sentences justifying the scores"
}

competition: 10 means underexplored/fresh on YouTube, 0 means saturated with many similar videos.
factual_reliability: how likely this topic can be covered with real, verifiable sources (not
how sensational it sounds)."""


def score_topic_llm(topic: str, category: str | None = None) -> dict:
    user = f"Topic: {topic}"
    if category:
        user += f"\nCategory: {category}"
    data, _ = llm_client.chat_json(_LLM_SYSTEM_PROMPT, user, max_tokens=500)
    scores = {k: float(data["scores"][k]) for k in CRITERIA}
    overall = round(sum(scores.values()) / len(scores), 2)
    return {
        "topic": topic,
        "category": category,
        "backend": "llm",
        "scores": scores,
        "overall": overall,
        "notes": data.get("notes", ""),
    }


def score_topic(topic: str, category: str | None = None, use_llm: bool = False) -> dict:
    if use_llm:
        try:
            return score_topic_llm(topic, category=category)
        except Exception as e:
            logger.warning("[SCORER] LLM scoring failed (%s), falling back to heuristic", e)
    return score_topic_heuristic(topic, category=category)
