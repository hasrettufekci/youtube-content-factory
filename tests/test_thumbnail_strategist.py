"""
youtube/thumbnail_strategist.py tests.

No I/O, no network, no LLM - build_thumbnail_strategy() is a pure function
over already-loaded script/research/metadata dicts. Covers:
    1  strategist schema
    2  missing headline
    3  excessive headline length
    4  no secondary text
    5  deterministic position mapping
   19  background prompt contains NO TEXT instruction
   20  no hardcoded Wow! Signal production headline
plus determinism, word-boundary emotion matching (regression for the
"Alien" / "lie" false-positive caught during development), and the
"never calls an LLM" static/behavioral guarantees.
"""
from __future__ import annotations

import ast

import pytest

from youtube import thumbnail_strategist as ts
from youtube.thumbnail_renderer import VALID_POSITIONS


def _script(title: str) -> dict:
    return {"title": title, "hook": "hook text", "sections": [], "full_narration": ""}


# ---------------------------------------------------------------------------
# 1 - strategist schema
# ---------------------------------------------------------------------------

def test_strategy_has_all_required_keys():
    strategy = ts.build_thumbnail_strategy(_script("The Wow! Signal: An Alien Message?"))
    assert set(ts.STRATEGY_KEYS) <= set(strategy.keys())


def test_strategy_field_types():
    strategy = ts.build_thumbnail_strategy(_script("The Wow! Signal: An Alien Message?"))
    assert isinstance(strategy["headline"], str)
    assert strategy["secondary_text"] is None or isinstance(strategy["secondary_text"], str)
    assert isinstance(strategy["visual_concept"], str)
    assert isinstance(strategy["subject"], str)
    assert isinstance(strategy["emotion"], str)
    assert isinstance(strategy["composition"], str)
    assert isinstance(strategy["background_prompt"], str)
    assert isinstance(strategy["text_position"], str)
    assert isinstance(strategy["text_style"], str)
    assert isinstance(strategy["avoid"], list)


# ---------------------------------------------------------------------------
# 2 - missing headline (no usable title at all)
# ---------------------------------------------------------------------------

def test_missing_title_key_raises():
    with pytest.raises(ts.ThumbnailStrategistError, match="usable title"):
        ts.build_thumbnail_strategy({"hook": "no title here"})


def test_blank_title_raises():
    with pytest.raises(ts.ThumbnailStrategistError, match="usable title"):
        ts.build_thumbnail_strategy(_script("   "))


def test_empty_script_raises():
    with pytest.raises(ts.ThumbnailStrategistError):
        ts.build_thumbnail_strategy({})


# ---------------------------------------------------------------------------
# 3 - excessive headline length is capped, never left unbounded
# ---------------------------------------------------------------------------

def test_long_single_segment_title_is_capped_to_preferred_word_count():
    long_title = "What Really Happened During The Great Unexplained Disappearance Of The Entire Expedition Team"
    strategy = ts.build_thumbnail_strategy(_script(long_title))
    assert len(strategy["headline"].split()) <= ts.MAX_HEADLINE_WORDS


def test_headline_never_exceeds_hard_word_ceiling():
    long_title = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    strategy = ts.build_thumbnail_strategy(_script(long_title))
    assert len(strategy["headline"].split()) <= ts.MAX_HEADLINE_WORDS_HARD


def test_excessively_long_secondary_segment_is_dropped_not_forced():
    title = "The Signal: A Very Long Explanatory Subtitle That Nobody Would Want On A Thumbnail"
    strategy = ts.build_thumbnail_strategy(_script(title))
    # the second segment has way more than MAX_SECONDARY_WORDS words - must be dropped, not truncated onto the thumbnail
    assert strategy["secondary_text"] is None


# ---------------------------------------------------------------------------
# 4 - no secondary text when the title doesn't naturally split
# ---------------------------------------------------------------------------

def test_single_segment_title_has_no_secondary_text():
    strategy = ts.build_thumbnail_strategy(_script("The Antikythera Mechanism"))
    assert strategy["secondary_text"] is None


def test_colon_split_title_produces_both_headline_and_secondary():
    strategy = ts.build_thumbnail_strategy(_script("The Wow! Signal: An Alien Message?"))
    assert strategy["headline"] == "THE WOW! SIGNAL"
    assert strategy["secondary_text"] == "AN ALIEN MESSAGE?"


def test_dash_split_title_also_works():
    strategy = ts.build_thumbnail_strategy(_script("Roanoke — Lost Forever"))
    assert strategy["headline"]
    assert "ROANOKE" in strategy["headline"] or "LOST FOREVER" in strategy["headline"]


# ---------------------------------------------------------------------------
# 5 - deterministic position mapping
# ---------------------------------------------------------------------------

def test_position_mapping_deterministic_same_input_same_output():
    strategy_a = ts.build_thumbnail_strategy(_script("The Dyatlov Pass Incident"))
    strategy_b = ts.build_thumbnail_strategy(_script("The Dyatlov Pass Incident"))
    assert strategy_a["text_position"] == strategy_b["text_position"]


def test_position_mapping_varies_across_different_subjects():
    positions = {
        ts.build_thumbnail_strategy(_script(title))["text_position"]
        for title in ["The Wow! Signal", "Roanoke", "The Mary Celeste", "D.B. Cooper", "Kaspar Hauser"]
    }
    assert len(positions) > 1, "position selection should not collapse every project onto the same value"


def test_position_always_a_renderer_valid_position():
    for title in ["Alpha", "Beta Gamma", "The Long Winding Mystery Of Something Or Other"]:
        strategy = ts.build_thumbnail_strategy(_script(title))
        assert strategy["text_position"] in VALID_POSITIONS


def test_invalid_position_mapping_would_be_rejected_by_own_validation(monkeypatch):
    monkeypatch.setattr(ts, "_select_text_position", lambda seed: "middle-of-nowhere")
    with pytest.raises(ts.ThumbnailStrategistError, match="text_position"):
        ts.build_thumbnail_strategy(_script("Some Topic"))


# ---------------------------------------------------------------------------
# 19 - background prompt contains the required "NO TEXT" instruction
# ---------------------------------------------------------------------------

def test_background_prompt_contains_no_text_instruction():
    strategy = ts.build_thumbnail_strategy(_script("The Wow! Signal: An Alien Message?"))
    assert "NO TEXT" in strategy["background_prompt"]


def test_background_prompt_contains_all_required_negative_constraints():
    strategy = ts.build_thumbnail_strategy(_script("Some Topic"))
    prompt_lower = strategy["background_prompt"].lower()
    for required in ("no text", "no letters", "no numbers", "no logos", "no watermark", "no ui", "no typography"):
        assert required in prompt_lower, f"missing required constraint: {required!r}"


def test_background_prompt_never_contains_the_headline_text_itself():
    """AI must not render the final thumbnail text - the background prompt
    must not smuggle the literal headline string into the image prompt."""
    strategy = ts.build_thumbnail_strategy(_script("The Wow! Signal: An Alien Message?"))
    assert strategy["headline"] not in strategy["background_prompt"]


# ---------------------------------------------------------------------------
# 20 - no hardcoded Wow! Signal production headline
# ---------------------------------------------------------------------------

def test_source_has_no_hardcoded_wow_signal_strings():
    source = open(ts.__file__, encoding="utf-8").read()
    for banned in ("THE WOW! SIGNAL", "ALIEN MESSAGE", "the-wow-signal", "wow_signal"):
        assert banned.lower() not in source.lower(), f"found hardcoded production string: {banned!r}"


def test_headline_is_genuinely_derived_not_special_cased():
    """The exact same derivation algorithm must produce a DIFFERENT
    headline for a different title - proving there is no special case
    keyed on the Wow! Signal project specifically."""
    wow = ts.build_thumbnail_strategy(_script("The Wow! Signal: An Alien Message?"))
    other = ts.build_thumbnail_strategy(_script("The Roswell Incident: Cover-Up Or Coincidence?"))
    assert wow["headline"] != other["headline"]
    assert other["headline"] == "THE ROSWELL INCIDENT"
    assert other["secondary_text"] == "COVER-UP OR COINCIDENCE?"


# ---------------------------------------------------------------------------
# Never calls an LLM (Claude/OpenAI) or the research stage
# ---------------------------------------------------------------------------

def test_strategist_module_imports_no_llm_or_research_or_network_client():
    source = open(ts.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported_modules = set()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
            imported_names.update(alias.name for alias in node.names)
    banned_modules = {"httpx", "requests", "socket", "llm_client", "researcher", "tts"}
    assert not (imported_modules & banned_modules), f"unexpected import(s): {imported_modules & banned_modules}"
    banned_names = {"llm_client", "researcher", "chat", "chat_json"}
    assert not (imported_names & banned_names), f"unexpected imported name(s): {imported_names & banned_names}"


# ---------------------------------------------------------------------------
# Word-boundary emotion matching (regression: "Alien" must never match "lie")
# ---------------------------------------------------------------------------

def test_emotion_keyword_matching_is_word_bounded_not_substring():
    """'Alien' contains the substring 'lie' - a naive `in` check would
    misfire on the hoax/fraud emotion category. Must not happen."""
    strategy = ts.build_thumbnail_strategy(_script("An Alien Signal From Deep Space"))
    assert strategy["emotion"] != "suspicion and skepticism"


def test_emotion_keyword_matching_still_finds_real_whole_word_matches():
    strategy = ts.build_thumbnail_strategy(_script("The Great Financial Fraud"))
    assert strategy["emotion"] == "suspicion and skepticism"


# ---------------------------------------------------------------------------
# Determinism / faithfulness (no invented words)
# ---------------------------------------------------------------------------

def test_build_thumbnail_strategy_is_deterministic():
    script = _script("The Wow! Signal: An Alien Message?")
    research = {"topic": "The Wow! Signal", "summary": "A 1977 unexplained signal."}
    a = ts.build_thumbnail_strategy(script, research=research)
    b = ts.build_thumbnail_strategy(script, research=research)
    assert a == b


def test_headline_words_are_all_present_in_the_source_title():
    title = "The Wow! Signal: An Alien Message?"
    strategy = ts.build_thumbnail_strategy(_script(title))
    title_words = {w.strip(".,!?–—:").upper() for w in title.split()}
    for word in strategy["headline"].split():
        assert word.strip(".,!?–—:") in title_words, f"headline word {word!r} not present in the title"


def test_uses_research_topic_over_script_title_for_subject_when_available():
    script = _script("A Very Different Working Title")
    research = {"topic": "The Real Subject", "summary": ""}
    strategy = ts.build_thumbnail_strategy(script, research=research)
    assert strategy["subject"] == "The Real Subject"


def test_falls_back_to_script_title_for_subject_without_research():
    strategy = ts.build_thumbnail_strategy(_script("Standalone Title"))
    assert strategy["subject"] == "Standalone Title"
