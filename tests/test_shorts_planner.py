"""
youtube/shorts_planner.py tests - angle selection + script condensation.

Covers spec scenarios:
    1  Long -> 3 Shorts planning
    3  exactly 3 Shorts
    4  Short numbering 1/2/3
    5  Short script generation
    6  source facts remain within Long research/script (no second research
       call, no invented facts - grounding is enforced by construction:
       generate_short_script() is only ever given ONE section's own text)
"""
from __future__ import annotations

import pytest

from youtube import llm_client, shorts_planner


def _script():
    return {
        "title": "The Signal Nobody Can Explain",
        "sections": [
            {"type": "hook", "title": "Hook", "narration": "One night in 1977 a telescope in Ohio heard something impossible."},
            {"type": "introduction", "title": "Intro", "narration": "The observatory had been listening for years."},
            {"type": "story_development", "title": "Story", "narration": "Researchers scrambled to understand the reading."},
            {"type": "evidence", "title": "Evidence", "narration": "The printout itself became a famous artifact."},
            {"type": "mystery", "title": "Mystery", "narration": "Nobody has ever detected the same signal again."},
            {"type": "reveal", "title": "Reveal", "narration": "The signal matched no known satellite or interference source."},
            {"type": "conclusion", "title": "Conclusion", "narration": "The case remains open to this day."},
            {"type": "cta", "title": "CTA", "narration": "Subscribe for more."},
        ],
    }


# ---------------------------------------------------------------------------
# 1/3/4 - plan_short_angles() is pure (no LLM call) and always produces
# exactly SHORTS_PER_LONG entries numbered 1..N
# ---------------------------------------------------------------------------

def test_plan_short_angles_produces_three_numbered_entries_no_llm_call(monkeypatch):
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the LLM")))
    plans = shorts_planner.plan_short_angles(_script())
    assert [p["short_number"] for p in plans] == [1, 2, 3]


def test_plan_short_angles_maps_to_hook_reveal_mystery():
    plans = shorts_planner.plan_short_angles(_script())
    by_number = {p["short_number"]: p for p in plans}
    assert by_number[1]["section_type"] == "hook"
    assert by_number[2]["section_type"] == "reveal"
    assert by_number[3]["section_type"] == "mystery"


def test_plan_short_angles_respects_shorts_per_long_override():
    plans = shorts_planner.plan_short_angles(_script(), shorts_per_long=1)
    assert len(plans) == 1
    assert plans[0]["short_number"] == 1


def test_plan_short_angles_raises_on_missing_section():
    script = _script()
    script["sections"] = [s for s in script["sections"] if s["type"] != "reveal"]
    with pytest.raises(ValueError):
        shorts_planner.plan_short_angles(script)


def test_plan_short_angles_raises_when_shorts_per_long_exceeds_defined_angles():
    with pytest.raises(ValueError):
        shorts_planner.plan_short_angles(_script(), shorts_per_long=5)


# ---------------------------------------------------------------------------
# 5 - Short script generation (ONE LLM call per Short, mocked)
# ---------------------------------------------------------------------------

def test_generate_short_script_calls_llm_once_and_returns_narration_and_title(monkeypatch):
    calls = []

    def fake_chat_json(system, user, max_tokens=2000, model=None, use_web_search=False):
        calls.append((system, user))
        return {"title": "A Punchy Short Title", "narration": "This is the condensed short narration."}, []

    monkeypatch.setattr(shorts_planner.llm_client, "chat_json", fake_chat_json)
    plan = shorts_planner.plan_short_angles(_script())[0]
    result = shorts_planner.generate_short_script("The Signal Nobody Can Explain", plan)
    assert len(calls) == 1
    assert result["title"] == "A Punchy Short Title"
    assert result["narration"] == "This is the condensed short narration."
    assert result["word_count"] == len(result["narration"].split())


def test_generate_short_script_falls_back_to_section_title_when_llm_omits_title(monkeypatch):
    monkeypatch.setattr(shorts_planner.llm_client, "chat_json",
                         lambda *a, **k: ({"narration": "Some narration text."}, []))
    plan = shorts_planner.plan_short_angles(_script())[0]
    result = shorts_planner.generate_short_script("Title", plan)
    assert result["title"] == plan["section_title"]


def test_generate_short_script_raises_on_empty_narration(monkeypatch):
    monkeypatch.setattr(shorts_planner.llm_client, "chat_json", lambda *a, **k: ({"narration": ""}, []))
    plan = shorts_planner.plan_short_angles(_script())[0]
    with pytest.raises(RuntimeError):
        shorts_planner.generate_short_script("Title", plan)


# ---------------------------------------------------------------------------
# 6 - source facts remain within Long research/script: each Short's LLM
# call is only ever given ITS OWN section's text, never the full research
# or another section's text - no second research call is possible because
# researcher.py/llm_client's web-search path is never invoked here.
# ---------------------------------------------------------------------------

def test_generate_short_script_prompt_only_contains_its_own_section_text(monkeypatch):
    captured = {}

    def fake_chat_json(system, user, max_tokens=2000, model=None, use_web_search=False):
        captured["user"] = user
        captured["use_web_search"] = use_web_search
        return {"title": "t", "narration": "n"}, []

    monkeypatch.setattr(shorts_planner.llm_client, "chat_json", fake_chat_json)
    script = _script()
    plans = shorts_planner.plan_short_angles(script)
    reveal_plan = next(p for p in plans if p["section_type"] == "reveal")
    shorts_planner.generate_short_script(script["title"], reveal_plan)

    # The reveal section's own text is present...
    assert "no known satellite or interference source" in captured["user"]
    # ...but text unique to OTHER sections is not - the LLM call cannot
    # draw on material outside this one section.
    assert "printout itself became a famous artifact" not in captured["user"]
    assert captured["use_web_search"] is False  # chat_json's default - no research call


def test_plan_and_generate_shorts_end_to_end(monkeypatch):
    monkeypatch.setattr(shorts_planner.llm_client, "chat_json",
                         lambda system, user, **k: ({"title": "T", "narration": "Condensed narration text here."}, []))
    results = shorts_planner.plan_and_generate_shorts(_script())
    assert [r["short_number"] for r in results] == [1, 2, 3]
    assert all(r["narration"] == "Condensed narration text here." for r in results)


# ---------------------------------------------------------------------------
# DURATION HARD-GATE PATCH - planner word-budget alignment (settings.
# SHORTS_WORDS_PER_SECOND) and long-script visibility logging. The planner
# never mechanically truncates, pads, or otherwise post-processes the
# narration - see module LENGTH guidance in SYSTEM_PROMPT and the
# assertions below that the returned narration is byte-for-byte what the
# LLM returned.
# ---------------------------------------------------------------------------

def test_word_budget_constants_derive_from_shorts_words_per_second():
    from core.config import settings
    assert shorts_planner.TARGET_WORDS_MIN == int(settings.SHORTS_TARGET_SECONDS_MIN * settings.SHORTS_WORDS_PER_SECOND)
    assert shorts_planner.TARGET_WORDS_MAX == int(settings.SHORTS_TARGET_SECONDS_MAX * settings.SHORTS_WORDS_PER_SECOND)
    assert shorts_planner.PREFERRED_WORDS_MIN == int(settings.SHORTS_PREFERRED_SECONDS_MIN * settings.SHORTS_WORDS_PER_SECOND)
    assert shorts_planner.PREFERRED_WORDS_MAX == int(settings.SHORTS_PREFERRED_SECONDS_MAX * settings.SHORTS_WORDS_PER_SECOND)


def test_system_prompt_states_hard_and_preferred_bands():
    from core.config import settings
    assert str(settings.SHORTS_TARGET_SECONDS_MIN) in shorts_planner.SYSTEM_PROMPT
    assert str(settings.SHORTS_TARGET_SECONDS_MAX) in shorts_planner.SYSTEM_PROMPT
    assert str(settings.SHORTS_PREFERRED_SECONDS_MIN) in shorts_planner.SYSTEM_PROMPT
    assert str(settings.SHORTS_PREFERRED_SECONDS_MAX) in shorts_planner.SYSTEM_PROMPT


def test_generate_short_script_reproduces_the_real_122_word_overshoot_and_does_not_alter_it(monkeypatch, caplog):
    # Reproduces the real production case (Short 2: 122 words -> 55.2s of
    # real audio, outside the default 30-50s band) at the planner level: a
    # too-long LLM response is logged loudly but returned completely
    # unmodified - no mechanical truncation, no mid-sentence cut, no
    # artificial change of any kind. The actual hard rejection happens
    # downstream in youtube/shorts_qc.py against the REAL rendered audio,
    # never here against a word-count guess.
    long_narration = " ".join(["word"] * 122) + "."
    monkeypatch.setattr(shorts_planner.llm_client, "chat_json",
                         lambda system, user, **k: ({"title": "T", "narration": long_narration}, []))
    plan = shorts_planner.plan_short_angles(_script())[1]  # short_number=2, mirrors the real incident

    with caplog.at_level("WARNING", logger="shorts_planner"):
        result = shorts_planner.generate_short_script("The Signal Nobody Can Explain", plan)

    assert result["narration"] == long_narration  # untouched - no truncation
    assert result["word_count"] == 122
    assert any("outside hard target word band" in r.message for r in caplog.records)


def test_generate_short_script_does_not_warn_for_a_word_count_inside_the_hard_band(monkeypatch, caplog):
    # 80 words sits comfortably inside the default hard band (66-110 words
    # at settings.SHORTS_WORDS_PER_SECOND=2.2 for a 30-50s target) even
    # though it is short of the preferred sweet spot - only the hard band
    # is a WARNING; the preferred band is a quieter INFO note.
    in_band_narration = " ".join(["word"] * 80) + "."
    monkeypatch.setattr(shorts_planner.llm_client, "chat_json",
                         lambda system, user, **k: ({"title": "T", "narration": in_band_narration}, []))
    plan = shorts_planner.plan_short_angles(_script())[0]

    with caplog.at_level("WARNING", logger="shorts_planner"):
        shorts_planner.generate_short_script("Title", plan)

    assert not any("outside hard target word band" in r.message for r in caplog.records)
