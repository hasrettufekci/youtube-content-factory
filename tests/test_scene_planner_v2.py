"""
Scene Planner V2 - Phase 1 ONLY (V2 scene classification + deterministic
strategy resolution + validation). See scene_planner.py's V2 section
docstring for the full design.

Covers exactly the 10 categories requested for Phase 1:
    - V2 schema generation
    - scene_type validation
    - evidence_level validation
    - data_type validation
    - hypothesis validation
    - exact_text derivation
    - visual relevance validation
    - deterministic strategy resolution
    - backward compatibility
    - invalid scene rejection

Deliberately does NOT test anything past Phase 1 scope: no visual_prompt,
no FAL/image_gen calls, no overlay/scientific-graphics, no pipeline.py
wiring - none of that exists yet.
"""
import inspect
import json
import re
from pathlib import Path

import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import scene_planner as sp


def _fake_script_en(word_count: int = 1400) -> dict:
    sentences = []
    words_left = word_count
    while words_left > 0:
        n = min(12, words_left)
        sentences.append(" ".join(["word"] * n) + ".")
        words_left -= n
    return {"full_narration": " ".join(sentences)}


def _valid_data_point(**overrides) -> dict:
    dp = {"data_type": "scalar_fact", "value": "42", "exact_render": False}
    dp.update(overrides)
    return dp


def _valid_v2_scene(**overrides) -> dict:
    """A minimal, internally-consistent V2 scene - every field a hand-built
    test scene needs, with a sane default for anything not overridden."""
    scene_type = overrides.pop("scene_type", "real_location")
    evidence_level = overrides.pop("evidence_level", "documented_fact")
    data_points = overrides.pop("data_points", [])
    resolved_scene_type, visual_strategy = sp.resolve_visual_strategy(scene_type, evidence_level, data_points)
    scene = {
        "scene_id": 1,
        "schema_version": 2,
        "claim": "A specific factual claim.",
        "scene_type": resolved_scene_type,
        "evidence_level": evidence_level,
        "data_points": data_points,
        "visual_strategy": visual_strategy,
        "exact_text": sp._derive_exact_text(data_points),
        "visual_relevance_reason": "Depicts the exact radio telescope dish described in this claim.",
    }
    scene.update(overrides)
    return scene


# ---------------------------------------------------------------------------
# Deterministic strategy resolution
# ---------------------------------------------------------------------------

def test_resolve_visual_strategy_defaults_to_ai_generated():
    scene_type, strategy = sp.resolve_visual_strategy("real_location", "documented_fact", [])
    assert scene_type == "real_location"
    assert strategy == "ai_generated"


def test_resolve_visual_strategy_hypothesis_forces_diagram_and_strategy():
    scene_type, strategy = sp.resolve_visual_strategy("real_location", "hypothesis", [])
    assert scene_type == "hypothesis_diagram"
    assert strategy == "hypothesis_visualization"


def test_resolve_visual_strategy_hypothesis_keeps_already_allowed_scene_type():
    scene_type, strategy = sp.resolve_visual_strategy("scientific_visualization", "hypothesis", [])
    assert scene_type == "scientific_visualization"
    assert strategy == "hypothesis_visualization"


def test_resolve_visual_strategy_exact_render_forces_overlay():
    data_points = [_valid_data_point(data_type="exact_text", value="1420.4556 MHz", exact_render=True)]
    scene_type, strategy = sp.resolve_visual_strategy("scientific_graphic", "documented_fact", data_points)
    assert strategy == "ai_generated_plus_overlay"
    assert scene_type == "scientific_graphic"


def test_resolve_visual_strategy_programmatic_data_type_forces_graphic():
    for data_type in sp._PROGRAMMATIC_GRAPHIC_DATA_TYPES:
        data_points = [_valid_data_point(data_type=data_type, exact_render=False)]
        _, strategy = sp.resolve_visual_strategy("scientific_graphic", "documented_fact", data_points)
        assert strategy == "programmatic_graphic", data_type


def test_resolve_visual_strategy_archival_source_always_normalizes_to_document_recreation():
    # V2.0 has no real source_asset_ref pipeline - archival_source must
    # never survive resolution down ANY path, not just the default one.
    scene_type, strategy = sp.resolve_visual_strategy("archival_source", "documented_fact", [])
    assert scene_type == "document_recreation"
    assert strategy == "ai_generated"

    # Even combined with an exact-render data point.
    data_points = [_valid_data_point(data_type="exact_text", value="X", exact_render=True)]
    scene_type, strategy = sp.resolve_visual_strategy("archival_source", "documented_fact", data_points)
    assert scene_type == "document_recreation"
    assert strategy == "ai_generated_plus_overlay"


def test_resolve_visual_strategy_exact_render_takes_priority_over_programmatic():
    # Rule ordering: exact_render (rule 2) beats programmatic data types
    # (rule 3) when a scene somehow has both.
    data_points = [
        _valid_data_point(data_type="time_series", exact_render=False),
        _valid_data_point(data_type="exact_text", value="X", exact_render=True),
    ]
    _, strategy = sp.resolve_visual_strategy("scientific_graphic", "documented_fact", data_points)
    assert strategy == "ai_generated_plus_overlay"


def test_resolve_visual_strategy_is_idempotent():
    # This property is what lets validate_scenes_v2() recompute from a
    # scene's OWN stored fields as a single self-consistency check.
    for scene_type in sp.SCENE_TYPES_V2:
        for evidence_level in sp.EVIDENCE_LEVELS_V2:
            data_points = [_valid_data_point(data_type="time_series")]
            once = sp.resolve_visual_strategy(scene_type, evidence_level, data_points)
            twice = sp.resolve_visual_strategy(once[0], evidence_level, data_points)
            assert once == twice, (scene_type, evidence_level)


# ---------------------------------------------------------------------------
# exact_text derivation
# ---------------------------------------------------------------------------

def test_derive_exact_text_collects_exact_text_and_exact_render_values():
    data_points = [
        _valid_data_point(data_type="exact_text", value="6EQUJ5"),
        _valid_data_point(data_type="measurement", value="1420.4556 MHz", exact_render=True),
        _valid_data_point(data_type="scalar_fact", value="ignored", exact_render=False),
    ]
    assert sp._derive_exact_text(data_points) == ["6EQUJ5", "1420.4556 MHz"]


def test_derive_exact_text_deduplicates_preserving_order():
    data_points = [
        _valid_data_point(data_type="exact_text", value="dup"),
        _valid_data_point(data_type="exact_text", value="unique"),
        _valid_data_point(data_type="exact_text", value="dup"),
    ]
    assert sp._derive_exact_text(data_points) == ["dup", "unique"]


def test_derive_exact_text_empty_when_no_data_points():
    assert sp._derive_exact_text([]) == []
    assert sp._derive_exact_text(None) == []


def test_derive_text_render_sets_hypothesis_badge_only_for_hypothesis():
    assert sp._derive_text_render(["X"], "hypothesis")["hypothesis_badge"] is True
    assert sp._derive_text_render(["X"], "documented_fact")["hypothesis_badge"] is False
    assert sp._derive_text_render(["X"], "hypothesis")["items"] == ["X"]


# ---------------------------------------------------------------------------
# V2 schema generation (plan_scenes_v2)
# ---------------------------------------------------------------------------

def _fake_chat_json_v2(scene_type="real_location", evidence_level="documented_fact", data_points=None):
    data_points = data_points if data_points is not None else []

    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        # input_chunk_index is parsed from each line's own real number
        # prefix (not a positional 0..k-1 counter) - correct for both the
        # normal full-batch call (numbered 1..N) and a targeted-repair
        # sub-call (numbered with the real, non-contiguous chunk indices
        # being repaired) - see _classify_chunks_v2 in scene_planner.py.
        indices = [int(re.match(r"^(\d+)\.", l).group(1)) for l in chunk_lines]
        return [
            {
                "input_chunk_index": idx,
                "claim": f"Claim for chunk {i}",
                "scene_type": scene_type,
                "evidence_level": evidence_level,
                "data_points": data_points,
                "visual_relevance_reason": f"Depicts the specific subject named in chunk {i}'s narration.",
                "prompt_segments": {
                    "subject": "s", "setting": "s", "lighting": "l",
                    "mood": "m", "composition": "c", "style": "s",
                },
            }
            for i, idx in enumerate(indices)
        ], []
    return fake


def test_plan_scenes_v2_produces_schema_version_2_scenes_with_no_visual_prompt(monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    script_en = _fake_script_en()

    scenes = sp.plan_scenes_v2(script_en)

    assert len(scenes) >= sp.MIN_SCENES
    for scene in scenes:
        assert scene["schema_version"] == 2
        assert "visual_prompt" not in scene  # Phase 2 scope, not Phase 1
        assert scene["visual_strategy"] == "ai_generated"
        assert scene["scene_type"] == "real_location"
        assert scene["claim"]
        assert scene["visual_relevance_reason"]
        assert scene["visual_type"] in sp.VISUAL_TYPES  # backward-compat field still populated


def test_plan_scenes_v2_derives_exact_text_end_to_end(monkeypatch):
    data_points = [_valid_data_point(data_type="exact_text", value="6EQUJ5", exact_render=True)]
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2(data_points=data_points))
    script_en = _fake_script_en()

    scenes = sp.plan_scenes_v2(script_en)

    for scene in scenes:
        assert scene["exact_text"] == ["6EQUJ5"]
        assert scene["visual_strategy"] == "ai_generated_plus_overlay"
        assert scene["text_render"]["items"] == ["6EQUJ5"]


def test_plan_scenes_v2_attaches_localized_narrations_like_v1(monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    script_en = _fake_script_en(1200)
    script_tr = {"full_narration": _fake_script_en(1200)["full_narration"]}

    scenes = sp.plan_scenes_v2(script_en, localized_scripts={"tr": script_tr})

    assert any("narration_tr" in scene for scene in scenes)


def test_plan_scenes_v2_raises_on_wrong_number_of_classifications(monkeypatch, tmp_path):
    # Redirect the (TEMPORARY) cardinality diagnostic writer away from the
    # real project tree for every test that triggers a mismatch - otherwise
    # running this test suite would litter debug_audit/ with real files.
    monkeypatch.setattr(sp, "_DEBUG_AUDIT_DIR", tmp_path)

    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        return [{"claim": "only one"}], []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError, match="expected"):
        sp.plan_scenes_v2(_fake_script_en())


def test_plan_scenes_v2_raises_when_llm_output_is_not_a_list(monkeypatch):
    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"not": "a list"}, []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError, match="JSON array"):
        sp.plan_scenes_v2(_fake_script_en())


def test_system_prompt_v2_states_absolute_cardinality_rule():
    # Regression guard for the explicit input/output cardinality instruction
    # in SYSTEM_PROMPT_V2_CLASSIFICATION: N input chunks must always produce
    # exactly N output objects, positionally aligned. plan_scenes_v2()
    # already hard-enforces this at runtime (see
    # test_plan_scenes_v2_raises_on_wrong_number_of_classifications above) -
    # this only guards that the LLM is actually TOLD the rule, so a future
    # edit to the prompt can't silently drop it and rely on the runtime
    # check alone to catch every mismatch after the fact.
    # Normalize whitespace first: the prompt is hand-wrapped across multiple
    # source lines, so a phrase spanning a wrap point contains a literal
    # newline rather than a space in the raw string.
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    assert "ABSOLUTE RULE" in prompt
    assert "EXACTLY N classification objects" in prompt
    assert "output[0] -> chunk 1" in prompt
    assert "output[N-1] -> chunk N" in prompt
    assert "internally count the input chunks" in prompt


# ---------------------------------------------------------------------------
# Scene Planner V2 stabilization: general (non-scene-ID-specific) scene_type/
# evidence_level classification guidance added to
# SYSTEM_PROMPT_V2_CLASSIFICATION. These are prompt-content regression
# guards (the LLM's actual scene_type choice can't be unit-tested without a
# real API call - see youtube/scene_audit.py for the audit-side matcher that
# catches classification quality issues after a real run), same pattern as
# test_system_prompt_v2_states_absolute_cardinality_rule above.
# ---------------------------------------------------------------------------

def test_system_prompt_v2_gives_research_evidence_trigger_phrases():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    for phrase in (
        "researchers reanalyzed", "a new analysis showed", "published in",
        "archival data revealed", "a preprint suggested", "observational data indicated",
    ):
        assert phrase in prompt, f"missing research_evidence trigger phrase: {phrase!r}"


def test_system_prompt_v2_restricts_hypothesis_diagram_to_genuinely_speculative_language():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    assert "RESTRICTED" in prompt
    for phrase in ("could be", "one possibility", "another explanation", "competing hypothesis"):
        assert phrase in prompt, f"missing hypothesis_diagram trigger phrase: {phrase!r}"
    # The explicit guard against the exact failure mode described in the
    # stabilization task: a research finding is not automatically a hypothesis.
    assert "NOT automatically a hypothesis just because it's scientific" in prompt


def test_system_prompt_v2_restricts_document_recreation_to_actual_documents():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    assert "Never use document_recreation just because the narration states a historical fact" in prompt


def test_system_prompt_v2_prefers_scientific_graphic_for_numeric_claims():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    for phrase in ("signal-to-noise ratio", "frequency", "spectrum"):
        assert phrase in prompt


def test_system_prompt_v2_describes_real_location_as_an_actual_physical_place():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    assert "a specific facility, building, site, or region" in prompt


def test_system_prompt_v2_gives_evidence_level_scene_type_consistency_guidance():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    assert "primary_source claims usually pair" in prompt
    assert "pairs with hypothesis_diagram (enforced)" in prompt
    assert "atmospheric always" in prompt


def test_system_prompt_v2_tells_llm_to_put_numeric_data_in_data_points_not_prose():
    prompt = " ".join(sp.SYSTEM_PROMPT_V2_CLASSIFICATION.split())
    assert "never only as prose in prompt_segments" in prompt


def test_system_prompt_v2_new_guidance_has_no_scene_id_or_topic_hardcoding():
    # The whole point of the stabilization: these are GENERAL classification
    # principles, never special-cased to a specific scene_id or to the Wow!
    # Signal script's own subject matter.
    prompt = sp.SYSTEM_PROMPT_V2_CLASSIFICATION
    assert "scene_id ==" not in prompt
    assert "Wow!" not in prompt
    assert "Sagittarius" not in prompt
    assert "6EQUJ5" not in prompt


def test_plan_scenes_v2_passes_the_real_system_prompt_to_chat_json(monkeypatch):
    # Confirms the cardinality-rule-bearing SYSTEM_PROMPT_V2_CLASSIFICATION
    # constant is what actually reaches the LLM call, not some other string.
    captured = {}

    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        captured["system"] = system
        return _fake_chat_json_v2()(system, user, max_tokens=max_tokens, model=model, use_web_search=use_web_search)

    monkeypatch.setattr(sp.llm_client, "chat_json", fake)
    sp.plan_scenes_v2(_fake_script_en())

    assert captured["system"] == sp.SYSTEM_PROMPT_V2_CLASSIFICATION


# ---------------------------------------------------------------------------
# cardinality-mismatch diagnostic (TEMPORARY - see _write_cardinality_
# diagnostic() in youtube/scene_planner.py; covers a real-world failure
# where a real Claude call returned more classification objects than input
# chunks despite the ABSOLUTE RULE above)
# ---------------------------------------------------------------------------

def test_write_cardinality_diagnostic_writes_expected_json_structure(tmp_path):
    chunks = ["First chunk narration.", "Second chunk narration."]
    classifications = [
        {
            "claim": "Claim A",
            "scene_type": "real_location",
            "evidence_level": "documented_fact",
            "data_points": [],
            "visual_relevance_reason": "Depicts the specific place named.",
            "prompt_segments": {"subject": "s"},
        },
        {
            "claim": "Claim B - split from chunk 2",
            "scene_type": "real_location",
            "evidence_level": "documented_fact",
            "data_points": [],
            "visual_relevance_reason": "Depicts a second aspect of the same place.",
            "prompt_segments": {"subject": "s2"},
        },
        {
            "claim": "Claim C - split from chunk 2",
            "scene_type": "evidence_closeup",
            "evidence_level": "documented_fact",
            "data_points": [],
            "visual_relevance_reason": "Closeup on a detail of the same place.",
            "prompt_segments": {"subject": "s3"},
        },
    ]

    path = sp._write_cardinality_diagnostic(chunks, classifications, output_dir=tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["diagnostic_type"] == "scene_planner_v2_cardinality_mismatch"
    assert data["expected_object_count"] == 2
    assert data["returned_object_count"] == 3
    assert [c["text"] for c in data["input_chunks"]] == chunks
    assert data["input_chunks"][0]["chunk_index"] == 1
    assert data["input_chunks"][0]["word_count"] == 3

    returned = data["returned_classifications"]
    assert len(returned) == 3
    assert [r["returned_index"] for r in returned] == [0, 1, 2]
    assert [r["claim"] for r in returned] == ["Claim A", "Claim B - split from chunk 2", "Claim C - split from chunk 2"]
    assert [r["scene_type"] for r in returned] == ["real_location", "real_location", "evidence_closeup"]
    assert [r["evidence_level"] for r in returned] == ["documented_fact"] * 3
    # prompt_segments/data_points/visual_relevance_reason are carried through
    # in full - these are exactly what would let a human spot that objects 1
    # and 2 both plausibly came from the same single input chunk.
    assert returned[1]["prompt_segments"] == {"subject": "s2"}
    assert returned[2]["visual_relevance_reason"] == "Closeup on a detail of the same place."


def test_write_cardinality_diagnostic_never_contains_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant-super-secret-do-not-leak")
    chunks = ["A narration chunk."]
    classifications = [{"claim": "x", "scene_type": "real_location", "evidence_level": "documented_fact"}]

    path = sp._write_cardinality_diagnostic(chunks, classifications, output_dir=tmp_path)

    raw = path.read_text(encoding="utf-8")
    assert "sk-ant-super-secret-do-not-leak" not in raw
    assert "ANTHROPIC_API_KEY" not in raw


def test_write_cardinality_diagnostic_handles_non_dict_classification_entries(tmp_path):
    # Defensive: a malformed real response could contain a null or a string
    # instead of an object at some index - must not crash the diagnostic.
    chunks = ["Chunk one.", "Chunk two."]
    classifications = [None, "not a dict"]

    path = sp._write_cardinality_diagnostic(chunks, classifications, output_dir=tmp_path)

    data = json.loads(path.read_text(encoding="utf-8"))
    returned = data["returned_classifications"]
    assert len(returned) == 2
    assert returned[0]["claim"] is None
    assert returned[0]["raw_keys"] is None
    assert returned[1]["claim"] is None


def test_write_cardinality_diagnostic_swallows_write_failures(tmp_path, monkeypatch, caplog):
    # "Safely captures": a failure to write the diagnostic file itself must
    # never raise out of this function - the caller's real RuntimeError
    # about the mismatch is what has to surface, not an IOError from here.
    def broken_mkdir(self, *args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(Path, "mkdir", broken_mkdir)

    path = sp._write_cardinality_diagnostic(["a"], [{"claim": "x"}, {"claim": "y"}], output_dir=tmp_path)

    assert not path.exists()  # write never succeeded, but no exception escaped


def test_plan_scenes_v2_writes_diagnostic_and_still_hard_fails_on_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "_DEBUG_AUDIT_DIR", tmp_path)

    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        # 3 chunks in, 4 classifications out - the real-world "35 -> 36" shape.
        return [
            {"claim": f"Claim {i}", "scene_type": "real_location", "evidence_level": "documented_fact",
             "data_points": [], "visual_relevance_reason": f"Reason {i} about the specific place shown.",
             "prompt_segments": {"subject": "s", "setting": "s", "lighting": "l", "mood": "m",
                                  "composition": "c", "style": "s"}}
            for i in range(4)
        ], []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError, match=r"expected \d+ classifications, got 4"):
        sp.plan_scenes_v2(_fake_script_en())

    written = list(tmp_path.glob("cardinality_mismatch_*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text(encoding="utf-8"))
    assert data["returned_object_count"] == 4
    assert len(data["returned_classifications"]) == 4


def test_plan_scenes_v2_does_not_write_diagnostic_when_counts_match(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "_DEBUG_AUDIT_DIR", tmp_path)
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())

    sp.plan_scenes_v2(_fake_script_en())

    assert list(tmp_path.glob("cardinality_mismatch_*.json")) == []


# ---------------------------------------------------------------------------
# CTA/end-screen separation (root-cause fix for the real "expected 35
# classifications, got 36" mismatch the diagnostic above captured - see
# _split_documentary_and_cta()/_build_cta_scene() in scene_planner.py)
# ---------------------------------------------------------------------------

_REAL_CTA_TEXT = (
    "If you found this investigation into the Wow! Signal as fascinating as we did, "
    "consider subscribing and let us know in the comments. Thanks for watching."
)


def _fake_script_en_with_sections(
    documentary_word_count: int = 1400,
    cta_narration: str = _REAL_CTA_TEXT,
) -> dict:
    """Mirrors script_generator.py's REAL output shape ("sections" tagged
    by "type", "full_narration" = every section's narration joined) -
    unlike _fake_script_en() above (bare {"full_narration": ...}), which
    exercises the backward-compat fallback instead. The CTA-separation fix
    only activates when "sections" is present, so this fixture is what
    every test in this section needs."""
    sentences = []
    words_left = documentary_word_count
    while words_left > 0:
        n = min(12, words_left)
        sentences.append(" ".join(["word"] * n) + ".")
        words_left -= n
    documentary_narration = " ".join(sentences)
    sections = [
        {"type": "hook", "title": "Hook", "narration": documentary_narration},
        {"type": "cta", "title": "CTA", "narration": cta_narration},
    ]
    full_narration = "\n\n".join(s["narration"] for s in sections)
    return {
        "title": "Test Documentary",
        "sections": sections,
        "full_narration": full_narration,
        "word_count": len(full_narration.split()),
    }


def test_split_documentary_and_cta_separates_by_section_type():
    script = {
        "sections": [
            {"type": "hook", "narration": "Hook narration."},
            {"type": "mystery", "narration": "Mystery narration."},
            {"type": "cta", "narration": "Subscribe and comment below."},
        ],
    }
    documentary, cta = sp._split_documentary_and_cta(script)
    assert documentary == "Hook narration.\n\nMystery narration."
    assert cta == "Subscribe and comment below."
    assert "Subscribe" not in documentary
    assert "Hook" not in cta and "Mystery" not in cta


def test_split_documentary_and_cta_falls_back_when_no_sections():
    # Backward compatibility: a script shaped like every pre-existing test
    # fixture (_fake_script_en()) has no "sections" at all - the entire
    # full_narration must still be treated as documentary text, with no CTA.
    script = {"full_narration": "word word word."}
    documentary, cta = sp._split_documentary_and_cta(script)
    assert documentary == "word word word."
    assert cta == ""


def test_split_documentary_and_cta_falls_back_when_sections_is_empty_list():
    script = {"sections": [], "full_narration": "word word word."}
    documentary, cta = sp._split_documentary_and_cta(script)
    assert documentary == "word word word."
    assert cta == ""


def test_split_documentary_and_cta_joins_multiple_cta_sections():
    # Defensive edge case: script_generator.py's spec requires exactly one
    # cta section, but the split logic itself doesn't assume that.
    script = {
        "sections": [
            {"type": "hook", "narration": "Hook."},
            {"type": "cta", "narration": "First CTA."},
            {"type": "cta", "narration": "Second CTA."},
        ],
    }
    documentary, cta = sp._split_documentary_and_cta(script)
    assert documentary == "Hook."
    assert cta == "First CTA.\n\nSecond CTA."


def test_build_cta_scene_returns_none_for_empty_cta_text():
    assert sp._build_cta_scene("", scene_id=5, localized_cta={}) is None
    assert sp._build_cta_scene("   ", scene_id=5, localized_cta={}) is None


def test_build_cta_scene_produces_a_valid_v2_scene():
    scene = sp._build_cta_scene(_REAL_CTA_TEXT, scene_id=36, localized_cta={})

    assert scene["scene_id"] == 36
    assert scene["schema_version"] == 2
    assert scene["is_cta"] is True
    assert scene["narration_en"] == _REAL_CTA_TEXT
    assert scene["scene_type"] == "atmospheric_establishing"
    assert scene["evidence_level"] == "atmospheric"
    assert scene["visual_strategy"] == "ai_generated"
    assert scene["claim"]
    assert scene["visual_relevance_reason"]
    # Never raises - a real, self-consistent V2 scene, not a special case
    # validate_scenes_v2() has to know about.
    sp.validate_scenes_v2([scene])


def test_build_cta_scene_attaches_localized_narrations():
    scene = sp._build_cta_scene(_REAL_CTA_TEXT, scene_id=1, localized_cta={"tr": "Abone olun ve yorum yapın."})
    assert scene["narration_tr"] == "Abone olun ve yorum yapın."


def test_chunk_localized_narrations_v2_excludes_cta_from_documentary_chunks():
    loc_script = {
        "sections": [
            {"type": "hook", "narration": "Localized hook sentence one. Localized hook sentence two."},
            {"type": "cta", "narration": "Localized CTA sentence."},
        ],
    }
    localized_chunks, localized_cta = sp._chunk_localized_narrations_v2({"tr": loc_script}, target_count=2)

    assert localized_cta["tr"] == "Localized CTA sentence."
    for chunk in localized_chunks["tr"]:
        assert "Localized CTA sentence" not in chunk


def test_plan_scenes_v2_excludes_cta_narration_from_the_llm_call(monkeypatch):
    captured = {}

    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        captured["user"] = user
        return _fake_chat_json_v2()(system, user, max_tokens=max_tokens, model=model, use_web_search=use_web_search)

    monkeypatch.setattr(sp.llm_client, "chat_json", fake)
    sp.plan_scenes_v2(_fake_script_en_with_sections())

    # The whole point of the fix: the CTA text must never reach the LLM at
    # all, so it cannot cause (or be blamed for) a cardinality mismatch.
    assert "subscribing" not in captured["user"]
    assert "Thanks for watching" not in captured["user"]


def test_plan_scenes_v2_appends_exactly_one_deterministic_cta_scene(monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    script = _fake_script_en_with_sections(documentary_word_count=1400)

    scenes = sp.plan_scenes_v2(script)

    documentary_word_count = 1400
    expected_documentary_count = sp._target_scene_count(documentary_word_count)
    assert len(scenes) == expected_documentary_count + 1

    documentary_scenes, cta_scenes = scenes[:-1], scenes[-1:]
    assert all(s["is_cta"] is False for s in documentary_scenes)
    assert len(cta_scenes) == 1
    cta_scene = cta_scenes[0]
    assert cta_scene["is_cta"] is True
    assert cta_scene["narration_en"] == _REAL_CTA_TEXT
    assert cta_scene["scene_id"] == expected_documentary_count + 1
    # scene_id sequence has no gap/overlap between documentary and CTA scenes.
    assert [s["scene_id"] for s in scenes] == list(range(1, len(scenes) + 1))


def test_plan_scenes_v2_target_scene_count_uses_documentary_word_count_only(monkeypatch):
    # The scene-count TARGET itself must reflect documentary content only -
    # not be inflated by CTA words that no longer go through chunking.
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    script = _fake_script_en_with_sections(documentary_word_count=1400)

    scenes = sp.plan_scenes_v2(script)

    documentary_scene_count = len(scenes) - 1  # minus the one CTA scene
    assert documentary_scene_count == sp._target_scene_count(1400)


def test_plan_scenes_v2_no_cta_scene_for_script_without_sections(monkeypatch):
    # Locks in existing behavior (goal: "existing 35-scene target behavior
    # remains intentional") for any script shaped like the pre-fix input.
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())

    scenes = sp.plan_scenes_v2(_fake_script_en())

    assert all(s.get("is_cta") is False for s in scenes)
    assert len(scenes) == sp._target_scene_count(1400)


def test_plan_scenes_v2_cta_scene_carries_localized_cta_narration(monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    script_en = _fake_script_en_with_sections(documentary_word_count=1400)
    script_tr = _fake_script_en_with_sections(
        documentary_word_count=1400, cta_narration="Abone olun ve yorum yapın."
    )

    scenes = sp.plan_scenes_v2(script_en, localized_scripts={"tr": script_tr})

    cta_scene = scenes[-1]
    assert cta_scene["is_cta"] is True
    assert cta_scene["narration_tr"] == "Abone olun ve yorum yapın."
    # Documentary scenes still get their own aligned narration_tr, same as
    # before this fix - localized alignment for real content is preserved.
    documentary_scenes = scenes[:-1]
    assert all("narration_tr" in s for s in documentary_scenes)


# ---------------------------------------------------------------------------
# Dated-beat chunk rebalancing (real Wow! Signal audit: "expected 35
# classifications, got 36" - chunk 17 bundled two independently-dated
# documentary beats). See scene_planner.py's section banner above
# plan_scenes_v2() for the full design.
# ---------------------------------------------------------------------------

# Real sentences from the diagnosed chunk 17 mismatch (the "!" in "Wow!
# signal" is dropped here only because it would itself be a sentence-
# splitting point for _split_sentences - immaterial to the year-conflict
# logic under test).
_KRAUS_SENTENCE = "John Kraus published the first general description in Cosmic Search in 1979."
_OHIO_SENTENCE = (
    "The Ohio State SETI program operated from 1973 to 1997 and detected "
    "no comparable Wow signal during nearly 24 years."
)
_FILLER_SENTENCE = "This finding sparked new interest across the wider search community once again."


def test_has_conflicting_dated_beats_true_for_units_with_different_years():
    assert sp._has_conflicting_dated_beats([_KRAUS_SENTENCE, _OHIO_SENTENCE]) is True


def test_has_conflicting_dated_beats_false_when_years_overlap():
    same_year_followup = "That same year, 1979, the paper was widely cited."
    assert sp._has_conflicting_dated_beats([_KRAUS_SENTENCE, same_year_followup]) is False


def test_has_conflicting_dated_beats_false_for_a_single_dated_unit():
    assert sp._has_conflicting_dated_beats([_KRAUS_SENTENCE, _FILLER_SENTENCE]) is False


def test_has_conflicting_dated_beats_false_when_no_unit_is_dated():
    assert sp._has_conflicting_dated_beats([_FILLER_SENTENCE, "Another undated sentence here."]) is False


def test_split_chunk_at_dated_beats_splits_the_real_chunk_17_case():
    chunk = f"{_KRAUS_SENTENCE} {_OHIO_SENTENCE}"
    result = sp._split_chunk_at_dated_beats(chunk)
    assert result == [_KRAUS_SENTENCE, _OHIO_SENTENCE]
    # Lossless: same guarantee _split_into_chunks/_split_clauses make.
    assert " ".join(result).split() == chunk.split()


def test_split_chunk_at_dated_beats_no_op_without_conflict():
    same_year_followup = "That same year, 1979, the paper was widely cited."
    chunk = f"{_KRAUS_SENTENCE} {same_year_followup}"
    assert sp._split_chunk_at_dated_beats(chunk) == [chunk]


def test_split_chunk_at_dated_beats_no_op_for_undated_filler():
    chunk = f"{_FILLER_SENTENCE} Another undated sentence here."
    assert sp._split_chunk_at_dated_beats(chunk) == [chunk]


def test_rebalance_dated_beat_chunks_splits_and_safely_remerges_to_preserve_target_count():
    chunk_a = "Word word word word."
    conflicting_chunk = f"{_KRAUS_SENTENCE} {_OHIO_SENTENCE}"
    chunk_c = "More filler words appear here now."
    chunks = [chunk_a, conflicting_chunk, chunk_c]

    result = sp._rebalance_dated_beat_chunks(chunks, target_count=3)

    assert len(result) == 3  # target preserved via a safe adjacent merge
    # Kraus (1979) and Ohio State (1973-1997) never end up back together.
    assert not any("1979" in c and "1973" in c for c in result)
    # Lossless and order-preserving.
    assert " ".join(result).split() == " ".join(chunks).split()


def test_rebalance_dated_beat_chunks_is_a_no_op_when_no_conflicts():
    chunks = ["Word word word.", "More plain filler text.", "Yet more filler text here."]
    assert sp._rebalance_dated_beat_chunks(chunks, target_count=3) == chunks


def test_rebalance_dated_beat_chunks_accepts_overage_when_no_safe_merge_exists():
    # Three consecutive, pairwise-conflicting dated beats: every possible
    # adjacent merge would recombine two independent beats, so none is
    # safe. The function must NOT force one - it should return all 3
    # pieces even though target_count asks for 1.
    a = "The first expedition departed the harbor in 1930."
    b = "A second, unrelated expedition set sail in 1955."
    c = "The final voyage of the series began in 1988."
    chunks = [f"{a} {b} {c}"]

    result = sp._rebalance_dated_beat_chunks(chunks, target_count=1)

    assert len(result) == 3
    assert result == [a, b, c]


def test_split_into_chunks_v2_reproduces_and_fixes_the_real_chunk_17_case():
    narration = f"{_KRAUS_SENTENCE} {_OHIO_SENTENCE} {_FILLER_SENTENCE}"
    target_count = 2

    # Confirm this reproduces the real bug shape at the baseline (v1,
    # unmodified) level first: the word-count-fit heuristic packs the two
    # independently-dated sentences into the SAME chunk, exactly like the
    # real chunk 17 - proving this test scenario is faithful to the audit.
    baseline_chunks = sp._split_into_chunks(narration, target_count)
    assert len(baseline_chunks) == target_count
    assert _KRAUS_SENTENCE in baseline_chunks[0] and _OHIO_SENTENCE in baseline_chunks[0]

    # The V2 chunker must not reproduce that: the two dated beats end up in
    # different chunks, the target chunk count is still hit (via the one
    # available safe merge, Ohio State + the filler sentence), and every
    # word survives in order.
    fixed_chunks = sp._split_into_chunks_v2(narration, target_count)
    assert len(fixed_chunks) == target_count
    assert not any(_KRAUS_SENTENCE in c and _OHIO_SENTENCE in c for c in fixed_chunks)
    assert " ".join(fixed_chunks).split() == narration.split()


def test_split_into_chunks_v2_matches_v1_when_no_dated_conflicts():
    # No year tokens anywhere -> the new rebalancing pass is a pure no-op,
    # byte-for-byte the same output _split_into_chunks already gave (every
    # pre-existing V2 test fixture is exactly this shape).
    narration = _fake_script_en(1400)["full_narration"]
    target = sp._target_scene_count(1400)
    assert sp._split_into_chunks_v2(narration, target) == sp._split_into_chunks(narration, target)


def test_split_into_chunks_v2_never_returns_a_chunk_with_conflicting_dated_beats():
    import random

    dated_sentences = [
        _KRAUS_SENTENCE,
        _OHIO_SENTENCE,
        "The Big Ear telescope was dismantled in 1998 to make way for a golf course.",
        "Jerry Ehman circled the printout on that August night in 1977, calling it Wow.",
    ]
    undated_sentences = [
        _FILLER_SENTENCE,
        "Astronomers around the world took notice of the unusual reading.",
        "The story would go on to capture public imagination for decades.",
    ]
    rng = random.Random(99)
    for target_count in (2, 3, 4, 5):
        sentences = dated_sentences + undated_sentences
        rng.shuffle(sentences)
        narration = " ".join(sentences)

        chunks = sp._split_into_chunks_v2(narration, target_count)

        for chunk in chunks:
            assert sp._has_conflicting_dated_beats(sp._split_sentences(chunk)) is False
        assert " ".join(chunks).split() == narration.split()


def _llm_that_splits_multi_beat_chunks(system, user, max_tokens=6000, model=None, use_web_search=False):
    """Simulates the REAL Anthropic model's actual (bug-triggering)
    behavior: given a chunk that bundles two independently-dated
    documentary beats, it - correctly, from a storytelling standpoint -
    returns TWO classification objects for that one chunk, exactly what
    caused the real "expected 35, got 36" mismatch. Every other chunk gets
    exactly one classification, as the cardinality rule requires. Proves
    the chunker - not another prompt instruction - is what prevents the
    mismatch: if _split_into_chunks_v2 ever again let a conflicting chunk
    through, this fake reproduces the crash and these tests fail."""
    chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
    out = []
    for line in chunk_lines:
        idx = int(re.match(r"^(\d+)\.", line).group(1))
        chunk_text = re.sub(r"^\d+\.\s*", "", line)
        beats = [ys for ys in (sp._dated_beat_years(s) for s in sp._split_sentences(chunk_text)) if ys]
        conflict = any(not (beats[a] & beats[b]) for a in range(len(beats)) for b in range(a + 1, len(beats)))
        for _ in range(2 if conflict else 1):
            out.append({
                "input_chunk_index": idx,
                "claim": f"Claim for chunk {idx}",
                "scene_type": "real_location",
                "evidence_level": "documented_fact",
                "data_points": [],
                "visual_relevance_reason": f"Depicts the specific subject named in chunk {idx}'s narration.",
                "prompt_segments": {
                    "subject": "s", "setting": "s", "lighting": "l",
                    "mood": "m", "composition": "c", "style": "s",
                },
            })
    return out, []


def test_plan_scenes_v2_end_to_end_survives_the_real_dated_beat_llm_split_behavior(monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _llm_that_splits_multi_beat_chunks)
    narration = (
        f"{_KRAUS_SENTENCE} {_OHIO_SENTENCE} {_FILLER_SENTENCE} "
        + _fake_script_en(400)["full_narration"]
    )
    script = {"full_narration": narration}

    # Must not raise: the chunker must have already kept every dated beat
    # in its own chunk before this fake ever saw the narration, so the fake
    # never actually returns an extra classification.
    scenes = sp.plan_scenes_v2(script)
    assert all(s["schema_version"] == 2 for s in scenes)


def test_plan_scenes_v2_dated_beat_rebalancing_works_alongside_cta_separation(monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _llm_that_splits_multi_beat_chunks)
    documentary = (
        f"{_KRAUS_SENTENCE} {_OHIO_SENTENCE} {_FILLER_SENTENCE} "
        + _fake_script_en(400)["full_narration"]
    )
    script = {
        "title": "Test Documentary",
        "sections": [
            {"type": "hook", "title": "Hook", "narration": documentary},
            {"type": "cta", "title": "CTA", "narration": _REAL_CTA_TEXT},
        ],
        "full_narration": documentary + "\n\n" + _REAL_CTA_TEXT,
        "word_count": 0,
    }

    scenes = sp.plan_scenes_v2(script)

    assert scenes[-1]["is_cta"] is True
    assert all(s["schema_version"] == 2 for s in scenes)


def test_v1_split_into_chunks_is_completely_unaffected_by_dated_beat_rebalancing():
    # Locks in the architecture goal: v1's plan_scenes()/_split_into_chunks
    # still call the ORIGINAL word-count-only packer directly, with no
    # awareness of _has_conflicting_dated_beats at all - it would still
    # reproduce the pre-fix bundling behavior on this exact input.
    narration = f"{_KRAUS_SENTENCE} {_OHIO_SENTENCE} {_FILLER_SENTENCE}"
    chunks = sp._split_into_chunks(narration, target_count=2)
    assert _KRAUS_SENTENCE in chunks[0] and _OHIO_SENTENCE in chunks[0]


# ---------------------------------------------------------------------------
# invalid scene rejection (plan_scenes_v2 runs validate_scenes_v2 itself)
# ---------------------------------------------------------------------------

def test_plan_scenes_v2_rejects_generic_visual_relevance_reason(monkeypatch):
    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        indices = [int(re.match(r"^(\d+)\.", l).group(1)) for l in chunk_lines]
        return [
            {
                "input_chunk_index": idx,
                "claim": "A claim", "scene_type": "real_location",
                "evidence_level": "documented_fact", "data_points": [],
                "visual_relevance_reason": "sets the mood",
                "prompt_segments": {},
            }
            for idx in indices
        ], []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError, match="generic/filler"):
        sp.plan_scenes_v2(_fake_script_en())


def test_plan_scenes_v2_rejects_missing_claim(monkeypatch):
    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        indices = [int(re.match(r"^(\d+)\.", l).group(1)) for l in chunk_lines]
        return [
            {
                "input_chunk_index": idx,
                "claim": "", "scene_type": "real_location",
                "evidence_level": "documented_fact", "data_points": [],
                "visual_relevance_reason": "Shows the specific dish named in the claim.",
                "prompt_segments": {},
            }
            for idx in indices
        ], []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError, match="missing/empty 'claim'"):
        sp.plan_scenes_v2(_fake_script_en())


# ---------------------------------------------------------------------------
# validate_scenes_v2 - scene_type validation
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_accepts_a_well_formed_scene():
    sp.validate_scenes_v2([_valid_v2_scene()])  # must not raise


def test_validate_scenes_v2_rejects_invalid_scene_type():
    scene = _valid_v2_scene()
    scene["scene_type"] = "not_a_real_type"
    with pytest.raises(RuntimeError, match="invalid scene_type"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_rejects_missing_scene_type():
    scene = _valid_v2_scene()
    del scene["scene_type"]
    with pytest.raises(RuntimeError, match="invalid scene_type"):
        sp.validate_scenes_v2([scene])


# ---------------------------------------------------------------------------
# validate_scenes_v2 - evidence_level validation
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_rejects_invalid_evidence_level():
    scene = _valid_v2_scene()
    scene["evidence_level"] = "somewhat_sure"
    with pytest.raises(RuntimeError, match="invalid evidence_level"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_accepts_every_valid_evidence_level():
    for level in sp.EVIDENCE_LEVELS_V2:
        scene_type = "hypothesis_diagram" if level == "hypothesis" else "real_location"
        scene = _valid_v2_scene(scene_type=scene_type, evidence_level=level)
        sp.validate_scenes_v2([scene])  # must not raise


# ---------------------------------------------------------------------------
# validate_scenes_v2 - data_type validation
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_rejects_invalid_data_type():
    scene = _valid_v2_scene(data_points=[_valid_data_point(data_type="not_a_type")])
    with pytest.raises(RuntimeError, match="invalid data_type"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_rejects_data_points_not_a_list():
    scene = _valid_v2_scene()
    scene["data_points"] = "not a list"
    with pytest.raises(RuntimeError, match="data_points must be a list"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_rejects_exact_text_data_point_with_no_value():
    scene = _valid_v2_scene(data_points=[_valid_data_point(data_type="exact_text", value="")])
    with pytest.raises(RuntimeError, match="exact_text cannot be derived"):
        sp.validate_scenes_v2([scene])


# ---------------------------------------------------------------------------
# validate_scenes_v2 - hypothesis validation
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_rejects_hypothesis_with_incompatible_scene_type():
    scene = _valid_v2_scene()
    # Hand-construct an inconsistent scene directly (bypassing
    # resolve_visual_strategy, the way a malformed/hand-edited scenes.json
    # would) rather than through the helper, so scene_type/visual_strategy
    # stay exactly as an LLM-proposed-but-unresolved scene would.
    scene["evidence_level"] = "hypothesis"
    scene["scene_type"] = "real_location"
    scene["visual_strategy"] = "ai_generated"
    with pytest.raises(RuntimeError, match="evidence_level=hypothesis but scene_type"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_accepts_hypothesis_with_allowed_scene_type():
    scene = _valid_v2_scene(scene_type="hypothesis_diagram", evidence_level="hypothesis")
    sp.validate_scenes_v2([scene])  # must not raise


# ---------------------------------------------------------------------------
# validate_scenes_v2 - visual relevance validation
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_rejects_empty_visual_relevance_reason():
    scene = _valid_v2_scene(visual_relevance_reason="")
    with pytest.raises(RuntimeError, match="empty visual_relevance_reason"):
        sp.validate_scenes_v2([scene])


@pytest.mark.parametrize("phrase", sorted(sp._GENERIC_RELEVANCE_PHRASES))
def test_validate_scenes_v2_rejects_every_known_generic_phrase(phrase):
    scene = _valid_v2_scene(visual_relevance_reason=f"This scene {phrase} for the viewer.")
    with pytest.raises(RuntimeError, match="generic/filler"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_accepts_a_specific_relevance_reason():
    scene = _valid_v2_scene(visual_relevance_reason="Shows the Big Ear radio telescope's actual dish structure.")
    sp.validate_scenes_v2([scene])  # must not raise


# ---------------------------------------------------------------------------
# validate_scenes_v2 - deterministic strategy self-consistency
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_rejects_mismatched_strategy():
    scene = _valid_v2_scene()
    scene["visual_strategy"] = "programmatic_graphic"  # doesn't match resolve_visual_strategy's output
    with pytest.raises(RuntimeError, match="does not match what resolve_visual_strategy"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_rejects_illegally_stored_archival_source_scene_type():
    scene = _valid_v2_scene()
    scene["scene_type"] = "archival_source"  # resolution never produces this
    scene["visual_strategy"] = "ai_generated"
    with pytest.raises(RuntimeError, match="does not match what resolve_visual_strategy"):
        sp.validate_scenes_v2([scene])


def test_validate_scenes_v2_collects_multiple_violations_at_once():
    scene = _valid_v2_scene()
    scene["scene_type"] = "bogus"
    scene["evidence_level"] = "bogus"
    scene["claim"] = ""
    with pytest.raises(RuntimeError) as excinfo:
        sp.validate_scenes_v2([scene])
    message = str(excinfo.value)
    assert "invalid scene_type" in message
    assert "invalid evidence_level" in message
    assert "missing/empty 'claim'" in message


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------

def test_validate_scenes_v2_skips_legacy_v1_scenes_with_no_schema_version():
    legacy_v1_scene = {
        "scene_id": 1, "duration_hint": 5.0, "narration_en": "text",
        "visual_prompt": "a cinematic shot", "visual_type": "cinematic",
        # no schema_version at all, no scene_type/evidence_level/data_points
    }
    sp.validate_scenes_v2([legacy_v1_scene])  # must not raise


def test_validate_scenes_v2_skips_scenes_with_non_2_schema_version():
    scene = {"scene_id": 1, "schema_version": 1}
    sp.validate_scenes_v2([scene])  # must not raise


def test_validate_scenes_v2_checks_v2_scenes_in_a_mixed_list_but_leaves_v1_alone():
    legacy_v1_scene = {"scene_id": 1, "visual_prompt": "x", "visual_type": "cinematic"}
    bad_v2_scene = _valid_v2_scene()
    bad_v2_scene["scene_id"] = 2
    bad_v2_scene["scene_type"] = "bogus"
    with pytest.raises(RuntimeError, match="scene 2"):
        sp.validate_scenes_v2([legacy_v1_scene, bad_v2_scene])


def test_v1_plan_scenes_is_completely_unaffected_by_the_v2_addition(monkeypatch):
    """Regression guard: v1's plan_scenes() must still produce the exact
    old shape (no schema_version, no scene_type, still just visual_prompt/
    visual_type) - the V2 addition above must be 100% additive."""
    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        return [
            {"scene_id": i + 1, "visual_prompt": f"prompt {i}", "visual_type": "cinematic"}
            for i in range(len(chunk_lines))
        ], []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake_chat_json)

    scenes = sp.plan_scenes(_fake_script_en())

    for scene in scenes:
        assert "schema_version" not in scene
        assert "scene_type" not in scene
        assert "claim" not in scene
        assert scene["visual_type"] in sp.VISUAL_TYPES
        assert scene["visual_prompt"]


# =============================================================================
# Scene Planner V2 - Phase 2 ONLY (deterministic prompt builder). See
# scene_planner.py's Phase 2 section docstring for the full design.
#
# Covers the 16 categories requested for Phase 2:
#     - exact_text -> AI render prohibition
#     - exact_text -> overlay requirement
#     - coordinate -> overlay/programmatic requirement
#     - measurement -> accuracy hint
#     - hypothesis -> hypothesis visualization
#     - hypothesis -> uncertainty preserved
#     - programmatic graphic strategy
#     - archival without source -> document recreation
#     - research evidence -> avoid generic scientist/computer
#     - visual relevance validation
#     - suspicious numeric leakage
#     - render_hash determinism
#     - same input -> same visual_prompt
#     - same input -> same render_hash
#     - changed strategy/data -> changed render_hash
#     - backward compatibility
#
# Real Wow! Signal fixtures are used throughout, per the explicit request:
# 6EQUJ5, 1420.726 MHz, Sagittarius coordinates, signal intensity curve,
# comet hypothesis, magnetar hypothesis, Arecibo 2024/2025 reanalysis,
# historical printout / Cosmic Search.
#
# Deliberately does NOT test anything past Phase 2 scope: no FAL call, no
# image_gen change, no overlay renderer, no scientific-graphics renderer,
# no pipeline.py wiring - none of that exists yet.
# =============================================================================

_PROMPT_SEGMENTS = {
    "subject": "a specific documentary subject", "setting": "a specific real setting",
    "lighting": "moody natural lighting", "mood": "investigative",
    "composition": "wide establishing shot", "style": "cinematic documentary, photorealistic",
}


def _6equj5_scene(**overrides) -> dict:
    """Real Wow! Signal fixture: the '6EQUJ5' printout code (evidence_closeup,
    primary_source, exact_text)."""
    return _valid_v2_scene(
        scene_type="evidence_closeup", evidence_level="primary_source",
        data_points=[_valid_data_point(data_type="exact_text", value="6EQUJ5", exact_render=True)],
        claim="Buried in the printout data was the sequence 6EQUJ5.",
        visual_relevance_reason="Closes in on the exact '6EQUJ5' character sequence from the real printout.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="closeup on a computer printout showing alphanumeric characters"),
        **overrides,
    )


def _frequency_scene(exact_render: bool = True, **overrides) -> dict:
    """Real Wow! Signal fixture: the corrected 1420.726 MHz frequency
    (scientific_graphic, documented_fact, measurement)."""
    return _valid_v2_scene(
        scene_type="scientific_graphic", evidence_level="documented_fact",
        data_points=[_valid_data_point(data_type="measurement", value="1420.726 MHz", exact_render=exact_render)],
        claim="The 2025 Arecibo Wow! reanalysis revised the frequency to approximately 1420.726 MHz.",
        visual_relevance_reason="Shows the corrected 2025 frequency value landing on hydrogen's natural emission line.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="radio spectrum chart with a peak near the hydrogen emission line"),
        **overrides,
    )


def _coordinates_scene(**overrides) -> dict:
    """Real Wow! Signal fixture: the refined Sagittarius RA/Dec coordinates
    (astronomical_map, documented_fact, coordinate, exact_render)."""
    return _valid_v2_scene(
        scene_type="astronomical_map", evidence_level="documented_fact",
        data_points=[
            _valid_data_point(data_type="coordinate", value="RA 19h25m02s, Dec -26d57m", exact_render=True),
        ],
        claim="The signal's origin was refined to precise coordinates toward Sagittarius.",
        visual_relevance_reason="Plots the exact refined coordinates on a sky map centered on Sagittarius.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="star map of Sagittarius with a coordinate grid overlay"),
        **overrides,
    )


def _signal_curve_scene(**overrides) -> dict:
    """Real Wow! Signal fixture: the intensity bell-curve matching a point
    source (scientific_graphic, primary_source, curve -> programmatic_graphic)."""
    return _valid_v2_scene(
        scene_type="scientific_graphic", evidence_level="primary_source",
        data_points=[_valid_data_point(data_type="curve", value="signal intensity bell curve", exact_render=False)],
        claim="The signal's intensity rose and fell in a bell-curve pattern matching a point source.",
        visual_relevance_reason="Plots the actual rise-and-fall intensity curve confirming a point-source origin.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="line chart of signal intensity over time"),
        **overrides,
    )


def _comet_hypothesis_scene(**overrides) -> dict:
    """Real Wow! Signal fixture: Antonio Paris's 2017 comet 266/P Christensen
    hypothesis (hypothesis_diagram, hypothesis)."""
    return _valid_v2_scene(
        scene_type="hypothesis_diagram", evidence_level="hypothesis",
        data_points=[_valid_data_point(data_type="exact_text", value="comet 266/P Christensen", exact_render=True)],
        claim="In 2017, Antonio Paris proposed the signal came from comet 266/P Christensen.",
        visual_relevance_reason="Diagrams the specific comet hypothesis, marked as speculative, not established fact.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="diagram of a comet's path intersecting the telescope's line of sight"),
        **overrides,
    )


def _magnetar_hypothesis_scene(**overrides) -> dict:
    """Real Wow! Signal fixture: the magnetar-triggered hydrogen-maser
    hypothesis (hypothesis_diagram, hypothesis, no data_points)."""
    return _valid_v2_scene(
        scene_type="hypothesis_diagram", evidence_level="hypothesis", data_points=[],
        claim="The magnetar hypothesis proposes a magnetar flare could brighten a cold hydrogen cloud, producing the signal.",
        visual_relevance_reason="Diagrams the magnetar-flare-triggering-a-hydrogen-cloud mechanism, marked as a hypothesis.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="diagram of a magnetar emitting a burst of radiation toward a hydrogen cloud"),
        **overrides,
    )


def _arecibo_research_scene(year: str = "2024", **overrides) -> dict:
    """Real Wow! Signal fixture: the Arecibo 2024/2025 reanalysis research
    (research_evidence, documented_fact/scientific_interpretation)."""
    return _valid_v2_scene(
        scene_type="research_evidence", evidence_level="documented_fact",
        data_points=[_valid_data_point(data_type="exact_text", value=f"August {year}", exact_render=False)],
        claim=f"In {year}, researchers published analysis of archival Big Ear and Arecibo data.",
        visual_relevance_reason=f"Shows the real {year} research effort analyzing archival Big Ear and Arecibo data.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="researcher reviewing archival radio telescope data"),
        **overrides,
    )


def _cosmic_search_scene(**overrides) -> dict:
    """Real Wow! Signal fixture: the 1979 Cosmic Search journal publication /
    historical printout (archival_source -> document_recreation)."""
    return _valid_v2_scene(
        scene_type="archival_source", evidence_level="documented_fact",
        data_points=[_valid_data_point(data_type="exact_text", value="Cosmic Search, 1979", exact_render=True)],
        claim="John Kraus first published a description in the journal Cosmic Search in 1979.",
        visual_relevance_reason="Recreates the actual 1979 Cosmic Search journal publication where Kraus first described the detection.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="1970s scientific journal pages with an article about the detection"),
        **overrides,
    )


# ---------------------------------------------------------------------------
# exact_text -> AI render prohibition / overlay requirement
# ---------------------------------------------------------------------------

def test_exact_text_scene_visual_prompt_forbids_ai_rendering_of_exact_text():
    built = sp.build_scene_prompt(_6equj5_scene())
    assert sp._NO_EXACT_TEXT_CONSTRAINT in built["visual_prompt"]
    assert "never rendered by the image model" in sp._NO_EXACT_TEXT_CONSTRAINT.lower()


def test_exact_text_scene_visual_prompt_never_contains_the_literal_exact_text():
    built = sp.build_scene_prompt(_6equj5_scene())
    assert "6equj5" not in built["visual_prompt"].lower()


def test_exact_text_scene_requires_overlay():
    built = sp.build_scene_prompt(_6equj5_scene())
    assert built["visual_strategy"] == "ai_generated_plus_overlay"
    assert built["overlay_required"] is True


def test_non_exact_text_scene_does_not_require_overlay():
    scene = _valid_v2_scene(scene_type="real_location", evidence_level="documented_fact", data_points=[])
    built = sp.build_scene_prompt(scene)
    assert built["overlay_required"] is False
    assert sp._NO_EXACT_TEXT_CONSTRAINT not in built["visual_prompt"]


# ---------------------------------------------------------------------------
# coordinate -> overlay/programmatic requirement
# ---------------------------------------------------------------------------

def test_coordinate_scene_with_exact_render_goes_to_overlay():
    built = sp.build_scene_prompt(_coordinates_scene())
    assert built["scene_type"] == "astronomical_map"
    assert built["visual_strategy"] == "ai_generated_plus_overlay"
    assert built["overlay_required"] is True
    assert sp._NO_EXACT_TEXT_CONSTRAINT in built["visual_prompt"]
    assert "19h25m02s" not in built["visual_prompt"]


def test_coordinate_scene_without_exact_render_gets_accuracy_hint_not_overlay():
    scene = _valid_v2_scene(
        scene_type="astronomical_map", evidence_level="documented_fact",
        data_points=[_valid_data_point(data_type="coordinate", value="RA 19h25m02s", exact_render=False)],
        visual_relevance_reason="Plots the general sky position toward Sagittarius.",
        prompt_segments=dict(_PROMPT_SEGMENTS, subject="star map of Sagittarius"),
    )
    built = sp.build_scene_prompt(scene)
    assert built["overlay_required"] is False
    assert any("coordinate figures" in hint for hint in built["accuracy_hints"])


# ---------------------------------------------------------------------------
# deterministic scrubbing of exact_text leaking through the LLM's own
# prompt_segments (real Phase 2.2 audit finding: scene 13, exact_text=
# ["Sagittarius"], leaked into visual_prompt via prompt_segments.setting -
# see _scrub_exact_text_from_segments()/build_visual_prompt() above).
#
# Root cause this section pins down: _coordinates_scene() above already
# covered an exact_text NUMBER leaking (a coordinate figure) - but a
# coordinate figure like "19h25m02s" almost never occurs naturally in an
# LLM's own descriptive prose. The real bug was a plain WORD (a
# constellation name) declared as exact_text - and a documentary
# cinematography description is very likely to naturally use that same
# word when describing the scene's setting, which is exactly what slipped
# through undetected by the existing test fixtures.
# ---------------------------------------------------------------------------

def _sagittarius_constellation_name_scene(**overrides) -> dict:
    """Real Phase 2.2 audit finding (scene 13): the LLM classified the
    constellation NAME "Sagittarius" itself (not just the numeric RA/Dec
    figure _coordinates_scene() above covers) as an exact_text data point -
    and its own prompt_segments.setting also naturally named "Sagittarius"
    when describing the scene, exactly as a documentary cinematographer
    would. That combination is what leaked into the real visual_prompt."""
    return _valid_v2_scene(
        scene_type="astronomical_map", evidence_level="documented_fact",
        data_points=[_valid_data_point(data_type="exact_text", value="Sagittarius", exact_render=True)],
        claim="The signal's origin was refined to a location toward the constellation Sagittarius.",
        visual_relevance_reason="Plots the refined sky position toward the constellation the signal came from.",
        prompt_segments=dict(
            _PROMPT_SEGMENTS,
            subject="a radio telescope dish pointed skyward",
            setting="the night sky in the direction of the Sagittarius constellation",
        ),
        **overrides,
    )


def test_scrub_exact_text_from_segments_removes_value_from_every_segment():
    segments = {
        "subject": "a view of Sagittarius rising",
        "setting": "somewhere near Sagittarius",
        "lighting": "dim",
        "mood": "quiet",
        "composition": "wide",
        "style": "cinematic",
    }
    scrubbed = sp._scrub_exact_text_from_segments(segments, ["Sagittarius"])
    for key in ("subject", "setting"):
        assert "sagittarius" not in scrubbed[key].lower()
    # Segments that never mentioned the value are left byte-identical.
    assert scrubbed["lighting"] == "dim"
    assert scrubbed["mood"] == "quiet"


def test_scrub_exact_text_from_segments_is_case_insensitive():
    scrubbed = sp._scrub_exact_text_from_segments({"subject": "toward SAGITTARIUS"}, ["Sagittarius"])
    assert "sagittarius" not in scrubbed["subject"].lower()


def test_scrub_exact_text_from_segments_no_op_when_exact_text_empty():
    segments = {"subject": "a calm scene", "setting": "a real place"}
    assert sp._scrub_exact_text_from_segments(segments, []) == segments
    assert sp._scrub_exact_text_from_segments(segments, None) == segments


def test_scrub_exact_text_from_segments_never_mutates_input():
    segments = {"subject": "toward Sagittarius"}
    sp._scrub_exact_text_from_segments(segments, ["Sagittarius"])
    assert segments["subject"] == "toward Sagittarius"  # the caller's dict is untouched


def test_astronomical_map_scene_with_leaking_setting_builds_a_clean_prompt():
    # The real bug, reproduced: prompt_segments.setting legitimately named
    # the same word the LLM separately declared as exact_text.
    scene = _sagittarius_constellation_name_scene()
    built = sp.build_scene_prompt(scene)

    assert "sagittarius" not in built["visual_prompt"].lower()
    assert sp._exact_text_leak(built["visual_prompt"], built["exact_text"]) == []
    # The overlay-relevant metadata is untouched by the scrub - never
    # deleted, never blanked - only the AI-image-prompt text is scrubbed.
    assert built["exact_text"] == ["Sagittarius"]
    assert built["data_points"][0]["value"] == "Sagittarius"


def test_astronomical_map_scene_with_leaking_setting_passes_validate_prompts_v2():
    # End-to-end: this exact shape is what raised RuntimeError from
    # validate_prompts_v2() in the real audit - confirms the hard gate
    # (left completely unchanged) no longer trips for it, because the leak
    # no longer happens upstream, not because the check got weaker.
    built = sp.build_scene_prompt(_sagittarius_constellation_name_scene())
    sp.validate_prompts_v2([built])  # must not raise


def test_scrub_is_a_no_op_for_a_scene_with_no_leak():
    # A normal, correctly-behaved classification (no overlap between
    # exact_text and prompt_segments) must build exactly as before this fix.
    built = sp.build_scene_prompt(_6equj5_scene())
    assert "printout showing alphanumeric characters" in built["visual_prompt"]


# ---------------------------------------------------------------------------
# measurement -> accuracy hint
# ---------------------------------------------------------------------------

def test_measurement_without_exact_render_gets_accuracy_hint():
    hints = sp._build_accuracy_hints([_valid_data_point(data_type="measurement", value="1420.726 MHz", exact_render=False)])
    assert len(hints) == 1
    assert "exact digits or numbers" in hints[0]
    assert "1420.726" not in hints[0]  # never quotes the literal value


def test_measurement_with_exact_render_gets_no_accuracy_hint_only_overlay():
    hints = sp._build_accuracy_hints([_valid_data_point(data_type="measurement", value="1420.726 MHz", exact_render=True)])
    assert hints == []


def test_frequency_scene_end_to_end_accuracy_hint_when_not_exact_render():
    built = sp.build_scene_prompt(_frequency_scene(exact_render=False))
    assert any("exact digits" in hint for hint in built["accuracy_hints"])
    assert "1420.726" not in built["visual_prompt"]


def test_frequency_scene_end_to_end_overlay_when_exact_render():
    built = sp.build_scene_prompt(_frequency_scene(exact_render=True))
    assert built["overlay_required"] is True
    assert "1420.726" not in built["visual_prompt"]


# ---------------------------------------------------------------------------
# hypothesis -> hypothesis_visualization / uncertainty preserved
# ---------------------------------------------------------------------------

def test_comet_hypothesis_resolves_to_hypothesis_visualization():
    built = sp.build_scene_prompt(_comet_hypothesis_scene())
    assert built["visual_strategy"] == "hypothesis_visualization"
    assert built["scene_type"] == "hypothesis_diagram"


def test_magnetar_hypothesis_resolves_to_hypothesis_visualization():
    built = sp.build_scene_prompt(_magnetar_hypothesis_scene())
    assert built["visual_strategy"] == "hypothesis_visualization"


def test_hypothesis_scenes_preserve_uncertainty_via_avoid_strategy():
    for scene in (_comet_hypothesis_scene(), _magnetar_hypothesis_scene()):
        built = sp.build_scene_prompt(scene)
        joined_avoid = " ".join(built["avoid_strategy"]).lower()
        assert "definitive" in joined_avoid or "confirmed" in joined_avoid
        assert "Avoid:" in built["visual_prompt"]


# ---------------------------------------------------------------------------
# programmatic graphic strategy
# ---------------------------------------------------------------------------

def test_signal_curve_scene_resolves_to_programmatic_graphic():
    built = sp.build_scene_prompt(_signal_curve_scene())
    assert built["visual_strategy"] == "programmatic_graphic"


def test_programmatic_graphic_scene_has_no_visual_prompt():
    built = sp.build_scene_prompt(_signal_curve_scene())
    assert built["visual_prompt"] is None


def test_programmatic_graphic_scene_still_gets_a_render_hash():
    built = sp.build_scene_prompt(_signal_curve_scene())
    assert isinstance(built["render_hash"], str) and len(built["render_hash"]) == 64


# ---------------------------------------------------------------------------
# archival without source -> document recreation
# ---------------------------------------------------------------------------

def test_cosmic_search_archival_source_resolves_to_document_recreation():
    built = sp.build_scene_prompt(_cosmic_search_scene())
    assert built["scene_type"] == "document_recreation"  # never archival_source - no real asset pipeline in V2.0


def test_document_recreation_avoids_claiming_to_be_genuine_archival_footage():
    built = sp.build_scene_prompt(_cosmic_search_scene())
    joined_avoid = " ".join(built["avoid_strategy"]).lower()
    assert "genuine" in joined_avoid or "unaltered archival" in joined_avoid
    assert "Avoid:" in built["visual_prompt"]


# ---------------------------------------------------------------------------
# research evidence -> avoid generic scientist/computer
# ---------------------------------------------------------------------------

def test_research_evidence_avoids_generic_scientist_at_laptop_cliche():
    for scene in (_arecibo_research_scene("2024"), _arecibo_research_scene("2025")):
        built = sp.build_scene_prompt(scene)
        joined_avoid = " ".join(built["avoid_strategy"]).lower()
        assert "laptop" in joined_avoid or "stock-office" in joined_avoid


def test_research_evidence_avoids_repeating_the_same_visual_setup():
    built = sp.build_scene_prompt(_arecibo_research_scene("2024"))
    joined_avoid = " ".join(built["avoid_strategy"]).lower()
    assert "repeating the same visual setup" in joined_avoid


def test_arecibo_2024_and_2025_scenes_both_build_cleanly():
    for year in ("2024", "2025"):
        built = sp.build_scene_prompt(_arecibo_research_scene(year))
        assert built["visual_prompt"]
        assert year not in built["visual_prompt"]  # the exact_text date is never quoted in the AI prompt


# ---------------------------------------------------------------------------
# visual relevance validation
# ---------------------------------------------------------------------------

def test_build_scene_prompt_requires_visual_relevance_reason():
    scene = _valid_v2_scene(visual_relevance_reason="")
    with pytest.raises(RuntimeError, match="visual_relevance_reason"):
        sp.build_scene_prompt(scene)


def test_build_scene_prompt_requires_visual_relevance_reason_not_just_whitespace():
    scene = _valid_v2_scene(visual_relevance_reason="   ")
    with pytest.raises(RuntimeError, match="visual_relevance_reason"):
        sp.build_scene_prompt(scene)


# ---------------------------------------------------------------------------
# exact_text leakage - hard validation (validate_prompts_v2)
# ---------------------------------------------------------------------------

def test_validate_prompts_v2_raises_when_visual_prompt_leaks_exact_text():
    built = sp.build_scene_prompt(_6equj5_scene())
    # Simulate drift - as if the prompt builder (or a future change) had
    # leaked the literal exact_text value into the prompt despite the design.
    leaked = dict(built)
    leaked["visual_prompt"] = built["visual_prompt"] + "\nSubject: a printout reading 6EQUJ5 clearly"
    with pytest.raises(RuntimeError, match="leaks exact_text"):
        sp.validate_prompts_v2([leaked])


def test_validate_prompts_v2_passes_for_correctly_built_scenes():
    scenes = [
        _6equj5_scene(), _frequency_scene(), _coordinates_scene(), _signal_curve_scene(),
        _comet_hypothesis_scene(), _magnetar_hypothesis_scene(),
        _arecibo_research_scene("2024"), _cosmic_search_scene(),
    ]
    built = [sp.build_scene_prompt(s) for s in scenes]
    sp.validate_prompts_v2(built)  # must not raise


def test_validate_prompts_v2_rejects_missing_visual_prompt_for_non_programmatic_strategy():
    built = sp.build_scene_prompt(_6equj5_scene())
    broken = dict(built)
    broken["visual_prompt"] = None
    with pytest.raises(RuntimeError, match="missing visual_prompt"):
        sp.validate_prompts_v2([broken])


def test_validate_prompts_v2_rejects_a_visual_prompt_on_a_programmatic_graphic_scene():
    built = sp.build_scene_prompt(_signal_curve_scene())
    broken = dict(built)
    broken["visual_prompt"] = "should not exist"
    with pytest.raises(RuntimeError, match="must not have a visual_prompt"):
        sp.validate_prompts_v2([broken])


# ---------------------------------------------------------------------------
# suspicious numeric/text leakage - soft warning
# ---------------------------------------------------------------------------

def test_scan_suspicious_leakage_detects_alphanumeric_code():
    hits = sp._scan_suspicious_leakage("Subject: a printout showing 6EQUJ5 on the page")
    assert any("alphanumeric code" in h for h in hits)


def test_scan_suspicious_leakage_detects_digit_sequences():
    hits = sp._scan_suspicious_leakage("Subject: a dial reading 1420 exactly")
    assert any("digit sequence" in h for h in hits)


def test_scan_suspicious_leakage_clean_prompt_has_no_hits():
    built = sp.build_scene_prompt(_magnetar_hypothesis_scene())
    assert sp._scan_suspicious_leakage(built["visual_prompt"]) == []


def test_validate_prompts_v2_logs_but_does_not_raise_on_suspicious_leakage(monkeypatch, caplog):
    built = sp.build_scene_prompt(_magnetar_hypothesis_scene())
    suspicious = dict(built)
    # Injects a number NOT present in this scene's own exact_text - the hard
    # check (validate_prompts_v2's leak check) must stay silent since it only
    # looks for this scene's declared exact_text; only the soft scan should fire.
    suspicious["visual_prompt"] = built["visual_prompt"] + "\nSubject: a dial reading 1420 exactly"
    import logging
    with caplog.at_level(logging.WARNING):
        sp.validate_prompts_v2([suspicious])  # must not raise
    assert any("suspicious" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# render_hash determinism / same input -> same output / changed input -> changed hash
# ---------------------------------------------------------------------------

def test_same_input_produces_same_visual_prompt():
    scene = _6equj5_scene()
    first = sp.build_visual_prompt(sp.build_scene_prompt(scene))
    second = sp.build_visual_prompt(sp.build_scene_prompt(scene))
    assert first == second


def test_same_input_produces_same_render_hash():
    scene = _6equj5_scene()
    first = sp.compute_render_hash(sp.build_scene_prompt(scene))
    second = sp.compute_render_hash(sp.build_scene_prompt(scene))
    assert first == second


def test_render_hash_is_a_64_char_hex_sha256():
    built = sp.build_scene_prompt(_6equj5_scene())
    assert len(built["render_hash"]) == 64
    int(built["render_hash"], 16)  # must be valid hex


def test_changed_visual_strategy_changes_render_hash():
    base = sp.build_scene_prompt(_6equj5_scene())
    changed = dict(base)
    changed["visual_strategy"] = "ai_generated"
    assert sp.compute_render_hash(changed) != sp.compute_render_hash(base)


def test_changed_data_points_changes_render_hash():
    base = sp.build_scene_prompt(_frequency_scene(exact_render=True))
    changed = sp.build_scene_prompt(_frequency_scene(exact_render=False))
    assert sp.compute_render_hash(base) != sp.compute_render_hash(changed)


def test_changed_scene_id_or_narration_does_not_change_render_hash():
    # render_hash is deliberately about RENDERING inputs only - renumbering a
    # scene or tweaking its narration text must not invalidate a cached render.
    scene_a = _6equj5_scene(scene_id=5, narration_en="Version A of the narration.")
    scene_b = _6equj5_scene(scene_id=9, narration_en="A completely different version B.")
    built_a = sp.build_scene_prompt(scene_a)
    built_b = sp.build_scene_prompt(scene_b)
    assert sp.compute_render_hash(built_a) == sp.compute_render_hash(built_b)


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------

def test_build_scene_prompt_passes_legacy_v1_scenes_through_untouched():
    legacy = {"scene_id": 1, "visual_prompt": "an existing v1 prompt", "visual_type": "cinematic"}
    result = sp.build_scene_prompt(legacy)
    assert result is legacy  # untouched, same object, no new fields added
    assert "render_hash" not in result
    assert "avoid_strategy" not in result or result.get("avoid_strategy") is None


def test_build_prompts_v2_leaves_v1_scenes_alone_in_a_mixed_list():
    legacy = {"scene_id": 1, "visual_prompt": "an existing v1 prompt", "visual_type": "cinematic"}
    v2_scene = _6equj5_scene()
    v2_scene["scene_id"] = 2
    result = sp.build_prompts_v2([legacy, v2_scene])
    assert result[0] is legacy
    assert result[1]["render_hash"]


def test_build_prompts_v2_end_to_end_on_wow_signal_fixtures():
    scenes = [
        _6equj5_scene(scene_id=1), _frequency_scene(scene_id=2), _coordinates_scene(scene_id=3),
        _signal_curve_scene(scene_id=4), _comet_hypothesis_scene(scene_id=5),
        _magnetar_hypothesis_scene(scene_id=6), _arecibo_research_scene("2024", scene_id=7),
        _arecibo_research_scene("2025", scene_id=8), _cosmic_search_scene(scene_id=9),
    ]
    built = sp.build_prompts_v2(scenes)  # runs validate_scenes_v2 + validate_prompts_v2 internally, must not raise
    assert len(built) == 9
    for scene in built:
        assert "render_hash" in scene
        assert "avoid_strategy" in scene


def test_v1_plan_scenes_and_v1_run_and_save_unaffected_by_phase_2(monkeypatch):
    """Regression guard mirroring Phase 1's own: the Phase 2 addition above
    must be 100% additive on top of both v1 AND Phase 1 - nothing about
    plan_scenes()'s output shape changes."""
    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        return [
            {"scene_id": i + 1, "visual_prompt": f"prompt {i}", "visual_type": "cinematic"}
            for i in range(len(chunk_lines))
        ], []
    monkeypatch.setattr(sp.llm_client, "chat_json", fake_chat_json)

    scenes = sp.plan_scenes(_fake_script_en())
    for scene in scenes:
        assert "render_hash" not in scene
        assert "overlay_required" not in scene


# ---------------------------------------------------------------------------
# run_and_save_v2 - Visual Generator V2 integration's explicit V2 pipeline
# entry point (pipeline.py's "scenes" stage now calls this, not v1's
# run_and_save). Mirrors test_scene_planner.py's v1 run_and_save tests
# (same storage primitives, same state, same failure path) plus confirms
# it actually runs BOTH Phase 1 (plan_scenes_v2) and Phase 2
# (build_prompts_v2) - not just Phase 1.
# ---------------------------------------------------------------------------

def test_run_and_save_v2_records_failed_state_when_script_json_missing(isolated_storage):
    """Same regression guard as v1's run_and_save: the precondition check
    must go through save_failure(), not raise before db.start_run()'s try
    block, or the video is left stuck with no FAILED state and no
    diagnostic trail."""
    paths, _ = ps.create_project("Missing Script JSON Topic V2")
    # script.json deliberately not written

    with pytest.raises(RuntimeError, match="no script.json found"):
        sp.run_and_save_v2(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "scene_plan"


def test_run_and_save_v2_writes_scenes_json_via_existing_storage_writer(isolated_storage, monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    paths, _ = ps.create_project("Run And Save V2 Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    result = sp.run_and_save_v2(paths)

    assert result  # non-empty
    on_disk = ps.load_json(paths.scenes_json)  # the SAME path v1 writes to
    assert on_disk == result
    assert ps.load_state(paths) == PipelineState.SCENES_PLANNED


def test_run_and_save_v2_runs_phase_1_and_phase_2_not_just_classification(isolated_storage, monkeypatch):
    """Phase 1 alone (plan_scenes_v2) never sets visual_prompt - that's
    Phase 2's (build_prompts_v2) job. If run_and_save_v2 only wired in
    Phase 1, every scene would still have visual_prompt missing/None even
    for ai_generated scenes, which would then crash image_gen.py. This is
    the test that would have caught that mistake."""
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    paths, _ = ps.create_project("Phase 1 And 2 Wired Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    scenes = sp.run_and_save_v2(paths)

    for scene in scenes:
        assert scene["schema_version"] == 2
        assert scene["visual_strategy"] == "ai_generated"
        assert scene.get("visual_prompt")  # Phase 2 ran - non-empty for ai_generated
        assert "render_hash" in scene  # Phase 2-only field
        assert "avoid_strategy" in scene  # Phase 2-only field


def test_run_and_save_v2_preserves_programmatic_graphic_none_visual_prompt(isolated_storage, monkeypatch):
    """The one scene shape where visual_prompt is SUPPOSED to stay None -
    run_and_save_v2 (via build_prompts_v2) must not paper over that with a
    placeholder string; image_gen.py's strategy dispatch is what handles
    None for this one strategy."""
    data_points = [{"data_type": "time_series", "value": "curve", "exact_render": False}]
    monkeypatch.setattr(
        sp.llm_client, "chat_json",
        _fake_chat_json_v2(scene_type="scientific_graphic", data_points=data_points),
    )
    paths, _ = ps.create_project("Programmatic Graphic Wired Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    scenes = sp.run_and_save_v2(paths)

    for scene in scenes:
        assert scene["visual_strategy"] == "programmatic_graphic"
        assert scene.get("visual_prompt") is None


def test_run_and_save_v2_does_not_touch_v1_run_and_save_or_plan_scenes(isolated_storage, monkeypatch):
    """run_and_save_v2 is an ADDITIVE sibling, not a replacement - v1's
    run_and_save/plan_scenes must remain independently callable and
    produce their own unrelated (schema_version-less) scene shape."""
    def fake_chat_json_v1(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        return [
            {"visual_prompt": f"prompt {i}", "visual_type": "cinematic"}
            for i in range(len(chunk_lines))
        ], []

    monkeypatch.setattr(sp.llm_client, "chat_json", fake_chat_json_v1)
    v1_scenes = sp.plan_scenes(_fake_script_en())
    for scene in v1_scenes:
        assert "schema_version" not in scene
        assert "visual_strategy" not in scene

    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2())
    paths, _ = ps.create_project("V1 V2 Coexistence Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    v2_scenes = sp.run_and_save_v2(paths)
    for scene in v2_scenes:
        assert scene["schema_version"] == 2


# ---------------------------------------------------------------------------
# Indexed classification hardening (cardinality drift fix for the real
# production "[SCENES][V2] cardinality mismatch: expected 35 classifications,
# got 36" failure - forensic diagnostic
# cardinality_mismatch_20260919T122149171224Z.json proved the LLM split
# input chunk 18 into two classification objects and absorbed chunk 19's
# content into the second half of that split).
#
# Covers exactly the 20 categories requested for this hardening pass:
#   1  normal N->N                              11 targeted repair wrong index
#   2  reordered but complete N->N               12 targeted repair failure -> full-batch retry
#   3  exact reproduction of the real failure     13 full-batch retry success
#   4  N-1 missing index                         14 full-batch retry failure
#   5  invalid index: string                     15 retry ceiling / no infinite loop
#   6  invalid index: None                       16 diagnostic enrichment
#   7  invalid index: 0                          17 reordered classifications -> same scene order
#   8  invalid index: N+1                        18 no scene_id/topic-specific logic
#   9  missing input_chunk_index on all objects  19 V1 compatibility
#  10  targeted repair success                   20 full existing test suite (run separately, not a unit test)
#
# _analyze_classification_indices is pure/deterministic and never calls the
# LLM, so categories 1-9 exercise it directly. Categories 10-16 exercise
# _classify_chunks_v2 (the recovery ladder) with a scripted chat_json fake
# that returns one response per call, in order, so each stage of the ladder
# can be verified independently. Categories 17-19 exercise the integration
# points (plan_scenes_v2 / plan_scenes) that sit around this machinery.
# ---------------------------------------------------------------------------

_TEST_CHUNKS = [
    "First chunk text here now.",
    "Second chunk text right here.",
    "Third chunk text over here.",
]


def _classification(idx, claim=None, scene_type="real_location", evidence_level="documented_fact"):
    """Builds one well-formed V2 classification object for input_chunk_index
    `idx` - the same field shape _fake_chat_json_v2() above uses, since
    these tests need the same content to satisfy plan_scenes_v2's
    downstream validate_scenes_v2() whenever a test goes through the full
    plan_scenes_v2() entrypoint rather than calling _classify_chunks_v2
    directly."""
    return {
        "input_chunk_index": idx,
        "claim": claim or f"Claim for chunk {idx}",
        "scene_type": scene_type,
        "evidence_level": evidence_level,
        "data_points": [],
        "visual_relevance_reason": f"Depicts the specific subject named in chunk {idx}'s narration.",
        "prompt_segments": {
            "subject": "s", "setting": "s", "lighting": "l",
            "mood": "m", "composition": "c", "style": "s",
        },
    }


def _scripted_chat_json(*responses):
    """Returns (fake, state). `fake` is a chat_json-compatible callable
    that returns each of `responses` in order on successive calls (each
    response is a (classifications, sources) tuple, matching chat_json's
    real return shape), repeating the last response if called more times
    than scripted - this only happens if a test's own assertions about
    call count are wrong, since _classify_chunks_v2 itself never calls the
    LLM more than three times (see
    test_classify_chunks_v2_never_calls_the_llm_more_than_three_times).
    state["calls"] collects every `user` prompt seen, in call order, so a
    test can assert exactly which chunks were sent to the LLM at each
    stage (e.g. that a targeted repair call includes ONLY the affected
    chunks, never surrounding context)."""
    state = {"calls": []}

    def fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        state["calls"].append(user)
        call_index = len(state["calls"]) - 1
        return responses[call_index] if call_index < len(responses) else responses[-1]

    return fake, state


# --- categories 1-9: _analyze_classification_indices (pure, no LLM) -------

def test_analyze_classification_indices_clean_when_all_present_once():
    classifications = [_classification(1), _classification(2), _classification(3)]
    analysis = sp._analyze_classification_indices(classifications, {1, 2, 3})
    assert sp._classification_indices_are_clean(analysis)
    assert analysis["duplicate_indices"] == []
    assert analysis["missing_indices"] == []
    assert analysis["invalid_indices"] == []


def test_analyze_classification_indices_clean_regardless_of_array_order():
    """Category 2: a complete, non-duplicated N->N response is clean even
    when the model's array order has nothing to do with chunk order -
    _ordered_classifications_by_index is what re-establishes chunk order,
    never the model's own array position (requirement D)."""
    classifications = [_classification(3), _classification(1), _classification(2)]
    analysis = sp._analyze_classification_indices(classifications, {1, 2, 3})
    assert sp._classification_indices_are_clean(analysis)
    ordered = sp._ordered_classifications_by_index(classifications, analysis, {1, 2, 3})
    assert [c["input_chunk_index"] for c in ordered] == [1, 2, 3]


def test_analyze_classification_indices_reproduces_the_real_dup18_missing19_shape():
    """Category 3: exact reproduction of the real 35 -> 36 production
    failure's index shape - duplicate at 18, missing at 19 - including the
    real diagnostic's exact total object count (36 for 35 input chunks):
    34 objects covering every index except 19, plus two extra objects for
    index 18 (the split)."""
    expected_indices = set(range(1, 36))
    classifications = [_classification(i) for i in range(1, 36) if i != 19]
    classifications.append(_classification(18, claim="Big Ear follow-up (split half 2)"))
    classifications.append(_classification(18, claim="Big Ear follow-up (split half 3, absorbing chunk 19)"))
    assert len(classifications) == 36

    analysis = sp._analyze_classification_indices(classifications, expected_indices)
    assert analysis["duplicate_indices"] == [18]
    assert analysis["missing_indices"] == [19]
    assert analysis["invalid_indices"] == []
    assert not sp._classification_indices_are_clean(analysis)


def test_analyze_classification_indices_detects_a_single_missing_index():
    classifications = [_classification(1), _classification(3)]  # index 2 never returned
    analysis = sp._analyze_classification_indices(classifications, {1, 2, 3})
    assert analysis["missing_indices"] == [2]
    assert analysis["duplicate_indices"] == []
    assert analysis["invalid_indices"] == []


def test_analyze_classification_indices_flags_string_index_as_invalid():
    bad = _classification(1)
    bad["input_chunk_index"] = "1"
    analysis = sp._analyze_classification_indices([bad, _classification(2), _classification(3)], {1, 2, 3})
    assert analysis["invalid_indices"] == [{"returned_index": 0, "value": "1"}]
    assert analysis["missing_indices"] == [1]
    assert analysis["duplicate_indices"] == []


def test_analyze_classification_indices_flags_none_index_as_invalid():
    bad = _classification(1)
    bad["input_chunk_index"] = None
    analysis = sp._analyze_classification_indices([bad, _classification(2), _classification(3)], {1, 2, 3})
    assert analysis["invalid_indices"] == [{"returned_index": 0, "value": None}]
    assert analysis["missing_indices"] == [1]


def test_analyze_classification_indices_flags_zero_index_as_invalid():
    bad = _classification(1)
    bad["input_chunk_index"] = 0
    analysis = sp._analyze_classification_indices([bad, _classification(2), _classification(3)], {1, 2, 3})
    assert analysis["invalid_indices"] == [{"returned_index": 0, "value": 0}]
    assert analysis["missing_indices"] == [1]


def test_analyze_classification_indices_flags_out_of_range_index_as_invalid():
    bad = _classification(1)
    bad["input_chunk_index"] = 4  # one past the last valid index (N=3)
    analysis = sp._analyze_classification_indices([bad, _classification(2), _classification(3)], {1, 2, 3})
    assert analysis["invalid_indices"] == [{"returned_index": 0, "value": 4}]
    assert analysis["missing_indices"] == [1]


def test_analyze_classification_indices_all_missing_field_are_all_invalid():
    classifications = [{"claim": "a"}, {"claim": "b"}, {"claim": "c"}]  # no input_chunk_index at all
    analysis = sp._analyze_classification_indices(classifications, {1, 2, 3})
    assert len(analysis["invalid_indices"]) == 3
    assert analysis["missing_indices"] == [1, 2, 3]
    assert analysis["duplicate_indices"] == []


# --- categories 10-16: _classify_chunks_v2 (the recovery ladder) ----------

def test_classify_chunks_v2_targeted_repair_success(monkeypatch):
    original = [_classification(1), _classification(2), _classification(2)]  # dup 2, missing 3
    repaired = [_classification(2, claim="repaired 2"), _classification(3, claim="repaired 3")]
    fake, state = _scripted_chat_json((original, []), (repaired, []))
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    result = sp._classify_chunks_v2(_TEST_CHUNKS)

    assert [c["input_chunk_index"] for c in result] == [1, 2, 3]
    assert result[1]["claim"] == "repaired 2"
    assert result[2]["claim"] == "repaired 3"
    assert len(state["calls"]) == 2  # original + one targeted repair, no full-batch retry needed
    # the targeted repair call must include ONLY the affected chunks (2 and
    # 3) - not chunk 1, not any surrounding context.
    assert "1. First chunk" not in state["calls"][1]
    assert "2. Second chunk" in state["calls"][1]
    assert "3. Third chunk" in state["calls"][1]


def test_classify_chunks_v2_targeted_repair_wrong_index_falls_through(monkeypatch):
    """Category 11: a repair response that still doesn't cleanly cover the
    exact requested index set (here, it labels both objects "2" instead of
    "2" and "3") must not be spliced in - it falls through to the
    full-batch retry stage instead of silently accepting a still-broken
    repair."""
    original = [_classification(1), _classification(2), _classification(2)]  # dup 2, missing 3
    wrong_repair = [_classification(2, claim="repaired 2"), _classification(2, claim="wrong, still 2 not 3")]
    full_retry = [_classification(1), _classification(2), _classification(3)]
    fake, state = _scripted_chat_json((original, []), (wrong_repair, []), (full_retry, []))
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    result = sp._classify_chunks_v2(_TEST_CHUNKS)

    assert [c["input_chunk_index"] for c in result] == [1, 2, 3]
    assert len(state["calls"]) == 3  # original + failed targeted repair + full-batch retry


def test_classify_chunks_v2_targeted_repair_non_list_response_falls_through_to_full_batch_retry(monkeypatch):
    original = [_classification(1), _classification(2), _classification(2)]  # dup 2, missing 3
    full_retry = [_classification(1), _classification(2), _classification(3)]
    fake, state = _scripted_chat_json((original, []), ({"not": "a list"}, []), (full_retry, []))
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    result = sp._classify_chunks_v2(_TEST_CHUNKS)

    assert [c["input_chunk_index"] for c in result] == [1, 2, 3]
    assert len(state["calls"]) == 3


def test_classify_chunks_v2_full_batch_retry_success_when_repair_is_disqualified(monkeypatch):
    """Category 13: when the original response has an invalid index (not
    just a duplicate/missing one), targeted repair is skipped entirely -
    an untrustworthy index gives nothing reliable to anchor a narrow
    re-ask on - and the ladder goes straight to the full-batch retry."""
    bad = _classification(1)
    bad["input_chunk_index"] = "not-an-int"
    original = [bad, _classification(2), _classification(2)]  # invalid + dup 2 + missing 3
    full_retry = [_classification(1), _classification(2), _classification(3)]
    fake, state = _scripted_chat_json((original, []), (full_retry, []))
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    result = sp._classify_chunks_v2(_TEST_CHUNKS)

    assert [c["input_chunk_index"] for c in result] == [1, 2, 3]
    assert len(state["calls"]) == 2  # original + full-batch retry - no targeted repair call at all


def test_classify_chunks_v2_full_batch_retry_failure_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "_DEBUG_AUDIT_DIR", tmp_path)
    original = [_classification(1), _classification(2), _classification(2)]  # dup 2, missing 3
    still_broken_repair = [_classification(2), _classification(2)]  # still dup 2, still missing 3
    still_broken_retry = [_classification(1), _classification(2), _classification(2)]  # same mismatch again
    fake, state = _scripted_chat_json((original, []), (still_broken_repair, []), (still_broken_retry, []))
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError, match=r"expected 3 classifications, got 3"):
        sp._classify_chunks_v2(_TEST_CHUNKS)

    assert len(state["calls"]) == 3  # original + targeted repair + full-batch retry, then hard fail
    written = list(tmp_path.glob("cardinality_mismatch_*.json"))
    assert len(written) == 1


def test_classify_chunks_v2_never_calls_the_llm_more_than_three_times(monkeypatch, tmp_path):
    """Category 15: retry ceiling / no infinite loop. However many broken
    responses are scripted, _classify_chunks_v2 must stop after exactly
    three calls (original, targeted repair, full-batch retry) and raise -
    never loop indefinitely trying to coax a clean response out of the
    model."""
    monkeypatch.setattr(sp, "_DEBUG_AUDIT_DIR", tmp_path)
    broken = [_classification(1), _classification(2), _classification(2)]  # dup 2, missing 3, forever
    fake, state = _scripted_chat_json(
        (broken, []), (broken, []), (broken, []), (broken, []), (broken, []),
    )
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError):
        sp._classify_chunks_v2(_TEST_CHUNKS)

    assert len(state["calls"]) == 3


def test_classify_chunks_v2_final_failure_diagnostic_is_enriched(monkeypatch, tmp_path):
    """Category 16: the diagnostic written on final failure must gain the
    new duplicate_indices/missing_indices/invalid_indices/repair_attempted/
    repair_indices/retry_stage fields additively, without breaking any of
    the pre-existing fields _write_cardinality_diagnostic already wrote."""
    monkeypatch.setattr(sp, "_DEBUG_AUDIT_DIR", tmp_path)
    original = [_classification(1), _classification(2), _classification(2)]  # dup 2, missing 3
    still_broken_repair = [_classification(2), _classification(2)]
    still_broken_retry = [_classification(1), _classification(2), _classification(2)]
    fake, state = _scripted_chat_json((original, []), (still_broken_repair, []), (still_broken_retry, []))
    monkeypatch.setattr(sp.llm_client, "chat_json", fake)

    with pytest.raises(RuntimeError):
        sp._classify_chunks_v2(_TEST_CHUNKS)

    written = list(tmp_path.glob("cardinality_mismatch_*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text(encoding="utf-8"))
    # pre-existing fields - must not have been broken by the additive extension
    assert data["expected_object_count"] == 3
    assert data["returned_object_count"] == 3
    # newly-added, additive fields from the indexed-classification hardening pass
    assert data["duplicate_indices"] == [2]
    assert data["missing_indices"] == [3]
    assert data["invalid_indices"] == []
    assert data["repair_attempted"] is True
    assert data["repair_indices"] == [2, 3]
    assert data["retry_stage"] == "full_batch_retry"


# --- categories 17-19: integration points ----------------------------------

def test_plan_scenes_v2_produces_same_scene_order_regardless_of_llm_array_order(monkeypatch):
    """Category 17: the scene-building loop in plan_scenes_v2 zips chunks
    with classifications positionally - it can only do that safely because
    _classify_chunks_v2 already re-sorted the classifications by
    input_chunk_index (never trusting the model's own array order). This
    proves that guarantee holds end to end: the LLM returning its
    classifications in reverse order must not change which claim ends up
    on which scene."""
    def reversed_order_fake(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        indices = [int(re.match(r"^(\d+)\.", l).group(1)) for l in chunk_lines]
        classifications = [_classification(idx, claim=f"Claim for chunk {idx}") for idx in indices]
        return list(reversed(classifications)), []

    monkeypatch.setattr(sp.llm_client, "chat_json", reversed_order_fake)
    scenes = sp.plan_scenes_v2(_fake_script_en())

    documentary_scenes = [s for s in scenes if not s.get("is_cta")]
    for i, scene in enumerate(documentary_scenes, start=1):
        assert scene["claim"] == f"Claim for chunk {i}"


def test_cardinality_hardening_has_no_scene_id_or_topic_specific_logic():
    """Category 18: this hardening pass must be a fully generic,
    deterministic mechanism - it must never special-case a scene_id, a
    specific topic string, or anything related to the Wow! Signal video
    whose real classification run produced the forensic failure this was
    built to fix."""
    functions = [
        sp._numbered_chunk_prompt, sp._request_classifications,
        sp._analyze_classification_indices, sp._classification_indices_are_clean,
        sp._ordered_classifications_by_index, sp._attempt_targeted_repair,
        sp._splice_repaired_classifications, sp._classify_chunks_v2,
    ]
    banned_terms = ["wow", "signal", "sagittarius", "6eq", "the-wow-signal", "big ear", "ohio state"]
    for fn in functions:
        source = inspect.getsource(fn).lower()
        for term in banned_terms:
            assert term not in source, f"{fn.__name__} contains topic-specific term {term!r}"
        assert "scene_id" not in source, f"{fn.__name__} references scene_id, which it must never hardcode"


def test_v1_plan_scenes_never_touches_the_new_indexed_classification_machinery(monkeypatch):
    """Category 19: v1's plan_scenes() must keep working exactly as
    before - it never requires input_chunk_index, never goes through
    _classify_chunks_v2 or the repair/retry ladder, and must not raise or
    behave differently just because the classification objects it
    receives happen to lack that field entirely."""
    def fake_chat_json_v1(system, user, max_tokens=6000, model=None, use_web_search=False):
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        return [
            {"visual_prompt": f"prompt {i}", "visual_type": "cinematic"}
            for i in range(len(chunk_lines))
        ], []

    monkeypatch.setattr(sp.llm_client, "chat_json", fake_chat_json_v1)
    scenes = sp.plan_scenes(_fake_script_en())
    assert scenes
    for scene in scenes:
        assert "schema_version" not in scene
        assert "input_chunk_index" not in scene

# Category 20 (full existing test suite) is not a unit test - it is the
# full-suite pytest run performed and reported separately, per the explicit
# instruction "Do not stop after unit tests. Run the full suite before
# reporting success."
