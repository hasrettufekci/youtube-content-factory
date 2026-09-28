"""
Scene Planner V2 stabilization - general-purpose, scene-ID-independent
semantic audit matcher (youtube/scene_audit.py).

Fixes a real weakness in the standalone Wow! Signal audit script's original
matcher: it only ever reported "found" (a scene exists at the requested
scene_id) and a weak "any one keyword present" signal, with no way to tell
"this scene is obviously wrong", "this scene is obviously right", or "the
expected content simply moved to a different scene_id" (which every
chunking/cardinality fix in this project legitimately does - chunk
boundaries are content-derived, not fixed).

None of these tests hardcode a "real" scene_id (5/13/14/19/25/29/32, the
Wow! Signal script's specific audit targets) into scene_audit.py's own
logic - that config lives only in the standalone Windows script. These
tests use arbitrary scene_ids to prove the matcher's behavior is a pure
function of its arguments, not of any particular scene_id.
"""
from youtube import scene_audit as sa


def _scene(scene_id, claim="", narration_en="", visual_prompt="", prompt_segments=None, scene_type="real_location"):
    return {
        "scene_id": scene_id,
        "claim": claim,
        "narration_en": narration_en,
        "visual_prompt": visual_prompt,
        "prompt_segments": prompt_segments or {},
        "scene_type": scene_type,
    }


# ---------------------------------------------------------------------------
# match_scene_semantics - single-scene scoring
# ---------------------------------------------------------------------------

def test_match_scene_semantics_correct_content_passes_both_signals():
    scene = _scene(
        13,
        claim="The signal's origin lies in Sagittarius, given by its right ascension and declination coordinates.",
        narration_en="Astronomers pinpointed the source using Sagittarius coordinates.",
    )
    keywords = ["sagittarius", "coordinate", "right ascension", "declination"]

    result = sa.match_scene_semantics(scene, keywords)

    assert result["keyword_sanity_match"] is True
    assert result["semantic_match"] is True
    assert set(result["matched_keywords"]) == {"sagittarius", "coordinate", "right ascension", "declination"}
    assert result["missing_keywords"] == []


def test_match_scene_semantics_wrong_scene_fails_both_signals():
    # Exactly the scenario the stabilization task described: expected
    # Sagittarius/coordinates content, but the scene at this id is really
    # about SNR/antenna pattern - zero keyword overlap.
    scene = _scene(
        13,
        claim="The antenna pattern shows a signal-to-noise ratio consistent with a real detection.",
        narration_en="The SNR curve peaked sharply during the observation window.",
    )
    keywords = ["sagittarius", "coordinate", "right ascension", "declination"]

    result = sa.match_scene_semantics(scene, keywords)

    assert result["keyword_sanity_match"] is False
    assert result["semantic_match"] is False
    assert result["matched_keywords"] == []
    assert result["missing_keywords"] == keywords


def test_match_scene_semantics_single_incidental_keyword_passes_weak_not_strong():
    # A single, incidental word overlap (one keyword out of four) must NOT
    # be treated as real content confirmation - this is exactly the
    # ambiguity the original matcher couldn't express.
    scene = _scene(
        13,
        claim="The team compared several coordinate systems used across radio astronomy.",
        narration_en="Coordinate conventions vary between observatories.",
    )
    keywords = ["sagittarius", "coordinate", "right ascension", "declination"]

    result = sa.match_scene_semantics(scene, keywords)

    assert result["keyword_sanity_match"] is True  # weak signal: not obviously unrelated
    assert result["semantic_match"] is False  # strong signal: not actually confirmed
    assert result["matched_keywords"] == ["coordinate"]


def test_match_scene_semantics_never_checks_visual_prompt_or_prompt_segments():
    # The claim/narration carry no expected content at all, but the visual
    # prompt and prompt_segments happen to mention "Sagittarius" (e.g. via
    # incidental phrasing or leakage elsewhere) - this must NOT count,
    # because the visual prompt is not the underlying documentary claim.
    scene = _scene(
        13,
        claim="The antenna pattern shows a strong signal-to-noise ratio.",
        narration_en="The SNR curve peaked sharply.",
        visual_prompt="A radio telescope dish pointed toward the region of Sagittarius at night.",
        prompt_segments={"setting": "the constellation Sagittarius region of the night sky"},
    )
    keywords = ["sagittarius"]

    result = sa.match_scene_semantics(scene, keywords)

    assert result["keyword_sanity_match"] is False
    assert result["semantic_match"] is False


def test_match_scene_semantics_scales_required_count_for_short_keyword_lists():
    # A 1-keyword target (e.g. a short code) must still be reachable at
    # semantic_match - the default min_matches=2 threshold is scaled down,
    # not a hard floor that short keyword lists can never clear.
    scene = _scene(5, claim="The recorded signal read 6EQUJ5 on the printout.")
    result = sa.match_scene_semantics(scene, ["6equj5"])

    assert result["keyword_sanity_match"] is True
    assert result["semantic_match"] is True


def test_match_scene_semantics_min_matches_is_configurable():
    scene = _scene(13, claim="Sagittarius and coordinate data were both mentioned once.")
    keywords = ["sagittarius", "coordinate", "right ascension", "declination"]

    lenient = sa.match_scene_semantics(scene, keywords, min_matches=1)
    strict = sa.match_scene_semantics(scene, keywords, min_matches=3)

    assert lenient["semantic_match"] is True   # 2 matches >= required 1
    assert strict["semantic_match"] is False   # 2 matches < required 3


# ---------------------------------------------------------------------------
# find_best_semantic_match - scene-ID-independent search
# ---------------------------------------------------------------------------

def test_find_best_semantic_match_locates_content_that_moved_to_a_different_scene_id():
    scenes = [
        _scene(13, claim="An unrelated claim about antenna hardware."),
        _scene(16, claim="The source lies in Sagittarius, given by right ascension and declination coordinates."),
        _scene(17, claim="A single passing mention of coordinate systems."),
    ]
    keywords = ["sagittarius", "coordinate", "right ascension", "declination"]

    best = sa.find_best_semantic_match(scenes, keywords)

    assert best is not None
    assert best["scene_id"] == 16
    assert best["matched_keyword_count"] == 4


def test_find_best_semantic_match_returns_none_when_nothing_matches():
    scenes = [_scene(1, claim="Nothing relevant here."), _scene(2, claim="Nor here.")]
    assert sa.find_best_semantic_match(scenes, ["sagittarius"]) is None


def test_find_best_semantic_match_breaks_ties_by_first_occurrence():
    scenes = [
        _scene(1, claim="Mentions coordinate once."),
        _scene(2, claim="Also mentions coordinate once."),
    ]
    best = sa.find_best_semantic_match(scenes, ["coordinate"])
    assert best["scene_id"] == 1


# ---------------------------------------------------------------------------
# audit_target_scene - the combined report a caller actually uses
# ---------------------------------------------------------------------------

def test_audit_target_scene_reports_found_false_when_scene_id_missing():
    scenes = [_scene(1, claim="Something.")]
    result = sa.audit_target_scene(scenes, requested_scene_id=99, expected_keywords=["sagittarius"])

    assert result["found"] is False
    assert result["keyword_sanity_match"] is False
    assert result["semantic_match"] is False
    assert result["missing_keywords"] == ["sagittarius"]


def test_audit_target_scene_content_moved_false_when_match_is_at_requested_id():
    scenes = [
        _scene(13, claim="The source lies in Sagittarius, right ascension and declination coordinates given."),
    ]
    result = sa.audit_target_scene(scenes, 13, ["sagittarius", "coordinate", "right ascension", "declination"])

    assert result["found"] is True
    assert result["semantic_match"] is True
    assert result["content_moved"] is False
    assert result["best_semantic_match"]["scene_id"] == 13


def test_audit_target_scene_reports_content_moved_when_best_match_is_elsewhere():
    # Exactly the real-world drift scenario: scene_id 13 exists (found=True)
    # but no longer carries the expected content, which has landed at a
    # different scene_id after a chunking change.
    scenes = [
        _scene(13, claim="The antenna pattern shows a strong signal-to-noise ratio."),
        _scene(16, claim="The source lies in Sagittarius, right ascension and declination coordinates given."),
    ]
    keywords = ["sagittarius", "coordinate", "right ascension", "declination"]

    result = sa.audit_target_scene(scenes, 13, keywords, expected_content="Sagittarius / RA-Dec coordinates")

    assert result["found"] is True
    assert result["keyword_sanity_match"] is False
    assert result["semantic_match"] is False
    assert result["content_moved"] is True
    assert result["best_semantic_match"]["scene_id"] == 16
    assert result["expected_content"] == "Sagittarius / RA-Dec coordinates"


def test_audit_target_scene_backward_compatible_field_names():
    # The original ad hoc matcher's report only ever had "found" and
    # "keyword_sanity_match" - both must keep their exact original meaning
    # so any existing report-reading code (or a human comparing old/new
    # reports) isn't broken by this stabilization.
    scenes = [_scene(1, claim="Mentions sagittarius once.")]
    result = sa.audit_target_scene(scenes, 1, ["sagittarius"])

    assert "found" in result and result["found"] is True
    assert "keyword_sanity_match" in result and result["keyword_sanity_match"] is True


def test_audit_target_scene_is_a_pure_function_of_its_arguments_not_scene_id():
    # No hardcoding anywhere in scene_audit.py: relabeling every scene_id in
    # the input (and the requested id to match) must produce an identical
    # report except for the scene_id values themselves.
    keywords = ["sagittarius", "coordinate"]

    def build(offset):
        return [
            _scene(1 + offset, claim="Unrelated antenna claim."),
            _scene(2 + offset, claim="Sagittarius coordinate data."),
        ]

    result_a = sa.audit_target_scene(build(0), requested_scene_id=1, expected_keywords=keywords)
    result_b = sa.audit_target_scene(build(1000), requested_scene_id=1001, expected_keywords=keywords)

    for key in ("found", "keyword_sanity_match", "semantic_match", "content_moved"):
        assert result_a[key] == result_b[key]
    assert result_a["best_semantic_match"]["scene_id"] == 2
    assert result_b["best_semantic_match"]["scene_id"] == 1002
