"""
Scene Planner V2 - stabilization: general-purpose, scene-ID-independent
semantic audit matcher.

Background: the real Wow! Signal audit uses a small standalone script
(audit_scene_planner_v2_7scenes.py, not part of this repo/pipeline) that
checks whether specific expected documentary content ("Sagittarius / RA-Dec
coordinates", "2025 reanalysis", ...) actually landed at the scene_id it was
originally observed at. That script's ORIGINAL matcher had a real weakness:
it only ever reported two flat facts -

    found = True   (a scene exists at the requested scene_id at all)
    keyword_sanity_match = <any ONE expected keyword found in that scene>

- with no distinction between "this scene is obviously correct", "this
  scene is obviously wrong", and "one incidental word happens to overlap but
  this probably isn't the right scene at all". A single keyword hit is weak
  evidence on its own (a generic word can appear anywhere), and neither flag
  says anything about content that has simply MOVED to a different scene_id
  - which every chunking/cardinality fix in this project has done, by
  design (chunk boundaries are content-derived, not fixed).

This module fixes that generally, not for any one script or scene_id:

  - match_scene_semantics(): scores ONE scene against a list of expected
    keywords, split into a weak "keyword_sanity_match" (>=1 keyword) and a
    stronger "semantic_match" (a configurable minimum number of DISTINCT
    keywords, scaled down for short keyword lists) - looking ONLY at the
    scene's own claim/narration_en, never its visual_prompt/prompt_segments
    (a word landing in the AI IMAGE PROMPT is not evidence the underlying
    documentary CLAIM is correct - it could be incidental phrasing or
    leakage from elsewhere).
  - find_best_semantic_match(): scene-ID-independent search across EVERY
    produced scene for whichever one best matches the expected keywords -
    this is what lets an audit report "expected at scene 13, but this
    content is now actually at scene 16" instead of just silently
    reporting a mismatch with no further information.
  - audit_target_scene(): combines both into the single report shape a
    caller (the Windows audit script, or a test) wants, taking scene_id and
    expected_keywords as plain arguments - nothing here special-cases any
    particular scene_id, topic, or script.

Deliberately not wired into plan_scenes_v2()/validate_scenes_v2() - this is
an AUDIT/QA helper (comparing produced output against externally-supplied
expectations), not a production pipeline gate.
"""
from __future__ import annotations

# Default minimum number of DISTINCT expected keywords that must be present
# for the STRONG "semantic_match" signal - deliberately more than 1 (unlike
# the weak "keyword_sanity_match", which only needs one) so that a single
# incidental word overlap can never look like a real content match on its
# own. Scaled down automatically (see _required_match_count) for a target
# that only has 1-2 expected keywords to begin with.
DEFAULT_MIN_SEMANTIC_MATCHES = 2


def _haystack(scene: dict) -> str:
    """The authoritative text a scene's expected DOCUMENTARY content should
    be checked against: its claim and its own narration - deliberately
    NEVER visual_prompt or prompt_segments (those are the AI IMAGE PROMPT,
    not the underlying claim, and can contain incidental or leaked wording
    that has nothing to do with whether this is really the right scene)."""
    return f"{scene.get('claim', '') or ''} {scene.get('narration_en', '') or ''}".lower()


def _required_match_count(expected_keywords: list[str], min_matches: int) -> int:
    """Never requires more distinct keyword hits than the target actually
    has keywords for (a 1-keyword target, e.g. "6equj5", can still reach
    semantic_match on that single keyword) - but never less than 1 either."""
    return max(1, min(min_matches, len(expected_keywords)))


def match_scene_semantics(
    scene: dict,
    expected_keywords: list[str],
    min_matches: int = DEFAULT_MIN_SEMANTIC_MATCHES,
) -> dict:
    """Scores ONE scene against expected_keywords. Returns:

        matched_keywords / missing_keywords - for a human-readable diff.
        keyword_sanity_match - True if AT LEAST ONE expected keyword is
            present (weak signal: "not obviously unrelated").
        semantic_match - True only if at least _required_match_count()
            DISTINCT expected keywords are present (strong signal: "this
            scene actually carries the expected claim", not just one
            incidental word in common).

    General-purpose: takes a scene dict and a keyword list, nothing else -
    no scene_id, no script-specific logic anywhere in this function.
    """
    haystack = _haystack(scene)
    matched = [kw for kw in expected_keywords if kw.lower() in haystack]
    missing = [kw for kw in expected_keywords if kw not in matched]
    required = _required_match_count(expected_keywords, min_matches)
    return {
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_sanity_match": len(matched) > 0,
        "semantic_match": len(matched) >= required,
    }


def find_best_semantic_match(scenes: list[dict], expected_keywords: list[str]) -> dict | None:
    """Scene-ID-independent search: scores EVERY produced scene (using the
    same claim+narration_en haystack as match_scene_semantics) and returns
    a summary of whichever one has the most distinct keyword hits. Returns
    None if not a single scene matches even one keyword.

    This is the piece that makes the audit robust to chunk-boundary drift -
    a normal, expected side effect of any chunking/cardinality change (see
    _split_into_chunks_v2 in scene_planner.py) - by finding WHERE the
    expected content actually ended up, instead of only ever looking at the
    one scene_id it used to be at."""
    best_scene: dict | None = None
    best_count = 0
    for scene in scenes:
        haystack = _haystack(scene)
        matched = [kw for kw in expected_keywords if kw.lower() in haystack]
        if len(matched) > best_count:
            best_count = len(matched)
            best_scene = scene

    if best_scene is None or best_count == 0:
        return None
    return {
        "scene_id": best_scene.get("scene_id"),
        "matched_keyword_count": best_count,
        "matched_keywords": [kw for kw in expected_keywords if kw.lower() in _haystack(best_scene)],
        "claim": best_scene.get("claim"),
        "scene_type": best_scene.get("scene_type"),
    }


def audit_target_scene(
    scenes: list[dict],
    requested_scene_id: int,
    expected_keywords: list[str],
    expected_content: str = "",
    min_matches: int = DEFAULT_MIN_SEMANTIC_MATCHES,
) -> dict:
    """The full, general-purpose audit report for ONE expected piece of
    content against ONE requested scene_id, combining match_scene_semantics
    (checked at that scene_id) with find_best_semantic_match (checked
    across every scene, so drift is always visible). Takes scene_id and
    expected_keywords as plain arguments - never hardcodes either.

    Backward compatible with the original ad hoc matcher's report shape:
    "found" and "keyword_sanity_match" keep their original meaning exactly
    (found = a scene exists at requested_scene_id at all; keyword_sanity_match
    = at least one expected keyword present in THAT scene). "semantic_match",
    "matched_keywords", "missing_keywords", and "best_semantic_match" are
    new fields, additive only.
    """
    by_id = {s.get("scene_id"): s for s in scenes}
    scene = by_id.get(requested_scene_id)

    entry: dict = {
        "requested_scene_id": requested_scene_id,
        "expected_content": expected_content,
        "found": scene is not None,
    }

    if scene is not None:
        entry.update(match_scene_semantics(scene, expected_keywords, min_matches=min_matches))
    else:
        entry.update({
            "keyword_sanity_match": False,
            "semantic_match": False,
            "matched_keywords": [],
            "missing_keywords": list(expected_keywords),
        })

    # Always computed, whether or not the requested scene_id itself matched -
    # this is what makes the report actionable when content has moved.
    best = find_best_semantic_match(scenes, expected_keywords)
    entry["best_semantic_match"] = best
    if best is not None and best.get("scene_id") != requested_scene_id:
        entry["content_moved"] = True
    else:
        entry["content_moved"] = False

    return entry
