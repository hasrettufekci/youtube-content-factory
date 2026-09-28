"""
Visual Generator V2 integration - pipeline.py wiring (spec categories A/B):

    A - the "scenes" stage in pipeline.py's _stage_plan() really goes
        through plan_scenes_v2() (V2 classification), not v1's plan_scenes()
    B - build_prompts_v2() is wired in too (not just Phase 1 classification)
        - i.e. going through the pipeline-level stage produces scenes with
        visual_prompt/render_hash/avoid_strategy already set, not just
        schema_version=2 with Phase 2 fields missing.

Deliberately isolates the "scenes" stage's run() callable out of
_stage_plan() rather than running the whole pipeline end-to-end - the full
resume/ordering behavior (confirming run_and_save_v2 is the exact attribute
pipeline.py calls, is skipped when already done, etc.) is already covered by
tests/test_pipeline_resume.py; this file is specifically about WHAT that
stage produces when it does run.
"""
import re

import pytest

from core.state import PipelineState
from storage import project_store as ps
from youtube import scene_planner as sp
import pipeline as pipeline_module


def _fake_script_en(word_count: int = 1400) -> dict:
    sentences = []
    words_left = word_count
    while words_left > 0:
        n = min(12, words_left)
        sentences.append(" ".join(["word"] * n) + ".")
        words_left -= n
    return {"full_narration": " ".join(sentences)}


def _fake_chat_json_v2(system, user, max_tokens=6000, model=None, use_web_search=False):
    chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
    indices = [int(re.match(r"^(\d+)\.", l).group(1)) for l in chunk_lines]
    return [
        {
            "input_chunk_index": idx,
            "claim": f"Claim for chunk {i}",
            "scene_type": "real_location",
            "evidence_level": "documented_fact",
            "data_points": [],
            "visual_relevance_reason": f"Depicts the specific subject named in chunk {i}'s narration.",
            "prompt_segments": {
                "subject": "s", "setting": "s", "lighting": "l",
                "mood": "m", "composition": "c", "style": "s",
            },
        }
        for i, idx in enumerate(indices)
    ], []


def _scenes_stage_run(paths):
    """Pulls the exact ("scenes", is_done, run) triple out of pipeline.py's
    real _stage_plan() and returns its `run` callable - this is what
    pipeline.produce_video()'s stage loop actually invokes for that stage,
    so calling it here exercises the real wiring rather than re-implementing it."""
    stages = pipeline_module._stage_plan(paths, topic="unused", category=None)
    for name, is_done, run in stages:
        if name == "scenes":
            return run
    raise AssertionError('no "scenes" stage found in _stage_plan()')


def test_pipeline_scenes_stage_uses_plan_scenes_v2_not_v1(isolated_storage, monkeypatch):
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2)
    paths, _ = ps.create_project("Pipeline Scenes Stage V2 Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    run = _scenes_stage_run(paths)
    run()

    scenes = ps.load_json(paths.scenes_json)
    assert scenes
    for scene in scenes:
        # schema_version=2/visual_strategy only exist on plan_scenes_v2's
        # output - v1's plan_scenes() never sets either field (see
        # test_v1_plan_scenes_and_v1_run_and_save_unaffected_by_phase_2 in
        # test_scene_planner_v2.py).
        assert scene["schema_version"] == 2
        assert scene["visual_strategy"] == "ai_generated"
    assert ps.load_state(paths) == PipelineState.SCENES_PLANNED


def test_pipeline_scenes_stage_also_runs_build_prompts_v2_phase_2(isolated_storage, monkeypatch):
    """Guards against a wiring mistake where only Phase 1 (classification)
    is wired into the pipeline stage and Phase 2 (build_prompts_v2, which
    sets visual_prompt/render_hash/avoid_strategy) is skipped - that would
    leave every ai_generated scene with no visual_prompt at all, which
    would then crash image_gen.py's strategy dispatch."""
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2)
    paths, _ = ps.create_project("Pipeline Scenes Stage Phase 2 Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    run = _scenes_stage_run(paths)
    run()

    scenes = ps.load_json(paths.scenes_json)
    for scene in scenes:
        assert scene.get("visual_prompt")  # Phase 2 field, non-empty for ai_generated
        assert "render_hash" in scene
        assert "avoid_strategy" in scene
        assert "overlay_required" in scene


def test_pipeline_scenes_stage_writes_through_existing_storage_writer(isolated_storage, monkeypatch):
    """No parallel storage system - the stage must write to the exact same
    paths.scenes_json path v1 uses, via the same ps.save_json/ps.save_state
    primitives, so every downstream reader (image_gen.py, assemble.py,
    pipeline.py's own _images_done()/_scenes_synced_for()) needs zero
    changes regardless of which planner version produced the file."""
    monkeypatch.setattr(sp.llm_client, "chat_json", _fake_chat_json_v2)
    paths, _ = ps.create_project("Pipeline Scenes Stage Storage Topic")
    ps.save_json(paths.script_json, _fake_script_en())

    run = _scenes_stage_run(paths)
    run()

    assert paths.scenes_json.exists()
    assert pipeline_module._images_done(paths) is False  # no images yet, but the check itself works unmodified
