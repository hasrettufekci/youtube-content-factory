"""
Resume/idempotency (spec Round 2 section 12; extended for multilingual
production in Round 3 section 11/21):

    - partial-pipeline resume: research done / script done / scenes done /
      images done / tts NOT done -> only tts onward reruns, nothing
      upstream is recomputed
    - duplicate-topic behavior at the pipeline level: producing the same
      topic twice reuses the same video_id and re-does none of the
      already-completed stages (not just at the storage layer - see
      test_storage.py's test_create_project_is_idempotent_for_same_topic
      for that - but end to end through pipeline.produce_video()).
    - multilingual resume: a missing localization/narration/metadata for
      ONE language never re-triggers research, the English master script,
      scene planning, or (most importantly) visual generation - see
      test_multilingual_pipeline.py for the dedicated per-language cases.
"""
import pytest

import pipeline as pipeline_module
from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import researcher, script_generator, localizer, scene_planner, image_gen, tts, assemble, thumbnail, metadata, quality_check


def _never_call(name):
    def _fail(*a, **k):
        raise AssertionError(f"{name}.run_and_save must NOT be called - this stage's artifact already exists on disk")
    return _fail


def _record_and_write(monkeypatch, calls, module, name, path_attr, content=b"x", is_json=False, attr="run_and_save"):
    # `attr` lets a caller target a different module attribute than the
    # default "run_and_save" - needed for scene_planner, whose pipeline.py
    # entry point is now the V2-specific run_and_save_v2 (Visual Generator
    # V2 integration; v1's run_and_save is preserved but no longer what
    # pipeline.py's "scenes" stage calls).
    def _fake(paths, *a, **k):
        calls.append(name)
        path = getattr(paths, path_attr)
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_json:
            ps.save_json(path, {"ok": True})
        else:
            path.write_bytes(content)
        return {}
    monkeypatch.setattr(module, attr, _fake)


def _record_and_write_by_language(monkeypatch, calls, module, prefix, path_method, content=b"x", is_json=False):
    """Like _record_and_write, but for a stage whose run_and_save() is
    called once per language (tts.run_and_save, metadata.run_and_save,
    localizer.run_and_save) - a single monkeypatch on the module handles
    every language's call, dispatching by the `language` kwarg pipeline.py
    passes, since all of them share the same module object."""
    def _fake(paths, language="en", **kwargs):
        calls.append(f"{prefix}_{language}")
        path = getattr(paths, path_method)(language)
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_json:
            ps.save_json(path, {"ok": True, "language": language})
        else:
            path.write_bytes(content)
        return {}
    monkeypatch.setattr(module, "run_and_save", _fake)


def _passing_quality_report():
    return quality_check.QualityReport(passed=True, checks={"ok": {"passed": True, "detail": ""}})


def test_partial_pipeline_resume_reruns_only_missing_stage_onward(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    topic = "Partial Resume Topic"
    paths, _ = ps.create_project(topic)

    # research / script_en / localization_tr / scenes / images already
    # done (on disk).
    ps.save_json(paths.research_json, {"topic": topic, "sources": []})
    ps.save_json(paths.script_json, {"full_narration": "text", "sections": []})
    ps.save_json(paths.script_tr_json, {"full_narration": "metin", "sections": []})
    scenes = [{"scene_id": 1, "visual_prompt": "p"}]
    ps.save_json(paths.scenes_json, scenes)
    paths.scene_image(1).write_bytes(b"fake png")
    # narration_en.mp3 (and everything after it) NOT done yet.

    calls = []
    monkeypatch.setattr(researcher, "run_and_save", _never_call("researcher"))
    monkeypatch.setattr(script_generator, "run_and_save", _never_call("script_generator"))
    monkeypatch.setattr(localizer, "run_and_save", _never_call("localizer"))
    monkeypatch.setattr(scene_planner, "run_and_save_v2", _never_call("scene_planner"))
    monkeypatch.setattr(image_gen, "run_and_save", _never_call("image_gen"))
    _record_and_write_by_language(monkeypatch, calls, tts, "narration", "narration_mp3_for")
    _record_and_write(monkeypatch, calls, assemble, "assemble", "video_mp4")
    _record_and_write(monkeypatch, calls, thumbnail, "thumbnail", "thumbnail_png")
    _record_and_write_by_language(monkeypatch, calls, metadata, "metadata", "metadata_json_for", is_json=True)
    monkeypatch.setattr(pipeline_module.quality_check, "run_and_enforce", lambda paths, languages=None: _passing_quality_report())

    result_paths = pipeline_module.produce_video(topic, video_id=paths.video_id)

    # Only the stages that were actually missing ran, in the right order -
    # everything upstream of the missing stage (including the already-done
    # Turkish localization) was left completely alone.
    assert calls == ["narration_en", "narration_tr", "assemble", "thumbnail", "metadata_en", "metadata_tr"]
    assert ps.load_state(result_paths) == PipelineState.PENDING_APPROVAL


def test_producing_the_same_topic_twice_reuses_video_id_and_reruns_nothing(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    topic = "Duplicate Topic End To End"

    calls = []
    for module, name, path_attr, is_json, attr in [
        (researcher, "researcher", "research_json", True, "run_and_save"),
        (script_generator, "script_generator", "script_json", True, "run_and_save"),
        # scene_planner: pipeline.py's "scenes" stage now calls run_and_save_v2
        # (Visual Generator V2 integration), not v1's run_and_save.
        (scene_planner, "scene_planner", "scenes_json", True, "run_and_save_v2"),
        (assemble, "assemble", "video_mp4", False, "run_and_save"),
        (thumbnail, "thumbnail", "thumbnail_png", False, "run_and_save"),
    ]:
        _record_and_write(monkeypatch, calls, module, name, path_attr, is_json=is_json, attr=attr)
    _record_and_write_by_language(monkeypatch, calls, localizer, "localization", "script_localized_json", is_json=True)
    _record_and_write_by_language(monkeypatch, calls, tts, "narration", "narration_mp3_for")
    _record_and_write_by_language(monkeypatch, calls, metadata, "metadata", "metadata_json_for", is_json=True)

    # image_gen.run_and_save's signature/behavior differs (writes one file
    # per scene, keyed off scenes.json) - stub it directly rather than
    # through the generic helper above.
    def _fake_images(paths):
        calls.append("image_gen")
        scenes = ps.load_json(paths.scenes_json) or [{"scene_id": 1}]
        paths.scene_image(1).write_bytes(b"fake png")
        return [paths.scene_image(1)]

    monkeypatch.setattr(image_gen, "run_and_save", _fake_images)
    monkeypatch.setattr(pipeline_module.quality_check, "run_and_enforce", lambda paths, languages=None: _passing_quality_report())

    first_paths = pipeline_module.produce_video(topic)
    first_call_counts = list(calls)
    assert first_call_counts == [
        "researcher", "script_generator", "localization_tr", "scene_planner", "image_gen",
        "narration_en", "narration_tr", "assemble", "thumbnail", "metadata_en", "metadata_tr",
    ]

    # Second production of the exact same topic (no force_new): must reuse
    # the same video_id/project, and since every stage's artifact already
    # exists on disk, none of the stage functions should run again.
    second_paths = pipeline_module.produce_video(topic)

    assert second_paths.video_id == first_paths.video_id
    assert second_paths.root == first_paths.root
    assert calls == first_call_counts  # no new calls appended
    assert ps.load_state(second_paths) == PipelineState.PENDING_APPROVAL


def test_force_new_on_a_previously_produced_topic_creates_a_fresh_video_and_reruns_everything(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    topic = "Force New Rerun Topic"

    calls = []
    for module, name, path_attr, is_json, attr in [
        (researcher, "researcher", "research_json", True, "run_and_save"),
        (script_generator, "script_generator", "script_json", True, "run_and_save"),
        # scene_planner: pipeline.py's "scenes" stage now calls run_and_save_v2
        # (Visual Generator V2 integration), not v1's run_and_save.
        (scene_planner, "scene_planner", "scenes_json", True, "run_and_save_v2"),
        (assemble, "assemble", "video_mp4", False, "run_and_save"),
        (thumbnail, "thumbnail", "thumbnail_png", False, "run_and_save"),
    ]:
        _record_and_write(monkeypatch, calls, module, name, path_attr, is_json=is_json, attr=attr)
    _record_and_write_by_language(monkeypatch, calls, localizer, "localization", "script_localized_json", is_json=True)
    _record_and_write_by_language(monkeypatch, calls, tts, "narration", "narration_mp3_for")
    _record_and_write_by_language(monkeypatch, calls, metadata, "metadata", "metadata_json_for", is_json=True)

    def _fake_images(paths):
        calls.append("image_gen")
        paths.scene_image(1).write_bytes(b"fake png")
        return [paths.scene_image(1)]

    monkeypatch.setattr(image_gen, "run_and_save", _fake_images)
    monkeypatch.setattr(pipeline_module.quality_check, "run_and_enforce", lambda paths, languages=None: _passing_quality_report())

    first_paths = pipeline_module.produce_video(topic)
    assert len(calls) == 11  # research, script_en, localization_tr, scenes, images, narration x2, assemble, thumbnail, metadata x2

    second_paths = pipeline_module.produce_video(topic, force_new=True)

    assert second_paths.video_id != first_paths.video_id
    assert len(calls) == 22  # every stage ran again for the new video
