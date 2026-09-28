"""
youtube/shorts_demo.py tests - the safe local/demo CLI.

Plan-only mode makes no LLM/FAL/TTS call at all (pure, free, read-only);
--produce mode uses mock providers only - no real network in this file.
"""
from __future__ import annotations

from core.config import settings
from storage import project_store as ps
from youtube import llm_client, shorts_demo


def _script():
    sections = [
        ("hook", "A strange signal was heard one night by a lonely radio telescope."),
        ("introduction", "The observatory kept careful records of every reading it took."),
        ("story_development", "Investigators pored over the printout for weeks on end."),
        ("evidence", "Nothing in the archives matched the pattern they eventually found."),
        ("mystery", "To this day nobody has explained where the signal came from."),
        ("reveal", "The most surprising detail was that it never repeated again."),
        ("conclusion", "The mystery endures as one of astronomy's greatest puzzles."),
        ("cta", "Subscribe for the full story."),
    ]
    return {"title": "The Signal", "sections": [{"type": t, "title": t, "narration": n} for t, n in sections]}


def test_plan_only_makes_no_llm_call_and_returns_three_plans(isolated_storage, monkeypatch):
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call allowed")))
    paths, _ = ps.create_project("Demo CLI Topic")
    ps.save_json(paths.script_en_json, _script())

    plans = shorts_demo.plan_only(paths.video_id)
    assert [p["short_number"] for p in plans] == [1, 2, 3]


def test_plan_only_raises_for_unknown_video_id(isolated_storage):
    import pytest
    with pytest.raises(RuntimeError):
        shorts_demo.plan_only(99999)


def test_main_plan_only_prints_angles_and_exits_zero(isolated_storage, monkeypatch, capsys):
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call allowed")))
    paths, _ = ps.create_project("Demo CLI Print Topic")
    ps.save_json(paths.script_en_json, _script())

    exit_code = shorts_demo.main(["--video-id", str(paths.video_id)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "short_number=1" in captured.out
    assert "hook" in captured.out


def test_main_handles_missing_video_id_cleanly(isolated_storage, capsys):
    exit_code = shorts_demo.main(["--video-id", "123456"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.err


def test_produce_uses_mock_providers_end_to_end(isolated_storage, monkeypatch):
    from youtube import image_gen, shorts_planner

    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")
    monkeypatch.setattr(shorts_planner.llm_client, "chat_json",
                         lambda system, user, **k: ({"title": "T", "narration": "Condensed narration text."}, []))

    script = _script()
    paths, _ = ps.create_project("Demo CLI Produce Topic")
    ps.save_json(paths.script_en_json, script)
    doc_words = " ".join(s["narration"] for s in script["sections"] if s["type"] != "cta").split()
    scenes = [{"scene_id": 1, "schema_version": 2, "duration_hint": 4.0,
               "narration_en": " ".join(doc_words), "visual_type": "cinematic", "visual_prompt": "a shot"}]
    ps.save_json(paths.scenes_json, scenes)
    provider = image_gen.get_provider()
    paths.scene_image(1).write_bytes(provider.generate("a shot", settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT))

    results = shorts_demo.produce(paths.video_id)
    assert len(results) == 3
