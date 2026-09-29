"""
youtube/shorts_engine.py tests - end-to-end Shorts production orchestration.

Covers spec scenarios:
    1  Long -> 3 Shorts planning (end to end, via produce_shorts_for_video)
   19  Short QC (each Short gets its own report, persisted to storage/shorts)
   20  one Short QC failure blocks package approval (all_shorts_ready())
plus resilience (one Short's production failure never stops the others)
and resumability (an already-READY/UPLOADED Short is never reproduced).

Providers are mocked (IMAGE_PROVIDER=mock, TTS_PROVIDER=mock) and
llm_client.chat_json is monkeypatched - no real network call is ever made
by this file, matching this codebase's standard testing convention.
"""
from __future__ import annotations

import json

import pytest
from PIL import Image

from core.config import settings
from storage import project_store as ps
from storage import shorts as shorts_store
from youtube import image_gen, shorts_engine, shorts_planner

# See the DURATION HARD-GATE PATCH note in _script() below: 90 words at the
# mock TTS's own MOCK_WORDS_PER_SECOND=2.5 pace (youtube/tts.py) renders to
# a real ~36s video - safely inside the default 30-50s hard-gate band with
# margin on both sides, without needing a full 122-word production script.
SHORTS_PLANNER_TEST_WORD_CAP = 90


def _script():
    # DURATION HARD-GATE PATCH note: hook/reveal/mystery (the three
    # sections shorts_planner.py actually draws from - see ANGLE_PLAN) are
    # deliberately full-length (~110-125 words) documentary paragraphs, not
    # one-liners, so that shorts_env's fake_chat_json condensation (capped
    # at SHORTS_PLANNER_TEST_WORD_CAP words below) produces narration long
    # enough for the mock TTS to render a REAL duration inside the default
    # 30-50s hard-gate band - matching realistic production section length
    # rather than an artificially short test fixture that the new gate
    # would otherwise (correctly) reject. Other sections are unused by the
    # planner and stay short.
    sections = [
        ("hook", "A strange signal was heard one night by a lonely radio telescope stationed deep in the "
                 "cornfields, arriving without warning on a frequency nobody on duty was expecting to monitor "
                 "that evening. The operator very nearly missed it completely, distracted by a coffee break and "
                 "a stack of unfinished paperwork, until the automated printer suddenly began chattering out a "
                 "long string of numbers that made no obvious sense at first glance. Within minutes the entire "
                 "observatory staff had gathered around the console, comparing notes and checking every known "
                 "source of possible interference, from passing satellites to nearby radio stations and even "
                 "stray equipment on site, and finding absolutely nothing that could explain what they were "
                 "looking at that night."),
        ("introduction", "The observatory had been quietly listening to the sky for years before this happened."),
        ("story_development", "Investigators pored over the printout for weeks trying to make sense of it."),
        ("evidence", "Nothing in the archives matched the pattern researchers found that night."),
        ("mystery", "To this day nobody has ever truly explained where the signal actually came from, and that "
                    "stubborn silence is exactly what keeps researchers coming back to the case every few years "
                    "with fresh equipment and fresh theories. Some point to natural cosmic phenomena that happen "
                    "to briefly mimic an artificial transmission, while others quietly wonder about something "
                    "stranger that current instruments simply were not built to detect or measure properly. The "
                    "observatory itself still receives letters and emails from curious strangers asking for "
                    "updates, even though there has been nothing genuinely new to report since that single night "
                    "decades ago. Whatever it truly was, it left behind only a printout, a story, and a mystery "
                    "that refuses to close."),
        ("reveal", "The most surprising detail to emerge afterward was that the signal never repeated again "
                   "anywhere, despite months of careful monitoring by observatories across several countries and "
                   "continents, all straining their equipment toward that exact patch of sky at that exact time "
                   "of night. Engineers eventually ruled out every conventional explanation they could think of, "
                   "from military hardware and commercial satellites to a stray reflection bouncing off orbital "
                   "debris, none of which matched the signal's unusual shape or its remarkably narrow bandwidth. "
                   "What remained, once every ordinary possibility had been carefully tested and discarded one "
                   "by one, was a single unexplained reading that still sits unresolved in the archive decades "
                   "later, printed out on a strip of paper nobody has ever managed to fully explain."),
        ("conclusion", "The mystery endures as one of astronomy's greatest unsolved puzzles."),
        ("cta", "Subscribe for the full story on AI FOUND THIS."),
    ]
    script = {
        "title": "The Signal Nobody Can Explain",
        "sections": [{"type": t, "title": t.title(), "narration": n} for t, n in sections],
    }
    script["full_narration"] = "\n\n".join(n for _, n in sections)
    return script


def _scenes(script):
    doc_words = " ".join(s["narration"] for s in script["sections"] if s["type"] != "cta").split()
    n_scenes = 6
    chunk = len(doc_words) // n_scenes + 1
    types = ["cinematic", "wide", "closeup", "map", "archival", "portrait"]
    scenes = []
    for i in range(n_scenes):
        words = doc_words[i * chunk:(i + 1) * chunk] or doc_words[-1:]
        scenes.append({
            "scene_id": i + 1, "schema_version": 2, "duration_hint": 4.0,
            "narration_en": " ".join(words), "visual_type": types[i % len(types)],
            "visual_prompt": f"documentary shot {i + 1}",
        })
    return scenes


@pytest.fixture
def shorts_env(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")
    # Test-speed only: a real 30-50s vertical video is now genuinely
    # rendered by several tests in this file (see the DURATION HARD-GATE
    # PATCH note in _script() above) to exercise the real ffprobe-measured
    # duration gate end to end. A quarter-size frame keeps real ffmpeg
    # encoding fast without changing anything about the duration/QC logic
    # under test - width/height are read from settings at call time by
    # shorts_engine.py, shorts_qc.py and shorts_visual.py alike, so this is
    # fully consistent everywhere it's checked (e.g. resolution_1080x1920).
    monkeypatch.setattr(settings, "SHORTS_WIDTH", 270)
    monkeypatch.setattr(settings, "SHORTS_HEIGHT", 480)

    def fake_chat_json(system, user, max_tokens=2000, model=None, use_web_search=False):
        source = user.split("SOURCE TEXT (the only material you may draw facts from):")[-1].strip()
        return {"title": "Fake Short Title", "narration": " ".join(source.split()[:SHORTS_PLANNER_TEST_WORD_CAP])}, []

    monkeypatch.setattr(shorts_planner.llm_client, "chat_json", fake_chat_json)

    script = _script()
    paths, _ = ps.create_project("End To End Shorts Topic")
    ps.save_json(paths.script_en_json, script)
    scenes = _scenes(script)
    ps.save_json(paths.scenes_json, scenes)

    provider = image_gen.get_provider()
    for s in scenes:
        img_bytes = provider.generate(s["visual_prompt"], settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT)
        paths.scene_image(s["scene_id"]).write_bytes(img_bytes)

    return paths, script, scenes


# ---------------------------------------------------------------------------
# 1 - Long -> 3 Shorts, end to end
# ---------------------------------------------------------------------------

def test_produce_shorts_for_video_produces_three_ready_shorts(shorts_env):
    paths, script, scenes = shorts_env
    results = shorts_engine.produce_shorts_for_video(paths, script, scenes)
    assert [r["short_number"] for r in results] == [1, 2, 3]
    assert all(r["status"] == shorts_store.STATUS_READY for r in results)
    assert shorts_engine.all_shorts_ready(results) is True


def test_produce_shorts_for_video_persists_rows_linked_to_parent(shorts_env):
    paths, script, scenes = shorts_env
    shorts_engine.produce_shorts_for_video(paths, script, scenes)
    rows = shorts_store.list_shorts_for_parent(paths.video_id)
    assert len(rows) == 3
    for row in rows:
        assert row["parent_video_id"] == paths.video_id
        assert row["video_path"] and __import__("pathlib").Path(row["video_path"]).exists()
        assert row["thumbnail_path"] and __import__("pathlib").Path(row["thumbnail_path"]).exists()


def test_produce_shorts_for_video_respects_shorts_per_long_override(shorts_env):
    paths, script, scenes = shorts_env
    results = shorts_engine.produce_shorts_for_video(paths, script, scenes, shorts_per_long=1)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 19/20 - Short QC + one failure blocks the package
# ---------------------------------------------------------------------------

def test_all_shorts_ready_false_if_any_short_failed_qc(shorts_env, monkeypatch):
    from youtube import shorts_qc

    paths, script, scenes = shorts_env

    real_run = shorts_qc.run_short_quality_check
    call_count = {"n": 0}

    def failing_on_second(video_path, thumbnail_path, narration, title, source_text, width=None, height=None):
        call_count["n"] += 1
        report = real_run(video_path, thumbnail_path, narration, title, source_text, width=width, height=height)
        if call_count["n"] == 2:
            report.passed = False
            report.errors.append("simulated_failure: forced for test")
        return report

    monkeypatch.setattr(shorts_engine.shorts_qc, "run_short_quality_check", failing_on_second)

    results = shorts_engine.produce_shorts_for_video(paths, script, scenes)
    assert shorts_engine.all_shorts_ready(results) is False
    statuses = {r["short_number"]: r["status"] for r in results}
    assert statuses[2] == shorts_store.STATUS_FAILED
    assert statuses[1] == shorts_store.STATUS_READY
    assert statuses[3] == shorts_store.STATUS_READY  # QC failure on #2 does not affect #1/#3


def test_all_shorts_ready_false_for_empty_results():
    assert shorts_engine.all_shorts_ready([]) is False


# ---------------------------------------------------------------------------
# resilience: one Short's production exception never stops the others
# ---------------------------------------------------------------------------

def test_one_short_production_exception_does_not_stop_the_others(shorts_env, monkeypatch):
    from youtube import shorts_visual

    paths, script, scenes = shorts_env

    real_build = shorts_visual.build_vertical_visual
    call_count = {"n": 0}

    def flaky_build(paths_, scene, dest, width=None, height=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transient FAL failure")
        return real_build(paths_, scene, dest, width=width, height=height)

    monkeypatch.setattr(shorts_engine.shorts_visual, "build_vertical_visual", flaky_build)

    results = shorts_engine.produce_shorts_for_video(paths, script, scenes)
    statuses = [r["status"] for r in results]
    assert shorts_store.STATUS_FAILED in statuses  # the one that failed
    assert statuses.count(shorts_store.STATUS_READY) == 2  # the other two still succeeded
    assert len(results) == 3  # never crashed/short-circuited the whole batch


# ---------------------------------------------------------------------------
# resumability: an already-ready/uploaded Short is never reproduced
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DURATION HARD-GATE PATCH - full production-path integration: a too-long
# LLM narration (never truncated by the planner) flows all the way through
# mock TTS -> real ffmpeg assembly -> real ffprobe duration measurement ->
# the new shorts_qc.py hard gate, reproducing the real 55.2s/30-50s
# production incident end to end without touching any real artifact or
# calling a real paid API.
# ---------------------------------------------------------------------------

def test_short_that_narrates_longer_than_former_target_is_still_ready(shorts_env, monkeypatch):
    paths, script, scenes = shorts_env

    # 140 words produces roughly 56 seconds with MockTTS.
    # Duration is no longer a Shorts QC hard gate.
    long_narration = " ".join(f"word{i}" for i in range(140)) + "."

    def fake_chat_json(system, user, max_tokens=2000, model=None, use_web_search=False):
        return {"title": "An Overly Long Short", "narration": long_narration}, []

    monkeypatch.setattr(shorts_planner.llm_client, "chat_json", fake_chat_json)

    results = shorts_engine.produce_shorts_for_video(
        paths,
        script,
        scenes,
        shorts_per_long=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result["status"] == shorts_store.STATUS_READY
    assert result["qc_passed"] is True
    assert not any(
        "duration_within_target_range" in error
        for error in result.get("qc_errors", [])
    )

    row = shorts_store.get_short(result["short_id"])

    assert row["duration_seconds"] is not None
    assert row["duration_seconds"] > settings.SHORTS_TARGET_SECONDS_MAX

def test_already_ready_short_is_not_reproduced(shorts_env, monkeypatch):
    from youtube import shorts_visual

    paths, script, scenes = shorts_env
    shorts_engine.produce_shorts_for_video(paths, script, scenes)

    call_count = {"n": 0}
    real_build = shorts_visual.build_vertical_visual

    def spy(paths_, scene, dest, width=None, height=None):
        call_count["n"] += 1
        return real_build(paths_, scene, dest, width=width, height=height)

    monkeypatch.setattr(shorts_engine.shorts_visual, "build_vertical_visual", spy)

    results = shorts_engine.produce_shorts_for_video(paths, script, scenes)
    assert all(r["status"] == shorts_store.STATUS_READY for r in results)
    assert call_count["n"] == 0  # nothing was rebuilt - all three were already READY
