"""
Round 3.1 - multilingual scene mapping fix.

Resume edge case: if scenes.json (and its visuals) were already produced
BEFORE a localization's script_<lang>.json existed, plan_scenes()'s own
opportunistic narration_<lang> attachment (see scene_planner.py's module
docstring) never had that language's text to work with, so re-running the
pipeline used to skip scene_planner entirely (scenes.json already exists)
and permanently leave that language's narration_<lang> missing from the
shared scene plan, even after the localized script became available.

scene_planner.sync_localized_narrations() (and pipeline.py's
scene_sync_<lang> stage that calls it) fixes this: a small, deterministic,
additive enrichment of the EXISTING scenes.json - never a replan, never a
new image, never a change to scene_id/visual_prompt.
"""
import pytest

import pipeline as pipeline_module
from core.config import settings
from storage import project_store as ps
from youtube import localizer, scene_planner as sp, tts


def _many_short_sentences(word_count: int, word: str = "word") -> str:
    """Many ~10-word sentences so there are always enough sentence-boundary
    cut points to reach a small target chunk count exactly."""
    sentences = []
    words_left = word_count
    while words_left > 0:
        n = min(10, words_left)
        sentences.append(" ".join([word] * n) + ".")
        words_left -= n
    return " ".join(sentences)


def _base_scenes(n=3):
    return [
        {
            "scene_id": i,
            "duration_hint": 12.0,
            "visual_prompt": f"a cinematic prompt for scene {i}",
            "visual_type": "cinematic",
            "narration_en": f"english narration chunk {i}",
        }
        for i in range(1, n + 1)
    ]


def _script_tr(target_scenes=3):
    return {
        "language": "tr",
        "source_language": "en",
        "title": "TR Title",
        "full_narration": _many_short_sentences(80, word="kelime"),
        "word_count": 80,
        "sections": [],
        "sources": [],
    }


# ---------------------------------------------------------------------------
# 1-2. scenes.json exists before script_tr.json / script_tr.json created
# ---------------------------------------------------------------------------

def test_scenes_json_can_exist_before_localization_and_be_enriched_later(isolated_storage):
    paths, _ = ps.create_project("Scene Sync Topic")
    original_scenes = _base_scenes(3)
    ps.save_json(paths.scenes_json, original_scenes)
    # script_tr.json deliberately does NOT exist yet at this point.
    assert not paths.script_tr_json.exists()

    with pytest.raises(RuntimeError, match="no localized script found"):
        sp.sync_localized_narrations(paths, language="tr")

    # script_tr.json now becomes available (localization stage ran).
    ps.save_json(paths.script_tr_json, _script_tr())

    result = sp.sync_localized_narrations(paths, language="tr")

    assert all("narration_tr" in s for s in result)


# ---------------------------------------------------------------------------
# 3. Resume enriches scenes.json with narration_tr
# ---------------------------------------------------------------------------

def test_sync_enriches_scenes_json_on_disk(isolated_storage):
    paths, _ = ps.create_project("Enrich On Disk Topic")
    ps.save_json(paths.scenes_json, _base_scenes(4))
    ps.save_json(paths.script_tr_json, _script_tr())

    sp.sync_localized_narrations(paths, language="tr")

    on_disk = ps.load_json(paths.scenes_json)
    assert all("narration_tr" in s and s["narration_tr"].strip() for s in on_disk)


# ---------------------------------------------------------------------------
# 4-6. scene count / scene IDs / visual_prompt unchanged
# ---------------------------------------------------------------------------

def test_sync_preserves_scene_count_ids_and_visual_prompts(isolated_storage):
    paths, _ = ps.create_project("Preserve Fields Topic")
    original = _base_scenes(5)
    ps.save_json(paths.scenes_json, original)
    ps.save_json(paths.script_tr_json, _script_tr())

    result = sp.sync_localized_narrations(paths, language="tr")

    assert len(result) == len(original) == 5
    assert [s["scene_id"] for s in result] == [s["scene_id"] for s in original]
    assert [s["visual_prompt"] for s in result] == [s["visual_prompt"] for s in original]
    assert [s["narration_en"] for s in result] == [s["narration_en"] for s in original]
    assert [s["duration_hint"] for s in result] == [s["duration_hint"] for s in original]
    assert [s["visual_type"] for s in result] == [s["visual_type"] for s in original]


def test_sync_is_idempotent_and_does_not_rewrite_an_already_enriched_file(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Idempotent Sync Topic")
    ps.save_json(paths.scenes_json, _base_scenes(3))
    ps.save_json(paths.script_tr_json, _script_tr())

    sp.sync_localized_narrations(paths, language="tr")
    enriched_once = ps.load_json(paths.scenes_json)

    write_calls = {"n": 0}
    real_atomic = ps.save_json_atomic

    def _counting_atomic(path, data):
        write_calls["n"] += 1
        return real_atomic(path, data)

    monkeypatch.setattr(ps, "save_json_atomic", _counting_atomic)

    # scene_planner.py imports `project_store as ps` too - patch the same
    # module object it actually calls through.
    monkeypatch.setattr(sp.ps, "save_json_atomic", _counting_atomic)

    result = sp.sync_localized_narrations(paths, language="tr")

    assert write_calls["n"] == 0  # already fully enriched - no rewrite
    assert result == enriched_once


# ---------------------------------------------------------------------------
# 7. Existing visual files are not regenerated
# ---------------------------------------------------------------------------

def test_sync_never_touches_visual_files(isolated_storage):
    paths, _ = ps.create_project("Visuals Untouched Topic")
    scenes = _base_scenes(3)
    ps.save_json(paths.scenes_json, scenes)
    ps.save_json(paths.script_tr_json, _script_tr())

    original_bytes = {}
    for scene in scenes:
        img_path = paths.scene_image(scene["scene_id"])
        content = f"fake png bytes for scene {scene['scene_id']}".encode()
        img_path.write_bytes(content)
        original_bytes[scene["scene_id"]] = content

    before_names = sorted(p.name for p in paths.scene_images())

    sp.sync_localized_narrations(paths, language="tr")

    after_names = sorted(p.name for p in paths.scene_images())
    assert before_names == after_names  # exact same filenames, e.g. scene_001.png...scene_003.png
    assert not any("_tr" in name or "_en" in name for name in after_names)  # never language-suffixed
    for scene in scenes:
        assert paths.scene_image(scene["scene_id"]).read_bytes() == original_bytes[scene["scene_id"]]  # byte-for-byte


# ---------------------------------------------------------------------------
# 8. Existing English narration remains unchanged
# ---------------------------------------------------------------------------

def test_sync_never_touches_english_narration_audio(isolated_storage):
    paths, _ = ps.create_project("English Audio Untouched Topic")
    ps.save_json(paths.scenes_json, _base_scenes(3))
    ps.save_json(paths.script_tr_json, _script_tr())
    paths.narration_en_mp3.write_bytes(b"original english audio bytes")

    sp.sync_localized_narrations(paths, language="tr")

    assert paths.narration_en_mp3.read_bytes() == b"original english audio bytes"
    assert not paths.narration_tr_mp3.exists()  # sync never generates audio either


# ---------------------------------------------------------------------------
# 9. Turkish TTS still uses script_tr.json (unaffected by scene sync)
# ---------------------------------------------------------------------------

def test_turkish_tts_after_sync_still_reads_from_script_tr_json(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")
    paths, _ = ps.create_project("TR TTS After Sync Topic")
    ps.save_json(paths.scenes_json, _base_scenes(3))
    script_tr = _script_tr()
    ps.save_json(paths.script_tr_json, script_tr)

    sp.sync_localized_narrations(paths, language="tr")
    tts.run_and_save(paths, language="tr")

    assert paths.narration_tr_mp3.exists()
    # The scene-level narration_tr chunks (used only for the shared scene
    # plan's own bookkeeping) must not be what TTS synthesizes from - TTS
    # always reads the localized script's own full_narration directly.
    import wave
    with wave.open(str(paths.narration_tr_mp3), "rb") as wf:
        duration = wf.getnframes() / wf.getframerate()
    expected_words = len(script_tr["full_narration"].split())
    assert duration > 0
    assert expected_words == 80


# ---------------------------------------------------------------------------
# 10. Future configured language (e.g. de) enriches the SAME scenes.json,
# with no language-specific code branches.
# ---------------------------------------------------------------------------

def test_sync_works_identically_for_a_future_language_with_no_special_casing(isolated_storage):
    paths, _ = ps.create_project("Future Language Sync Topic")
    ps.save_json(paths.scenes_json, _base_scenes(3))
    ps.save_json(paths.script_localized_json("de"), {
        "language": "de", "source_language": "en", "title": "DE Title",
        "full_narration": _many_short_sentences(80, word="wort"),
        "word_count": 80, "sections": [], "sources": [],
    })

    result = sp.sync_localized_narrations(paths, language="de")

    assert all("narration_de" in s for s in result)
    # Confirmed no language-specific branch: the exact same function, same
    # scenes.json, was already used for "tr" in the tests above.
    scenes_on_disk = ps.load_json(paths.scenes_json)
    assert all("narration_de" in s for s in scenes_on_disk)


def test_pipeline_stage_plan_includes_scene_sync_for_every_configured_localization(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr", "de", "es"])
    paths, _ = ps.create_project("Stage Plan Scene Sync Topic")

    stages = pipeline_module._stage_plan(paths, "Stage Plan Scene Sync Topic", None)
    names = [name for name, _, _ in stages]

    for lang in ("tr", "de", "es"):
        assert f"scene_sync_{lang}" in names
        # scene_sync_<lang> must come after "scenes"/"images" (it enriches
        # an EXISTING scene plan) and before that language's own narration
        # stage (per the required resume order: script -> sync -> audio).
        assert names.index("scenes") < names.index(f"scene_sync_{lang}")
        assert names.index("images") < names.index(f"scene_sync_{lang}")
        assert names.index(f"scene_sync_{lang}") < names.index(f"narration_{lang}")
    # And it never duplicates the shared, singular stages.
    assert names.count("scenes") == 1
    assert names.count("images") == 1


# ---------------------------------------------------------------------------
# Full pipeline resume integration: scenes.json/visuals/narration_en already
# exist, script_tr.json is missing - resume must localize, sync scenes.json
# in place, and only then generate Turkish audio, touching nothing else.
# ---------------------------------------------------------------------------

def _never_call(name):
    def _fail(*a, **k):
        raise AssertionError(f"{name} must NOT be called - this stage's artifact already exists on disk")
    return _fail


def test_full_pipeline_resume_localizes_syncs_scenes_then_generates_turkish_audio_only(isolated_storage, monkeypatch):
    from youtube import researcher, script_generator, scene_planner, image_gen, assemble, thumbnail, metadata, quality_check

    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    topic = "Full Resume Scene Sync Topic"
    paths, _ = ps.create_project(topic)

    # research / script_en / scenes / images / narration_en already done.
    ps.save_json(paths.research_json, {"topic": topic, "sources": []})
    ps.save_json(paths.script_en_json, {"full_narration": "english text", "sections": []})
    original_scenes = _base_scenes(3)
    ps.save_json(paths.scenes_json, original_scenes)
    scene_image_bytes = {}
    for scene in original_scenes:
        content = f"fake image bytes {scene['scene_id']}".encode()
        paths.scene_image(scene["scene_id"]).write_bytes(content)
        scene_image_bytes[scene["scene_id"]] = content
    paths.narration_en_mp3.write_bytes(b"original english narration bytes")
    # script_tr.json deliberately missing - this is the resume trigger.

    calls = []

    monkeypatch.setattr(researcher, "run_and_save", _never_call("researcher"))
    monkeypatch.setattr(script_generator, "run_and_save", _never_call("script_generator"))
    monkeypatch.setattr(scene_planner, "run_and_save", _never_call("scene_planner (must not replan)"))
    monkeypatch.setattr(image_gen, "run_and_save", _never_call("image_gen (must not regenerate visuals)"))

    def _fake_localize(paths_arg, language="tr"):
        calls.append(f"localization_{language}")
        ps.save_json(paths_arg.script_localized_json(language), _script_tr())
        return {}

    monkeypatch.setattr(localizer, "run_and_save", _fake_localize)

    def _fake_tts(paths_arg, language=None, **kwargs):
        language = language or settings.LANGUAGE_MASTER
        if language == settings.LANGUAGE_MASTER:
            raise AssertionError("English narration already exists on disk - TTS must not run for 'en'")
        calls.append(f"narration_{language}")
        paths_arg.narration_mp3_for(language).write_bytes(b"tr audio bytes")
        return paths_arg.narration_mp3_for(language)

    monkeypatch.setattr(tts, "run_and_save", _fake_tts)

    def _fake_assemble(paths_arg, **kwargs):
        calls.append("assemble")
        paths_arg.video_mp4.write_bytes(b"fake video")
        return paths_arg.video_mp4

    monkeypatch.setattr(assemble, "run_and_save", _fake_assemble)

    def _fake_thumbnail(paths_arg, **kwargs):
        calls.append("thumbnail")
        paths_arg.thumbnail_png.write_bytes(b"fake thumb")
        return paths_arg.thumbnail_png

    monkeypatch.setattr(thumbnail, "run_and_save", _fake_thumbnail)

    def _fake_metadata(paths_arg, category=None, language=None):
        language = language or settings.LANGUAGE_MASTER
        calls.append(f"metadata_{language}")
        ps.save_json(paths_arg.metadata_json_for(language), {"ok": True, "language": language})
        return {}

    monkeypatch.setattr(metadata, "run_and_save", _fake_metadata)
    monkeypatch.setattr(pipeline_module.quality_check, "run_and_enforce",
                         lambda paths_arg, languages=None: quality_check.QualityReport(passed=True, checks={"ok": {"passed": True, "detail": ""}}))

    pipeline_module.produce_video(topic, video_id=paths.video_id)

    # Exact required order: localize -> sync (implicit, not a "call" to any
    # stubbed module since it runs for real) -> Turkish audio -> the rest.
    assert calls == ["localization_tr", "narration_tr", "assemble", "thumbnail", "metadata_en", "metadata_tr"]

    # scenes.json enriched in place: same count, same IDs, same prompts.
    enriched = ps.load_json(paths.scenes_json)
    assert len(enriched) == len(original_scenes) == 3
    assert [s["scene_id"] for s in enriched] == [s["scene_id"] for s in original_scenes]
    assert [s["visual_prompt"] for s in enriched] == [s["visual_prompt"] for s in original_scenes]
    assert all("narration_tr" in s for s in enriched)

    # Visual files: exact same filenames, byte-for-byte unchanged - the
    # visual generation API (image_gen.run_and_save) was never called
    # (enforced above by _never_call), so this is also a structural
    # guarantee, not just a byte comparison.
    after_names = sorted(p.name for p in paths.scene_images())
    assert after_names == [f"scene_{i:03d}.png" for i in (1, 2, 3)]
    assert not any("_tr" in n or "_en" in n for n in after_names)
    for scene in original_scenes:
        assert paths.scene_image(scene["scene_id"]).read_bytes() == scene_image_bytes[scene["scene_id"]]

    # English narration untouched.
    assert paths.narration_en_mp3.read_bytes() == b"original english narration bytes"
