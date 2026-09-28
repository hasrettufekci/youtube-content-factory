"""
Dedicated tests for the Round 3 multilingual single-video architecture
(spec section 19's 12 required areas):

    1. English master script
    2. Turkish localization
    3. shared scene plan
    4. shared visual assets (no duplicate generation)
    5. English TTS
    6. Turkish TTS
    7. language-specific audio durations (never forced equal)
    8. resume when Turkish script is missing
    9. resume when Turkish audio is missing
    10. no duplicate visual generation
    11. multilingual metadata
    12. future-language configuration

test_pipeline_resume.py already covers resume end-to-end through
pipeline.produce_video(); the tests here focus on each stage module in
isolation, at the unit level, so a break in any one piece of the
multilingual design is pinpointed without going through the full pipeline.
"""
import importlib

import pytest

import pipeline as pipeline_module
from core.config import settings
from storage import project_store as ps
from youtube import localizer, metadata, scene_planner as sp, script_generator as sg, tts, image_gen


def _many_short_sentences(word_count: int, word: str = "word") -> str:
    """Many ~12-word sentences (not one giant run-on sentence), so there are
    always enough natural sentence-boundary cut points for scene_planner to
    reach its scene-count target - mirrors test_scene_planner.py's
    `_fake_narration` helper."""
    sentences = []
    words_left = word_count
    while words_left > 0:
        n = min(12, words_left)
        sentences.append(" ".join([word] * n) + ".")
        words_left -= n
    return " ".join(sentences)


EN_SECTIONS = [
    {"type": t, "title": t, "narration": _many_short_sentences(150)}
    for t in sg.REQUIRED_SECTION_TYPES
]


def _fake_script_en() -> dict:
    full_narration = "\n\n".join(s["narration"] for s in EN_SECTIONS)
    return {
        "title": "The English Master Title",
        "hook": EN_SECTIONS[0]["narration"],
        "sections": EN_SECTIONS,
        "full_narration": full_narration,
        "word_count": len(full_narration.split()),
        "sources": [{"url": "https://example.com/source", "title": "A Source"}],
        "language": "en",
    }


# ---------------------------------------------------------------------------
# 1. English master script
# ---------------------------------------------------------------------------

def test_english_master_script_is_the_canonical_source_of_truth(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Master Script Topic")
    ps.save_json(paths.research_json, {"topic": "Master Script Topic", "sources": []})

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "A Real Documentary Title", "sections": EN_SECTIONS}, []

    monkeypatch.setattr(sg.llm_client, "chat_json", fake_chat_json)

    script = sg.run_and_save(paths)

    # Written to script_en.json on disk - script_json is just a
    # backward-compat alias for the very same path.
    assert paths.script_en_json.exists()
    assert paths.script_json == paths.script_en_json
    assert ps.load_json(paths.script_en_json)["title"] == script["title"]


# ---------------------------------------------------------------------------
# 2. Turkish localization
# ---------------------------------------------------------------------------

def test_localize_script_preserves_structure_and_recomputes_narration(monkeypatch):
    script_en = _fake_script_en()
    tr_sections = [
        {"type": s["type"], "title": s["title"], "narration": f"Türkçe metin: {s['narration']}"}
        for s in EN_SECTIONS
    ]

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "Türkçe Belgesel Başlığı", "sections": tr_sections}, []

    monkeypatch.setattr(localizer.llm_client, "chat_json", fake_chat_json)

    localized = localizer.localize_script(script_en, language="tr")

    assert localized["language"] == "tr"
    assert localized["source_language"] == "en"
    assert localized["title"] == "Türkçe Belgesel Başlığı"
    # full_narration is recomputed in code, never trusted from the LLM.
    assert localized["full_narration"] == "\n\n".join(s["narration"] for s in tr_sections)
    assert localized["word_count"] == len(localized["full_narration"].split())
    # Same section count/order/types as the English master - required for
    # scene-relationship alignment downstream.
    assert [s["type"] for s in localized["sections"]] == [s["type"] for s in script_en["sections"]]
    # Sources are carried over unchanged, not "translated".
    assert localized["sources"] == script_en["sources"]


def test_localize_script_rejects_mismatched_section_types(monkeypatch):
    script_en = _fake_script_en()
    # Drop one section - desyncs from the English master's structure.
    bad_sections = [
        {"type": s["type"], "title": s["title"], "narration": "x"} for s in EN_SECTIONS[:-1]
    ]

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "T", "sections": bad_sections}, []

    monkeypatch.setattr(localizer.llm_client, "chat_json", fake_chat_json)

    with pytest.raises(RuntimeError, match="do not match"):
        localizer.localize_script(script_en, language="tr")


def test_localizer_run_and_save_writes_script_tr_json_and_leaves_english_untouched(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Localization Run Topic")
    ps.save_json(paths.script_en_json, _fake_script_en())

    tr_sections = [
        {"type": s["type"], "title": s["title"], "narration": f"TR: {s['narration']}"}
        for s in EN_SECTIONS
    ]

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "TR Title", "sections": tr_sections}, []

    monkeypatch.setattr(localizer.llm_client, "chat_json", fake_chat_json)

    localizer.run_and_save(paths, language="tr")

    assert paths.script_tr_json.exists()
    assert paths.script_en_json.exists()  # untouched
    saved = ps.load_json(paths.script_tr_json)
    assert saved["language"] == "tr"


def test_localizer_requires_english_master_script_first(isolated_storage):
    paths, _ = ps.create_project("No English Script Yet Topic")
    with pytest.raises(RuntimeError, match="script_en.json"):
        localizer.run_and_save(paths, language="tr")


# ---------------------------------------------------------------------------
# 3. Shared scene plan
# ---------------------------------------------------------------------------

def test_shared_scene_plan_attaches_both_language_narrations_to_the_same_scenes(monkeypatch):
    script_en = _fake_script_en()
    # A Turkish narration long enough (many short sentences) to be chunked
    # the same way as the English master.
    script_tr = {"full_narration": _many_short_sentences(1200, word="kelime")}

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        # one visual prompt per numbered chunk in the user prompt
        import re
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        return [
            {"scene_id": i + 1, "visual_prompt": f"prompt {i+1}", "visual_type": "cinematic"}
            for i in range(len(chunk_lines))
        ], []

    monkeypatch.setattr(sp.llm_client, "chat_json", fake_chat_json)

    scenes = sp.plan_scenes(script_en, localized_scripts={"tr": script_tr})

    assert len(scenes) >= sp.MIN_SCENES
    for scene in scenes:
        # visual_prompt is language-independent - exactly one shared field,
        # never duplicated per language.
        assert "visual_prompt" in scene
        assert "narration_en" in scene
        # Every scene here should have a narration_tr too, since the
        # Turkish script was fully available at planning time.
        assert "narration_tr" in scene
        assert scene["narration_en"] != scene["narration_tr"]

    # Scene count == shared visual count implicitly: one visual_prompt per
    # scene, not one per (scene, language) pair.
    assert len(scenes) == len({s["scene_id"] for s in scenes})


def test_shared_scene_plan_still_builds_from_english_alone_when_localization_missing(monkeypatch):
    """A language whose script isn't available yet at planning time must
    never block or alter scene/visual planning - narration_<lang> is simply
    absent from every scene until a later resume run fills it in."""
    script_en = _fake_script_en()

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        import re
        chunk_lines = [l for l in user.split("\n") if re.match(r"^\d+\.", l)]
        return [
            {"scene_id": i + 1, "visual_prompt": f"prompt {i+1}", "visual_type": "cinematic"}
            for i in range(len(chunk_lines))
        ], []

    monkeypatch.setattr(sp.llm_client, "chat_json", fake_chat_json)

    scenes = sp.plan_scenes(script_en, localized_scripts=None)

    assert len(scenes) >= sp.MIN_SCENES
    for scene in scenes:
        assert "narration_en" in scene
        assert "narration_tr" not in scene  # not available yet, not fabricated


# ---------------------------------------------------------------------------
# 4 & 10. Shared visual assets / no duplicate generation
# ---------------------------------------------------------------------------

def test_visual_generation_count_equals_scene_count_not_doubled_per_language(isolated_storage, monkeypatch):
    """The core Round 3 invariant (spec section 26): shared visual count
    MUST equal the number of shared scenes - never 2x for two languages.
    image_gen.py has no language parameter at all - it only ever knows
    about scenes.json, never about which languages are configured."""
    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    paths, _ = ps.create_project("Shared Visuals Topic")

    scenes = [
        {"scene_id": i, "visual_prompt": f"a cinematic scene number {i}", "visual_type": "cinematic"}
        for i in range(1, 6)
    ]
    ps.save_json(paths.scenes_json, scenes)

    result_paths = image_gen.generate_scene_images(paths, scenes)

    assert len(result_paths) == len(scenes) == 5
    assert len(paths.scene_images()) == 5  # not 10 (5 scenes x 2 languages)

    # image_gen's public surface never takes a language argument - it is
    # structurally incapable of generating per-language images.
    import inspect
    assert "language" not in inspect.signature(image_gen.generate_scene_images).parameters
    assert "language" not in inspect.signature(image_gen.run_and_save).parameters


def test_existing_scene_images_are_not_regenerated_on_resume(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    paths, _ = ps.create_project("Resume Visuals Topic")

    scenes = [{"scene_id": 1, "visual_prompt": "prompt one", "visual_type": "cinematic"}]
    ps.save_json(paths.scenes_json, scenes)
    paths.scene_image(1).write_bytes(b"already generated bytes")
    original_bytes = paths.scene_image(1).read_bytes()

    calls = {"n": 0}
    real_generate = image_gen.MockImageProvider.generate

    def _counting_generate(self, prompt, width, height):
        calls["n"] += 1
        return real_generate(self, prompt, width, height)

    monkeypatch.setattr(image_gen.MockImageProvider, "generate", _counting_generate)

    image_gen.generate_scene_images(paths, scenes)

    assert calls["n"] == 0  # provider never called - existing file left alone
    assert paths.scene_image(1).read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# 5 & 6. English TTS / Turkish TTS
# ---------------------------------------------------------------------------

def test_english_and_turkish_tts_write_independent_files_from_independent_scripts(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")
    paths, _ = ps.create_project("Bilingual TTS Topic")
    ps.save_json(paths.script_en_json, {"full_narration": " ".join(["word"] * 100)})
    ps.save_json(paths.script_tr_json, {"full_narration": " ".join(["kelime"] * 40)})

    tts.run_and_save(paths, language="en")
    tts.run_and_save(paths, language="tr")

    assert paths.narration_en_mp3.exists()
    assert paths.narration_tr_mp3.exists()
    assert paths.narration_en_mp3 != paths.narration_tr_mp3
    # English narration used script_en.json's text, Turkish used
    # script_tr.json's - independently, not a shared/derived source.
    assert paths.narration_en_mp3.read_bytes() != paths.narration_tr_mp3.read_bytes()


def test_tts_run_and_save_defaults_to_master_language(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")
    paths, _ = ps.create_project("Default Language TTS Topic")
    ps.save_json(paths.script_en_json, {"full_narration": "some english narration text"})

    tts.run_and_save(paths)  # no language given -> settings.LANGUAGE_MASTER ("en")

    assert paths.narration_en_mp3.exists()
    assert paths.narration_mp3 == paths.narration_en_mp3  # alias


def test_tts_for_a_localization_reads_that_localizations_own_script_not_english(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")
    paths, _ = ps.create_project("TR Reads TR Script Topic")
    ps.save_json(paths.script_en_json, {"full_narration": " ".join(["word"] * 200)})
    # deliberately no script_tr.json yet

    with pytest.raises(RuntimeError, match="localized script"):
        tts.run_and_save(paths, language="tr")

    assert not paths.narration_tr_mp3.exists()
    # English narration must be completely unaffected by the Turkish failure.
    assert not paths.narration_en_mp3.exists()  # never even attempted here


# ---------------------------------------------------------------------------
# 7. Language-specific audio durations (never forced equal)
# ---------------------------------------------------------------------------

def test_mock_tts_durations_differ_by_text_length_and_are_never_equalized():
    provider = tts.MockTTSProvider()
    short_bytes = provider.synthesize(" ".join(["word"] * 20), voice="")
    long_bytes = provider.synthesize(" ".join(["kelime"] * 260), voice="")

    import wave
    from io import BytesIO

    with wave.open(BytesIO(short_bytes), "rb") as wf:
        short_duration = wf.getnframes() / wf.getframerate()
    with wave.open(BytesIO(long_bytes), "rb") as wf:
        long_duration = wf.getnframes() / wf.getframerate()

    # No cropping/stretching/speed-matching logic exists anywhere in tts.py -
    # each language's duration is whatever its own text produces.
    assert long_duration > short_duration
    assert abs(long_duration - short_duration) > 5.0


def test_quality_check_reports_audio_durations_per_language_separately(isolated_storage, monkeypatch):
    from PIL import Image

    from youtube import quality_check as qc

    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    paths, _ = ps.create_project("Per Language Duration Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [{"scene_id": 1}])
    ps.save_json(paths.research_json, {"sources": [{"url": "https://example.com"}]})
    ps.save_json(paths.script_en_json, {"title": "t"})
    ps.save_json(paths.script_tr_json, {"title": "t"})
    ps.save_json(paths.metadata_en_json, {"title": "t"})
    ps.save_json(paths.metadata_tr_json, {"title": "t"})
    paths.narration_en_mp3.write_bytes(b"fake")
    paths.narration_tr_mp3.write_bytes(b"fake")
    paths.scene_image(1).write_bytes(b"fake")
    Image.new("RGB", (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), (0, 0, 0)).save(paths.thumbnail_png, format="PNG")

    durations = {"en": 481.2, "tr": 512.7}  # deliberately different, on purpose
    video_probe = {"format": {"duration": "480.0"}, "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT},
        {"codec_type": "audio", "codec_name": "aac"},
    ]}

    def _fake_probe(path):
        for lang, seconds in durations.items():
            if str(path).endswith(f"narration_{lang}.mp3"):
                return {"format": {"duration": str(seconds)}, "streams": []}
        return video_probe

    monkeypatch.setattr(qc, "_ffprobe_json", _fake_probe)

    report = qc.run_quality_check(paths)

    assert report.audio_durations["en"] == pytest.approx(481.2)
    assert report.audio_durations["tr"] == pytest.approx(512.7)
    assert report.audio_durations["en"] != report.audio_durations["tr"]


# ---------------------------------------------------------------------------
# 8 & 9. Resume when Turkish script / Turkish audio is missing
# ---------------------------------------------------------------------------

def test_stage_plan_resumes_at_localization_when_only_turkish_script_missing(isolated_storage, monkeypatch):
    """Direct, unit-level check of pipeline._stage_plan()'s is_done()
    predicates (complementing test_pipeline_resume.py's full
    monkeypatched integration test): research/script_en/scenes/images/
    narration_en all present, script_tr absent -> only localization_tr
    (and everything after it) should report is_done()==False."""
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    paths, _ = ps.create_project("TR Script Missing Topic")
    ps.save_json(paths.research_json, {"topic": "x", "sources": []})
    ps.save_json(paths.script_en_json, {"full_narration": "x", "sections": []})
    ps.save_json(paths.scenes_json, [{"scene_id": 1, "visual_prompt": "p"}])
    paths.scene_image(1).write_bytes(b"fake png")
    paths.narration_en_mp3.write_bytes(b"fake")

    stages = pipeline_module._stage_plan(paths, "TR Script Missing Topic", None)
    status = {name: is_done() for name, is_done, _ in stages}

    assert status["research"] is True
    assert status["script_en"] is True
    assert status["localization_tr"] is False  # exactly the missing piece
    assert status["scenes"] is True
    assert status["images"] is True
    assert status["narration_en"] is True


def test_stage_plan_resumes_at_narration_tr_when_only_turkish_audio_missing(isolated_storage, monkeypatch):
    """Everything through the Turkish script is already done; only
    narration_tr.mp3 is missing -> only that stage (and metadata_tr, which
    naturally isn't written yet either) should be outstanding, and every
    upstream stage (including the already-done Turkish localization)
    must report is_done()==True so it is never rerun."""
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    paths, _ = ps.create_project("TR Audio Missing Topic")
    ps.save_json(paths.research_json, {"topic": "x", "sources": []})
    ps.save_json(paths.script_en_json, {"full_narration": "x", "sections": []})
    ps.save_json(paths.script_tr_json, {"full_narration": "metin", "sections": []})
    ps.save_json(paths.scenes_json, [{"scene_id": 1, "visual_prompt": "p"}])
    paths.scene_image(1).write_bytes(b"fake png")
    paths.narration_en_mp3.write_bytes(b"fake")
    # narration_tr.mp3 deliberately not written

    stages = pipeline_module._stage_plan(paths, "TR Audio Missing Topic", None)
    status = {name: is_done() for name, is_done, _ in stages}

    assert status["localization_tr"] is True  # already done, must not rerun
    assert status["narration_en"] is True      # already done, must not rerun
    assert status["narration_tr"] is False      # exactly the missing piece


# ---------------------------------------------------------------------------
# 11. Multilingual metadata
# ---------------------------------------------------------------------------

def test_multilingual_metadata_describes_the_same_story_in_each_language():
    research = {
        "topic": "The Dyatlov Pass Incident",
        "sources": [{"url": "https://example.com/a", "title": "A"}, {"url": "https://example.com/b", "title": "B"}],
    }
    script_en = _fake_script_en()
    script_tr = dict(script_en)
    script_tr["title"] = "Dyatlov Geçidi Olayı"
    script_tr["language"] = "tr"

    meta_en = metadata.build_metadata(script_en, research, category="Mysteries", language="en")
    meta_tr = metadata.build_metadata(script_tr, research, category="Mysteries", language="tr")

    # Same underlying story: same sources, same category-derived source count.
    assert meta_en["source_count"] == meta_tr["source_count"] == 2
    # But properly localized presentation: different language tag, different
    # (localized) title, Turkish-specific tag suffixes.
    assert meta_en["language"] == "en"
    assert meta_tr["language"] == "tr"
    assert meta_en["title"] != meta_tr["title"]
    assert "belgesel" in meta_tr["tags"]  # Turkish tag suffix table used
    assert "documentary" in meta_en["tags"]  # English tag suffix table used
    assert "gizem" not in meta_en["tags"]


def test_metadata_run_and_save_reads_the_correct_language_specific_script(isolated_storage, monkeypatch):
    """Regression coverage for the real bug fixed in this round: language
    was accepted as a parameter but the script source was NOT branched on
    it, so a Turkish metadata call would have silently built its output
    from the English script instead."""
    paths, _ = ps.create_project("Metadata Reads Right Script Topic")
    ps.save_json(paths.research_json, {"topic": "x", "sources": []})
    ps.save_json(paths.script_en_json, {**_fake_script_en(), "title": "English Only Title"})
    ps.save_json(paths.script_tr_json, {**_fake_script_en(), "title": "Sadece Türkçe Başlık"})

    meta_en = metadata.run_and_save(paths, language="en")
    meta_tr = metadata.run_and_save(paths, language="tr")

    assert meta_en["title"] == "English Only Title"
    assert meta_tr["title"] == "Sadece Türkçe Başlık"
    assert ps.load_json(paths.metadata_en_json)["title"] == "English Only Title"
    assert ps.load_json(paths.metadata_tr_json)["title"] == "Sadece Türkçe Başlık"


# ---------------------------------------------------------------------------
# 12. Future-language configuration
# ---------------------------------------------------------------------------

def test_adding_a_future_language_to_config_extends_the_stage_plan_with_zero_code_changes(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr", "de"])
    paths, _ = ps.create_project("Future Language Topic")

    stages = pipeline_module._stage_plan(paths, "Future Language Topic", None)
    names = [name for name, _, _ in stages]

    assert "localization_de" in names
    assert "narration_de" in names
    assert "metadata_de" in names
    # And German never displaces the existing Turkish stages.
    assert "localization_tr" in names
    assert "narration_tr" in names
    assert "metadata_tr" in names
    # Nothing upstream duplicates per language - research/scenes/images stay singular.
    assert names.count("research") == 1
    assert names.count("scenes") == 1
    assert names.count("images") == 1


def test_adding_a_language_never_adds_a_second_research_or_visual_stage(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr", "de", "fr", "es"])
    paths, _ = ps.create_project("Many Languages Topic")

    stages = pipeline_module._stage_plan(paths, "Many Languages Topic", None)
    names = [name for name, _, _ in stages]

    research_and_visual_stages = [n for n in names if n in ("research", "script_en", "scenes", "images")]
    assert research_and_visual_stages == ["research", "script_en", "scenes", "images"]  # exactly one each

    # Cost control (spec section 21): exactly one localization/narration/
    # metadata triple per configured localization language.
    for lang in ("tr", "de", "fr", "es"):
        assert names.count(f"localization_{lang}") == 1
        assert names.count(f"narration_{lang}") == 1
        assert names.count(f"metadata_{lang}") == 1


def test_tts_voice_for_is_configurable_per_language_and_falls_back_cleanly(monkeypatch):
    monkeypatch.delenv("TTS_VOICE_DE", raising=False)
    monkeypatch.setenv("TTS_VOICE_TR", "shimmer")
    monkeypatch.setattr(settings, "OPENAI_TTS_VOICE", "onyx")

    assert settings.tts_voice_for("tr") == "shimmer"       # explicit per-language override
    assert settings.tts_voice_for("de") == "onyx"          # no override configured -> falls back cleanly
    assert settings.tts_voice_for("en") == "onyx"


def test_tts_instructions_for_is_configurable_per_language_and_falls_back_cleanly(monkeypatch):
    """Round 3.2.1 TTS production update: OPENAI_TTS_INSTRUCTIONS gained the
    exact same per-language-override-with-fallback shape as OPENAI_TTS_VOICE
    above (TTS_INSTRUCTIONS_<LANG>, via tts_instructions_for()) - reusing
    the project's existing configuration mechanism, not a new one."""
    monkeypatch.delenv("TTS_INSTRUCTIONS_DE", raising=False)
    monkeypatch.setenv("TTS_INSTRUCTIONS_TR", "energetic and playful, faster pace")
    monkeypatch.setattr(settings, "OPENAI_TTS_INSTRUCTIONS", "deep documentary default style")

    assert settings.tts_instructions_for("tr") == "energetic and playful, faster pace"  # explicit per-language override
    assert settings.tts_instructions_for("de") == "deep documentary default style"      # no override -> falls back cleanly
    assert settings.tts_instructions_for("en") == "deep documentary default style"


def test_localization_languages_helper_never_double_counts_master_language(monkeypatch):
    # Defensive config-mistake case: master accidentally also listed as a
    # localization. Must not produce a redundant/self-localization stage.
    monkeypatch.setattr(settings, "LANGUAGE_MASTER", "en")
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["en", "tr"])

    assert pipeline_module._localization_languages() == ["tr"]
