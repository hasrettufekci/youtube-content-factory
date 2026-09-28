"""
Quality check depth (spec Round 2 section 9): must actually verify file
exists, valid MP4, duration, 1920x1080, video stream, audio stream, H264,
AAC, thumbnail exists (+ correct size), metadata exists - not just "the
file exists on disk". Also covers the pipeline-level fix (section 8/9's
"ffmpeg/quality-check failure must transition to FAILED"): a crash inside
quality_check must never propagate out of pipeline.produce_video()
uncaught, or the video would be left stuck, unresumable, with no FAILED
state ever recorded.
"""
import json

import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import quality_check as qc


def _fake_probe_json(monkeypatch, video_streams=None, audio_streams=None, duration=480.0):
    payload = {
        "format": {"duration": str(duration)},
        "streams": (video_streams or []) + (audio_streams or []),
    }

    def _fake(path):
        return payload

    monkeypatch.setattr(qc, "_ffprobe_json", _fake)
    return payload


def _valid_video_stream():
    return {"codec_type": "video", "codec_name": "h264", "width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}


def _valid_audio_stream():
    return {"codec_type": "audio", "codec_name": "aac"}


def test_corrupt_mp4_reported_as_failed_check_not_a_crash(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Corrupt MP4 Topic")
    paths.video_mp4.write_bytes(b"not actually a valid mp4 container")

    def _raise(path):
        raise RuntimeError("[FFMPEG] ffprobe failed on video.mp4: moov atom not found")

    monkeypatch.setattr(qc, "_ffprobe_json", _raise)

    # Must not raise - a corrupt file is a reported, named check failure.
    report = qc.run_quality_check(paths)

    assert report.passed is False
    assert report.checks["valid_mp4_container"]["passed"] is False
    assert "moov atom" in report.checks["valid_mp4_container"]["detail"]


def test_wrong_video_codec_fails_check(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Wrong Video Codec Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [])
    ps.save_json(paths.metadata_json, {"title": "t"})
    paths.narration_mp3.write_bytes(b"fake")
    paths.thumbnail_png.write_bytes(b"fake")

    wrong_codec_video = {"codec_type": "video", "codec_name": "mpeg4",
                          "width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}
    _fake_probe_json(monkeypatch, video_streams=[wrong_codec_video], audio_streams=[_valid_audio_stream()])

    report = qc.run_quality_check(paths)

    assert report.checks["video_codec_h264"]["passed"] is False
    assert "mpeg4" in report.checks["video_codec_h264"]["detail"]
    assert report.passed is False


def test_wrong_audio_codec_fails_check(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Wrong Audio Codec Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [])
    ps.save_json(paths.metadata_json, {"title": "t"})
    paths.narration_mp3.write_bytes(b"fake")
    paths.thumbnail_png.write_bytes(b"fake")

    wrong_codec_audio = {"codec_type": "audio", "codec_name": "mp3"}
    _fake_probe_json(monkeypatch, video_streams=[_valid_video_stream()], audio_streams=[wrong_codec_audio])

    report = qc.run_quality_check(paths)

    assert report.checks["audio_codec_aac"]["passed"] is False
    assert "mp3" in report.checks["audio_codec_aac"]["detail"]


def test_missing_video_stream_fails_resolution_and_codec_checks(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("No Video Stream Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [])
    ps.save_json(paths.metadata_json, {"title": "t"})
    paths.narration_mp3.write_bytes(b"fake")
    paths.thumbnail_png.write_bytes(b"fake")

    _fake_probe_json(monkeypatch, video_streams=[], audio_streams=[_valid_audio_stream()])

    report = qc.run_quality_check(paths)

    assert report.checks["video_stream_present"]["passed"] is False
    assert report.checks["resolution_1920x1080"]["passed"] is False
    assert report.checks["video_codec_h264"]["passed"] is False


def test_thumbnail_wrong_dimensions_fails_check(isolated_storage, monkeypatch):
    from PIL import Image

    paths, _ = ps.create_project("Wrong Thumbnail Size Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [])
    ps.save_json(paths.metadata_json, {"title": "t"})
    paths.narration_mp3.write_bytes(b"fake")

    # 640x360 instead of the required 1280x720
    Image.new("RGB", (640, 360), (0, 0, 0)).save(paths.thumbnail_png, format="PNG")

    _fake_probe_json(monkeypatch, video_streams=[_valid_video_stream()], audio_streams=[_valid_audio_stream()])

    report = qc.run_quality_check(paths)

    assert report.checks["thumbnail_dimensions_1280x720"]["passed"] is False
    assert "640x360" in report.checks["thumbnail_dimensions_1280x720"]["detail"]


def test_correct_thumbnail_dimensions_pass_check(isolated_storage, monkeypatch):
    from PIL import Image

    paths, _ = ps.create_project("Correct Thumbnail Size Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [])
    ps.save_json(paths.metadata_json, {"title": "t"})
    paths.narration_mp3.write_bytes(b"fake")

    Image.new("RGB", (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), (0, 0, 0)).save(paths.thumbnail_png, format="PNG")

    _fake_probe_json(monkeypatch, video_streams=[_valid_video_stream()], audio_streams=[_valid_audio_stream()])

    report = qc.run_quality_check(paths)

    assert report.checks["thumbnail_dimensions_1280x720"]["passed"] is True


def test_all_checks_pass_for_a_fully_correct_video(isolated_storage, monkeypatch):
    from PIL import Image

    # Pinned rather than relying on the ambient/.env-derived default (see
    # the "LANGUAGE_LOCALIZATIONS ambient-config fix" note further below in
    # this file) - this test writes exactly the Turkish artifacts, so it
    # must require exactly ["tr"], regardless of what ANY real deployment's
    # own .env happens to set that value to.
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])

    paths, _ = ps.create_project("Fully Correct Video Topic")
    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [{"scene_id": 1}])
    # Multilingual (Round 3 section 18): "fully correct" now also requires
    # the master AND every configured localization's own script/narration/
    # metadata to exist - the default config localizes to Turkish.
    ps.save_json(paths.script_en_json, {"title": "t"})
    ps.save_json(paths.script_tr_json, {"title": "t"})
    ps.save_json(paths.metadata_en_json, {"title": "t"})
    ps.save_json(paths.metadata_tr_json, {"title": "t"})
    ps.save_json(paths.research_json, {"sources": [{"url": "https://example.com"}]})
    paths.narration_en_mp3.write_bytes(b"fake")
    paths.narration_tr_mp3.write_bytes(b"fake")
    paths.scene_image(1).write_bytes(b"fake")
    Image.new("RGB", (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), (0, 0, 0)).save(paths.thumbnail_png, format="PNG")

    call_count = {"n": 0}
    video_probe = {"format": {"duration": "480.0"}, "streams": [_valid_video_stream(), _valid_audio_stream()]}
    narration_probe = {"format": {"duration": "479.0"}, "streams": []}

    def _fake(path):
        call_count["n"] += 1
        return video_probe if str(path).endswith("video.mp4") else narration_probe

    monkeypatch.setattr(qc, "_ffprobe_json", _fake)

    report = qc.run_quality_check(paths)

    assert report.passed is True, report.errors
    assert all(c["passed"] for c in report.checks.values())


def _write_common_artifacts(paths, monkeypatch, include_tr=True):
    """Shared setup for the multilingual-vs-english-only QC tests below:
    everything a fully-correct video needs EXCEPT the localized (Turkish)
    artifacts, which are only written when include_tr=True."""
    from PIL import Image

    paths.video_mp4.write_bytes(b"fake")
    ps.save_json(paths.scenes_json, [{"scene_id": 1}])
    ps.save_json(paths.script_en_json, {"title": "t"})
    ps.save_json(paths.metadata_en_json, {"title": "t"})
    ps.save_json(paths.research_json, {"sources": [{"url": "https://example.com"}]})
    paths.narration_en_mp3.write_bytes(b"fake")
    paths.scene_image(1).write_bytes(b"fake")
    Image.new("RGB", (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), (0, 0, 0)).save(paths.thumbnail_png, format="PNG")

    if include_tr:
        ps.save_json(paths.script_tr_json, {"title": "t"})
        ps.save_json(paths.metadata_tr_json, {"title": "t"})
        paths.narration_tr_mp3.write_bytes(b"fake")

    video_probe = {"format": {"duration": "480.0"}, "streams": [_valid_video_stream(), _valid_audio_stream()]}
    narration_probe = {"format": {"duration": "479.0"}, "streams": []}

    def _fake(path):
        return video_probe if str(path).endswith("video.mp4") else narration_probe

    monkeypatch.setattr(qc, "_ffprobe_json", _fake)


# ---------------------------------------------------------------------------
# English-only / QC integration fix (production bug: a real --english-only
# pipeline run - research/script/scenes/images/TTS/assembly/thumbnail/
# metadata all genuinely succeeding with only English artifacts produced -
# still failed QC with script_tr_exists/narration_tr_exists/
# narration_tr_valid/metadata_tr_exists, because run_quality_check() read
# settings.LANGUAGE_LOCALIZATIONS directly instead of the per-run
# `languages` override every other pipeline stage already respects.
# ---------------------------------------------------------------------------

def test_english_only_qc_passes_without_turkish_artifacts(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("English Only QC Topic")
    _write_common_artifacts(paths, monkeypatch, include_tr=False)

    report = qc.run_quality_check(paths, languages=[])

    assert report.passed is True, report.errors
    # Not just "passed" - the Turkish checks must not even be required/present,
    # per "QC must validate only the artifacts actually requested/generated".
    assert "script_tr_exists" not in report.checks
    assert "narration_tr_exists" not in report.checks
    assert "narration_tr_valid" not in report.checks
    assert "metadata_tr_exists" not in report.checks
    # The master-language checks are still very much required.
    assert report.checks["script_en_exists"]["passed"] is True
    assert report.checks["narration_en_exists"]["passed"] is True
    assert report.checks["metadata_en_exists"]["passed"] is True


def test_english_only_qc_still_requires_english_artifacts(isolated_storage, monkeypatch):
    """English-only mode narrows what's required to the master language -
    it does not waive the master language's own checks."""
    paths, _ = ps.create_project("English Only Missing English Topic")
    _write_common_artifacts(paths, monkeypatch, include_tr=False)
    paths.script_en_json.unlink()  # the one artifact this run genuinely needs and is missing

    report = qc.run_quality_check(paths, languages=[])

    assert report.passed is False
    assert report.checks["script_en_exists"]["passed"] is False


# AMBIENT-CONFIG TEST FIX (2026-09): the three "default behavior" tests
# below assert what languages=None/omitted requires. Before this fix they
# relied on whatever settings.LANGUAGE_LOCALIZATIONS happened to resolve to
# from the real environment (core/config.py: os.getenv("LANGUAGE_LOCALIZATIONS",
# "tr")) instead of pinning it themselves - so they only ever proved "the
# default equals settings.LANGUAGE_LOCALIZATIONS" on a machine whose .env
# (or absence of one) happens to default that setting to exactly ["tr"].
# On a real deployment whose .env sets LANGUAGE_LOCALIZATIONS to anything
# else (empty, a different language, a different order), these tests would
# fail with exactly the symptoms reported: report.passed=True with no
# Turkish requirement, and a KeyError on "script_tr_exists" because that
# check was never registered for a localization list that never contained
# "tr" - NOT because youtube/quality_check.py's _qc_languages() stopped
# reading settings.LANGUAGE_LOCALIZATIONS for the None case (it still does,
# unchanged - see that function's own docstring). The fix is here, in the
# tests: pin settings.LANGUAGE_LOCALIZATIONS explicitly (the same pattern
# test_qc_languages_helper_mirrors_pipeline_localization_languages below
# already uses), so "default behavior" is asserted deterministically
# against a known config instead of silently depending on whatever real
# .env happens to be loaded wherever pytest runs.

def test_default_multilingual_qc_still_requires_turkish_artifacts_when_missing(isolated_storage, monkeypatch):
    """Default (no `languages` override, matching every caller before the
    185c2ba fix and the standalone `python -m youtube.quality_check` CLI)
    behavior is unchanged: for a deployment configured with Turkish as a
    localization, Turkish artifacts are still mandatory and their absence
    still fails QC exactly as before - pinned explicitly (see the
    AMBIENT-CONFIG TEST FIX note above), not inferred from the real .env."""
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])
    paths, _ = ps.create_project("Default Multilingual Missing TR Topic")
    _write_common_artifacts(paths, monkeypatch, include_tr=False)

    report = qc.run_quality_check(paths)  # languages omitted - the pre-fix default path

    assert report.passed is False
    assert report.checks["script_tr_exists"]["passed"] is False
    assert report.checks["narration_tr_exists"]["passed"] is False
    assert report.checks["narration_tr_valid"]["passed"] is False
    assert report.checks["metadata_tr_exists"]["passed"] is False


def test_default_multilingual_qc_passes_when_turkish_artifacts_present(isolated_storage, monkeypatch):
    """Companion to test_all_checks_pass_for_a_fully_correct_video above -
    confirms passing `languages=None` explicitly is identical to omitting
    it (both mean "use settings.LANGUAGE_LOCALIZATIONS", unchanged) -
    pinned explicitly (see the AMBIENT-CONFIG TEST FIX note above)."""
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr"])
    paths, _ = ps.create_project("Default Multilingual With TR Topic")
    _write_common_artifacts(paths, monkeypatch, include_tr=True)

    report_omitted = qc.run_quality_check(paths)
    report_explicit_none = qc.run_quality_check(paths, languages=None)

    assert report_omitted.passed is True, report_omitted.errors
    assert report_explicit_none.passed is True, report_explicit_none.errors
    assert report_omitted.checks["script_tr_exists"]["passed"] is True
    assert report_explicit_none.checks["script_tr_exists"]["passed"] is True


def test_explicit_multilingual_languages_override_requires_turkish_artifacts(isolated_storage, monkeypatch):
    """Case B (explicit multilingual caller): languages=["en", "tr"] must
    require Turkish artifacts exactly like the default does - and must do
    so regardless of settings.LANGUAGE_LOCALIZATIONS, since an explicit
    override always wins over the ambient config (see _qc_languages())."""
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["de"])  # deliberately NOT "tr" - proves the override, not the ambient default, drives this
    paths, _ = ps.create_project("Explicit EN TR Override Topic")
    _write_common_artifacts(paths, monkeypatch, include_tr=False)

    missing_report = qc.run_quality_check(paths, languages=["en", "tr"])
    assert missing_report.passed is False
    assert missing_report.checks["script_tr_exists"]["passed"] is False

    _write_common_artifacts(paths, monkeypatch, include_tr=True)
    present_report = qc.run_quality_check(paths, languages=["en", "tr"])
    assert present_report.passed is True, present_report.errors
    assert present_report.checks["script_tr_exists"]["passed"] is True


def test_qc_languages_helper_mirrors_pipeline_localization_languages(monkeypatch):
    """_qc_languages() must agree with pipeline._localization_languages()
    on every input, since QC and the stage plan need to always check/
    produce the exact same set of languages for a given run."""
    monkeypatch.setattr(settings, "LANGUAGE_MASTER", "en")
    monkeypatch.setattr(settings, "LANGUAGE_LOCALIZATIONS", ["tr", "de"])

    # default (None): master + every configured localization
    assert qc._qc_languages(None) == ["en", "tr", "de"]
    # explicit override (e.g. --english-only): master only
    assert qc._qc_languages([]) == ["en"]
    # explicit subset override
    assert qc._qc_languages(["de"]) == ["en", "de"]


def test_run_and_enforce_passes_languages_through_to_run_quality_check(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Run And Enforce Languages Topic")
    _write_common_artifacts(paths, monkeypatch, include_tr=False)

    report = qc.run_and_enforce(paths, languages=[])

    assert report.passed is True, report.errors


def test_pipeline_english_only_run_passes_qc_without_turkish_artifacts(isolated_storage, monkeypatch):
    """Direct regression test for the reported production bug, exercised
    through the real pipeline entrypoint (not just the qc module directly):
    a real --english-only pipeline run - research/script/scenes/images/
    TTS/assembly/thumbnail/metadata all genuinely succeeding, producing
    only English artifacts - must reach PENDING_APPROVAL, not fail QC over
    Turkish artifacts it never asked for."""
    import pipeline as pipeline_module
    from PIL import Image
    from core.state import PipelineState
    from youtube import researcher, script_generator, scene_planner, image_gen, tts, assemble, thumbnail, metadata

    paths, _ = ps.create_project("English Only Pipeline QC Regression Topic")

    def fake_research(p, topic, category=None):
        ps.save_json(p.research_json, {"sources": [{"url": "https://example.com"}]})
    monkeypatch.setattr(researcher, "run_and_save", fake_research)

    def fake_script(p):
        ps.save_json(p.script_en_json, {"title": "t"})
    monkeypatch.setattr(script_generator, "run_and_save", fake_script)

    def fake_scenes(p):
        ps.save_json(p.scenes_json, [{"scene_id": 1}])
    monkeypatch.setattr(scene_planner, "run_and_save_v2", fake_scenes)

    def fake_images(p):
        p.scene_image(1).write_bytes(b"fake")
    monkeypatch.setattr(image_gen, "run_and_save", fake_images)

    def fake_tts(p, language="en"):
        p.narration_mp3_for(language).parent.mkdir(parents=True, exist_ok=True)
        p.narration_mp3_for(language).write_bytes(b"fake")
    monkeypatch.setattr(tts, "run_and_save", fake_tts)

    def fake_assemble(p):
        p.video_mp4.write_bytes(b"fake")
    monkeypatch.setattr(assemble, "run_and_save", fake_assemble)

    def fake_thumbnail(p):
        Image.new("RGB", (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), (0, 0, 0)).save(p.thumbnail_png, format="PNG")
    monkeypatch.setattr(thumbnail, "run_and_save", fake_thumbnail)

    def fake_metadata(p, category=None, language="en"):
        ps.save_json(p.metadata_json_for(language), {"title": "t"})
    monkeypatch.setattr(metadata, "run_and_save", fake_metadata)

    video_probe = {"format": {"duration": "480.0"}, "streams": [_valid_video_stream(), _valid_audio_stream()]}
    monkeypatch.setattr(qc, "_ffprobe_json", lambda path: video_probe)
    monkeypatch.setattr(pipeline_module, "_notify_telegram", lambda paths, **kwargs: None)

    result_paths = pipeline_module.produce_video(
        "English Only Pipeline QC Regression Topic", video_id=paths.video_id, languages=[],
    )

    assert ps.load_state(result_paths) == PipelineState.PENDING_APPROVAL
    assert not result_paths.script_localized_json("tr").exists()
    assert not result_paths.narration_mp3_for("tr").exists()
    assert not result_paths.metadata_json_for("tr").exists()


def test_pipeline_never_leaves_video_stuck_when_quality_check_itself_crashes(isolated_storage, monkeypatch):
    """The bug this guards against: pipeline.produce_video() previously
    called quality_check.run_and_enforce() with no exception handling at
    all (unlike every other stage), so an unexpected crash inside it -
    e.g. an uncaught bug, not a normal "some checks failed" report -
    would propagate out of produce_video() uncaught: never recorded as
    FAILED, and since every upstream artifact already exists on disk,
    silently un-resumable (every retry would hit the same crash forever)."""
    import pipeline as pipeline_module
    from youtube import researcher, script_generator, scene_planner, image_gen, tts, assemble, thumbnail, metadata

    paths, _ = ps.create_project("QC Crash Must Not Get Lost Topic")

    # Stub every upstream stage as already-done so the stage loop is a
    # no-op and we reach the quality_check call directly. Multilingual
    # (Round 3): this now also includes the default Turkish localization's
    # own script/narration/metadata, or the new localization_tr/
    # narration_tr/metadata_tr stages would run for real first.
    monkeypatch.setattr(pipeline_module, "_images_done", lambda p: True)
    for stage_module, path_attr in [
        (researcher, "research_json"), (script_generator, "script_json"), (scene_planner, "scenes_json"),
        (tts, "narration_mp3"), (assemble, "video_mp4"), (thumbnail, "thumbnail_png"), (metadata, "metadata_json"),
        (script_generator, "script_tr_json"), (tts, "narration_tr_mp3"), (metadata, "metadata_tr_json"),
    ]:
        getattr(paths, path_attr).parent.mkdir(parents=True, exist_ok=True)
        if not getattr(paths, path_attr).exists():
            getattr(paths, path_attr).write_bytes(b"{}") if path_attr.endswith("_json") else getattr(paths, path_attr).write_bytes(b"x")
    # scenes.json needs real content (not just "{}") so pipeline._scenes_synced_for()
    # sees narration_tr already present and skips the scene_sync_tr stage too -
    # otherwise that stage would run for real before reaching the quality_check
    # stub, same reasoning as every other stage stubbed as already-done above.
    ps.save_json(paths.scenes_json, [{"scene_id": 1, "visual_prompt": "p", "narration_en": "x", "narration_tr": "y"}])

    def _crash(paths_arg, languages=None):
        raise ValueError("simulated unexpected bug inside quality_check")

    monkeypatch.setattr(pipeline_module.quality_check, "run_and_enforce", _crash)

    with pytest.raises(pipeline_module.PipelineFailed):
        pipeline_module.produce_video("QC Crash Must Not Get Lost Topic", video_id=paths.video_id)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "quality_check"
    assert "simulated unexpected bug" in data["error"]
