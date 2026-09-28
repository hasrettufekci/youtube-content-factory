"""
pipeline.py CLI argument-mapping tests for --english-only.

Unit-level (monkeypatches pipeline.produce_video itself to record exactly
what it was called with) rather than subprocess-based like
tests/test_cli_help.py, because the point here is asserting the mapped
`languages` value, not just that --help prints cleanly. No real API call,
no real pipeline stage, no real Telegram/YouTube call.
"""
from __future__ import annotations

import types

import pytest

import pipeline as pipeline_module


@pytest.fixture
def recording_produce_video(monkeypatch):
    """Replaces pipeline.produce_video with a fake that records its kwargs
    and returns a minimal stand-in with the two attributes main() prints
    (video_id, video_mp4) - never touches ps.create_project/research/etc."""
    calls = []

    def _fake(topic, **kwargs):
        calls.append({"topic": topic, **kwargs})
        return types.SimpleNamespace(video_id=1, video_mp4="storage/2026/09/fake-slug/video.mp4")

    monkeypatch.setattr(pipeline_module, "produce_video", _fake)
    return calls


# ---------------------------------------------------------------------------
# --english-only is accepted by argparse (no "unrecognized arguments" error)
# ---------------------------------------------------------------------------

def test_english_only_flag_is_accepted_with_topic(recording_produce_video, capsys):
    exit_code = pipeline_module.main(["--topic", "Some New Topic", "--english-only"])
    assert exit_code == 0
    err = capsys.readouterr().err
    assert "unrecognized arguments" not in err


def test_english_only_flag_is_accepted_with_video_id(recording_produce_video, capsys):
    exit_code = pipeline_module.main(["--video-id", "42", "--english-only"])
    assert exit_code == 0
    err = capsys.readouterr().err
    assert "unrecognized arguments" not in err


# ---------------------------------------------------------------------------
# --english-only maps to languages=[]
# ---------------------------------------------------------------------------

def test_english_only_maps_to_empty_languages_list_with_topic(recording_produce_video):
    pipeline_module.main(["--topic", "Some New Topic", "--english-only"])
    assert recording_produce_video[0]["languages"] == []


def test_english_only_maps_to_empty_languages_list_with_video_id(recording_produce_video):
    pipeline_module.main(["--video-id", "42", "--english-only"])
    assert recording_produce_video[0]["languages"] == []


# ---------------------------------------------------------------------------
# Default CLI behavior (no --english-only) is unchanged: languages=None,
# i.e. "use settings.LANGUAGE_LOCALIZATIONS exactly as configured".
# ---------------------------------------------------------------------------

def test_default_behavior_without_flag_passes_languages_none_with_topic(recording_produce_video):
    pipeline_module.main(["--topic", "Some New Topic"])
    call = recording_produce_video[0]
    assert call["languages"] is None
    # every other kwarg this branch has always passed stays exactly as before
    assert call["topic"] == "Some New Topic"
    assert call["force_new"] is False
    assert call["background_music"] is False


def test_default_behavior_without_flag_passes_languages_none_with_video_id(recording_produce_video):
    pipeline_module.main(["--video-id", "42"])
    call = recording_produce_video[0]
    assert call["languages"] is None
    assert call["video_id"] == 42


def test_english_only_defaults_to_false_when_omitted(recording_produce_video):
    pipeline_module.main(["--topic", "Some New Topic"])
    # argparse store_true default: absent flag never raises AttributeError,
    # and never accidentally behaves as if the flag were passed.
    assert recording_produce_video[0]["languages"] is None
