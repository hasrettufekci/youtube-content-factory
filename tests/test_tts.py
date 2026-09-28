"""
TTS narration (spec Round 2 section 7):

    - real provider correctness: text/voice/model/instructions sent
      correctly to OpenAI's /v1/audio/speech
    - mock/real cleanly separated (TTS_PROVIDER=openai|mock)
    - output is a genuine, valid audio file with a retrievable duration
    - a TTS failure must never cause the pipeline stage to be marked
      successful
"""
import wave
from io import BytesIO

import httpx
import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import tts


def _fake_response(status_code=200, content=b"fake-mp3-bytes", text=""):
    class _Resp:
        def __init__(self):
            self.status_code = status_code
            self.content = content
            self.text = text
    return _Resp()


def test_openai_payload_includes_model_voice_text_and_instructions(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

    captured = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _fake_response()

    monkeypatch.setattr(tts.httpx, "Client", _FakeClient)

    result = tts.OpenAITTSProvider().synthesize(
        "This is the narration text.", voice="onyx", instructions="calm documentary tone"
    )

    assert result == b"fake-mp3-bytes"
    assert captured["url"] == tts.OPENAI_TTS_URL
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "gpt-4o-mini-tts"
    assert captured["json"]["voice"] == "onyx"
    assert captured["json"]["input"] == "This is the narration text."
    assert captured["json"]["instructions"] == "calm documentary tone"
    assert captured["json"]["response_format"] == "mp3"
    # the API key itself must never leak into the JSON payload
    assert "test-key" not in str(captured["json"])


def test_openai_omits_instructions_for_legacy_tts_models_that_reject_it(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "OPENAI_TTS_MODEL", "tts-1")

    captured = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return _fake_response()

    monkeypatch.setattr(tts.httpx, "Client", _FakeClient)

    tts.OpenAITTSProvider().synthesize("text", voice="onyx", instructions="calm tone")

    assert "instructions" not in captured["json"]


def test_openai_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(tts, "TTS_MAX_RETRIES", 3)
    monkeypatch.setattr(tts.time, "sleep", lambda *_: None)

    attempts = {"n": 0}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return _fake_response(status_code=503, content=b"", text="service unavailable")
            return _fake_response(status_code=200, content=b"real-audio-bytes")

    monkeypatch.setattr(tts.httpx, "Client", _FakeClient)

    result = tts.OpenAITTSProvider().synthesize("text", voice="onyx")

    assert attempts["n"] == 2
    assert result == b"real-audio-bytes"


def test_openai_does_not_retry_non_retryable_4xx(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(tts, "TTS_MAX_RETRIES", 3)
    monkeypatch.setattr(tts.time, "sleep", lambda *_: None)

    attempts = {"n": 0}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            attempts["n"] += 1
            return _fake_response(status_code=400, content=b"", text="bad voice name")

    monkeypatch.setattr(tts.httpx, "Client", _FakeClient)

    with pytest.raises(RuntimeError, match="400"):
        tts.OpenAITTSProvider().synthesize("text", voice="not-a-real-voice")

    assert attempts["n"] == 1  # no retries burned on a non-retryable 4xx


def test_openai_empty_response_body_raises(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(tts, "TTS_MAX_RETRIES", 1)
    monkeypatch.setattr(tts.time, "sleep", lambda *_: None)

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            return _fake_response(status_code=200, content=b"")

    monkeypatch.setattr(tts.httpx, "Client", _FakeClient)

    with pytest.raises(RuntimeError, match="empty response"):
        tts.OpenAITTSProvider().synthesize("text", voice="onyx")


def test_mock_provider_produces_a_valid_wav_scaled_to_word_count():
    provider = tts.MockTTSProvider()
    short_audio = provider.synthesize("just a few words here", voice="")
    long_audio = provider.synthesize(" ".join(["word"] * 300), voice="")

    with wave.open(BytesIO(short_audio), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        short_duration = wf.getnframes() / wf.getframerate()
    with wave.open(BytesIO(long_audio), "rb") as wf:
        long_duration = wf.getnframes() / wf.getframerate()

    assert short_duration >= tts.MockTTSProvider.MIN_SECONDS
    assert long_duration > short_duration  # more words -> longer clip, not fixed-length


def test_generate_narration_defaults_to_configured_production_voice_and_instructions(isolated_storage, monkeypatch):
    """Round 3.2.1 TTS production update ("Deep Documentary" A/B test
    result): when the caller doesn't pass an explicit voice/instructions,
    generate_narration() must resolve BOTH from settings (per-language
    override, falling back to the global default) - instructions now
    follows the exact same fallback pattern voice already used."""
    paths, _ = ps.create_project("TTS Default Voice Topic")

    captured = {}

    class _CapturingProvider(tts.TTSProvider):
        def synthesize(self, text, voice, instructions=""):
            captured["voice"] = voice
            captured["instructions"] = instructions
            return b"fake-mp3-bytes"

    monkeypatch.setattr(tts, "get_provider", lambda: _CapturingProvider())

    tts.generate_narration(paths, "Some narration text.", language="en")

    assert captured["voice"] == settings.OPENAI_TTS_VOICE == "echo"
    assert captured["instructions"] == settings.OPENAI_TTS_INSTRUCTIONS
    assert "calm and authoritative" in captured["instructions"].lower()


def test_generate_narration_explicit_voice_and_instructions_override_the_default(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("TTS Explicit Override Topic")

    captured = {}

    class _CapturingProvider(tts.TTSProvider):
        def synthesize(self, text, voice, instructions=""):
            captured["voice"] = voice
            captured["instructions"] = instructions
            return b"fake-mp3-bytes"

    monkeypatch.setattr(tts, "get_provider", lambda: _CapturingProvider())

    tts.generate_narration(
        paths, "Some narration text.", language="en", voice="alloy", instructions="a totally different style"
    )

    assert captured["voice"] == "alloy"
    assert captured["instructions"] == "a totally different style"


def test_production_tts_default_is_echo_not_onyx():
    """Regression guard for the Round 3.2.1 voice-character decision: the
    previous default was onyx; production now standardizes on echo with
    the Deep Documentary style instructions (model stays gpt-4o-mini-tts,
    unchanged)."""
    assert settings.OPENAI_TTS_VOICE == "echo"
    assert settings.OPENAI_TTS_MODEL == "gpt-4o-mini-tts"
    assert "calm and authoritative" in settings.OPENAI_TTS_INSTRUCTIONS.lower()
    assert "never exaggerated or theatrical" in settings.OPENAI_TTS_INSTRUCTIONS.lower()


def test_run_and_save_marks_failure_and_reraises_never_masks_a_tts_error(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("TTS Failure Must Not Be Marked Success")
    ps.save_json(paths.script_json, {"full_narration": "some narration text", "scene_breakdown": []})

    class _AlwaysFailsProvider(tts.TTSProvider):
        def synthesize(self, text, voice, instructions=""):
            raise RuntimeError("[TTS] simulated OpenAI outage")

    monkeypatch.setattr(tts, "get_provider", lambda: _AlwaysFailsProvider())

    with pytest.raises(RuntimeError, match="simulated OpenAI outage"):
        tts.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    assert not paths.narration_mp3.exists()
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "narration"


def test_run_and_save_records_failed_state_when_script_json_missing(isolated_storage):
    """Regression: this precondition check used to raise BEFORE
    db.start_run()/the try block, so it never went through
    save_failure() - the video was left stuck with no FAILED state
    recorded and no diagnostic trail in state.json."""
    paths, _ = ps.create_project("TTS Missing Script JSON Topic")
    # script.json deliberately not written

    with pytest.raises(RuntimeError, match="no script.json found"):
        tts.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "narration"
