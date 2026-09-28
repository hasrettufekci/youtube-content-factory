"""
TTS narration (spec section 17; extended for multilingual production in
Round 3 section 6/8).

Provider abstraction so swapping the TTS backend later never touches
pipeline.py or assemble.py - only `TTS_PROVIDER` in .env changes.

Multilingual: `generate_narration()`/`run_and_save()` both take a
`language` argument (default = core.config.settings.LANGUAGE_MASTER, i.e.
English) - the SAME provider/model handles every language, nothing here is
hard-coded to Turkish or to any other specific language. Each language's
audio is synthesized independently from that language's own script (its
own full_narration, which is naturally a different length once spoken),
written to its own narration_<lang>.mp3, and is never assumed to share a
duration with any other language's track - the pipeline never speeds up,
slows down, crops, or stretches one language's audio to match another's.

- OpenAITTSProvider: real OpenAI TTS (gpt-4o-mini-tts by default - natural,
  documentary-appropriate). Voice/instructions come from the project config.
  Retries transient (429/5xx/timeout) failures; a real failure raises and is
  surfaced as a FAILED pipeline stage - it never gets treated as success.
- MockTTSProvider: fully offline, ZERO external dependencies (no API key, no
  network, no extra binary like espeak-ng - stdlib `wave` only). Not real
  speech - a deterministic sine-tone clip sized to roughly the narration's
  expected spoken length - but it's a genuine, valid audio file, so the rest
  of the pipeline (duration sync, ffmpeg mux, QC) can be exercised end to end
  without any provider account or anything beyond a bare Python install.

Standalone test:
    python -m youtube.tts --video-id 1
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import tempfile
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from core.config import settings, require
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps

logger = get_logger("tts")

OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
TTS_MAX_RETRIES = 3

# Mirrors youtube.scene_planner.WORDS_PER_SECOND intentionally (kept as its
# own constant, not imported, so the two stages stay decoupled) - used only
# to size the mock clip to roughly the real narration's expected duration.
MOCK_WORDS_PER_SECOND = 2.5


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, instructions: str = "") -> bytes:
        """Return narration audio bytes (a real audio file - mp3 for the real
        provider, a valid wav for the mock; ffmpeg/ffprobe read either
        regardless of the .mp3 filename they get saved under)."""


class _TTSTransientError(RuntimeError):
    """Internal marker for a retryable TTS failure (HTTP 429/5xx, or an
    unexpectedly empty response body on an otherwise-200 response). Kept
    distinct from the plain RuntimeError raised for non-retryable 4xx
    errors below so the retry loop's `except` can catch only this one -
    without it, a bad-request/bad-voice/auth error would get caught by a
    broad `except RuntimeError` and retried 3 times for no benefit before
    ultimately failing anyway."""


class OpenAITTSProvider(TTSProvider):
    def synthesize(self, text: str, voice: str, instructions: str = "") -> bytes:
        api_key = require("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": settings.OPENAI_TTS_MODEL,
            "voice": voice,
            "input": text,
            "response_format": "mp3",
        }
        if instructions and settings.OPENAI_TTS_MODEL not in ("tts-1", "tts-1-hd"):
            payload["instructions"] = instructions

        last_error: Exception | None = None
        for attempt in range(1, TTS_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(OPENAI_TTS_URL, headers=headers, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _TTSTransientError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    # Non-retryable (bad request, auth, bad voice name, ...) -
                    # fail immediately instead of burning through retries on
                    # an error that will never resolve itself.
                    raise RuntimeError(f"[TTS] OpenAI TTS failed ({resp.status_code}): {resp.text[:500]}")
                if not resp.content:
                    raise _TTSTransientError("[TTS] OpenAI TTS returned an empty response body")
                return resp.content
            except (httpx.TimeoutException, httpx.TransportError, _TTSTransientError) as e:
                last_error = e
                if attempt == TTS_MAX_RETRIES:
                    break
                wait = min(2 ** attempt, 15)
                logger.warning("[TTS] attempt %d/%d failed (%s), retrying in %ss", attempt, TTS_MAX_RETRIES, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"[TTS] OpenAI TTS failed after {TTS_MAX_RETRIES} attempts: {last_error}")


class MockTTSProvider(TTSProvider):
    """Offline stand-in - pure Python stdlib (`wave`), no external binary, no
    network. Generates a low-volume sine tone whose duration is derived from
    the narration's word count, so downstream duration-sync/QC logic gets a
    realistic (if silent-sounding, non-speech) clip to work with. Never used
    for real narration - only when TTS_PROVIDER=mock."""

    SAMPLE_RATE = 22050
    FREQUENCY_HZ = 220.0  # a low, unobtrusive tone
    AMPLITUDE = 0.15       # quiet - this is a placeholder, not content
    MIN_SECONDS = 1.0

    def synthesize(self, text: str, voice: str = "", instructions: str = "") -> bytes:
        word_count = max(len(text.split()), 1)
        duration = max(word_count / MOCK_WORDS_PER_SECOND, self.MIN_SECONDS)
        n_samples = int(duration * self.SAMPLE_RATE)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "mock_narration.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(self.SAMPLE_RATE)
                frames = bytearray()
                for i in range(n_samples):
                    # gentle fade in/out so it's not a harsh click at the edges
                    fade = min(1.0, i / 500, (n_samples - i) / 500)
                    sample = self.AMPLITUDE * fade * math.sin(2 * math.pi * self.FREQUENCY_HZ * i / self.SAMPLE_RATE)
                    frames += struct.pack("<h", int(sample * 32767))
                wav_file.writeframes(bytes(frames))
            return wav_path.read_bytes()


def get_provider() -> TTSProvider:
    if settings.TTS_PROVIDER == "mock":
        return MockTTSProvider()
    return OpenAITTSProvider()


def generate_narration(
    paths: ps.ProjectPaths,
    full_narration: str,
    language: str | None = None,
    voice: str | None = None,
    instructions: str = "",
    dest_path: Path | None = None,
) -> Path:
    """`dest_path` (Shorts Engine V1, additive): when given, narration is
    written there instead of paths.narration_mp3_for(language) - e.g.
    paths.short_narration_path(n) - reusing the exact same provider/voice/
    instructions/retry logic without duplicating this function. None (every
    pre-existing caller) is 100% unchanged behavior."""
    language = language or settings.LANGUAGE_MASTER
    provider = get_provider()
    voice = voice or settings.tts_voice_for(language)
    # Same fallback pattern as voice above: an explicit instructions=
    # argument always wins, otherwise fall back to the configured
    # per-language/default style (settings.OPENAI_TTS_INSTRUCTIONS,
    # overridable per language via TTS_INSTRUCTIONS_<LANG> - see
    # core/config.py's tts_instructions_for()). Production standard as of
    # the "Deep Documentary" A/B test: calm, authoritative, deliberately
    # paced, restrained.
    instructions = instructions or settings.tts_instructions_for(language)
    dest = dest_path or paths.narration_mp3_for(language)
    logger.info("[TTS] synthesizing narration (%d words, language=%s, provider=%s, voice=%s)",
                len(full_narration.split()), language, settings.TTS_PROVIDER, voice)
    audio_bytes = provider.synthesize(full_narration, voice, instructions)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(audio_bytes)
    return dest


def run_and_save(
    paths: ps.ProjectPaths,
    language: str | None = None,
    voice: str | None = None,
    instructions: str = "",
) -> Path:
    # The precondition check (missing script) lives inside the try block
    # below, not before it: every failure here must go through the same
    # save_failure() path as a real synthesis error, or the video is left
    # stuck with no FAILED state recorded and no diagnostic trail.
    language = language or settings.LANGUAGE_MASTER
    is_master = language == settings.LANGUAGE_MASTER
    stage_label = "narration" if is_master else f"narration_{language}"
    run_id = db.start_run(paths.video_id, stage_label)
    try:
        if is_master:
            script = ps.load_json(paths.script_en_json)
            if not script:
                raise RuntimeError(f"[TTS] no script.json found at {paths.script_json} - run script_generator.py first")
        else:
            script_path = paths.script_localized_json(language)
            script = ps.load_json(script_path)
            if not script:
                raise RuntimeError(
                    f"[TTS] no localized script found at {script_path} for language={language!r} - "
                    f"run the localization stage first"
                )

        result = generate_narration(paths, script["full_narration"], language=language, voice=voice, instructions=instructions)
        if is_master:
            # Only the master language's narration gates the pipeline's
            # coarse-grained state machine forward (video assembly is built
            # from the master track + shared visuals) - a localization
            # language's own audio is tracked via db run history and the
            # narration_<lang>.mp3 file itself, not a new PipelineState.
            ps.save_state(paths, PipelineState.AUDIO_GENERATED)
        db.finish_run(run_id, "success")
        return result
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, stage_label, str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate narration audio from script.json (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--language", default=None, help="Defaults to the configured master language (LANGUAGE_MASTER)")
    parser.add_argument("--voice", default=None)
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        result = run_and_save(paths, language=args.language, voice=args.voice)
        print(f"OK - narration saved to {result}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
