"""
YOUTUBE_CONTENT_FACTORY - central settings.

Independent project. Nothing here knows about any other product/brand -
only generic YouTube documentary production concepts (topics, research,
scripts, scenes, TTS, video assembly, thumbnails, upload).

All secrets are read from environment variables (.env, loaded once below).
Import of this module NEVER raises on a missing key - each provider checks
for the specific key(s) it needs at call time (see `require()`), so
`import core.config` / `import youtube.*` always works (tests, --help,
`python -m youtube.script_generator` on an already-researched topic, etc.)
even when most keys are unset. A CLI entrypoint that truly needs a key
will fail fast and loudly at the moment it is used, per spec section 27/40.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    return float(value)


class Settings:
    # --- LLM (Anthropic Claude) - research / script / scoring / metadata ---
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    CLAUDE_MAX_RETRIES: int = int(os.getenv("CLAUDE_MAX_RETRIES", "3"))
    # Per-request httpx timeout (seconds) for Claude API calls. Raised from the
    # old hard-coded 120s default to 240s: long classification calls (e.g.
    # Scene Planner V2's single-shot whole-script classification, max_tokens
    # up to 14000) can legitimately take longer than 120s to generate, which
    # was surfacing as spurious httpx read timeouts rather than a real
    # failure. Override via .env if a specific deployment needs a different
    # value; retry count/backoff (CLAUDE_MAX_RETRIES) is unrelated and unchanged.
    CLAUDE_TIMEOUT_SECONDS: float = _float("CLAUDE_TIMEOUT_SECONDS", 240.0)
    CLAUDE_WEB_SEARCH_ENABLED: bool = _bool("CLAUDE_WEB_SEARCH_ENABLED", True)
    # Research stage robustness (production incident: a real research call
    # for "The Ocean Sound Scientists Still Can't Explain" was cut off by
    # Claude's own max_tokens limit mid-JSON - the old hard-coded 4000 was
    # too low for a real web-searched topic's full summary+facts+timeline+
    # sources+uncertainties+search_notes payload (observed truncation at
    # ~14500 chars, right around where a 4000-token budget runs out).
    # Raised to a more generous default, and research now retries (a fresh
    # request each time, never parsing the truncated text) up to this many
    # attempts before giving up loudly. See youtube/researcher.py.
    CLAUDE_RESEARCH_MAX_TOKENS: int = int(os.getenv("CLAUDE_RESEARCH_MAX_TOKENS", "8000"))
    CLAUDE_RESEARCH_MAX_ATTEMPTS: int = int(os.getenv("CLAUDE_RESEARCH_MAX_ATTEMPTS", "3"))

    # --- Image generation (fal.ai) ---
    FAL_KEY: str = os.getenv("FAL_KEY", "")
    FAL_IMAGE_MODEL: str = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/dev")
    FAL_MAX_RETRIES: int = int(os.getenv("FAL_MAX_RETRIES", "3"))

    # --- TTS narration (OpenAI) ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    # Standard production voice/style (decided after an A/B test against the
    # previous "onyx" default - see OPENAI_TTS_INSTRUCTIONS below for the
    # matching "Deep Documentary" style guidance sent alongside it).
    OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "echo")
    # Style/delivery guidance sent as the OpenAI TTS "instructions" field
    # (gpt-4o-mini-tts and newer only - youtube/tts.py already omits it for
    # tts-1/tts-1-hd, which reject it). "Deep Documentary": calm, authoritative,
    # deliberately paced, restrained - never theatrical.
    OPENAI_TTS_INSTRUCTIONS: str = os.getenv(
        "OPENAI_TTS_INSTRUCTIONS",
        "Deep, warm, mature documentary narration. Calm and authoritative. "
        "Slow, deliberate pacing with natural pauses. Clear articulation, "
        "restrained emotion, cinematic atmosphere. The voice should feel "
        "intelligent, mysterious and trustworthy, never exaggerated or "
        "theatrical.",
    )

    # --- Telegram approval bot ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- YouTube Data API upload ---
    YOUTUBE_CLIENT_ID: str = os.getenv("YOUTUBE_CLIENT_ID", "")
    YOUTUBE_CLIENT_SECRET: str = os.getenv("YOUTUBE_CLIENT_SECRET", "")
    YOUTUBE_REFRESH_TOKEN: str = os.getenv("YOUTUBE_REFRESH_TOKEN", "")
    YOUTUBE_UPLOAD_PRIVACY: str = os.getenv("YOUTUBE_UPLOAD_PRIVACY", "private")  # private|unlisted|public
    YOUTUBE_CATEGORY_ID: str = os.getenv("YOUTUBE_CATEGORY_ID", "27")  # 27 = Education

    # --- Public base URL (optional; only needed if media is ever served over HTTP) ---
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "")

    # --- Storage / paths ---
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", str(PROJECT_ROOT / "storage")))
    MEDIA_MUSIC_DIR: Path = Path(os.getenv("MEDIA_MUSIC_DIR", str(PROJECT_ROOT / "storage" / "music")))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
    DB_PATH: Path = Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "storage" / "factory.db")))
    PROJECTS_DIR: Path = PROJECT_ROOT / "projects"

    # --- Video production defaults (can be overridden per project config) ---
    DEFAULT_PROJECT_ID: str = os.getenv("DEFAULT_PROJECT_ID", "mystery_youtube")
    VIDEO_WIDTH: int = int(os.getenv("VIDEO_WIDTH", "1920"))
    VIDEO_HEIGHT: int = int(os.getenv("VIDEO_HEIGHT", "1080"))
    VIDEO_FPS: int = int(os.getenv("VIDEO_FPS", "30"))
    THUMBNAIL_WIDTH: int = int(os.getenv("THUMBNAIL_WIDTH", "1280"))
    THUMBNAIL_HEIGHT: int = int(os.getenv("THUMBNAIL_HEIGHT", "720"))

    # --- Scene planning (spec section 15/37 - "explainable and config-driven,
    # not a blind hard-coded scene count"). An 8-10 minute video targets
    # 25-40 scenes; WORDS_PER_SCENE_TARGET is the words-of-narration-per-scene
    # that produces roughly that many scenes for the SCRIPT_TARGET word count
    # (1200-1600 words / ~45 words-per-scene ~= 27-36 scenes). Override any of
    # these via .env for a different pacing (e.g. a 12-20min extended format).
    SCENE_MIN_COUNT: int = int(os.getenv("SCENE_MIN_COUNT", "25"))
    SCENE_MAX_COUNT: int = int(os.getenv("SCENE_MAX_COUNT", "40"))
    SCENE_WORDS_PER_SCENE_TARGET: int = int(os.getenv("SCENE_WORDS_PER_SCENE_TARGET", "45"))
    SCENE_WORDS_PER_SECOND: float = float(os.getenv("SCENE_WORDS_PER_SECOND", "2.5"))  # ~150 wpm calm narration pace

    # --- Provider selection: "live" uses real paid APIs, "mock" uses local
    # stand-ins (no network, no cost) so the pipeline is runnable/testable
    # without any API keys at all. See youtube/image_gen.py and youtube/tts.py.
    IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "fal")     # fal|mock
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "openai")      # openai|mock

    # --- Multilingual production (Round 3): ONE research, ONE master
    # script, ONE shared scene plan, ONE shared visual production, and a
    # per-language narration/metadata for the master plus every configured
    # localization. LANGUAGE_MASTER is the canonical/global channel
    # language (English); LANGUAGE_LOCALIZATIONS is a comma-separated list
    # read once at startup. Adding a future language (German, Spanish, ...)
    # is exactly this: one new code in this env var - script_generator.py,
    # scene_planner.py, image_gen.py and assemble.py are untouched, and
    # pipeline.py's stage plan picks it up automatically (see pipeline.py's
    # `_stage_plan`).
    LANGUAGE_MASTER: str = os.getenv("LANGUAGE_MASTER", "en")
    LANGUAGE_LOCALIZATIONS: list[str] = [
        code.strip() for code in os.getenv("LANGUAGE_LOCALIZATIONS", "tr").split(",") if code.strip()
    ]

    # --- Daily Topic Discovery / Telegram Topic Selector (Phase A + B) ---
    # Purely additive: none of these are read by the existing production
    # pipeline, topic_scout.py, or scheduler.run_once()/run_daily() - see
    # youtube/topic_discovery.py and scheduler.py's new --discover-once/
    # --discover-daily flags.
    DAILY_TOPIC_SCOUT_ENABLED: bool = _bool("DAILY_TOPIC_SCOUT_ENABLED", True)
    DAILY_TOPIC_SCOUT_TIME: str = os.getenv("DAILY_TOPIC_SCOUT_TIME", "09:00")  # HH:MM, local to the timezone below
    DAILY_TOPIC_SCOUT_TIMEZONE: str = os.getenv("DAILY_TOPIC_SCOUT_TIMEZONE", "Europe/Istanbul")
    # How many final candidates are presented in Telegram (spec: exactly 5
    # under normal operation, but configurable rather than hard-coded).
    TOPIC_DISCOVERY_CANDIDATE_COUNT: int = int(os.getenv("TOPIC_DISCOVERY_CANDIDATE_COUNT", "5"))
    # How many RAW candidates to ask Claude for before dedup/filtering/
    # scoring narrows them down to TOPIC_DISCOVERY_CANDIDATE_COUNT - deliberately
    # larger so duplicates/already-produced/invalid topics can be discarded
    # while still reaching the target count under normal operation.
    TOPIC_DISCOVERY_RAW_COUNT: int = int(os.getenv("TOPIC_DISCOVERY_RAW_COUNT", "8"))
    # 2026-09: production hit ClaudeTruncatedResponseError (stop_reason=max_tokens)
    # at the old default of 4000 - root cause was verbose LLM output (bilingual
    # prose fields, an unused short_description, multi-source citations), fixed
    # primarily by making _DISCOVERY_SYSTEM_PROMPT enforce hard per-field word
    # limits and dropping short_description/multi-source entirely (see
    # youtube/topic_discovery.py's _DISCOVERY_SYSTEM_PROMPT). This default is
    # raised modestly alongside that prompt fix, as a safety margin only - the
    # concise 8-candidate payload is estimated at ~2500-3000 tokens, so 6000
    # leaves real headroom without relying on a bigger budget to paper over
    # verbose output.
    TOPIC_DISCOVERY_MAX_TOKENS: int = int(os.getenv("TOPIC_DISCOVERY_MAX_TOKENS", "6000"))
    TOPIC_DISCOVERY_MAX_ATTEMPTS: int = int(os.getenv("TOPIC_DISCOVERY_MAX_ATTEMPTS", "3"))
    # Off by default: the existing topic_scorer.score_topic(use_llm=...)
    # heuristic backend is deterministic and free - matches topic_scout.py's
    # own default (use_llm_scoring=False). Set true to score discovered
    # candidates with Claude instead.
    TOPIC_DISCOVERY_USE_LLM_SCORING: bool = _bool("TOPIC_DISCOVERY_USE_LLM_SCORING", False)

    # --- Smart Publish Scheduler V1 (AI FOUND THIS) ---
    # Purely additive/opt-in: when SMART_PUBLISH_ENABLED is false (the
    # library-level default, though the shipped .env.example turns it on),
    # production behavior is 100% unchanged - APPROVE -> immediate private
    # upload -> thumbnail -> Telegram update, exactly as before this
    # feature existed. See youtube/smart_publish.py for the decision engine
    # and youtube/telegram_approval.py's _upload_to_youtube()/
    # process_approval() for where this is consulted.
    SMART_PUBLISH_ENABLED: bool = _bool("SMART_PUBLISH_ENABLED", False)
    SMART_PUBLISH_TIMEZONE: str = os.getenv("SMART_PUBLISH_TIMEZONE", "Europe/Istanbul")
    SMART_PUBLISH_DEFAULT_HOUR: int = int(os.getenv("SMART_PUBLISH_DEFAULT_HOUR", "20"))
    SMART_PUBLISH_DEFAULT_MINUTE: int = int(os.getenv("SMART_PUBLISH_DEFAULT_MINUTE", "0"))
    # Never schedule sooner than this from "now" - gives the upload itself
    # (which can take minutes for a large file) headroom to finish well
    # before the chosen publish time.
    SMART_PUBLISH_MIN_DELAY_MINUTES: int = int(os.getenv("SMART_PUBLISH_MIN_DELAY_MINUTES", "60"))
    # How far into the future an audience-timing candidate slot may be
    # searched for before giving up on it and falling back to the
    # deterministic default hour (V1 has no real analytics provider yet, so
    # in practice this only bounds a future analytics-driven slot search -
    # see calculate_publish_time()).
    SMART_PUBLISH_LOOKAHEAD_DAYS: int = int(os.getenv("SMART_PUBLISH_LOOKAHEAD_DAYS", "7"))
    SMART_PUBLISH_MODE: str = os.getenv("SMART_PUBLISH_MODE", "auto")  # auto|manual (manual: reserved for a future UI override)
    SMART_PUBLISH_REQUIRE_APPROVAL: bool = _bool("SMART_PUBLISH_REQUIRE_APPROVAL", True)
    # V1 never auto-publishes public immediately even with Smart Publish
    # enabled - this flag is a reserved, explicit escape hatch for a future
    # "publish right now" override; nothing in V1 sets it, and honoring it
    # is intentionally NOT implemented yet (spec: never fabricate support
    # for a behavior that isn't real).
    SMART_PUBLISH_ALLOW_IMMEDIATE: bool = _bool("SMART_PUBLISH_ALLOW_IMMEDIATE", False)

    # --- YouTube Analytics Timing V1 (AI FOUND THIS) ---
    # Purely additive on top of Smart Publish V1: when disabled (the
    # library default), zero YouTube Analytics API calls are made and
    # calculate_publish_time() behaves exactly as it did before this
    # feature existed (bootstrap-or-explicit-analytics-provider only).
    YOUTUBE_ANALYTICS_ENABLED: bool = _bool("YOUTUBE_ANALYTICS_ENABLED", False)
    # Real YouTube Analytics API v2 has no hour-of-day dimension (verified
    # against https://developers.google.com/youtube/analytics/dimensions -
    # only "day"/"month" exist), so "direct" audience timing is structurally
    # unavailable via the public API. This flag is a documented, honest
    # off switch for that fact - it is never set True by this codebase, and
    # exists only so a future API change (or a private/undocumented
    # capability) has a place to be enabled without a code change to the
    # provider priority order in analytics_timing.py.
    YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE: bool = _bool(
        "YOUTUBE_ANALYTICS_USE_PERFORMANCE_INFERENCE", True
    )
    YOUTUBE_ANALYTICS_CACHE_TTL_HOURS: int = int(os.getenv("YOUTUBE_ANALYTICS_CACHE_TTL_HOURS", "24"))
    YOUTUBE_ANALYTICS_LOOKBACK_DAYS: int = int(os.getenv("YOUTUBE_ANALYTICS_LOOKBACK_DAYS", "90"))
    # Minimum number of published videos with usable stats before the
    # performance-inference provider will report ANY slots at all -
    # below this, it reports unavailable and Smart Publish falls back to
    # bootstrap rather than trust a too-small sample.
    YOUTUBE_ANALYTICS_MIN_VIDEOS: int = int(os.getenv("YOUTUBE_ANALYTICS_MIN_VIDEOS", "5"))
    YOUTUBE_ANALYTICS_TIMEZONE: str = os.getenv("YOUTUBE_ANALYTICS_TIMEZONE", "Europe/Istanbul")
    # confidence_for_video_count() tiers (distinct from smart_publish.py's
    # own confidence_for_sample_size(), which is unchanged and keeps its
    # own 10/30 thresholds for a different quantity - see that function's
    # docstring for why the two scales must never be conflated).
    YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW: int = int(os.getenv("YOUTUBE_ANALYTICS_MIN_SAMPLE_LOW", "5"))
    YOUTUBE_ANALYTICS_MIN_SAMPLE_MEDIUM: int = int(os.getenv("YOUTUBE_ANALYTICS_MIN_SAMPLE_MEDIUM", "10"))
    YOUTUBE_ANALYTICS_MIN_SAMPLE_HIGH: int = int(os.getenv("YOUTUBE_ANALYTICS_MIN_SAMPLE_HIGH", "20"))

    # --- Shorts Engine V1 (AI FOUND THIS) ---
    # Purely additive/opt-in: when SHORTS_ENABLED is false (the library
    # default), pipeline.py's stage plan never runs a single Shorts stage,
    # no `shorts` DB row is ever created, and the Telegram approval sent is
    # byte-for-byte the original single-video message - exactly as before
    # this feature existed. See youtube/shorts_engine.py (production),
    # youtube/telegram_approval.py's process_shorts_for_approval() (upload/
    # scheduling), and pipeline.py's _notify_telegram() (package approval).
    SHORTS_ENABLED: bool = _bool("SHORTS_ENABLED", False)
    SHORTS_PER_LONG: int = int(os.getenv("SHORTS_PER_LONG", "3"))
    SHORTS_TIMEZONE: str = os.getenv("SHORTS_TIMEZONE", "Europe/Istanbul")
    # Target vertical (9:16) render resolution - independent of
    # VIDEO_WIDTH/VIDEO_HEIGHT above (the Long's own 16:9 resolution, never
    # changed by this feature).
    SHORTS_WIDTH: int = int(os.getenv("SHORTS_WIDTH", "1080"))
    SHORTS_HEIGHT: int = int(os.getenv("SHORTS_HEIGHT", "1920"))
    # Target narration length band (spec: "~30-50 seconds, not forced exact").
    # This is the HARD QC GATE band - youtube/shorts_qc.py fails any Short
    # whose real (ffprobe-measured) MP4 duration falls outside it.
    SHORTS_TARGET_SECONDS_MIN: int = int(os.getenv("SHORTS_TARGET_SECONDS_MIN", "30"))
    SHORTS_TARGET_SECONDS_MAX: int = int(os.getenv("SHORTS_TARGET_SECONDS_MAX", "50"))
    # Preferred sweet spot WITHIN the hard band above (spec: "Tercihen:
    # 40-48 saniye") - youtube/shorts_planner.py's condensation prompt aims
    # for this narrower range so a natural, non-truncated script lands
    # comfortably inside the hard gate rather than right at its edge.
    SHORTS_PREFERRED_SECONDS_MIN: int = int(os.getenv("SHORTS_PREFERRED_SECONDS_MIN", "40"))
    SHORTS_PREFERRED_SECONDS_MAX: int = int(os.getenv("SHORTS_PREFERRED_SECONDS_MAX", "48"))
    # DURATION HARD-GATE PATCH (2026-09): a real production Short with 122
    # words of narration came out at 55.2s of real TTS audio - about
    # 2.21 words/sec, noticeably slower than the 2.5 words/sec figure used
    # elsewhere in this codebase (scene_planner.py's SCENE_WORDS_PER_SECOND,
    # tts.py's MOCK_WORDS_PER_SECOND) for scene-timing estimates and the
    # mock TTS tone's synthetic duration. This is a SEPARATE, more
    # conservative constant used only by shorts_planner.py's own word-budget
    # math, so it can be tuned to real observed Shorts narration pace
    # without touching those unrelated 2.5 wps usages (no unnecessary
    # refactor). Real production narration pace varies by voice/style, so
    # this remains .env-overridable rather than hardcoded into the planner.
    SHORTS_WORDS_PER_SECOND: float = _float("SHORTS_WORDS_PER_SECOND", 2.2)
    # Fixed daily publish slots (spec's own recommended default cadence -
    # configuration, never hard-coded into business logic; see
    # youtube/telegram_approval.py's _short_publish_decision(), which reuses
    # youtube.smart_publish's existing bootstrap-slot helper with these
    # values instead of settings.SMART_PUBLISH_DEFAULT_HOUR/MINUTE).
    SHORT_1_DEFAULT_HOUR: int = int(os.getenv("SHORT_1_DEFAULT_HOUR", "12"))
    SHORT_1_DEFAULT_MINUTE: int = int(os.getenv("SHORT_1_DEFAULT_MINUTE", "0"))
    SHORT_2_DEFAULT_HOUR: int = int(os.getenv("SHORT_2_DEFAULT_HOUR", "16"))
    SHORT_2_DEFAULT_MINUTE: int = int(os.getenv("SHORT_2_DEFAULT_MINUTE", "0"))
    SHORT_3_DEFAULT_HOUR: int = int(os.getenv("SHORT_3_DEFAULT_HOUR", "22"))
    SHORT_3_DEFAULT_MINUTE: int = int(os.getenv("SHORT_3_DEFAULT_MINUTE", "0"))

    # Fallback (hour, minute) used by short_default_hour_minute() below for
    # any short_number beyond the three explicit SHORT_N_DEFAULT_* pairs
    # above (e.g. a future SHORTS_PER_LONG=4) - read once at class-body
    # time like every other default in this file, never hard-coded into
    # business logic (env-overridable via SHORT_DEFAULT_FALLBACK_HOUR/MINUTE).
    SHORT_DEFAULT_FALLBACK_HOUR: int = int(os.getenv("SHORT_DEFAULT_FALLBACK_HOUR", "18"))
    SHORT_DEFAULT_FALLBACK_MINUTE: int = int(os.getenv("SHORT_DEFAULT_FALLBACK_MINUTE", "0"))

    def short_default_hour_minute(self, short_number: int) -> tuple[int, int]:
        """(hour, minute) for SHORT_<n>_DEFAULT_HOUR/MINUTE - generic in
        short_number so SHORTS_PER_LONG can grow past 3 later with only new
        env vars (SHORT_4_DEFAULT_HOUR/MINUTE, ...), never a code change
        here; falls back to SHORT_DEFAULT_FALLBACK_HOUR/MINUTE for any
        short_number without its own explicit pair."""
        default_hour = {1: self.SHORT_1_DEFAULT_HOUR, 2: self.SHORT_2_DEFAULT_HOUR,
                         3: self.SHORT_3_DEFAULT_HOUR}.get(short_number, self.SHORT_DEFAULT_FALLBACK_HOUR)
        default_minute = {1: self.SHORT_1_DEFAULT_MINUTE, 2: self.SHORT_2_DEFAULT_MINUTE,
                           3: self.SHORT_3_DEFAULT_MINUTE}.get(short_number, self.SHORT_DEFAULT_FALLBACK_MINUTE)
        hour = int(os.getenv(f"SHORT_{short_number}_DEFAULT_HOUR", str(default_hour)))
        minute = int(os.getenv(f"SHORT_{short_number}_DEFAULT_MINUTE", str(default_minute)))
        return hour, minute

    def __init__(self) -> None:
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.MEDIA_MUSIC_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    def tts_voice_for(self, language: str) -> str:
        """Per-language TTS voice override (env: TTS_VOICE_<LANG>, e.g.
        TTS_VOICE_TR) - falls back to OPENAI_TTS_VOICE, so a new
        localization language needs one new env var, never a code change
        (spec Round 3 section 8/9)."""
        return os.getenv(f"TTS_VOICE_{language.upper()}", self.OPENAI_TTS_VOICE)

    def tts_instructions_for(self, language: str) -> str:
        """Per-language TTS style/instructions override (env:
        TTS_INSTRUCTIONS_<LANG>) - falls back to OPENAI_TTS_INSTRUCTIONS.
        Mirrors tts_voice_for() exactly, for the same reason: a new
        localization language (or a one-off style override for an existing
        one) needs one new env var, never a code change."""
        return os.getenv(f"TTS_INSTRUCTIONS_{language.upper()}", self.OPENAI_TTS_INSTRUCTIONS)


settings = Settings()


def require(key: str) -> str:
    """Fetch a required secret by attribute name on `settings`, raising a
    clear, actionable error (per spec section 27) if it is missing. Used by
    providers at call time - never at import time."""
    value = getattr(settings, key, "")
    if not value:
        raise RuntimeError(
            f"[CONFIG] Missing required environment variable: {key}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value
