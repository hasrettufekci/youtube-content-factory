# YOUTUBE_CONTENT_FACTORY

Independent, fully automatable pipeline for producing 8-10 minute (extensible to 12-20
minute) faceless YouTube documentary videos: mysteries, unexplained places, history,
science, lost civilizations, and extraordinary human stories.

```
TOPIC -> RESEARCH -> FACT CHECK (built into RESEARCH) -> SCRIPT -> SCENE PLAN ->
AI VISUALS -> TTS NARRATION -> VIDEO ASSEMBLY -> THUMBNAIL -> TELEGRAM APPROVAL ->
YOUTUBE UPLOAD
```

This is a standalone project. It shares no branding, prompts, config, or code paths with
any other product - only generic documentary-production concepts.

## 1. Setup

```bash
python3 --version       # 3.10+
ffmpeg -version         # required for video assembly (and ffprobe, which ships with it)
pip install -r requirements.txt --break-system-packages   # drop the flag if using a venv
cp .env.example .env
```

You do **not** need every credential below before you can run anything - see "Your first
test" just below for the minimum to get a real video end to end. Fill in the rest as you
turn on each real provider:

- **`ANTHROPIC_API_KEY`** (required for every mode, mock or production - research/script/
  scene-prompt generation always calls the real Claude API; there is no offline substitute
  for that, since it's the part that actually has to be "smart"). Get one at
  https://console.anthropic.com/settings/keys.
- **`FAL_KEY`** (only needed once `IMAGE_PROVIDER=fal`, i.e. real scene images). Get one at
  https://fal.ai/dashboard/keys.
- **`OPENAI_API_KEY`** (only needed once `TTS_PROVIDER=openai`, i.e. real narration).
  Get one at https://platform.openai.com/api-keys.
- **`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`** (only needed for the Telegram approval
  step - you can approve/reject via the CLI instead, see section 3, and skip Telegram
  entirely). To set one up: message **@BotFather** on Telegram, run `/newbot`, and copy the
  token it gives you into `TELEGRAM_BOT_TOKEN`. Then message your new bot once (anything),
  and fetch `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser - your numeric
  `chat.id` is in the JSON response; put that in `TELEGRAM_CHAT_ID`.
- **`YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN`** (only needed
  to actually upload - everything before that works without them). See section 7 for the
  full OAuth setup (Google Cloud Console steps, required test-user step, scopes), how to
  verify which channel you're authenticated as without uploading anything
  (`--check-channel`), and how to route uploads to more than one YouTube channel without
  duplicating this project.

## 2. Your first test (no fal.ai/OpenAI/Telegram/YouTube account needed)

With just `ANTHROPIC_API_KEY` set in `.env`, and these two lines added (or left at their
`.env.example` defaults) to skip the paid image/voice providers:
```
IMAGE_PROVIDER=mock
TTS_PROVIDER=mock
```
run:
```bash
python pipeline.py --topic "The Dyatlov Pass Incident" --category "Mysteries"
```
This does real research + a real script via Claude, then a real ffmpeg assembly using
placeholder scene images and a placeholder narration tone - so you get a genuine
1920x1080 H.264/AAC MP4 end to end, for the cost of one Claude call, with no other
account required. It prints the new `video_id` and stops at **pending approval**
(nothing is ever uploaded automatically - see section 3 below for approving it). Swap
`IMAGE_PROVIDER`/`TTS_PROVIDER` to `fal`/`openai` (and set `FAL_KEY`/`OPENAI_API_KEY`)
whenever you want real visuals/narration instead of placeholders - see section 5 for the
full mock-vs-production picture.

## 3. Produce a video

```bash
python pipeline.py --topic "The Dyatlov Pass Incident" --category "Mysteries"
```

This runs research -> script -> scenes -> images -> narration -> video -> thumbnail ->
metadata -> quality check -> Telegram approval message, and stops. Nothing is uploaded
until a human taps **Approve** in Telegram (or runs `--approve` below).

Resume a failed/interrupted run (only the stage that failed is re-run - everything
already on disk is kept; see "Resumability" in section 6):
```bash
python pipeline.py --video-id 3
```

Approve / reject / regenerate without Telegram (useful for testing, or scripting, or if
you never configured `TELEGRAM_*` at all):
```bash
python pipeline.py --approve 3
python pipeline.py --reject 3
python pipeline.py --regenerate 3
```

Listen for Telegram Approve/Reject button presses (the production pipeline's own approval
step - Telegram Approval V1):
```bash
python -m youtube.telegram_approval_listener
```
This is the **only** listener `pipeline.py` itself talks to - it's what has to be running
for the Approve/Reject buttons on a pipeline-produced video's approval message to do
anything. It refuses to start a second time against the same storage root (a clear
`FAILED: ... already holds the lock` message, not a crash) - Telegram's Bot API allows only
one open `getUpdates` connection per bot token, so a second instance would otherwise
silently fight the first one and both would start failing with `409 Conflict`.

> **Legacy / deprecated - do not run together with Telegram Approval V1:**
> ```bash
> python telegram_bot.py --listen
> ```
> This is the older, `video_id`-based Approve/Reject/Regenerate/Preview listener that
> predates Telegram Approval V1 above. It still works on its own, but it polls the exact
> same bot token with no coordination with the listener above - running both at the same
> time is exactly what produces the `409 Conflict` errors described above, since Telegram
> rejects a second simultaneous `getUpdates` connection for one token. Use one or the
> other, never both. The **Preview** button (legacy listener only) sends the actual video
> file inline when it's small enough for Telegram's ~50MB upload limit; for a larger video
> it needs `PUBLIC_BASE_URL` set in `.env` to a URL that already serves your `storage/`
> folder publicly (e.g. nginx/Caddy/a static host pointed at it - this project doesn't run
> that server for you) - it will never send a local filesystem path. Without
> `PUBLIC_BASE_URL`, an oversized video just gets a clear "too large, here's how to approve
> from the CLI instead" message.

## 3.5 Daily Topic Discovery & Telegram Topic Selection

A layer on top of the existing pipeline (nothing above is changed by this): every day (or
on demand), the system finds 5 current, high-potential documentary topic opportunities,
sends them to Telegram, and waits for a human to pick one before any production starts.

```
daily schedule / "/topics"
        |
youtube/topic_discovery.py: discover (Claude + web search)
        -> filter/dedupe (drop duplicates + already-produced topics)
        -> score (EXISTING youtube.topic_scorer.score_topic - unchanged)
        -> top 5
        |
Telegram: "AI FOUND THIS" message + [1] [2] [3] [4] [5] inline buttons
        |
human taps a button
        |
topic_select:<candidate_id> callback -> pipeline.produce_video(topic, category)
        (the EXISTING, unchanged production entry point - same research -> script ->
         scenes -> images -> TTS -> assembly -> thumbnail -> metadata -> QC ->
         Telegram Approval V1 -> human APPROVE -> YouTube upload chain as section 3)
```

**Selecting a topic starts production - it never publishes.** Publishing still requires a
human tapping **Approve** on the existing Telegram Approval V1 message, exactly as in
section 3 - this layer only adds one new step *before* research: picking *which* topic to
research.

Dispatched from the **same, single** official listener as section 3 - no second polling
process:
```bash
python -m youtube.telegram_approval_listener
```
This listener now also recognizes a `topic_select:<candidate_id>` callback (distinct from
`approval:<approve|reject>:<slug>` - the two never collide) and a plain `/topics` text
message, alongside its original Telegram Approval V1 handling.

**Manual trigger** - run discovery immediately without waiting for the daily schedule:
```
/topics
```
sent as a message to the bot (the listener above must be running), or from the CLI:
```bash
python scheduler.py --discover-once
```
Both call the exact same `youtube.topic_discovery.run_daily_discovery()` - discovery,
scoring, and the Telegram message only, never production.

**Daily scheduler configuration** (`.env`, all optional - these are new, separate from the
existing `scheduler.py --daily-at`/`--once` production scheduler, which is untouched):
```
DAILY_TOPIC_SCOUT_ENABLED=true
DAILY_TOPIC_SCOUT_TIME=09:00
DAILY_TOPIC_SCOUT_TIMEZONE=Europe/Istanbul
TOPIC_DISCOVERY_CANDIDATE_COUNT=5
TOPIC_DISCOVERY_RAW_COUNT=8
```
Run the blocking daily discovery loop:
```bash
python scheduler.py --discover-daily
```
This **only ever runs discovery** at the configured time - it never calls
`pipeline.produce_video()` itself. Production only begins once a human selects a topic in
Telegram.

**How to test manually** (no real Claude/Telegram/YouTube calls needed for most of this):
```bash
pytest tests/test_topic_discovery.py tests/test_scheduler.py tests/test_telegram_approval_listener.py -q
```
To try the real flow end-to-end: start the listener (`python -m youtube.telegram_approval_listener`),
then either send `/topics` to the bot or run `python scheduler.py --discover-once`, and tap
one of the 5 buttons - production starts the same way `pipeline.py --topic "..."` already
does, and everything from research onward is unchanged.

## 3.6 Shorts Engine V1 (opt-in, additive)

Set `SHORTS_ENABLED=true` (default: `false`) to additionally produce **3 Shorts from the
SAME topic/research/script** as every Long video - no second research call, no separate
Shorts topic selection, no second human decision beyond the one APPROVE/REJECT that already
covers the whole package.

**1 Long -> 3 Shorts.** `youtube/shorts_planner.py` maps each Short onto one of the Long
script's own structured sections (`youtube/script_generator.py`'s `hook`/`reveal`/`mystery`
sections - already the strongest-hook / most-surprising / best-twist beats of the story),
then condenses ONLY that section's own text into a ~30-50s narration via one Claude call per
Short - grounded strictly in that section, so a Short can never introduce a fact the Long
script doesn't already state. `youtube/shorts_visual.py` reuses the Long's own scene images,
center-cropped to 9:16, whenever a scene's `visual_type` is safe to crop (anything except
`map`/`wide`); a dedicated vertical image is only generated (via the same
`image_gen.ImageProvider` used for the Long) when it isn't. Narration TTS
(`youtube/tts.py`), video assembly (`youtube/assemble.py`) and the thumbnail composer
(`youtube/thumbnail.py`) are all the exact same functions the Long pipeline already uses,
called with vertical (`SHORTS_WIDTH`/`SHORTS_HEIGHT`, default 1080x1920) overrides - nothing
here is a second implementation of TTS/ffmpeg/thumbnail logic.

**Storage / parent-child relationship.** Each Short is a row in its own `shorts` table
(`storage/shorts.py`, additive - `storage/db.py`'s existing tables are untouched), with
`parent_video_id` pointing at the Long's own `videos.id` and `short_number` (1/2/3) - the DB
always knows "this Short was generated from this Long", without duplicating the Long's row.
Its files live under the Long's own project folder: `storage/<year>/<month>/<slug>/shorts/
short_01.mp4`, `short_01.png`, `short_01_narration.mp3`, etc.

**Per-Short QC, then ONE combined Telegram approval.** Every Short gets its own QC pass
(`youtube/shorts_qc.py`: resolution/aspect ratio, playable audio/video, English-only, no
internal/debug labels, a lexical grounding check against its own source section, and a
**duration hard gate** - see below). If every Short passes, pipeline.py sends **exactly one**
Telegram approval for the whole
package - a single composed image (the Long's thumbnail + all 3 Short thumbnails) with one
APPROVE/REJECT pair, **never** the Long or any Short's video file (thumbnails-only keeps the
notification fast). If any Short fails QC, no package is sent as approve-ready - a plain
text failure notice is sent instead, and the Long's own separate, already-passed QC is
unaffected. See `youtube/telegram_approval.py`'s `build_package_approval_payload()` /
`send_new_package_approval()` / `send_shorts_production_failed_notice()`.

**APPROVE/REJECT semantics.** One APPROVE moves the Long AND all 3 Shorts to
upload/scheduled together; one REJECT rejects all four - never a partial approval, never a
separate approval record per Short (all Shorts share the Long's own `telegram_approvals.id`
via `shorts.approval_id`).

**Scheduling.** Shorts publish at fixed daily slots, independent of Smart Publish/Analytics
Timing (both completely unmodified by this feature) - `SHORT_1_DEFAULT_HOUR`/`MINUTE`,
`SHORT_2_DEFAULT_HOUR`/`MINUTE`, `SHORT_3_DEFAULT_HOUR`/`MINUTE` (`SHORTS_TIMEZONE`, default
`Europe/Istanbul`), reusing `youtube.smart_publish`'s own slot-computation helper rather than
a second scheduler.

**Retry / idempotency.** `youtube.telegram_approval.process_shorts_for_approval(approval_id)`
is independently re-callable at any time: it only (re)attempts Shorts not yet uploaded (by
their own `shorts.status` column, separate from the Long's own approval state machine), so a
transient FAL/TTS/YouTube failure on one Short is retryable without re-uploading an
already-uploaded Short or re-touching the Long. A duplicate Telegram APPROVE tap is a safe
no-op (the Long's own existing `approve()` guard blocks the whole callback before any upload
is attempted a second time).

**Duration hard gate (patch, 2026-09).** A real production Short at 55.2s (configured band:
30-50s) was found to still pass QC, because the only pre-existing duration check verified
just `duration > 0`, never the configured band. `youtube/shorts_qc.py` now checks
`duration_within_target_range` against the real, ffprobe-measured MP4 duration -
`settings.SHORTS_TARGET_SECONDS_MIN`/`SHORTS_TARGET_SECONDS_MAX`, read live, never
hardcoded - and hard-fails QC (blocking the whole package, per the rule above) when it's out
of range. `youtube/shorts_planner.py`'s condensation prompt was separately tightened to aim
for a narrower, configurable preferred sweet spot (`SHORTS_PREFERRED_SECONDS_MIN`/`MAX`,
default 40-48s) using a dedicated, more conservative real-narration-pace estimate
(`SHORTS_WORDS_PER_SECOND`, default 2.2 - independent of scene_planner.py's/tts.py's own 2.5
words/sec, which are unrelated and unchanged) - this only changes what the model is asked to
aim for; the narration text itself is never mechanically truncated, padded, or sped up.

**Try it without any paid API:**
```bash
SHORTS_ENABLED=true IMAGE_PROVIDER=mock TTS_PROVIDER=mock python -m youtube.shorts_demo --video-id 1
python -m youtube.shorts_demo --video-id 1 --produce   # actually builds the 3 Shorts (mock providers = free)
pytest tests/test_shorts_storage.py tests/test_shorts_planner.py tests/test_shorts_visual.py tests/test_shorts_qc.py tests/test_shorts_engine.py tests/test_shorts_demo.py tests/test_shorts_telegram.py tests/test_pipeline_shorts_integration.py -q
```

**Never touches:** the Long-only pipeline when `SHORTS_ENABLED=false` (the default - the
Telegram message sent is byte-for-byte the original single-video one), `core/state.py`'s
`PipelineState` machine, `storage/db.py`'s existing tables, `youtube/smart_publish.py` /
`youtube/analytics_timing.py`, the existing Telegram listener/single-instance lock, or any
already-produced/approved/uploaded video or Short.

## 4. Run one stage standalone (for testing/debugging)

Every stage is independently runnable once its input file exists:
```bash
python -m youtube.researcher --topic "..." --category Mysteries
python -m youtube.script_generator --video-id 1
python -m youtube.scene_planner --video-id 1
python -m youtube.image_gen --video-id 1
python -m youtube.tts --video-id 1
python -m youtube.assemble --video-id 1
python -m youtube.thumbnail --video-id 1
python -m youtube.metadata --video-id 1
python -m youtube.quality_check --video-id 1
python -m youtube.topic_scout --topic "..." --list
```

`--video-id` is printed by `researcher.py`/`pipeline.py`, or look it up with:
```python
from storage import db
for row in db.list_videos(): print(row["id"], row["slug"], row["state"])
```

## 5. Mock vs. production mode

Every paid provider is behind a swappable interface (`youtube/image_gen.py`,
`youtube/tts.py`), selected by one env var each - nothing else in the pipeline changes:

| Env var | `mock` (default-friendly, zero cost) | real value |
|---|---|---|
| `IMAGE_PROVIDER` | `mock` - Pillow-rendered placeholder scene images, clearly watermarked "MOCK IMAGE" so a mock output is never mistaken for a real one | `fal` - real fal.ai-generated 1920x1080 images |
| `TTS_PROVIDER` | `mock` - a pure-Python generated placeholder tone (stdlib `wave` only, no external binary, no network) | `openai` - real OpenAI TTS narration |

Both can be flipped independently, so you can mix (e.g. real images with mock narration)
while iterating. `ANTHROPIC_API_KEY` is **always** required in every mode, mock or
production - research/script/scene-prompt generation always calls the real Claude API,
since there's no offline substitute for that. There is no mock mode for Telegram or
YouTube upload either - those stages just don't run until you invoke them (Telegram) or
approve (YouTube upload), so you can develop everything upstream of them without ever
configuring either.

Run `pytest tests/` for the full test suite, which needs **no API key at all** (every
external call is mocked at the HTTP layer, not just at the provider-interface level like
`IMAGE_PROVIDER=mock`/`TTS_PROVIDER=mock` above).

## 6. Architecture

```
core/            settings (env-driven, no hard-coded secrets), pipeline state machine, logging
youtube/         one file per pipeline stage - researcher, topic_scout, topic_scorer,
                 topic_discovery (Daily Topic Discovery + Telegram Topic Selector, section
                 3.5), script_generator, scene_planner, image_gen, tts, assemble, thumbnail,
                 metadata, quality_check, uploader, llm_client (shared Claude wrapper),
                 telegram_approval + telegram_approval_listener (Telegram Approval V1)
storage/         SQLite (videos/topics/research_sources/pipeline_runs) + per-video
                 project folders under storage/<year>/<month>/<slug>/, plus
                 telegram_approvals.py and topic_batches.py (own tables, same DB file)
projects/        brand/content configs (mystery_youtube.json) - add a new project by
                 adding a new JSON file here, no Python changes needed
tests/           unit tests, fully mocked - no live API calls
pipeline.py      orchestrator / state machine / CLI entrypoint
telegram_bot.py  legacy approval message + button handling (deprecated - see 3.5/7)
scheduler.py     minimal daily-automation loop: production (--daily-at/--once, unchanged)
                 and, separately, Daily Topic Discovery (--discover-daily/--discover-once,
                 section 3.5) - the latter never starts production on its own
```

Each video gets its own folder:
```
storage/2026/09/the-dyatlov-pass-incident/
  research.json    script.json    scenes.json
  scene_001.png ... scene_NNN.png
  narration.mp3    video.mp4    thumbnail.png
  metadata.json    state.json
```

**Resumability**: `pipeline.py --video-id N` derives progress from which of those files
already exist on disk, not just the `state` label in `state.json` - so a failure at any
stage (a flaky fal.ai call, an ffmpeg crash, a killed process) never forces earlier,
often-paid stages (research, script, images already generated) to redo their work.
Producing the same topic twice (without `--force-new`) reuses the same video/folder the
same way. Scene planning bounds (`SCENE_MIN_COUNT`/`SCENE_MAX_COUNT`/
`SCENE_WORDS_PER_SCENE_TARGET`/`SCENE_WORDS_PER_SECOND`) and output dimensions
(`VIDEO_WIDTH`/`HEIGHT`/`FPS`, `THUMBNAIL_WIDTH`/`HEIGHT`) are all `.env`-tunable - see the
"Advanced tuning" section of `.env.example` for the full list and defaults.

## 7. YouTube: OAuth, scopes, and multi-channel setup

The content pipeline (research -> script -> scenes -> visuals -> narration -> video) is
always produced **once**, shared. This section is only about where a finished video gets
**uploaded** - one YouTube channel (the default/legacy setup) or several.

### 7.1 YouTube API setup

In [Google Cloud Console](https://console.cloud.google.com/):
1. Create (or pick) a project.
2. **APIs & Services -> Library** -> enable **YouTube Data API v3**.
3. **APIs & Services -> OAuth consent screen** -> configure it (External is fine for a
   personal/small-team project) and **add your own Google account under "Test users"**
   (see 7.3 - this is required, not optional, while the app is unverified).
4. **APIs & Services -> Credentials** -> **Create Credentials -> OAuth client ID** -> type
   **Desktop app**. Note the client ID and client secret it gives you.

The downloaded "OAuth client" JSON from this step is sometimes saved locally (e.g. as
`youtube/client_secret.json`) just for reference while copying values into `.env` - this
project's code never reads that file directly (see 7.9), but `.gitignore` keeps
`client_secret*.json` (anywhere in the repo) and a `youtube/credentials/` directory out of
git either way, in case you keep it around.

### 7.2 OAuth setup (authorize once, upload forever after)

With `YOUTUBE_CLIENT_ID`/`YOUTUBE_CLIENT_SECRET` (or a channel's own, see 7.6) set in
`.env`, run the one-time interactive consent flow on a machine with a browser:
```bash
python -m youtube.uploader --authorize                     # legacy/default channel
python -m youtube.uploader --authorize --channel english   # a configured channel (7.6)
```
It opens Google's consent screen, then prints a refresh token - put that in `.env` as
`YOUTUBE_REFRESH_TOKEN` (or `YOUTUBE_CHANNEL_ENGLISH_REFRESH_TOKEN`, etc.). Uploads are
fully non-interactive after this.

### 7.3 Test user requirement for OAuth testing

While your Google Cloud OAuth consent screen is in **"Testing"** publishing status (the
default until you submit it for verification), Google will refuse to complete the consent
flow for any Google account that isn't explicitly listed under **OAuth consent screen ->
Audience/Test users**. Add every Google account/channel you intend to authorize (including
each one used in a multi-channel setup) as a test user there before running `--authorize`
for it - otherwise the browser flow fails with an "app hasn't completed verification" /
"access blocked" screen, not a code bug.

### 7.4 Scopes

```python
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
```
`youtube.upload` is what `upload_video()`/`upload_thumbnail()` need. `youtube.readonly` is
what `--check-channel` needs to safely read the authenticated channel's identity
(`channels().list(mine=True)`) **without ever uploading anything** - added in Round 3.2
after a real test showed a token authorized with only `youtube.upload` gets a clear `403
Request had insufficient authentication scopes` from that read call. Only these two,
YouTube-specific scopes are requested - nothing broader.

**A refresh token only ever has the scope(s) it was issued with.** If you authorized before
this change (or are adding a scope later), the existing `YOUTUBE_REFRESH_TOKEN` (or a
channel's own) does **not** retroactively gain `youtube.readonly` - this project never
assumes otherwise. Just re-run `--authorize` (with the same `--channel`, if any) to get a
fresh token covering the current `SCOPES`.

### 7.5 Single-channel configuration (legacy/default)

Unchanged since before Round 3.2 - if you never set `YOUTUBE_CHANNELS`, everything uses:
```
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...
YOUTUBE_REFRESH_TOKEN=...
YOUTUBE_UPLOAD_PRIVACY=private
YOUTUBE_CATEGORY_ID=27
```
Every function in `youtube/uploader.py` (`upload_video`, `run_and_save`, `--check-channel`,
...) defaults to this when called with no channel specified. **You do not need to change
anything to keep using a single channel** - multi-channel config below is entirely opt-in.

### 7.6 Multi-channel configuration

For routing different languages/brands to different YouTube channels **without**
duplicating the project - one factory, one shared pipeline, multiple upload destinations:
```
                 CONTENT FACTORY (shared: research, script, scenes, visuals, narration)
                       |
          +------------+------------+
          |            |            |
       English      Turkish      Spanish
       Channel      Channel      Channel
          |            |            |
       OAuth A       OAuth B       OAuth C
```
Set a comma-separated list of stable internal channel keys, then define each channel's own
env vars (`KEY` uppercased):
```
YOUTUBE_CHANNELS=english,turkish

YOUTUBE_CHANNEL_ENGLISH_CLIENT_ID=...
YOUTUBE_CHANNEL_ENGLISH_CLIENT_SECRET=...
YOUTUBE_CHANNEL_ENGLISH_REFRESH_TOKEN=...
YOUTUBE_CHANNEL_ENGLISH_DISPLAY_NAME=AI Found This
YOUTUBE_CHANNEL_ENGLISH_ENABLED=true
YOUTUBE_CHANNEL_ENGLISH_LANGUAGE=en
YOUTUBE_CHANNEL_ENGLISH_PRIVACY=private
YOUTUBE_CHANNEL_ENGLISH_CATEGORY_ID=27
YOUTUBE_CHANNEL_ENGLISH_EXPECTED_CHANNEL_ID=UCxxxxxxxxxxxxxxxxxxxxxx   # optional, see 7.7

YOUTUBE_CHANNEL_TURKISH_CLIENT_ID=...
YOUTUBE_CHANNEL_TURKISH_CLIENT_SECRET=...
YOUTUBE_CHANNEL_TURKISH_REFRESH_TOKEN=...
YOUTUBE_CHANNEL_TURKISH_LANGUAGE=tr
```
Only `CLIENT_ID`/`CLIENT_SECRET`/`REFRESH_TOKEN` are required per channel; everything else
is optional and falls back to the global `YOUTUBE_UPLOAD_PRIVACY`/`YOUTUBE_CATEGORY_ID` (or
to `"en"` for language) when unset. `DISPLAY_NAME` is for humans only - it is never used to
select or verify a channel (see 7.7). Authorize and use a channel by its key:
```bash
python -m youtube.uploader --authorize --channel english
python -m youtube.uploader --video-id 3 --channel english
```
Or from Python, `upload_video(video_path, metadata, channel_key="english")` /
`run_and_save(paths, channel_key="english")` - both still work with `channel_key` omitted
(-> legacy channel), so no existing caller breaks (see `pipeline.py`'s `handle_approve()`,
which is intentionally unchanged this round).

### 7.7 Channel verification (read-only, never uploads)

Check which channel a token actually authenticates as, without touching anything:
```bash
python -m youtube.uploader --check-channel
python -m youtube.uploader --check-channel --channel english
```
```
Authenticated YouTube channel:
Name: AI FOUND THIS
Channel ID: UCxxxxxxxxxxxxxxxxxxxxxx
Handle: @aifoundthis
Subscribers: 1234
Videos: 56
```
This never calls the upload API - it's `channels().list(mine=True)` only. Channel
**identity** is always the API-returned `channel_id`, never the display name (a display
name can be changed or duplicated by anyone; a channel_id can't). If you set
`YOUTUBE_CHANNEL_<KEY>_EXPECTED_CHANNEL_ID` (get the value once from `--check-channel`'s
own output, then pin it), every upload to that channel re-verifies the authenticated
channel_id against it FIRST and **fails safe - refuses to upload - on any mismatch**,
instead of silently publishing to whichever Google account the token happens to belong to.

### 7.8 Safe private upload testing

`YOUTUBE_UPLOAD_PRIVACY` (global) / `YOUTUBE_CHANNEL_<KEY>_PRIVACY` (per-channel) default to
**`private`** and this project never changes that default - a video is only ever
`unlisted`/`public` if you explicitly set that yourself. For your first real upload test:
run the normal approval flow (`pipeline.py --approve N`, or the Telegram button) with
privacy left at `private`, then check the video in YouTube Studio yourself before ever
setting a channel to `unlisted`/`public`.

### 7.9 Credential security

- **Never commit** `.env`, any `client_secret*.json`, or any refresh/access token -
  `.gitignore` covers `.env`, `client_secret*.json` (anywhere in the repo), `*_token.json`,
  and a `credentials/`/`youtube/credentials/` directory if you use one for local reference
  files.
- Every OAuth secret (`client_secret`, `refresh_token`) is redacted (`***REDACTED***`) out
  of any log line or raised error before it's ever printed or written to `logs/app.log` -
  `--check-channel` and upload failures only ever surface the channel's public
  `channel_id`/`title`, never a secret.
- `--help` on every entrypoint, including `youtube.uploader --help`, works with zero
  environment variables set (see `tests/test_cli_help.py`) - a missing credential is only
  ever an error at the moment something actually tries to authenticate, never at import or
  `--help` time.

### 7.10 Adding another YouTube channel

1. Create a **separate** OAuth 2.0 Client ID for it in Google Cloud Console (7.1) - or reuse
   one client ID across channels if they're on the same Google Cloud project, but each
   channel still needs **its own** `--authorize` run against **its own** Google
   account/channel. Never reuse one refresh token across different channels.
2. Add the account as a test user (7.3) if the consent screen is still in Testing status.
3. Add its key to `YOUTUBE_CHANNELS` and set `YOUTUBE_CHANNEL_<KEY>_CLIENT_ID`/
   `_CLIENT_SECRET` in `.env` (7.6).
4. `python -m youtube.uploader --authorize --channel <key>` -> copy the printed refresh
   token into `YOUTUBE_CHANNEL_<KEY>_REFRESH_TOKEN`.
5. `python -m youtube.uploader --check-channel --channel <key>` -> confirm it's the right
   channel, then optionally copy its `Channel ID` into
   `YOUTUBE_CHANNEL_<KEY>_EXPECTED_CHANNEL_ID` for ongoing safety (7.7).
6. Upload to it with `--channel <key>`, or `channel_key="<key>"` from Python.

## 8. Roadmap (not built yet, per spec)

~~Phase 2: automatic topic discovery~~ - **done**, see section 3.5 (Daily Topic Discovery &
Telegram Topic Selection: `youtube/topic_discovery.py`, `storage/topic_batches.py`).
Phase 3: thumbnail A/B variants via Telegram. Phase 4: localization (`tr`, `es`, `pt`,
`id`) - `script_generator.py`/`tts.py` are already language-parameterizable. Phase 6:
YouTube Analytics feedback into topic scoring. **Phase C (not started):** Shorts generation
(9:16 rendering, Shorts-specific TTS/captions/upload/approval) - out of scope for the Daily
Topic Discovery / Telegram Topic Selector work above, which only prepares topics for the
existing long-form pipeline.
