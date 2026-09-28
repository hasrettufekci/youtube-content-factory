"""
Daily Topic Discovery + Telegram Topic Selector (Phase A + Phase B -
"YOUTUBE CONTENT FACTORY - DAILY TOPIC DISCOVERY + TELEGRAM TOPIC
SELECTOR").

    EVERY DAY (or on-demand via /topics)
            |
    discover_raw_candidates()   - Claude + web_search, current opportunities
            |
    _filter_and_dedupe()        - drop duplicates / already-produced topics
            |
    _score_candidates()         - EXISTING youtube.topic_scorer, unchanged
            |
    _select_top()                - top N (5 by default)
            |
    run_daily_discovery()        - persists a topic_batches/topic_candidates
            |                       batch, sends ONE Telegram message with
            |                       an inline keyboard
    Telegram: user taps [1]-[5]
            |
    handle_topic_select_callback() - atomic select (storage/topic_batches.py
            |                         try_select_candidate), idempotent
            |
    pipeline.produce_video(topic, category=...)   - THE EXISTING, UNCHANGED
                                                      production entry point

This module does NOT reimplement research, scoring, video production, or
approval - it only adds the discovery -> scoring -> presentation -> selection
layer described above, and its one call into the existing pipeline (see
handle_topic_select_callback() below). topic_scorer.py, pipeline.py, and
youtube/telegram_approval.py's own approve/reject flow are never modified.

CALLBACK NAMESPACE: "topic_select:<candidate_id>" - deliberately distinct
from both telegram_bot.py's own "approve:<video_id>" etc. and Telegram
Approval V1's "approval:<approve|reject>:<slug>" (see
youtube/telegram_approval.py's build_keyboard()), so all three coexist on
the same bot token without ever colliding. Dispatched from the existing
official listener - see youtube/telegram_approval_listener.py's
process_update(), extended (not replaced) to recognize this prefix and a
plain "/topics" text command.

PRODUCTION START IS SYNCHRONOUS (architecture note): handle_topic_select_
callback() calls pipeline.produce_video() directly, in-process, exactly
like scheduler.run_once() already does today - this project deliberately
has no job queue/worker system (see scheduler.py's own docstring: "not a
service/cron system, no Celery/Kubernetes"). This means the listener
process is busy for the full production duration after a topic is
selected, same as it already is whenever scheduler.run_once() runs a
production cycle. This is a conscious continuation of the existing
architecture, not a new limitation introduced here.

SELECTING A TOPIC NEVER PUBLISHES: pipeline.produce_video() ends by
calling its own existing _notify_telegram() (Telegram Approval V1) and
stops there, exactly as it does for every other caller - the existing
research -> ... -> QC -> Telegram approval -> human APPROVE -> YouTube
upload chain is completely unchanged and is not bypassed here.
"""
from __future__ import annotations

import telegram_bot as tb
from core.config import settings
from core.logging_setup import get_logger
from storage import db, project_store as ps, topic_batches as tbatch
from youtube import llm_client, topic_scorer

logger = get_logger("topic_discovery")

CALLBACK_PREFIX = "topic_select:"

_NUMBER_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣",
                 "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

_DISCOVERY_SYSTEM_PROMPT = """You are a daily topic scout for "AI FOUND THIS", a factual YouTube
documentary channel (mysteries, unexplained events, history, science, lost civilizations,
extraordinary human stories, unexplained places).

Your job: use web search to find CURRENT, real content opportunities worth turning into a
long-form documentary - not simply "the most-viewed videos today". Consider: current interest/
trend signal, freshness, curiosity, uniqueness, search potential, documentary potential, visual
potential, factual reliability, competition, title potential, whether it could become a strong
long-form documentary, and relevance to the AI FOUND THIS style.

Never invent a metric or number you cannot verify from a real source you found - describe the
signal qualitatively instead (e.g. "widely discussed on YouTube/news this week" rather than a
fabricated view count). Cite exactly ONE real source URL you actually found per candidate.

The AI FOUND THIS channel itself is ENGLISH-ONLY - title/topic/why_now/category/trend_signal
MUST be in English, since they are used for research and production exactly as given, never
translated. Separately, ALSO provide a Turkish translation of each of title/why_now/category/
trend_signal (as title_tr/why_now_tr/category_tr/trend_signal_tr) - these are shown ONLY in the
Telegram management interface so the channel owner (who reads Turkish) can understand and select
a topic; they are never used for research, script, or video production, and must be faithful
translations, not new/different content.

OUTPUT MUST BE EXTREMELY CONCISE - every field below has a hard word limit. Do not write prose,
do not add background/caveats/extra sentences, do not explain your reasoning. A response that
runs long gets cut off before it is valid JSON, which wastes the whole request - staying under
every limit is more important than sounding polished:
- title / title_tr: at most 12 words each
- why_now / why_now_tr: at most 25 words each
- category / category_tr: at most 4 words each
- trend_signal / trend_signal_tr: at most 20 words each
- sources: exactly ONE item, {"url": "..."} only - no title field, no extra fields
- do NOT include a "short_description" field - it is not used and wastes output budget

Return ONLY a single valid JSON object (no markdown, no code fences, no commentary) with this
exact shape:

{
  "candidates": [
    {
      "title": "documentary-style title, at most 12 words (English - used for production)",
      "title_tr": "Turkish translation of title, at most 12 words (Telegram display only)",
      "topic": "a concise, researchable version of the topic (English - used for production)",
      "why_now": "why this is a good opportunity right now, at most 25 words (English - used for production)",
      "why_now_tr": "Turkish translation of why_now, at most 25 words (Telegram display only)",
      "category": "e.g. Science / Mystery, History - at most 4 words (English - used for production)",
      "category_tr": "Turkish translation of category, at most 4 words (Telegram display only)",
      "trend_signal": "qualitative current-interest signal, at most 20 words, never a fabricated number (English - used for production)",
      "trend_signal_tr": "Turkish translation of trend_signal, at most 20 words (Telegram display only)",
      "sources": [{"url": "https://..."}]
    }
  ]
}

CRITICAL RULES:
- Never fabricate a topic, a source, or a trend metric.
- Every candidate must be something that could genuinely be researched with real sources.
- Do not propose a topic already covered ad nauseam with no fresh angle.
- title/topic/why_now/category/trend_signal must be English - never translate these, they are
  used for real production.
- title_tr/why_now_tr/category_tr/trend_signal_tr must be Turkish translations of their English
  counterparts - never invented content, never left empty when the English counterpart exists.
- Stay within every word limit above for every candidate - a long, uncut response is worthless if
  it gets truncated before the JSON closes.
- Output ONLY the JSON object. No prose outside it."""


class TopicDiscoveryError(RuntimeError):
    pass


# --- Phase A: discovery / filtering / scoring -------------------------------

def discover_raw_candidates(count: int | None = None) -> list[dict]:
    """One Claude + web_search call asking for `count` current candidate
    topics. Retries a FRESH request (never parses partial content) on
    truncation or unparseable JSON, mirroring youtube/researcher.py's own
    _chat_json_research() pattern exactly, up to
    settings.TOPIC_DISCOVERY_MAX_ATTEMPTS."""
    count = count or settings.TOPIC_DISCOVERY_RAW_COUNT
    user_prompt = (
        f"Find {count} current, high-potential documentary topic opportunities for "
        "AI FOUND THIS. Use web search to ground every candidate in something real "
        "and current. Keep every field within the word limits given in the system "
        "prompt - short and concise beats thorough here, since a response that runs "
        "long gets cut off before it can be used."
    )
    attempts = max(1, settings.TOPIC_DISCOVERY_MAX_ATTEMPTS)
    last_error: llm_client.ClaudeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            data, _sources = llm_client.chat_json(
                _DISCOVERY_SYSTEM_PROMPT, user_prompt,
                max_tokens=settings.TOPIC_DISCOVERY_MAX_TOKENS, use_web_search=True,
            )
            candidates = data.get("candidates")
            if not isinstance(candidates, list):
                raise llm_client.ClaudeError("[TOPIC_DISCOVERY] response missing a 'candidates' list")
            return candidates
        except llm_client.ClaudeTruncatedResponseError as e:
            last_error = e
            logger.warning(
                "[TOPIC_DISCOVERY] attempt %d/%d: Claude response truncated - %s",
                attempt, attempts, "retrying with a fresh request" if attempt < attempts else "giving up",
            )
        except llm_client.ClaudeError as e:
            last_error = e
            logger.warning(
                "[TOPIC_DISCOVERY] attempt %d/%d: Claude returned unusable JSON (%s) - %s",
                attempt, attempts, e, "retrying with a fresh request" if attempt < attempts else "giving up",
            )
    raise TopicDiscoveryError(
        f"[TOPIC_DISCOVERY] gave up after {attempts} attempt(s), all truncated or unusable: {last_error}"
    ) from last_error


def _filter_and_dedupe(raw_candidates: list[dict]) -> list[dict]:
    """Deduplicates by normalized topic (storage/project_store.normalize_topic
    - the same normalization used for production idempotency) and rejects
    any candidate whose normalized topic already has a produced video
    (storage/db.find_video_by_normalized_topic - the existing `videos`
    table, unchanged). Never invents replacement topics - a rejected
    candidate is simply dropped (spec section 6).

    Also fills in title_tr/why_now_tr/category_tr/trend_signal_tr with
    their English counterparts when Claude didn't provide a translation
    (defensive only - never crashes the Telegram message over a missing
    translation) - this falls back to displaying English, it never
    fabricates Turkish text."""
    seen_normalized: set[str] = set()
    result: list[dict] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        topic = (raw.get("topic") or raw.get("title") or "").strip()
        title = (raw.get("title") or topic).strip()
        if not topic or not title:
            logger.info("[TOPIC_DISCOVERY] rejecting candidate with no topic/title")
            continue
        normalized = ps.normalize_topic(topic)
        if normalized in seen_normalized:
            logger.info("[TOPIC_DISCOVERY] rejecting duplicate candidate topic=%r", topic)
            continue
        if db.find_video_by_normalized_topic(normalized) is not None:
            logger.info("[TOPIC_DISCOVERY] rejecting already-produced topic=%r", topic)
            continue
        seen_normalized.add(normalized)
        candidate = dict(raw)
        candidate["topic"] = topic
        candidate["title"] = title
        candidate["normalized_topic"] = normalized
        for tr_key, en_key in (
            ("title_tr", "title"), ("why_now_tr", "why_now"),
            ("category_tr", "category"), ("trend_signal_tr", "trend_signal"),
        ):
            if not (candidate.get(tr_key) or "").strip():
                fallback = candidate.get(en_key)
                if fallback:
                    logger.info(
                        "[TOPIC_DISCOVERY] no %s provided for topic=%r - falling back to English %s",
                        tr_key, topic, en_key,
                    )
                candidate[tr_key] = fallback
        result.append(candidate)
    return result


def _score_candidates(candidates: list[dict]) -> list[dict]:
    """Scores each candidate with the EXISTING youtube.topic_scorer -
    identical to what topic_scout.add_candidate() already calls for a
    manually-entered topic. Heuristic backend by default (deterministic,
    no API call), matching topic_scout.py's own default; set
    TOPIC_DISCOVERY_USE_LLM_SCORING=true to score with Claude instead."""
    for candidate in candidates:
        scored = topic_scorer.score_topic(
            candidate["topic"], category=candidate.get("category"),
            use_llm=settings.TOPIC_DISCOVERY_USE_LLM_SCORING,
        )
        candidate["score"] = scored["overall"]
        candidate["score_breakdown"] = scored["scores"]
    return candidates


def _select_top(candidates: list[dict], n: int | None = None) -> list[dict]:
    n = n or settings.TOPIC_DISCOVERY_CANDIDATE_COUNT
    return sorted(candidates, key=lambda c: c.get("score") or 0, reverse=True)[:n]


# --- Phase B: Telegram message / keyboard -----------------------------------

def _format_sources(sources: list[dict] | None) -> str:
    if not sources:
        return "-"
    lines = []
    for s in sources[:3]:
        url = s.get("url") if isinstance(s, dict) else None
        if url:
            lines.append(url)
    return "\n".join(lines) if lines else "-"


def _display_title(c: dict) -> str:
    """Turkish display title for the Telegram owner UI, falling back to the
    English title if no translation is available (never crashes, never
    fabricates)."""
    return (c.get("title_tr") or c.get("title") or "").strip()


def _score_label(score) -> str:
    """topic_scorer.py's overall score is 0-10 (see youtube/topic_scorer.py's
    CRITERIA average) - shown as "X/10" to match the Turkish UI example
    exactly. Never recomputes or rescales the score itself."""
    if score is None:
        return "-"
    return f"{score}/10"


def build_topic_batch_message(candidates: list[dict]) -> str:
    """Telegram MANAGEMENT/INTERFACE text only, in Turkish, for the channel
    owner - the AI FOUND THIS channel itself, and every field actually used
    for production (topic/category passed to pipeline.produce_video()),
    stays English and is never touched by this function. See
    _display_title()/*_tr fields for the Turkish text shown here."""
    lines = [
        "\U0001f916 <b>AI FOUND THIS</b>", "",
        "\U0001f525 <b>BUGÜN ÜRETMEYE DEĞER EN GÜÇLÜ 5 KONU</b>", "",
        "\"Bugün AI FOUND THIS için üretmeye değer en güçlü 5 konu bunlar.\"", "",
    ]
    for i, c in enumerate(candidates):
        emoji = _NUMBER_EMOJI[i] if i < len(_NUMBER_EMOJI) else f"{i + 1}."
        lines += [
            f"{emoji} <b>{_display_title(c).upper()}</b>", "",
            "Neden şimdi:", c.get("why_now_tr") or c.get("why_now") or "-", "",
            "Kategori:", c.get("category_tr") or c.get("category") or "Kategorisiz", "",
            "Trend sinyali:", c.get("trend_signal_tr") or c.get("trend_signal") or "-", "",
            "AI FOUND THIS skoru:", _score_label(c.get("score")), "",
            "Kaynaklar:", _format_sources(c.get("sources")), "",
        ]
    lines.append("\U0001f447 <b>KONU SEÇ</b>")
    return "\n".join(lines)


def build_topic_keyboard(candidates: list[dict]) -> dict:
    buttons = []
    for i, c in enumerate(candidates):
        emoji = _NUMBER_EMOJI[i] if i < len(_NUMBER_EMOJI) else str(i + 1)
        buttons.append({"text": emoji, "callback_data": f"{CALLBACK_PREFIX}{c['candidate_id']}"})
    return {"inline_keyboard": [buttons]}


# --- orchestration -----------------------------------------------------------

def run_daily_discovery(triggered_by: str = "scheduler", chat_id: str | None = None) -> dict | None:
    """The one entry point both the scheduler and /topics call. Discovers,
    filters, scores, persists a new batch, and sends ONE Telegram message -
    never starts production (spec section 11: "It must NOT start video
    production automatically"). Returns the batch dict, or None if fewer
    than 1 valid candidate could be found (logged, never invented - spec
    section 6)."""
    logger.info("[TOPIC_DISCOVERY] starting daily discovery (triggered_by=%s)", triggered_by)
    raw = discover_raw_candidates()
    logger.info("[TOPIC_DISCOVERY] discovered %d candidates", len(raw))

    filtered = _filter_and_dedupe(raw)
    logger.info("[TOPIC_SCORER] scoring candidates")
    scored = _score_candidates(filtered)
    top = _select_top(scored)

    if not top:
        logger.warning(
            "[TOPIC_DISCOVERY] no valid candidates found today (raw=%d, after filtering=%d) - nothing sent",
            len(raw), len(filtered),
        )
        return None
    if len(top) < settings.TOPIC_DISCOVERY_CANDIDATE_COUNT:
        logger.warning(
            "[TOPIC_DISCOVERY] only %d valid candidate(s) found today (target=%d) - sending fewer than target",
            len(top), settings.TOPIC_DISCOVERY_CANDIDATE_COUNT,
        )
    logger.info("[TOPIC_DISCOVERY] selected top %d", len(top))

    batch_id = tbatch.create_batch(triggered_by=triggered_by)
    for i, c in enumerate(top, start=1):
        candidate_id = tbatch.add_candidate(
            batch_id=batch_id, position=i, title=c["title"], topic=c["topic"],
            normalized_topic=c["normalized_topic"], short_description=c.get("short_description"),
            why_now=c.get("why_now"), category=c.get("category"), trend_signal=c.get("trend_signal"),
            score=c.get("score"), score_breakdown=c.get("score_breakdown"), sources=c.get("sources"),
            status="presented",
            title_tr=c.get("title_tr"), why_now_tr=c.get("why_now_tr"),
            category_tr=c.get("category_tr"), trend_signal_tr=c.get("trend_signal_tr"),
        )
        c["candidate_id"] = candidate_id

    text = build_topic_batch_message(top)
    keyboard = build_topic_keyboard(top)
    logger.info("[TOPIC_TELEGRAM] sending topic selection")
    message_id = tb.send_text(text, reply_markup=keyboard, chat_id=chat_id)
    tbatch.update_batch_fields(
        batch_id, telegram_chat_id=str(chat_id or settings.TELEGRAM_CHAT_ID), telegram_message_id=message_id,
    )
    return {"batch_id": batch_id, "candidates": top, "message_id": message_id}


def handle_topics_command(message: dict) -> None:
    """Dispatched by youtube/telegram_approval_listener.py's process_update()
    for an incoming "/topics" text message - runs the exact same discovery
    flow as the scheduler, on demand, for manual testing (spec section 12)."""
    chat = message.get("chat", {}) or {}
    chat_id = str(chat["id"]) if chat.get("id") is not None else None
    logger.info("[TOPIC_DISCOVERY] /topics command received")
    try:
        result = run_daily_discovery(triggered_by="manual", chat_id=chat_id)
        if result is None:
            tb.send_text("Şu anda geçerli konu adayı bulunamadı - sunucu loglarını kontrol edin.", chat_id=chat_id)
    except Exception:
        logger.exception("[TOPIC_DISCOVERY] /topics discovery failed")
        tb.send_text("⚠️ Konu keşfi başarısız oldu - ayrıntılar için sunucu loglarını kontrol edin.", chat_id=chat_id)


# --- Phase B: selection callback ---------------------------------------------
#
# Every string below is Telegram MANAGEMENT/INTERFACE text for the channel
# owner, in Turkish. row["title"]/row["topic"]/row["category"] (English)
# are never translated and are exactly what is passed to
# pipeline.produce_video() further down in handle_topic_select_callback() -
# the confirmation text below deliberately shows BOTH the Turkish display
# title (row["title_tr"]) and the original English production title
# (row["title"]) side by side, so the owner can always see which real
# English video title is actually being produced.

def _selected_confirmation_text(row) -> str:
    display_title = (row["title_tr"] or row["title"]).strip()
    return (
        "✅ <b>KONU SEÇİLDİ</b>\n\n"
        f"{display_title}\n\n"
        "İngilizce video başlığı:\n"
        f"{row['title']}\n\n"
        "\U0001f3ac Üretim başlatılıyor...\n\n"
        "Araştırma → Senaryo → Görseller → Seslendirme → Video → QC"
    )


def _already_handled_text(row) -> str:
    display_title = (row["title_tr"] or row["title"]).strip()
    status = row["status"]
    if status == "selected":
        return f"✅ Bu konu daha önce seçildi\n\n{display_title}\n\nÜretim zaten başladı (video_id={row['video_id']})."
    if status == "rejected":
        return "Bu konu seçilmedi (aynı gruptan başka bir konu seçildi)."
    if status == "expired":
        return "Konu seçimi için süre doldu. Yeni konular için /topics kullanın."
    return f"Bu konu daha önce işlendi (durum={status})."


def handle_topic_select_callback(callback_query: dict) -> None:
    """Entry point for one "topic_select:<candidate_id>" callback_query -
    mirrors youtube/telegram_approval.py's handle_approval_callback()
    structure (validate -> answerCallbackQuery -> atomic transition ->
    editMessageText -> start work), for this new namespace.

    Idempotency/duplicate protection (spec section 13): the atomic
    storage.topic_batches.try_select_candidate() below is what guarantees
    pipeline.produce_video() is called AT MOST ONCE per batch, however many
    times a callback for this batch's buttons is received (double-click,
    Telegram retrying an unanswered callback, two different buttons in the
    same batch clicked in quick succession)."""
    data = callback_query.get("data", "") or ""
    message = callback_query.get("message", {}) or {}
    message_id = message.get("message_id")
    chat = message.get("chat", {}) or {}
    chat_id = chat.get("id")
    cbq_id = callback_query.get("id")

    if not data.startswith(CALLBACK_PREFIX):
        logger.warning("[TOPIC_TELEGRAM] handle_topic_select_callback got non-matching data: %r", data)
        return

    candidate_id_str = data[len(CALLBACK_PREFIX):]
    try:
        candidate_id = int(candidate_id_str)
    except ValueError:
        logger.warning("[TOPIC_TELEGRAM] invalid topic_select callback data: %r", data)
        if cbq_id:
            tb._answer_callback(cbq_id, "Geçersiz konu.")
        return

    candidate = tbatch.get_candidate(candidate_id)
    if candidate is None:
        logger.warning("[TOPIC_TELEGRAM] topic_select callback for unknown candidate_id=%d", candidate_id)
        if cbq_id:
            tb._answer_callback(cbq_id, "Geçersiz konu.")
        return

    batch = tbatch.get_batch(candidate["batch_id"])
    # Security (mirrors youtube/telegram_approval.py's own chat/message
    # validation exactly): the callback must match the SAME chat + message
    # this batch was actually sent to - rejects a forged callback_data, or
    # a stale button on a superseded/old batch's message (spec section 9/13
    # "Candidate from an old batch cannot be selected").
    if batch is None or str(batch["telegram_chat_id"]) != str(chat_id) or batch["telegram_message_id"] != message_id:
        logger.warning(
            "[TOPIC_TELEGRAM] topic_select callback chat/message mismatch for candidate_id=%d", candidate_id,
        )
        if cbq_id:
            tb._answer_callback(cbq_id, "Konu seçimi için süre doldu.")
        return

    if cbq_id:
        tb._answer_callback(cbq_id, "Seçiliyor...")

    row = tbatch.try_select_candidate(candidate_id)
    if row is None:
        current = tbatch.get_candidate(candidate_id)
        logger.info(
            "[TOPIC_TELEGRAM] duplicate/late selection for candidate_id=%d (already status=%s) - ignoring",
            candidate_id, current["status"] if current else "unknown",
        )
        if message_id and current is not None:
            tb.edit_text(message_id, _already_handled_text(current))
        return

    logger.info("[TOPIC_TELEGRAM] candidate selected id=%d topic=%r", candidate_id, row["topic"])
    if message_id:
        tb.edit_text(message_id, _selected_confirmation_text(row))

    import pipeline as pipeline_module

    try:
        logger.info("[TOPIC_PIPELINE] starting production topic=%r", row["topic"])
        paths = pipeline_module.produce_video(row["topic"], category=row["category"])
        logger.info("[TOPIC_PIPELINE] starting production video_id=%d", paths.video_id)
        tbatch.update_candidate_fields(candidate_id, video_id=paths.video_id)
    except Exception:
        # Never forward raw exception text to Telegram (mirrors
        # youtube/telegram_approval.py's own rule) - production errors can
        # embed local filesystem paths.
        logger.exception("[TOPIC_PIPELINE] production failed for candidate_id=%d topic=%r", candidate_id, row["topic"])
        if message_id:
            display_title = (row["title_tr"] or row["title"]).strip()
            tb.edit_text(
                message_id,
                f"⚠️ Üretim başlatılamadı:\n\n{display_title}\n\nAyrıntılar için sunucu loglarını kontrol edin.",
            )
