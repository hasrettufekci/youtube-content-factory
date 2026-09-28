"""
Daily Topic Discovery + Telegram Topic Selector (Phase A + Phase B) tests.

Covers spec section 15's required scenarios A-K (L/M/N/O are covered in
test_telegram_approval_listener.py / test_scheduler.py, where the
dispatch/scheduler entry points actually live; P/Q/R are the existing
Telegram approval / YouTube upload / full-suite regression tests, run
separately - see the delivery report).

No real Claude, Telegram, or YouTube calls are ever made here.
"""
from __future__ import annotations

import pytest

from core.config import settings
from storage import db, topic_batches as tbatch
from youtube import llm_client, topic_discovery as td


def _raw_candidate(title="The Signal Scientists Can't Explain", topic=None, category="Science / Mystery"):
    return {
        "title": title,
        "title_tr": f"{title} (TR)",
        "topic": topic or title,
        "short_description": "A short description.",
        "why_now": "Recently discussed widely.",
        "why_now_tr": "Son zamanlarda yaygın şekilde konuşuluyor.",
        "category": category,
        "category_tr": "Bilim / Gizem",
        "trend_signal": "Growing interest across several outlets this week.",
        "trend_signal_tr": "Bu hafta birçok kaynakta artan ilgi.",
        "sources": [{"title": "Example", "url": "https://example.com/article"}],
    }


def _compact_raw_candidate(title="A Compact Topic", topic=None, category="History"):
    """The leaner LLM output shape after the truncation fix (real production
    incident: 3 attempts truncated at max_tokens=4000 mid-why_now_tr) - no
    short_description, exactly one source with only a url, and every text
    field short enough to respect the system prompt's word limits. Used to
    prove the rest of the pipeline (filtering, scoring, the Turkish Telegram
    message, selection, production) still works unchanged with this
    smaller payload."""
    return {
        "title": title,
        "title_tr": f"{title} (TR)",
        "topic": topic or title,
        "why_now": "Widely discussed this week across several outlets.",
        "why_now_tr": "Bu hafta birçok kaynakta yaygın şekilde konuşuluyor.",
        "category": category,
        "category_tr": "Tarih",
        "trend_signal": "Growing search interest over the past few days.",
        "trend_signal_tr": "Son günlerde artan arama ilgisi.",
        "sources": [{"url": "https://example.com/compact-source"}],
        # Deliberately no "short_description" key.
    }


@pytest.fixture
def fake_topic_telegram(monkeypatch):
    sent = {"texts": [], "edits": [], "answers": []}

    def fake_send_text(text, reply_markup=None, chat_id=None):
        message_id = 5000 + len(sent["texts"])
        sent["texts"].append({"text": text, "reply_markup": reply_markup, "chat_id": chat_id, "message_id": message_id})
        return message_id

    def fake_edit_text(message_id, text, reply_markup=None):
        sent["edits"].append((message_id, text))

    def fake_answer(cbq_id, text=""):
        sent["answers"].append((cbq_id, text))

    monkeypatch.setattr(td.tb, "send_text", fake_send_text)
    monkeypatch.setattr(td.tb, "edit_text", fake_edit_text)
    monkeypatch.setattr(td.tb, "_answer_callback", fake_answer)
    return sent


# ---------------------------------------------------------------------------
# A - discovery returns candidates
# ---------------------------------------------------------------------------

def test_discover_raw_candidates_returns_parsed_list(isolated_storage, monkeypatch):
    raw = [_raw_candidate(f"Topic {i}") for i in range(3)]
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: ({"candidates": raw}, []))

    result = td.discover_raw_candidates(count=3)

    assert result == raw


def test_discover_raw_candidates_retries_on_truncation_then_succeeds(isolated_storage, monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise llm_client.ClaudeTruncatedResponseError("truncated")
        return {"candidates": [_raw_candidate()]}, []

    monkeypatch.setattr(llm_client, "chat_json", flaky)
    monkeypatch.setattr(settings, "TOPIC_DISCOVERY_MAX_ATTEMPTS", 3)

    result = td.discover_raw_candidates()

    assert calls["n"] == 2
    assert len(result) == 1


def test_discover_raw_candidates_gives_up_after_max_attempts(isolated_storage, monkeypatch):
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: (_ for _ in ()).throw(llm_client.ClaudeError("bad")))
    monkeypatch.setattr(settings, "TOPIC_DISCOVERY_MAX_ATTEMPTS", 2)

    with pytest.raises(td.TopicDiscoveryError):
        td.discover_raw_candidates()


# ---------------------------------------------------------------------------
# B - scoring uses the existing TopicScorer
# ---------------------------------------------------------------------------

def test_score_candidates_uses_existing_topic_scorer(isolated_storage, monkeypatch):
    from youtube import topic_scorer

    calls = []
    real_score = topic_scorer.score_topic

    def spy(topic, category=None, use_llm=False):
        calls.append((topic, category, use_llm))
        return real_score(topic, category=category, use_llm=use_llm)

    monkeypatch.setattr(topic_scorer, "score_topic", spy)

    candidates = [dict(_raw_candidate(), normalized_topic="x")]
    scored = td._score_candidates(candidates)

    assert len(calls) == 1
    assert calls[0][2] is False  # heuristic backend by default
    assert "score" in scored[0] and "score_breakdown" in scored[0]


# ---------------------------------------------------------------------------
# C - exactly 5 valid candidates when 5+ exist
# ---------------------------------------------------------------------------

def test_select_top_returns_exactly_5_when_6_valid_candidates_exist(isolated_storage):
    candidates = [{"title": f"T{i}", "score": i} for i in range(6)]
    top = td._select_top(candidates, n=5)
    assert len(top) == 5
    assert [c["score"] for c in top] == [5, 4, 3, 2, 1]  # sorted desc, top 5


def test_select_top_returns_fewer_when_fewer_valid_candidates_exist(isolated_storage):
    candidates = [{"title": "T1", "score": 9}, {"title": "T2", "score": 8}]
    top = td._select_top(candidates, n=5)
    assert len(top) == 2


# ---------------------------------------------------------------------------
# D - duplicate topics removed
# ---------------------------------------------------------------------------

def test_filter_and_dedupe_removes_duplicate_topics(isolated_storage):
    raw = [_raw_candidate("The Wow Signal"), _raw_candidate("the   WOW signal!!")]
    result = td._filter_and_dedupe(raw)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# E - recently produced topics filtered
# ---------------------------------------------------------------------------

def test_filter_and_dedupe_rejects_already_produced_topics(isolated_storage):
    from storage import project_store as ps

    topic = "An Already Produced Topic"
    db.insert_video(
        slug="an-already-produced-topic", topic=topic, normalized_topic=ps.normalize_topic(topic),
        category=None, project_id="p", state="published", storage_path="/tmp/x",
    )
    raw = [_raw_candidate("An Already Produced Topic", topic=topic), _raw_candidate("A Fresh New Topic")]
    result = td._filter_and_dedupe(raw)
    assert len(result) == 1
    assert result[0]["topic"] == "A Fresh New Topic"


def test_filter_and_dedupe_drops_candidates_missing_topic(isolated_storage):
    result = td._filter_and_dedupe([{"title": "", "topic": ""}, _raw_candidate("Valid Topic")])
    assert len(result) == 1


def test_filter_and_dedupe_fills_english_fallback_for_missing_translations(isolated_storage):
    """When Claude omits a _tr field, _filter_and_dedupe() must fill it with
    the English counterpart (never leave it missing/None going into
    Telegram rendering, and never fabricate new Turkish text)."""
    raw = _raw_candidate("A Topic With No Translation")
    for key in ("title_tr", "why_now_tr", "category_tr", "trend_signal_tr"):
        raw.pop(key)

    result = td._filter_and_dedupe([raw])

    assert len(result) == 1
    c = result[0]
    assert c["title_tr"] == c["title"]
    assert c["why_now_tr"] == c["why_now"]
    assert c["category_tr"] == c["category"]
    assert c["trend_signal_tr"] == c["trend_signal"]


def test_filter_and_dedupe_keeps_provided_translation_distinct_from_english(isolated_storage):
    """The common case: a real Turkish translation is provided and must be
    kept as-is, not overwritten by the English fallback."""
    raw = _raw_candidate("A Fully Translated Topic")
    result = td._filter_and_dedupe([raw])
    assert len(result) == 1
    c = result[0]
    assert c["title_tr"] != c["title"]
    assert c["title_tr"] == "A Fully Translated Topic (TR)"


# ---------------------------------------------------------------------------
# F - Telegram topic message renders correctly
# ---------------------------------------------------------------------------

def test_build_topic_batch_message_renders_all_fields(isolated_storage):
    candidates = [dict(_raw_candidate(f"Topic {i}"), candidate_id=i, score=7.5) for i in range(1, 6)]
    text = td.build_topic_batch_message(candidates)

    # Telegram management/interface text is Turkish - the AI FOUND THIS
    # channel name itself is a proper noun and stays as-is.
    assert "AI FOUND THIS" in text
    assert "BUGÜN ÜRETMEYE DEĞER EN GÜÇLÜ 5 KONU" in text
    assert "Bugün AI FOUND THIS için üretmeye değer en güçlü 5 konu bunlar." in text
    for c in candidates:
        # Turkish display title (title_tr), not the English production title.
        assert c["title_tr"].upper() in text
        assert c["why_now_tr"] in text
        assert c["category_tr"] in text
        assert c["trend_signal_tr"] in text
        assert "7.5/10" in text
    assert "Neden şimdi:" in text
    assert "Kategori:" in text
    assert "Trend sinyali:" in text
    assert "AI FOUND THIS skoru:" in text
    assert "Kaynaklar:" in text
    assert "KONU SEÇ" in text


def test_build_topic_batch_message_falls_back_to_english_when_translation_missing(isolated_storage):
    """_display_title()/build_topic_batch_message() must never crash or
    render empty/blank text just because Claude left a _tr field empty -
    they fall back to the English production text instead (never fabricate
    Turkish)."""
    candidate = _raw_candidate("The Untranslated Topic")
    candidate["candidate_id"] = 1
    candidate["score"] = 6.0
    for key in ("title_tr", "why_now_tr", "category_tr", "trend_signal_tr"):
        candidate[key] = None

    text = td.build_topic_batch_message([candidate])

    assert "THE UNTRANSLATED TOPIC" in text
    assert candidate["why_now"] in text
    assert candidate["category"] in text
    assert candidate["trend_signal"] in text


# ---------------------------------------------------------------------------
# G - inline buttons contain correct callback_data
# ---------------------------------------------------------------------------

def test_build_topic_keyboard_has_correct_callback_data(isolated_storage):
    candidates = [dict(_raw_candidate(f"Topic {i}"), candidate_id=100 + i) for i in range(5)]
    keyboard = td.build_topic_keyboard(candidates)
    buttons = keyboard["inline_keyboard"][0]
    assert [b["callback_data"] for b in buttons] == [f"topic_select:{100 + i}" for i in range(5)]


# ---------------------------------------------------------------------------
# H - topic_select callback selects exactly one candidate
# ---------------------------------------------------------------------------

def _make_batch_with_candidates(n=5, chat_id="999", message_id=42):
    batch_id = tbatch.create_batch(triggered_by="scheduler")
    ids = []
    for i in range(1, n + 1):
        cid = tbatch.add_candidate(
            batch_id=batch_id, position=i, title=f"Topic {i}", topic=f"topic {i}",
            normalized_topic=f"topic {i}", category="Science", status="presented",
            # Deliberately distinct from the English fields above, so tests
            # can assert the Turkish display text and the English production
            # text never get confused with one another.
            title_tr=f"Konu {i} (TR)", why_now_tr="Türkçe neden.",
            category_tr="Bilim (TR)", trend_signal_tr="Türkçe trend.",
        )
        ids.append(cid)
    tbatch.update_batch_fields(batch_id, telegram_chat_id=chat_id, telegram_message_id=message_id)
    return batch_id, ids


def _topic_select_callback(candidate_id, chat_id="999", message_id=42, cbq_id="cbq1"):
    return {
        "id": cbq_id,
        "data": f"topic_select:{candidate_id}",
        "message": {"message_id": message_id, "chat": {"id": chat_id}},
    }


def test_handle_topic_select_callback_selects_exactly_one_candidate(isolated_storage, fake_topic_telegram, monkeypatch):
    import pipeline as pipeline_module

    calls = []

    class _FakePaths:
        video_id = 7

    def fake_produce(topic, category=None, **kwargs):
        calls.append((topic, category))
        return _FakePaths()

    monkeypatch.setattr(pipeline_module, "produce_video", fake_produce)

    batch_id, ids = _make_batch_with_candidates()
    td.handle_topic_select_callback(_topic_select_callback(ids[2]))

    assert calls == [("topic 3", "Science")]
    selected = tbatch.get_candidate(ids[2])
    assert selected["status"] == "selected"
    assert selected["video_id"] == 7
    for other_id in ids:
        if other_id != ids[2]:
            assert tbatch.get_candidate(other_id)["status"] == "rejected"
    assert tbatch.get_batch(batch_id)["status"] == "selected"


# ---------------------------------------------------------------------------
# I - duplicate callback does not start production twice
# ---------------------------------------------------------------------------

def test_handle_topic_select_callback_duplicate_does_not_start_production_twice(isolated_storage, fake_topic_telegram, monkeypatch):
    import pipeline as pipeline_module

    calls = []

    class _FakePaths:
        video_id = 7

    monkeypatch.setattr(pipeline_module, "produce_video", lambda topic, category=None, **k: calls.append(topic) or _FakePaths())

    batch_id, ids = _make_batch_with_candidates()
    callback = _topic_select_callback(ids[0])

    td.handle_topic_select_callback(callback)
    td.handle_topic_select_callback(callback)  # duplicate/late tap

    assert len(calls) == 1


def test_handle_topic_select_callback_two_different_buttons_only_produce_once(isolated_storage, fake_topic_telegram, monkeypatch):
    """Two different candidates from the same batch both clicked (race) -
    only the first (atomic) transition may proceed to production."""
    import pipeline as pipeline_module

    calls = []

    class _FakePaths:
        video_id = 7

    monkeypatch.setattr(pipeline_module, "produce_video", lambda topic, category=None, **k: calls.append(topic) or _FakePaths())

    batch_id, ids = _make_batch_with_candidates()
    td.handle_topic_select_callback(_topic_select_callback(ids[0]))
    td.handle_topic_select_callback(_topic_select_callback(ids[1]))  # already rejected by the first selection

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# J - invalid candidate ID is rejected
# ---------------------------------------------------------------------------

def test_handle_topic_select_callback_invalid_candidate_id_is_rejected(isolated_storage, fake_topic_telegram, monkeypatch):
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "produce_video", lambda *a, **k: pytest.fail("must not start production"))

    td.handle_topic_select_callback(_topic_select_callback(999999))  # never created

    assert any("Geçersiz konu." in a[1] for a in fake_topic_telegram["answers"])


def test_handle_topic_select_callback_malformed_data_is_rejected(isolated_storage, fake_topic_telegram, monkeypatch):
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "produce_video", lambda *a, **k: pytest.fail("must not start production"))

    callback = _topic_select_callback("not-a-number")
    td.handle_topic_select_callback(callback)

    assert any("Geçersiz konu." in a[1] for a in fake_topic_telegram["answers"])


# ---------------------------------------------------------------------------
# K - candidate from an old batch cannot be selected
# ---------------------------------------------------------------------------

def test_handle_topic_select_callback_rejects_candidate_from_stale_message(isolated_storage, fake_topic_telegram, monkeypatch):
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "produce_video", lambda *a, **k: pytest.fail("must not start production"))

    batch_id, ids = _make_batch_with_candidates(chat_id="999", message_id=42)
    # Same candidate id, but a DIFFERENT (stale/forged) message_id.
    stale_callback = _topic_select_callback(ids[0], chat_id="999", message_id=999999)

    td.handle_topic_select_callback(stale_callback)

    assert tbatch.get_candidate(ids[0])["status"] == "presented"  # unchanged
    assert any("Konu seçimi için süre doldu." in a[1] for a in fake_topic_telegram["answers"])


# ---------------------------------------------------------------------------
# Turkish Telegram Management UI patch - display/production separation.
#
# These pin down the core guarantee of the patch: Telegram shows Turkish,
# but pipeline.produce_video() (and everything downstream of it - research,
# script, narration, metadata, the actual YouTube video) only ever sees the
# original English topic/category, never the Turkish display text.
# ---------------------------------------------------------------------------

def test_selected_confirmation_shows_both_turkish_and_english_titles(isolated_storage, fake_topic_telegram, monkeypatch):
    import pipeline as pipeline_module

    class _FakePaths:
        video_id = 7

    monkeypatch.setattr(pipeline_module, "produce_video", lambda topic, category=None, **k: _FakePaths())

    batch_id, ids = _make_batch_with_candidates()
    td.handle_topic_select_callback(_topic_select_callback(ids[2]))

    edit_texts = [text for (_mid, text) in fake_topic_telegram["edits"]]
    assert any("Konu 3 (TR)" in t and "Topic 3" in t for t in edit_texts), edit_texts
    assert any("İngilizce video başlığı:" in t for t in edit_texts)
    assert any("KONU SEÇİLDİ" in t for t in edit_texts)


def test_handle_topic_select_callback_passes_only_english_to_production(isolated_storage, fake_topic_telegram, monkeypatch):
    """Strengthened invariant (Turkish Telegram Management UI patch): even
    though the candidate's Turkish display fields are clearly different text
    from the English production fields, produce_video() must receive
    EXACTLY the English topic/category - never the _tr values, and never a
    mix of the two."""
    import pipeline as pipeline_module

    calls = []

    class _FakePaths:
        video_id = 7

    def fake_produce(topic, category=None, **kwargs):
        calls.append((topic, category))
        return _FakePaths()

    monkeypatch.setattr(pipeline_module, "produce_video", fake_produce)

    batch_id = tbatch.create_batch(triggered_by="scheduler")
    cid = tbatch.add_candidate(
        batch_id=batch_id, position=1,
        title="The English Production Title", topic="the english production topic",
        normalized_topic="the english production topic", category="History",
        status="presented",
        title_tr="Bambaşka Türkçe Başlık", why_now_tr="Alakasız Türkçe metin.",
        category_tr="Tarih (Türkçe)", trend_signal_tr="Türkçe trend sinyali.",
    )
    tbatch.update_batch_fields(batch_id, telegram_chat_id="999", telegram_message_id=42)

    td.handle_topic_select_callback(_topic_select_callback(cid, chat_id="999", message_id=42))

    assert calls == [("the english production topic", "History")]
    for topic, category in calls:
        assert "Türkçe" not in topic and "Türkçe" not in (category or "")
        assert "Bambaşka" not in topic


# ---------------------------------------------------------------------------
# Truncation fix (real production incident: 3/3 discovery attempts hit
# ClaudeTruncatedResponseError, stop_reason=max_tokens=4000, truncated
# inside why_now_tr after several candidates had already been generated).
#
# Root cause was output volume, not a code defect - _DISCOVERY_SYSTEM_PROMPT
# now enforces hard per-field word limits and drops the unused
# short_description field / multi-source citations. These tests pin down
# that the leaner payload still flows correctly through every existing
# stage: parsing, filtering, the Turkish Telegram message, and selection ->
# production (still English-only) - none of which changed.
# ---------------------------------------------------------------------------

def test_discovery_system_prompt_enforces_concise_word_limits():
    prompt = td._DISCOVERY_SYSTEM_PROMPT
    assert "at most 12 words" in prompt
    assert "at most 25 words" in prompt
    assert "at most 4 words" in prompt
    assert "at most 20 words" in prompt
    assert "do NOT include a \"short_description\" field" in prompt
    assert "exactly ONE item" in prompt


def test_discover_raw_candidates_returns_parsed_list_of_compact_candidates(isolated_storage, monkeypatch):
    raw = [_compact_raw_candidate(f"Compact Topic {i}") for i in range(5)]
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: ({"candidates": raw}, []))

    result = td.discover_raw_candidates(count=5)

    assert result == raw
    assert len(result) == 5
    assert all("short_description" not in c for c in result)


def test_filter_and_dedupe_accepts_compact_candidates_without_short_description(isolated_storage):
    raw = [_compact_raw_candidate("A Concise Topic")]

    result = td._filter_and_dedupe(raw)

    assert len(result) == 1
    c = result[0]
    assert c.get("short_description") is None
    # Provided translations are kept as-is - the English-fallback path in
    # _filter_and_dedupe is only for a MISSING translation, not exercised
    # here since the compact fixture already supplies real ones.
    assert c["title_tr"] == "A Concise Topic (TR)"
    assert c["why_now_tr"]
    assert c["category_tr"] == "Tarih"
    assert c["trend_signal_tr"]


def test_build_topic_batch_message_unchanged_format_with_compact_candidates(isolated_storage):
    """Proves the Turkish Telegram message format is completely unaffected
    by the leaner candidate shape - short_description was never rendered
    there in the first place."""
    candidates = [dict(_compact_raw_candidate(f"Compact Topic {i}"), candidate_id=i, score=6.5) for i in range(1, 6)]

    text = td.build_topic_batch_message(candidates)

    assert "AI FOUND THIS" in text
    assert "BUGÜN ÜRETMEYE DEĞER EN GÜÇLÜ 5 KONU" in text
    assert "Bugün AI FOUND THIS için üretmeye değer en güçlü 5 konu bunlar." in text
    for c in candidates:
        assert c["title_tr"].upper() in text
        assert c["why_now_tr"] in text
        assert c["category_tr"] in text
        assert c["trend_signal_tr"] in text
        assert "6.5/10" in text
    assert "KONU SEÇ" in text


def test_run_daily_discovery_and_select_with_compact_candidates_uses_english_only(
    isolated_storage, fake_topic_telegram, monkeypatch,
):
    """End-to-end regression for the truncation fix: 5 compact candidates
    (the new leaner LLM output shape) flow through discovery -> filtering ->
    the EXISTING topic_scorer -> the Turkish Telegram message -> selection,
    and production still only ever receives the English topic/category -
    exactly as before this fix, none of which was touched by it."""
    import pipeline as pipeline_module

    raw = [_compact_raw_candidate(f"Compact Topic {i}", category="History") for i in range(5)]
    monkeypatch.setattr(llm_client, "chat_json", lambda *a, **k: ({"candidates": raw}, []))

    calls = []

    class _FakePaths:
        video_id = 42

    def fake_produce(topic, category=None, **kwargs):
        calls.append((topic, category))
        return _FakePaths()

    monkeypatch.setattr(pipeline_module, "produce_video", fake_produce)

    result = td.run_daily_discovery(triggered_by="scheduler", chat_id="999")

    assert result is not None
    assert len(result["candidates"]) == 5

    sent_text = fake_topic_telegram["texts"][0]["text"]
    assert "BUGÜN ÜRETMEYE DEĞER EN GÜÇLÜ 5 KONU" in sent_text
    assert "KONU SEÇ" in sent_text

    top = result["candidates"][0]
    callback = {
        "id": "cbq1",
        "data": f"topic_select:{top['candidate_id']}",
        "message": {"message_id": result["message_id"], "chat": {"id": "999"}},
    }
    td.handle_topic_select_callback(callback)

    assert calls == [(top["topic"], top["category"])]
    produced_topic, produced_category = calls[0]
    assert "(TR)" not in produced_topic
    assert "Tarih" not in (produced_category or "")
