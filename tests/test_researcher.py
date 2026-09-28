"""
Research / fact-check relationship (spec Round 2 section 11):

    - facts/sources/uncertainties: every source Claude cites must be
      merged into "sources" (never silently dropped), deduplicated by URL
    - required fields validated - a malformed research result must fail
      loudly, not ship half-empty
    - research.json actually persists to disk, and sources persist to the
      DB too (survives a resume even if research.json were lost)
    - source URLs must not be lost on the way into script.json
      (script_generator.py carries research["sources"] straight through)
"""
import pytest

from core.state import PipelineState
from storage import db, project_store as ps
from youtube import researcher, script_generator as sg


def _base_data(**overrides):
    data = {
        "topic": "The Dyatlov Pass Incident",
        "summary": "Nine hikers died in 1959 under disputed circumstances.",
        "facts": ["Nine hikers died in the Ural Mountains in February 1959."],
        "sources": [{"title": "Official report", "url": "https://example.com/report"}],
    }
    data.update(overrides)
    return data


def test_research_topic_merges_cited_sources_deduped_by_url(monkeypatch):
    self_reported = [{"title": "Official report", "url": "https://example.com/report"}]
    api_cited = [
        {"title": "Official report (dup)", "url": "https://example.com/report"},  # same URL - must not duplicate
        {"title": "Newspaper archive", "url": "https://example.com/newspaper"},   # new - must be added
    ]

    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        return _base_data(sources=self_reported), api_cited

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    data = researcher.research_topic("The Dyatlov Pass Incident")

    urls = [s["url"] for s in data["sources"]]
    assert urls.count("https://example.com/report") == 1  # deduped, not doubled
    assert "https://example.com/newspaper" in urls  # the API-cited-but-not-self-reported source wasn't dropped
    assert len(data["sources"]) == 2


def test_research_topic_raises_on_missing_required_field(monkeypatch):
    incomplete = _base_data()
    del incomplete["facts"]

    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        return incomplete, []

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    with pytest.raises(RuntimeError, match="facts"):
        researcher.research_topic("Some Topic")


def test_research_topic_defaults_optional_fields_to_empty_lists(monkeypatch):
    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        return _base_data(), []  # no timeline/uncertainties/search_notes supplied

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    data = researcher.research_topic("Some Topic")

    assert data["timeline"] == []
    assert data["uncertainties"] == []
    assert data["search_notes"] == []


def test_run_and_save_persists_research_json_and_db_sources(isolated_storage, monkeypatch):
    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        return _base_data(), []

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    paths, _ = ps.create_project("Persisted Research Topic")
    result = researcher.run_and_save(paths, "Persisted Research Topic")

    on_disk = ps.load_json(paths.research_json)
    assert on_disk == result
    assert on_disk["sources"][0]["url"] == "https://example.com/report"

    db_sources = db.get_research_sources(paths.video_id)
    assert len(db_sources) == 1
    assert db_sources[0]["url"] == "https://example.com/report"
    assert ps.load_state(paths) == PipelineState.RESEARCHED


def test_script_generator_carries_research_source_urls_through_unchanged(monkeypatch):
    research = _base_data(sources=[
        {"title": "A", "url": "https://a.example.com"},
        {"title": "B", "url": "https://b.example.com"},
    ])

    sections = [
        {"type": t, "title": t, "narration": " ".join(["word"] * 170) + "."}
        for t in sg.REQUIRED_SECTION_TYPES
    ]

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "A Real Title", "sections": sections}, []

    monkeypatch.setattr(sg.llm_client, "chat_json", fake_chat_json)

    script = sg.generate_script(research)

    assert script["sources"] == research["sources"]
    assert [s["url"] for s in script["sources"]] == ["https://a.example.com", "https://b.example.com"]


def test_script_generator_prompt_tells_the_model_not_to_invent_facts():
    """A cheap, durable guard against someone loosening the system prompt
    later without noticing they removed the anti-fabrication instruction -
    spec section 7/11's core requirement."""
    assert "Do not invent facts" in sg.SYSTEM_PROMPT
    assert "uncertainties" in sg.SYSTEM_PROMPT.lower()


def test_researcher_prompt_forbids_inventing_sources():
    assert "Never invent a source" in researcher.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Research JSON robustness fix (production incident: a real web-searched
# research call was cut off mid-JSON by Claude's own max_tokens limit,
# raising `[CLAUDE] invalid JSON returned: Expecting ',' delimiter...`).
# researcher._chat_json_research() now retries - a completely fresh request
# each time - on both a distinguishable truncation error and a generic
# malformed-JSON error, up to settings.CLAUDE_RESEARCH_MAX_ATTEMPTS, and
# never returns/accepts partial data: it either returns a fully parsed
# result or raises loudly once every attempt is exhausted.
# ---------------------------------------------------------------------------

def test_research_topic_retries_after_truncated_response_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise researcher.llm_client.ClaudeTruncatedResponseError("truncated mid-JSON (stop_reason=max_tokens)")
        return _base_data(), []

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    data = researcher.research_topic("The Ocean Sound Scientists Still Can't Explain")

    assert calls["n"] == 2  # first attempt truncated, second (fresh) attempt succeeded
    assert data["topic"] == _base_data()["topic"]


def test_research_topic_retries_after_malformed_json_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise researcher.llm_client.ClaudeError("[CLAUDE] invalid JSON returned: Expecting ',' delimiter")
        return _base_data(), []

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    data = researcher.research_topic("Some Topic")

    assert calls["n"] == 2
    assert data["facts"] == _base_data()["facts"]


def test_research_topic_gives_up_after_exhausting_all_retry_attempts(monkeypatch):
    calls = {"n": 0}

    def always_truncated(system, user, max_tokens=4000, model=None, use_web_search=False):
        calls["n"] += 1
        raise researcher.llm_client.ClaudeTruncatedResponseError(f"truncated attempt {calls['n']}")

    monkeypatch.setattr(researcher.llm_client, "chat_json", always_truncated)

    with pytest.raises(researcher.llm_client.ClaudeError, match="gave up after"):
        researcher.research_topic("Some Topic")

    assert calls["n"] == researcher.settings.CLAUDE_RESEARCH_MAX_ATTEMPTS  # bounded, never retried forever


def test_research_topic_never_returns_partial_data_when_all_attempts_fail(monkeypatch):
    """Incomplete research data must never be silently accepted - confirms
    failure raises loudly instead of research_topic() ever constructing (or
    returning) a result built from a truncated/malformed attempt."""
    def always_truncated(system, user, max_tokens=4000, model=None, use_web_search=False):
        raise researcher.llm_client.ClaudeTruncatedResponseError("truncated")

    monkeypatch.setattr(researcher.llm_client, "chat_json", always_truncated)

    with pytest.raises(researcher.llm_client.ClaudeError):
        researcher.research_topic("Some Topic")


def test_research_topic_each_retry_is_a_fresh_request_not_a_reparse(monkeypatch):
    """Each attempt must be a brand-new chat_json call with the same fresh
    prompt - never an attempt to salvage or continue a prior truncated
    response. This is exactly what "retry the research request rather than
    parsing partial content" means."""
    seen_prompts = []

    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        seen_prompts.append(user)
        if len(seen_prompts) == 1:
            raise researcher.llm_client.ClaudeTruncatedResponseError("truncated")
        return _base_data(), []

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    researcher.research_topic("Ocean Sound Topic")

    assert len(seen_prompts) == 2
    assert seen_prompts[0] == seen_prompts[1]  # identical fresh prompt, not a patched/continued partial one


def test_research_topic_uses_the_configured_research_max_tokens(monkeypatch):
    """Root-cause mitigation: the old hard-coded max_tokens=4000 was too low
    for a real web-searched topic's full payload - research now uses the
    configurable (and more generous) settings.CLAUDE_RESEARCH_MAX_TOKENS."""
    seen = {}

    def fake_chat_json(system, user, max_tokens=4000, model=None, use_web_search=False):
        seen["max_tokens"] = max_tokens
        return _base_data(), []

    monkeypatch.setattr(researcher.llm_client, "chat_json", fake_chat_json)

    researcher.research_topic("Some Topic")

    assert seen["max_tokens"] == researcher.settings.CLAUDE_RESEARCH_MAX_TOKENS
    assert seen["max_tokens"] > 4000  # strictly more generous than the value that caused the incident


def test_research_topic_respects_a_lower_configured_max_attempts(monkeypatch):
    monkeypatch.setattr(researcher.settings, "CLAUDE_RESEARCH_MAX_ATTEMPTS", 1)
    calls = {"n": 0}

    def always_truncated(system, user, max_tokens=4000, model=None, use_web_search=False):
        calls["n"] += 1
        raise researcher.llm_client.ClaudeTruncatedResponseError("truncated")

    monkeypatch.setattr(researcher.llm_client, "chat_json", always_truncated)

    with pytest.raises(researcher.llm_client.ClaudeError):
        researcher.research_topic("Some Topic")

    assert calls["n"] == 1  # no retry at all when max_attempts=1 - preserves single-shot behavior when configured
