import pytest

from core.state import PipelineState
from storage import project_store as ps
from youtube import script_generator as sg


def _valid_sections(words_per_section=150):
    return [
        {"type": t, "title": t, "narration": " ".join(["word"] * words_per_section) + "."}
        for t in sg.REQUIRED_SECTION_TYPES
    ]


def test_generate_script_recomputes_full_narration_from_sections(monkeypatch):
    sections = _valid_sections(words_per_section=170)  # 8 * 170 = 1360 words, inside target range

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "A Real Documentary Title", "sections": sections}, []

    monkeypatch.setattr(sg.llm_client, "chat_json", fake_chat_json)

    research = {"topic": "Test Topic", "summary": "s", "facts": ["f1"], "sources": [{"url": "https://x.test"}]}
    script = sg.generate_script(research)

    assert script["title"] == "A Real Documentary Title"
    assert script["full_narration"] == "\n\n".join(s["narration"] for s in sections)
    assert script["word_count"] == 1360
    assert script["sources"] == research["sources"]
    assert script["hook"] == sections[0]["narration"]


def test_generate_script_rejects_missing_section(monkeypatch):
    sections = _valid_sections()[:-1]  # drop "cta"

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "T", "sections": sections}, []

    monkeypatch.setattr(sg.llm_client, "chat_json", fake_chat_json)

    with pytest.raises(RuntimeError, match="missing required section"):
        sg.generate_script({"topic": "Test Topic"})


def test_generate_script_rejects_wildly_short_output(monkeypatch):
    sections = _valid_sections(words_per_section=5)  # only 40 words total - not a real script

    def fake_chat_json(system, user, max_tokens=6000, model=None, use_web_search=False):
        return {"title": "T", "sections": sections}, []

    monkeypatch.setattr(sg.llm_client, "chat_json", fake_chat_json)

    with pytest.raises(RuntimeError, match="word count"):
        sg.generate_script({"topic": "Test Topic"})


def test_run_and_save_records_failed_state_when_research_json_missing(isolated_storage):
    """Regression: this precondition check used to raise BEFORE
    db.start_run()/the try block, so it never went through
    save_failure() - the video was left stuck with no FAILED state
    recorded and no diagnostic trail in state.json."""
    paths, _ = ps.create_project("Missing Research JSON Topic")
    # research.json deliberately not written

    with pytest.raises(RuntimeError, match="no research.json found"):
        sg.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "script"
