"""
Metadata generation. Deterministic/templated (no LLM call) - see
youtube/metadata.py's module docstring. Round 2 audit found this was the
only stage without db.start_run()/finish_run() run tracking or a
save_failure() on error at all (every other stage has both); brought in
line with the rest of the pipeline so a bug or a missing upstream file
here is recorded the same way as everywhere else, not silently uncaught.
"""
import pytest

from core.state import PipelineState
from storage import db, project_store as ps
from youtube import metadata as md


def _script():
    return {
        "title": "The Dyatlov Pass Incident",
        "hook": "Nine hikers. One mystery.",
        "sections": [
            {"type": "hook", "narration": "Nine hikers."},
            {"type": "background", "narration": "In 1959, nine experienced hikers set out."},
        ],
    }


def _research():
    return {"topic": "Dyatlov Pass", "sources": [{"title": "Report", "url": "https://example.com/report"}]}


def test_run_and_save_builds_and_persists_metadata(isolated_storage):
    paths, _ = ps.create_project("Metadata Happy Path Topic")
    ps.save_json(paths.script_json, _script())
    ps.save_json(paths.research_json, _research())

    result = md.run_and_save(paths, category="Mysteries")

    assert result["title"] == "The Dyatlov Pass Incident"
    assert ps.load_json(paths.metadata_json) == result
    row = db.get_video(paths.video_id)
    assert row is not None  # sanity: project row exists and stage ran against it


def test_run_and_save_records_failed_state_when_upstream_json_missing(isolated_storage):
    """Regression: previously this had NO db.start_run()/finish_run() and
    NO save_failure() at all (unlike every other stage) - a missing
    script.json/research.json, or any bug in build_metadata(), would raise
    straight out with no FAILED state ever recorded and no run history."""
    paths, _ = ps.create_project("Metadata Missing Upstream Topic")
    # neither script.json nor research.json written

    with pytest.raises(RuntimeError, match="script.json and research.json"):
        md.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "metadata"


def test_run_and_save_records_failed_state_on_a_real_build_error(isolated_storage):
    paths, _ = ps.create_project("Metadata Build Error Topic")
    ps.save_json(paths.script_json, {"sections": []})  # missing required "title" key
    ps.save_json(paths.research_json, _research())

    with pytest.raises(KeyError):
        md.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "metadata"
