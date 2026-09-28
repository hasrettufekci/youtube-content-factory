from core.state import PipelineState
from storage import project_store as ps


def test_slugify_basic():
    assert ps.slugify("The Mysterious Place on Earth") == "the-mysterious-place-on-earth"
    assert ps.slugify("The Mysterious Place on Earth Where Nobody Is Allowed to Go") == "the-mysterious-place-on-earth-where"


def test_slugify_handles_unicode_and_empty():
    assert ps.slugify("İzmir'deki Gizemli Yer!") != ""
    assert ps.slugify("!!!") == "untitled"


def test_normalize_topic_is_loose():
    assert ps.normalize_topic("The Dyatlov Pass Incident!") == ps.normalize_topic("the dyatlov pass incident")


def test_create_project_creates_folder_and_db_row(isolated_storage):
    paths, created = ps.create_project("The Dyatlov Pass Incident", category="Mysteries")
    assert created is True
    assert paths.root.exists()
    assert paths.root.name.startswith("the-dyatlov")


def test_create_project_is_idempotent_for_same_topic(isolated_storage):
    paths1, created1 = ps.create_project("The Dyatlov Pass Incident")
    paths2, created2 = ps.create_project("the dyatlov pass incident!!")  # same topic, different casing/punctuation
    assert created1 is True
    assert created2 is False
    assert paths1.video_id == paths2.video_id
    assert paths1.root == paths2.root


def test_create_project_force_new_bypasses_idempotency(isolated_storage):
    paths1, _ = ps.create_project("Same Topic")
    paths2, created2 = ps.create_project("Same Topic", force_new=True)
    assert created2 is True
    assert paths1.video_id != paths2.video_id
    assert paths1.slug != paths2.slug


def test_json_round_trip(isolated_storage):
    paths, _ = ps.create_project("Round Trip Topic")
    data = {"topic": "x", "facts": ["a", "b"]}
    ps.save_json(paths.research_json, data)
    assert ps.load_json(paths.research_json) == data


def test_state_save_and_load(isolated_storage):
    paths, _ = ps.create_project("State Topic")
    assert ps.load_state(paths) == PipelineState.TOPIC_CREATED
    ps.save_state(paths, PipelineState.RESEARCHED)
    assert ps.load_state(paths) == PipelineState.RESEARCHED


def test_save_failure_sets_failed_state_with_reason(isolated_storage):
    paths, _ = ps.create_project("Failing Topic")
    ps.save_failure(paths, "video_assembly", "ffmpeg exited 1")
    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "video_assembly"
    assert "ffmpeg" in data["error"]


def test_scene_image_paths_are_zero_padded(isolated_storage):
    paths, _ = ps.create_project("Scene Path Topic")
    p1 = paths.scene_image(1)
    p32 = paths.scene_image(32)
    assert p1.name == "scene_001.png"
    assert p32.name == "scene_032.png"
