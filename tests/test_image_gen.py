"""
Scene image generation (spec Round 2 section 6):

    "Bir scene image generation basarisiz olursa basarili scene'leri tekrar
    uretme. Retry sadece basarisiz scene uzerinde yapilmalir."
    (If one scene image generation fails, don't regenerate successful
    scenes - retry only the failed scene.)

Also covers: fal.ai HTTP retry-with-backoff on transient errors (429/5xx/
timeout), atomic scene-image writes, and corrupt/empty existing-file
detection so a crash mid-write is never mistaken for a completed scene.
"""
import httpx
import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import image_gen


def _scenes(n=3):
    return [
        {"scene_id": i, "visual_prompt": f"prompt for scene {i}", "narration": f"narration {i}"}
        for i in range(1, n + 1)
    ]


class _CountingProvider(image_gen.ImageProvider):
    """Fails on scenes listed in `fail_on` the first time they're
    requested, succeeds on every other call. Records every prompt it was
    asked to generate, so tests can assert exactly which scenes were
    (re)generated."""

    def __init__(self, fail_once_on=()):
        self.fail_once_on = set(fail_once_on)
        self.already_failed = set()
        self.calls = []

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        self.calls.append(prompt)
        scene_id = int(prompt.split()[-1])
        if scene_id in self.fail_once_on and scene_id not in self.already_failed:
            self.already_failed.add(scene_id)
            raise RuntimeError(f"[FAL] simulated transient failure for scene {scene_id}")
        return f"fake-png-bytes-{scene_id}".encode()


def test_failed_scene_does_not_regenerate_already_successful_scenes(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Retry Only Failed Scene Topic")
    scenes = _scenes(3)
    provider = _CountingProvider(fail_once_on={2})
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    with pytest.raises(RuntimeError):
        image_gen.generate_scene_images(paths, scenes)

    # scene 1 succeeded and was written to disk; scene 2 failed (and, since
    # generation is sequential and the exception propagates immediately,
    # scene 3 was never even attempted in this run - both are absent).
    assert paths.scene_image(1).exists()
    assert not paths.scene_image(2).exists()
    assert not paths.scene_image(3).exists()
    assert provider.calls == ["prompt for scene 1", "prompt for scene 2"]

    # Second run (simulating pipeline resume/retry): scene 1 must NOT be
    # regenerated (it's already done); only scene 2 (previously failed)
    # and scene 3 (never attempted) should be requested from the provider.
    result = image_gen.generate_scene_images(paths, scenes)

    assert provider.calls == ["prompt for scene 1", "prompt for scene 2", "prompt for scene 2", "prompt for scene 3"]
    assert len(result) == 3
    assert all(p.exists() for p in result)


def test_empty_existing_file_is_treated_as_not_done_and_regenerated(isolated_storage, monkeypatch):
    """Simulates a crash that left a zero-byte scene_NNN.png on disk from
    an earlier interrupted run - it must be detected as incomplete and
    regenerated, never silently skipped as "already done"."""
    paths, _ = ps.create_project("Corrupt Existing Image Topic")
    scenes = _scenes(2)

    # Scene 1 has a real (complete) file already; scene 2 has a truncated
    # zero-byte file left over from an interrupted run.
    paths.scene_image(1).write_bytes(b"already-complete-image-bytes")
    paths.scene_image(2).touch()  # zero bytes
    assert paths.scene_image(2).stat().st_size == 0

    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    result = image_gen.generate_scene_images(paths, scenes)

    # Only the corrupt scene 2 was regenerated - scene 1's untouched bytes
    # prove it was never re-requested from the provider.
    assert provider.calls == ["prompt for scene 2"]
    assert paths.scene_image(1).read_bytes() == b"already-complete-image-bytes"
    assert paths.scene_image(2).stat().st_size > 0
    assert len(result) == 2


def test_scene_image_write_is_atomic_no_tmp_file_left_behind(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Atomic Write Topic")
    scenes = _scenes(1)
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    image_gen.generate_scene_images(paths, scenes)

    assert paths.scene_image(1).exists()
    assert not paths.scene_image(1).with_name(paths.scene_image(1).name + ".tmp").exists()


def test_provider_returning_empty_bytes_raises_instead_of_writing_empty_file(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Empty Provider Response Topic")
    scenes = _scenes(1)

    class _EmptyProvider(image_gen.ImageProvider):
        def generate(self, prompt, width, height):
            return b""

    monkeypatch.setattr(image_gen, "get_provider", lambda: _EmptyProvider())

    with pytest.raises(RuntimeError, match="empty image data"):
        image_gen.generate_scene_images(paths, scenes)

    assert not paths.scene_image(1).exists()


def test_fal_provider_retries_on_transient_5xx_then_succeeds(monkeypatch):
    """FalImageProvider.generate() must retry a transient 503 on the
    submit call (mirroring youtube/llm_client.py and youtube/tts.py's
    retry behavior) rather than failing on the first hiccup."""
    monkeypatch.setattr(settings, "FAL_KEY", "test-fal-key")
    monkeypatch.setattr(settings, "FAL_MAX_RETRIES", 3)
    monkeypatch.setattr(image_gen.time, "sleep", lambda *_: None)

    attempts = {"submit": 0}

    class _FakeResponse:
        def __init__(self, status_code=200, json_data=None, content=b"", text=""):
            self.status_code = status_code
            self._json = json_data or {}
            self.content = content
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://example.test")
                raise httpx.HTTPStatusError(
                    f"error {self.status_code}", request=request,
                    response=httpx.Response(self.status_code, request=request, text=self.text),
                )

        def json(self):
            return self._json

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            attempts["submit"] += 1
            if attempts["submit"] < 2:
                return _FakeResponse(status_code=503, text="service unavailable")
            return _FakeResponse(status_code=200, json_data={
                "status_url": "https://example.test/status",
                "response_url": "https://example.test/result",
            })

        def get(self, url, headers=None):
            if url == "https://example.test/status":
                return _FakeResponse(status_code=200, json_data={"status": "COMPLETED"})
            return _FakeResponse(status_code=200, json_data={"images": [{"url": "https://example.test/img.png"}]})

    monkeypatch.setattr(image_gen.httpx, "Client", _FakeClient)
    monkeypatch.setattr(
        image_gen.httpx, "get",
        lambda url, timeout=None: _FakeResponse(status_code=200, content=b"real-image-bytes"),
    )

    result = image_gen.FalImageProvider().generate("a prompt", 1920, 1080)

    assert attempts["submit"] == 2  # first 503 retried, second attempt succeeded
    assert result == b"real-image-bytes"


def test_fal_provider_does_not_retry_non_retryable_4xx(monkeypatch):
    monkeypatch.setattr(settings, "FAL_KEY", "test-fal-key")
    monkeypatch.setattr(settings, "FAL_MAX_RETRIES", 3)
    monkeypatch.setattr(image_gen.time, "sleep", lambda *_: None)

    attempts = {"submit": 0}

    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://example.test")
                raise httpx.HTTPStatusError(
                    f"error {self.status_code}", request=request,
                    response=httpx.Response(self.status_code, request=request, text=self.text),
                )

        def json(self):
            return {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            attempts["submit"] += 1
            return _FakeResponse(status_code=400, text="bad prompt")

        def get(self, url, headers=None):
            raise AssertionError("should never poll status after a non-retryable submit failure")

    monkeypatch.setattr(image_gen.httpx, "Client", _FakeClient)

    with pytest.raises(RuntimeError, match="400"):
        image_gen.FalImageProvider().generate("a prompt", 1920, 1080)

    assert attempts["submit"] == 1  # no retry burned on a non-retryable 4xx


def test_run_and_save_records_failed_state_when_scenes_json_missing(isolated_storage):
    """Regression: this precondition check used to raise BEFORE
    db.start_run()/the try block, so it never went through
    save_failure() - the video was left stuck with no FAILED state
    recorded and no diagnostic trail in state.json."""
    paths, _ = ps.create_project("Missing Scenes JSON Topic")
    # scenes.json deliberately not written

    with pytest.raises(RuntimeError, match="no scenes.json found"):
        image_gen.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "images"
