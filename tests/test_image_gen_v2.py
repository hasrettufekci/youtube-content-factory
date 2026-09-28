"""
Visual Generator V2 integration - image_gen.py's strategy-aware dispatch
layer (_generate_image_bytes_for_scene / programmatic_graphic_renderer
/ _write_image_atomic) sitting on top of the existing ImageProvider
abstraction.

Deliberately does NOT make any real FAL/network call - every test here uses
either a fake in-process ImageProvider or IMAGE_PROVIDER=mock (Pillow-based,
offline). See test_image_gen.py for the pre-existing v1/provider-mechanics
tests (retry, atomic write, corrupt-file detection, FAL HTTP retry rules),
all of which must keep passing unchanged after this refactor.

Covers spec categories C-N:
    C - ai_generated calls provider.generate()
    D - ai_generated_plus_overlay calls provider.generate(), overlay
        metadata (exact_text/data_points/overlay_required) is not lost
    E - programmatic_graphic NEVER calls provider.generate(None, ...)
        (in fact: never calls provider.generate() at all)
    F - programmatic_graphic produces a valid PNG fallback of the right size
    G - research_evidence scene_type fallback-transparency logging
    H - document_recreation scene_type fallback-transparency logging
        (stands in for the structurally-unreachable literal
        "archival_source" - see resolve_visual_strategy()'s rule 0 in
        scene_planner.py)
    I - exact_text/data_points never leak into the provider prompt
    J - an existing valid PNG is left alone (provider/placeholder not
        re-invoked), for both an AI strategy and programmatic_graphic
    K - MockImageProvider (IMAGE_PROVIDER=mock) handles every strategy
        network-free
    L - every scene ends with a file at the assembler's expected path,
        whatever strategy produced it
    M - v1-shaped scenes (no visual_strategy key at all) still work,
        defaulting to ai_generated - full backward compatibility
    N - resume/atomic-write behavior holds across a mix of strategies
"""
import pytest

from core.config import settings
from storage import project_store as ps
from youtube import image_gen


def _v2_scene(scene_id, visual_strategy="ai_generated", scene_type="real_location",
              visual_prompt="a prompt", exact_text=None, data_points=None, overlay_required=False):
    return {
        "scene_id": scene_id,
        "schema_version": 2,
        "visual_strategy": visual_strategy,
        "scene_type": scene_type,
        "visual_prompt": visual_prompt,
        "exact_text": exact_text or [],
        "data_points": data_points or [],
        "overlay_required": overlay_required,
        "claim": f"claim for scene {scene_id}",
        "narration_en": f"narration {scene_id}",
    }


class _CountingProvider(image_gen.ImageProvider):
    """Records every prompt it's asked to generate and returns a real,
    distinct, decodable PNG at the requested size - no network, no actual
    AI generation. Must be a genuine image (not opaque placeholder bytes)
    because the overlay renderer (youtube/overlay_renderer.py) opens
    whatever a provider returns with Pillow whenever overlay_required is
    True; a real provider always returns real image bytes, so this fake
    does too."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        from io import BytesIO
        from PIL import Image

        self.calls.append(prompt)
        color = (10 + len(self.calls) % 200, 20, 30)
        img = Image.new("RGB", (width, height), color)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


class _NeverCallProvider(image_gen.ImageProvider):
    def generate(self, prompt, width, height):
        raise AssertionError(f"provider.generate() must not be called here (got prompt={prompt!r})")


# ---------------------------------------------------------------------------
# C - ai_generated
# ---------------------------------------------------------------------------

def test_ai_generated_calls_provider_generate_with_visual_prompt(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("AI Generated Strategy Topic")
    scene = _v2_scene(1, visual_strategy="ai_generated", visual_prompt="a telescope at night")
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    image_gen.generate_scene_images(paths, [scene])

    assert provider.calls == ["a telescope at night"]
    assert paths.scene_image(1).exists()


# ---------------------------------------------------------------------------
# D - ai_generated_plus_overlay
# ---------------------------------------------------------------------------

def test_overlay_strategy_calls_provider_and_preserves_metadata(isolated_storage, monkeypatch, caplog):
    paths, _ = ps.create_project("Overlay Strategy Topic")
    scene = _v2_scene(
        7, visual_strategy="ai_generated_plus_overlay", scene_type="scientific_graphic",
        visual_prompt="a plot of signal intensity over time, no readable text or labels",
        exact_text=["6EQUJ5", "1420.726 MHz"],
        data_points=[{"data_type": "exact_text", "value": "6EQUJ5", "exact_render": True}],
        overlay_required=True,
    )
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    with caplog.at_level("INFO", logger="ycf.image_gen"):
        image_gen.generate_scene_images(paths, [scene])

    assert provider.calls == ["a plot of signal intensity over time, no readable text or labels"]
    # Overlay metadata is on the scene dict the caller passed in - the
    # dispatch layer must never mutate it away; a future overlay renderer
    # reads these fields off scenes.json, not off anything image_gen writes.
    assert scene["exact_text"] == ["6EQUJ5", "1420.726 MHz"]
    assert scene["overlay_required"] is True
    assert any("[IMAGE][OVERLAY]" in r.message and "rendered" in r.message for r in caplog.records)
    assert paths.scene_image(7).exists()
    # _CountingProvider always returns a single FLAT color - if the saved
    # file has more than one distinct color, the overlay renderer really
    # drew something on top of it, not just logged a claim that it did.
    from PIL import Image
    saved = Image.open(paths.scene_image(7)).convert("RGB")
    assert len(saved.getcolors(maxcolors=saved.width * saved.height)) > 1


# ---------------------------------------------------------------------------
# E / F - programmatic_graphic
# ---------------------------------------------------------------------------

def test_programmatic_graphic_never_calls_provider_generate(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Programmatic Never Calls Provider Topic")
    scene = _v2_scene(
        11, visual_strategy="programmatic_graphic", scene_type="scientific_graphic",
        visual_prompt=None, data_points=[{"data_type": "time_series", "value": "intensity curve"}],
    )
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    image_gen.generate_scene_images(paths, [scene])  # must not raise, must not call the provider

    assert paths.scene_image(11).exists()


def test_programmatic_graphic_produces_valid_png_with_correct_dimensions(isolated_storage, monkeypatch, caplog):
    paths, _ = ps.create_project("Programmatic Valid PNG Topic")
    scene = _v2_scene(
        11, visual_strategy="programmatic_graphic", visual_prompt=None,
        data_points=[{"data_type": "curve", "value": "x"}],
    )
    monkeypatch.setattr(image_gen, "get_provider", lambda: _CountingProvider())

    with caplog.at_level("INFO", logger="ycf.image_gen"):
        image_gen.generate_scene_images(paths, [scene])

    from PIL import Image
    img = Image.open(paths.scene_image(11))
    assert img.format == "PNG"
    assert img.size == (image_gen.WIDTH, image_gen.HEIGHT)
    assert any("[IMAGE][PROGRAMMATIC]" in r.message and "renderer=" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# G / H - scene_type-based fallback-transparency logging (orthogonal to
# visual_strategy routing - both still hit ai_generated here, since no real
# research/archival source system exists yet)
# ---------------------------------------------------------------------------

def test_research_evidence_scene_type_falls_back_to_ai_with_transparent_log(isolated_storage, monkeypatch, caplog):
    paths, _ = ps.create_project("Research Evidence Fallback Topic")
    scene = _v2_scene(
        25, visual_strategy="ai_generated", scene_type="research_evidence",
        visual_prompt="an archival research paper laid out on a desk",
    )
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    with caplog.at_level("INFO", logger="ycf.image_gen"):
        image_gen.generate_scene_images(paths, [scene])

    assert provider.calls == ["an archival research paper laid out on a desk"]
    assert any("[IMAGE][RESEARCH_EVIDENCE][FALLBACK=AI]" in r.message for r in caplog.records)


def test_document_recreation_scene_type_falls_back_to_ai_with_transparent_log(isolated_storage, monkeypatch, caplog):
    # document_recreation stands in for the literal "archival_source" -
    # resolve_visual_strategy()'s rule 0 (scene_planner.py) unconditionally
    # normalizes archival_source -> document_recreation, so a stored/
    # validated scene can never actually carry the literal value.
    paths, _ = ps.create_project("Archival Fallback Topic")
    scene = _v2_scene(
        29, visual_strategy="ai_generated", scene_type="document_recreation",
        visual_prompt="a recreated historical newspaper clipping",
    )
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    with caplog.at_level("INFO", logger="ycf.image_gen"):
        image_gen.generate_scene_images(paths, [scene])

    assert provider.calls == ["a recreated historical newspaper clipping"]
    assert any("[IMAGE][ARCHIVAL][FALLBACK=AI]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# I - exact_text leakage
# ---------------------------------------------------------------------------

def test_exact_text_never_leaks_into_provider_prompt(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Exact Text No Leak Topic")
    scene = _v2_scene(
        13, visual_strategy="ai_generated_plus_overlay", scene_type="scientific_graphic",
        visual_prompt="a radio telescope dish under a night sky, no readable text or labels",
        exact_text=["6EQUJ5", "1420.726 MHz", "RA 19h 25m", "Dec -27°03'"],
        data_points=[{"data_type": "exact_text", "value": "6EQUJ5", "exact_render": True}],
        overlay_required=True,
    )
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    image_gen.generate_scene_images(paths, [scene])

    sent_prompt = provider.calls[0]
    assert sent_prompt == "a radio telescope dish under a night sky, no readable text or labels"
    for forbidden in ("6EQUJ5", "1420.726 MHz", "RA 19h 25m", "Dec -27"):
        assert forbidden not in sent_prompt


# ---------------------------------------------------------------------------
# J - existing image -> provider/placeholder not re-invoked
# ---------------------------------------------------------------------------

def test_existing_valid_png_skips_regeneration_for_v2_ai_generated_scene(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Existing PNG V2 Skip Topic")
    scene = _v2_scene(3, visual_strategy="ai_generated", visual_prompt="should never be sent to the provider")
    paths.scene_image(3).write_bytes(b"already-there")
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    image_gen.generate_scene_images(paths, [scene])

    assert paths.scene_image(3).read_bytes() == b"already-there"


def test_existing_valid_png_skips_programmatic_renderer_regeneration(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Existing PNG Programmatic Skip Topic")
    scene = _v2_scene(11, visual_strategy="programmatic_graphic", visual_prompt=None)
    paths.scene_image(11).write_bytes(b"already-there-placeholder")

    calls = {"render": 0}
    original = image_gen.programmatic_graphic_renderer.render_programmatic_graphic

    def _spy(*a, **k):
        calls["render"] += 1
        return original(*a, **k)

    monkeypatch.setattr(image_gen.programmatic_graphic_renderer, "render_programmatic_graphic", _spy)
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    image_gen.generate_scene_images(paths, [scene])

    assert calls["render"] == 0
    assert paths.scene_image(11).read_bytes() == b"already-there-placeholder"


# ---------------------------------------------------------------------------
# K - MockImageProvider, every strategy, network-free
# ---------------------------------------------------------------------------

def test_mock_provider_handles_all_strategies_network_free(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "mock")
    paths, _ = ps.create_project("Mock All Strategies Topic")
    scenes = [
        _v2_scene(1, visual_strategy="ai_generated", visual_prompt="a real location scene"),
        _v2_scene(2, visual_strategy="ai_generated_plus_overlay", scene_type="scientific_graphic",
                   visual_prompt="a scene with overlay data pending", exact_text=["X"], overlay_required=True),
        _v2_scene(3, visual_strategy="programmatic_graphic", visual_prompt=None,
                   data_points=[{"data_type": "curve", "value": "y"}]),
        _v2_scene(4, visual_strategy="hypothesis_visualization", scene_type="hypothesis_diagram",
                   visual_prompt="a hypothesis diagram"),
        _v2_scene(5, visual_strategy="ai_generated", scene_type="research_evidence", visual_prompt="research desk"),
        _v2_scene(6, visual_strategy="ai_generated", scene_type="document_recreation", visual_prompt="archival doc"),
    ]

    result = image_gen.generate_scene_images(paths, scenes)

    assert len(result) == 6
    assert all(p.exists() and p.stat().st_size > 0 for p in result)


# ---------------------------------------------------------------------------
# L - assembler's contract: a file at scene_image(scene_id) for every scene
# ---------------------------------------------------------------------------

def test_every_scene_ends_with_a_file_at_its_expected_assembler_path(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Assembler Contract Topic")
    scenes = [
        _v2_scene(1, visual_strategy="ai_generated", visual_prompt="a"),
        _v2_scene(2, visual_strategy="ai_generated_plus_overlay", scene_type="scientific_graphic",
                   visual_prompt="b", exact_text=["X"]),
        _v2_scene(3, visual_strategy="programmatic_graphic", visual_prompt=None,
                   data_points=[{"data_type": "curve"}]),
    ]
    monkeypatch.setattr(image_gen, "get_provider", lambda: _CountingProvider())

    result = image_gen.generate_scene_images(paths, scenes)

    for scene in scenes:
        expected = paths.scene_image(scene["scene_id"])
        assert expected.exists()
        assert expected in result


# ---------------------------------------------------------------------------
# M - v1 backward compatibility (no visual_strategy key at all)
# ---------------------------------------------------------------------------

def test_v1_scenes_without_visual_strategy_default_to_ai_generated(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("V1 Backward Compat Topic")
    scene = {"scene_id": 1, "visual_prompt": "a v1-shaped scene, no visual_strategy key at all"}
    provider = _CountingProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    image_gen.generate_scene_images(paths, [scene])

    assert provider.calls == ["a v1-shaped scene, no visual_strategy key at all"]
    assert paths.scene_image(1).exists()


# ---------------------------------------------------------------------------
# N - resume/atomic-write behavior across a mix of strategies
# ---------------------------------------------------------------------------

def test_resume_retries_only_failed_scene_across_mixed_strategies(isolated_storage, monkeypatch):
    paths, _ = ps.create_project("Mixed Strategy Resume Topic")
    scenes = [
        _v2_scene(1, visual_strategy="ai_generated", visual_prompt="a"),
        _v2_scene(2, visual_strategy="programmatic_graphic", visual_prompt=None,
                   data_points=[{"data_type": "curve"}]),
        _v2_scene(3, visual_strategy="ai_generated", visual_prompt="c"),
    ]

    class _FailOnceProvider(image_gen.ImageProvider):
        def __init__(self):
            self.calls = []
            self.failed_once = False

        def generate(self, prompt, width, height):
            self.calls.append(prompt)
            if prompt == "c" and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("simulated transient failure")
            return f"bytes-{prompt}".encode()

    provider = _FailOnceProvider()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    with pytest.raises(RuntimeError):
        image_gen.generate_scene_images(paths, scenes)

    assert paths.scene_image(1).exists()      # ai_generated succeeded
    assert paths.scene_image(2).exists()      # programmatic placeholder succeeded, no provider call
    assert not paths.scene_image(3).exists()  # failed, not written

    result = image_gen.generate_scene_images(paths, scenes)  # resume

    assert len(result) == 3
    assert paths.scene_image(3).exists()
    assert not paths.scene_image(3).with_name(paths.scene_image(3).name + ".tmp").exists()
    # scene 1/2 were never re-requested from the provider on resume
    assert provider.calls == ["a", "c", "c"]
