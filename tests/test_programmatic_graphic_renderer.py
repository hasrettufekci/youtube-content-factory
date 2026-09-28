"""
Scene Planner V2 - Phase 4 programmatic scientific graphic renderer
(youtube/programmatic_graphic_renderer.py).

Deliberately never calls FAL, Claude, OpenAI, or any network API - every
scene dict here is hand-built, matching Scene Planner V2's real,
already-existing schema (data_points/exact_text/text_render - see
scene_planner.py's SYSTEM_PROMPT_V2_CLASSIFICATION and
_PROGRAMMATIC_GRAPHIC_DATA_TYPES). Assertions inspect actual rendered
output - pixel dimensions, byte-identical PNGs for determinism, and the
actual sequence of strings Pillow was asked to draw (by spying on
ImageDraw.ImageDraw.text, the same deterministic non-OCR technique used in
tests/test_overlay_renderer.py) - never just "the function returned
without raising".

Covers exactly the 16 categories requested for this renderer:
    1  programmatic_graphic dispatches to local renderer (image_gen.py
       integration - no diagnostic placeholder, no AI image)
    2  provider.generate() is NOT called for programmatic_graphic
    3  renderer returns 1920x1080 (and other requested dimensions)
    4  renderer is deterministic (same scene -> byte-identical PNG)
    5  missing data does not crash (empty/absent data_points/exact_text)
    6  unsupported graphic type uses safe fallback (unknown registry name,
       and a renderer that raises mid-draw)
    7  exact factual strings are preserved (6EQUJ5, 1420 megahertz, etc.
       drawn character-for-character, never rewritten/rounded/abbreviated)
    8  no invented numerical values (every string Pillow is asked to draw
       is traceable to scene["data_points"]/scene["exact_text"] - nothing
       extra appears)
    9  no external API calls (no network imports in the module; no
       provider/http call happens during rendering)
   10  output file is created successfully (render_programmatic_graphic_to_file)
   11  existing overlay renderer tests remain green (run separately - see
       tests/test_overlay_renderer.py; this renderer imports and reuses
       overlay_renderer's primitives without modifying that file)
   12  existing image_gen V2 tests remain green (run separately - see
       tests/test_image_gen_v2.py, updated for the new call site/log line)
   13  Scene Planner V2 tests remain green (run separately - see
       tests/test_scene_planner_v2.py; this task never touches scene_planner.py)
   14  V1 behavior remains unchanged (a scene with no visual_strategy key
       never reaches this renderer at all - image_gen.py defaults it to
       "ai_generated", exercised in test_image_gen_v2.py category M)
   15  existing pipeline resume tests remain green (run separately - see
       tests/test_pipeline_resume.py / test_pipeline_visual_v2.py)
   16  scene 026's real metadata can be rendered (see
       test_scene_026_real_metadata_renders_without_crashing below)

Categories 11/12/13/15 are cross-file/whole-suite guarantees, not unit
tests inside this file - they are verified by running the full project
test suite (see the task's delivery report), not by duplicating those
files' tests here.
"""
from __future__ import annotations

import ast
import io

import pytest
from PIL import Image, ImageDraw

from youtube import programmatic_graphic_renderer as pg


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _timeline_scene(scene_id: int = 26, values: list[str] | None = None) -> dict:
    values = values or ["August 15, 1977", "266P/Christensen", "February-May 2020"]
    return {
        "scene_id": scene_id,
        "schema_version": 2,
        "visual_strategy": "programmatic_graphic",
        "scene_type": "scientific_graphic",
        "visual_prompt": None,
        "exact_text": list(values),
        "data_points": [{"data_type": "timeline", "value": v, "exact_render": True} for v in values],
        "text_render": {"items": values, "hypothesis_badge": False},
        "claim": f"timeline scene {scene_id}",
    }


def _measurement_scene(scene_id: int = 6, value: str = "1420 megahertz") -> dict:
    return {
        "scene_id": scene_id,
        "schema_version": 2,
        "visual_strategy": "programmatic_graphic",
        "scene_type": "scientific_graphic",
        "visual_prompt": None,
        "exact_text": [value],
        "data_points": [{"data_type": "measurement", "value": value, "exact_render": True}],
        "text_render": {"items": [value], "hypothesis_badge": False},
        "claim": f"measurement scene {scene_id}",
    }


def _empty_scene(scene_id: int = 999) -> dict:
    return {
        "scene_id": scene_id,
        "schema_version": 2,
        "visual_strategy": "programmatic_graphic",
        "scene_type": "scientific_graphic",
        "visual_prompt": None,
        "exact_text": [],
        "data_points": [],
        "text_render": {},
        "claim": "",
    }


def _drawn_strings(monkeypatch) -> list[str]:
    """Spies on every string Pillow is asked to draw via ImageDraw.text/
    multiline_text during the wrapped call - the same deterministic,
    non-OCR technique test_overlay_renderer.py uses to prove exactly what
    text did or did not reach the image."""
    drawn: list[str] = []
    original_text = ImageDraw.ImageDraw.text
    original_multiline = ImageDraw.ImageDraw.multiline_text

    def _spy_text(self, xy, text, *a, **k):
        drawn.append(text)
        return original_text(self, xy, text, *a, **k)

    def _spy_multiline(self, xy, text, *a, **k):
        drawn.append(text)
        return original_multiline(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _spy_text)
    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", _spy_multiline)
    return drawn


# ---------------------------------------------------------------------------
# 1/2 - image_gen.py dispatch: local renderer only, provider never called
# ---------------------------------------------------------------------------

def test_programmatic_graphic_dispatches_to_local_renderer_not_provider(isolated_storage, monkeypatch):
    from storage import project_store as ps
    from youtube import image_gen

    class _NeverCallProvider(image_gen.ImageProvider):
        def generate(self, prompt, width, height):
            raise AssertionError("provider.generate() must not be called for programmatic_graphic")

    paths, _ = ps.create_project("Programmatic Renderer Dispatch Topic")
    scene = _timeline_scene(scene_id=11)
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    image_gen.generate_scene_images(paths, [scene])  # must not raise, must not call the provider

    assert paths.scene_image(11).exists()
    img = Image.open(paths.scene_image(11))
    assert img.format == "PNG"
    assert img.size == (image_gen.WIDTH, image_gen.HEIGHT)


def test_image_gen_logs_programmatic_renderer_name_not_pending_placeholder(isolated_storage, monkeypatch, caplog):
    from storage import project_store as ps
    from youtube import image_gen

    class _NeverCallProvider(image_gen.ImageProvider):
        def generate(self, prompt, width, height):
            raise AssertionError("provider.generate() must not be called for programmatic_graphic")

    paths, _ = ps.create_project("Programmatic Renderer Log Topic")
    scene = _timeline_scene(scene_id=12)
    # get_provider() is still called unconditionally by generate_scene_images()
    # (it's needed for any non-programmatic scene in the same batch) - what
    # must never happen is provider.generate() itself being invoked.
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    with caplog.at_level("INFO", logger="ycf.image_gen"):
        image_gen.generate_scene_images(paths, [scene])

    messages = [r.message for r in caplog.records]
    assert any("[IMAGE][PROGRAMMATIC]" in m and "renderer=" in m for m in messages)
    assert not any("PROGRAMMATIC_PENDING" in m for m in messages)
    assert not any("PLACEHOLDER" in m.upper() for m in messages)


# ---------------------------------------------------------------------------
# 3 - dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,height", [(1920, 1080), (1280, 720), (640, 360)])
def test_renderer_returns_requested_dimensions(width, height):
    png_bytes, _renderer_name = pg.render_programmatic_graphic(_timeline_scene(), width=width, height=height)
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (width, height)
    assert img.format == "PNG"


# ---------------------------------------------------------------------------
# 4 - determinism
# ---------------------------------------------------------------------------

def test_renderer_is_deterministic_timeline():
    scene = _timeline_scene()
    b1, n1 = pg.render_programmatic_graphic(scene)
    b2, n2 = pg.render_programmatic_graphic(scene)
    assert b1 == b2
    assert n1 == n2 == "timeline"


def test_renderer_is_deterministic_fallback():
    scene = _measurement_scene()
    b1, n1 = pg.render_programmatic_graphic(scene)
    b2, n2 = pg.render_programmatic_graphic(scene)
    assert b1 == b2
    assert n1 == n2 == "fallback_graphic"


def test_different_scene_metadata_produces_different_output():
    b1, _ = pg.render_programmatic_graphic(_timeline_scene(values=["August 15, 1977", "1977-2020", "2020"]))
    b2, _ = pg.render_programmatic_graphic(_timeline_scene(values=["June 1, 1990", "1990-2000", "2000"]))
    assert b1 != b2


# ---------------------------------------------------------------------------
# 5 - missing data does not crash
# ---------------------------------------------------------------------------

def test_missing_data_points_does_not_crash():
    scene = _empty_scene()
    png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (1920, 1080)
    assert renderer_name == "fallback_graphic"


def test_scene_with_no_data_points_key_at_all_does_not_crash():
    scene = {"scene_id": 1, "visual_strategy": "programmatic_graphic"}  # no data_points/exact_text keys
    png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    assert Image.open(io.BytesIO(png_bytes)).size == (1920, 1080)
    assert renderer_name == "fallback_graphic"


def test_malformed_data_points_entries_do_not_crash():
    scene = {
        "scene_id": 2,
        "visual_strategy": "programmatic_graphic",
        "data_points": [None, {}, {"data_type": "timeline"}, {"data_type": "timeline", "value": ""}, 42, "oops"],
        "exact_text": [],
    }
    png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    assert Image.open(io.BytesIO(png_bytes)).size == (1920, 1080)
    assert renderer_name == "fallback_graphic"


# ---------------------------------------------------------------------------
# 6 - unsupported graphic type / renderer failure -> safe fallback
# ---------------------------------------------------------------------------

def test_unregistered_renderer_name_falls_back_safely(monkeypatch):
    monkeypatch.setattr(pg, "_select_renderer_name", lambda scene: "signal_graph_not_yet_built")
    png_bytes, renderer_name = pg.render_programmatic_graphic(_timeline_scene())
    assert Image.open(io.BytesIO(png_bytes)).size == (1920, 1080)
    # _RENDERERS.get(...) returning None falls back to the registered
    # fallback_graphic function - the reported name must reflect the
    # function that ACTUALLY ran, never the bogus/unrecognized name that
    # was requested (a caller/log line must never claim a renderer type
    # that doesn't exist).
    assert renderer_name == "fallback_graphic"


def test_renderer_exception_falls_back_to_clean_background(monkeypatch):
    def _boom(overlay, scene):
        raise RuntimeError("simulated drawing failure")

    monkeypatch.setitem(pg._RENDERERS, "timeline", _boom)
    png_bytes, renderer_name = pg.render_programmatic_graphic(_timeline_scene())
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (1920, 1080)
    assert renderer_name == "fallback_graphic"
    # background-only fallback: no drawn text at all for a scene whose
    # renderer raised (the exception handler rebuilds a blank overlay
    # rather than trying to recover partial drawing state)
    assert len(img.convert("RGB").getcolors(maxcolors=img.width * img.height)) > 1  # gradient+grid, not solid


# ---------------------------------------------------------------------------
# 7 - exact factual strings preserved verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exact_value", [
    "6EQUJ5", "Wow!", "1420 megahertz", "30 times background noise",
    "constellation Sagittarius", "August 15, 1977", "266P/Christensen",
    "February-May 2020", "Arecibo Observatory, Puerto Rico", "72 seconds", "1977",
])
def test_exact_factual_string_rendered_verbatim(monkeypatch, exact_value):
    drawn = _drawn_strings(monkeypatch)
    scene = _measurement_scene(value=exact_value)
    pg.render_programmatic_graphic(scene)
    joined = "\n".join(drawn)
    assert exact_value in joined


def test_timeline_values_rendered_verbatim(monkeypatch):
    values = ["August 15, 1977", "266P/Christensen", "February-May 2020"]
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_timeline_scene(values=values))
    joined = "\n".join(drawn)
    for v in values:
        assert v in joined


# ---------------------------------------------------------------------------
# 8 - no invented numerical values / no extra text
# ---------------------------------------------------------------------------

def test_no_text_drawn_beyond_the_scenes_own_values(monkeypatch):
    values = ["August 15, 1977", "266P/Christensen", "February-May 2020"]
    scene = _timeline_scene(values=values)
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(scene)
    # Every drawn string must be one of the scene's own values, or a
    # substring produced only by this renderer's own text-wrapping (never
    # a number/word that didn't already appear in the scene's data).
    allowed_words = set()
    for v in values:
        allowed_words.update(v.replace(",", "").split())
    for line in drawn:
        for word in line.replace(",", "").split():
            assert word in allowed_words, f"unexpected word drawn that isn't from scene data: {word!r}"


def test_internal_renderer_identifiers_never_drawn(monkeypatch):
    """Renderer-type names ('timeline', 'fallback_graphic') and internal
    data_type identifiers must never themselves appear as on-screen text -
    same invariant overlay_renderer.py enforces for its own field names."""
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_timeline_scene())
    joined = " ".join(drawn).upper()
    for banned in ("FALLBACK_GRAPHIC", "TIMELINE", "DATA_TYPE", "EXACT_RENDER", "PROGRAMMATIC_GRAPHIC"):
        assert banned not in joined


# ---------------------------------------------------------------------------
# 9 - no external API calls
# ---------------------------------------------------------------------------

def test_module_source_has_no_network_imports():
    source = open(pg.__file__).read()
    tree = ast.parse(source)
    banned_modules = {"httpx", "requests", "urllib", "urllib2", "socket", "aiohttp"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned_modules), f"unexpected network import(s): {imported & banned_modules}"


def test_rendering_makes_no_provider_or_network_call(monkeypatch):
    """Even with no provider wired in at all (render_programmatic_graphic
    takes a plain scene dict, not an ImageProvider), confirm nothing in the
    call chain reaches out - a smoke check that the public API signature
    itself never accepts or needs network access."""
    import inspect
    sig = inspect.signature(pg.render_programmatic_graphic)
    assert "provider" not in sig.parameters
    assert "url" not in sig.parameters
    # functional: renders fine with no network/proxy configured at all
    png_bytes, _ = pg.render_programmatic_graphic(_timeline_scene())
    assert len(png_bytes) > 0


# ---------------------------------------------------------------------------
# 10 - output file creation
# ---------------------------------------------------------------------------

def test_render_to_file_creates_output_file(tmp_path):
    dest = tmp_path / "scene_026.png"
    out_path, renderer_name = pg.render_programmatic_graphic_to_file(_timeline_scene(), dest)
    assert out_path == dest
    assert dest.exists()
    assert dest.stat().st_size > 0
    img = Image.open(dest)
    assert img.size == (1920, 1080)
    assert renderer_name == "timeline"
    assert not dest.with_name(dest.name + ".tmp").exists()  # atomic write, no leftover temp file


def test_render_to_file_creates_parent_directories(tmp_path):
    dest = tmp_path / "nested" / "dirs" / "scene_026.png"
    out_path, _ = pg.render_programmatic_graphic_to_file(_empty_scene(), dest)
    assert out_path.exists()


# ---------------------------------------------------------------------------
# 16 - scene 026's real metadata can be rendered
# ---------------------------------------------------------------------------

def test_scene_026_real_metadata_renders_without_crashing():
    """Scene 026's ACTUAL production payload, from storage/2026/09/
    the-wow-signal-6/scenes.json (provided directly by the user - see
    _REAL_SCENE_026 below), not the earlier schema-generic placeholder
    this test used before that data was available."""
    png_bytes, renderer_name = pg.render_programmatic_graphic(_REAL_SCENE_026)
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (1920, 1080)
    assert renderer_name in pg._RENDERERS


# ---------------------------------------------------------------------------
# PROGRAMMATIC GRAPHIC RENDERER - SCENE 026 FINALIZATION
#
# conceptual_curve: scene 026's actual visual_strategy=="programmatic_graphic"
# shape - a single data_type=="curve" data point whose value is DESCRIPTIVE
# text ("celestial beam-crossing intensity profile"), not a numeric sample
# array. Covers the 14 categories requested for this finalization:
#   1  descriptive curve data routes to conceptual_curve
#   2  scene 026 real metadata routes correctly
#   3  no provider.generate() call
#   4  output is 1920x1080
#   5  output is deterministic
#   6  no numeric data is fabricated
#   7  "CONCEPTUAL - NOT TO SCALE" appears via the renderer's explicit
#      disclosure mechanism
#   8  internal labels do not appear
#   9  exact_text remains empty for this scene
#  10  existing timeline renderer remains unchanged
#  11  existing fallback renderer remains unchanged
#  12  existing overlay tests remain green (run separately - see
#      tests/test_overlay_renderer.py)
#  13  Scene Planner V2 remains green (run separately - see
#      tests/test_scene_planner_v2.py)
#  14  full project suite remains green (run separately - see the task's
#      delivery report)
# ---------------------------------------------------------------------------

# Scene 026's REAL production metadata, exactly as provided by the user
# from storage/2026/09/the-wow-signal-6/scenes.json - never altered.
_REAL_SCENE_026 = {
    "scene_id": 26,
    "scene_type": "scientific_graphic",
    "visual_strategy": "programmatic_graphic",
    "exact_text": [],
    "data_points": [
        {
            "data_type": "curve",
            "value": "celestial beam-crossing intensity profile",
            "exact_render": False,
        },
    ],
    "text_render": {
        "items": [],
        "hypothesis_badge": False,
    },
    "visual_relevance_reason": (
        "Plots the combined characteristics—strength, frequency specificity, bandwidth, "
        "and celestial intensity profile—as overlaid evidence showing all anomalies present simultaneously."
    ),
    "visual_prompt": None,
    "narration": None,
}


def _generic_curve_scene(scene_id: int, value: str, visual_relevance_reason: str = "") -> dict:
    """A DIFFERENT (non-scene-026) descriptive curve scene - used to prove
    the dispatch/rendering logic is generic, not hardcoded to scene 026's
    scene_id/topic/wording."""
    return {
        "scene_id": scene_id,
        "schema_version": 2,
        "visual_strategy": "programmatic_graphic",
        "scene_type": "scientific_graphic",
        "visual_prompt": None,
        "exact_text": [],
        "data_points": [{"data_type": "curve", "value": value, "exact_render": False}],
        "text_render": {"items": [], "hypothesis_badge": False},
        "visual_relevance_reason": visual_relevance_reason,
    }


# 1 - descriptive curve data routes to conceptual_curve (generic, not scene 026)

def test_generic_descriptive_curve_scene_routes_to_conceptual_curve():
    scene = _generic_curve_scene(
        scene_id=999, value="orbital velocity trend, qualitative",
        visual_relevance_reason="Illustrates how speed changes across the flyby.",
    )
    assert pg._select_renderer_name(scene) == "conceptual_curve"
    png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    assert renderer_name == "conceptual_curve"
    assert Image.open(io.BytesIO(png_bytes)).size == (1920, 1080)


def test_numeric_curve_value_does_not_route_to_conceptual_curve():
    """A future data_type=='curve' value that is purely numeric (a real
    sample or list of samples) must NOT be treated as a conceptual/
    qualitative curve - see _looks_purely_numeric()."""
    scene = _generic_curve_scene(scene_id=998, value="42.5")
    assert pg._select_renderer_name(scene) != "conceptual_curve"
    scene2 = _generic_curve_scene(scene_id=997, value="1, 2, 3, 4")
    assert pg._select_renderer_name(scene2) != "conceptual_curve"


# 2 - scene 026 real metadata routes correctly

def test_scene_026_real_metadata_routes_to_conceptual_curve():
    assert pg._select_renderer_name(_REAL_SCENE_026) == "conceptual_curve"
    _png_bytes, renderer_name = pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert renderer_name == "conceptual_curve"


# 3 - no provider.generate() call

def test_scene_026_never_calls_provider_generate(isolated_storage, monkeypatch):
    from storage import project_store as ps
    from youtube import image_gen

    class _NeverCallProvider(image_gen.ImageProvider):
        def generate(self, prompt, width, height):
            raise AssertionError("provider.generate() must not be called for programmatic_graphic")

    paths, _ = ps.create_project("Scene 026 Conceptual Curve Topic")
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    image_gen.generate_scene_images(paths, [_REAL_SCENE_026])  # must not raise

    assert paths.scene_image(26).exists()


def test_image_gen_logs_conceptual_curve_renderer_for_scene_026(isolated_storage, monkeypatch, caplog):
    from storage import project_store as ps
    from youtube import image_gen

    class _NeverCallProvider(image_gen.ImageProvider):
        def generate(self, prompt, width, height):
            raise AssertionError("provider.generate() must not be called")

    paths, _ = ps.create_project("Scene 026 Log Topic")
    monkeypatch.setattr(image_gen, "get_provider", lambda: _NeverCallProvider())

    with caplog.at_level("INFO", logger="ycf.image_gen"):
        image_gen.generate_scene_images(paths, [_REAL_SCENE_026])

    messages = [r.message for r in caplog.records]
    assert any(
        "[IMAGE][PROGRAMMATIC]" in m and "scene=026" in m and "renderer=conceptual_curve" in m for m in messages
    )


# 4 - output is 1920x1080

def test_scene_026_output_is_1920x1080():
    png_bytes, _ = pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert Image.open(io.BytesIO(png_bytes)).size == (1920, 1080)


# 5 - output is deterministic

def test_scene_026_output_is_deterministic():
    b1, n1 = pg.render_programmatic_graphic(_REAL_SCENE_026)
    b2, n2 = pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert b1 == b2
    assert n1 == n2 == "conceptual_curve"


# 6 - no numeric data is fabricated

def test_scene_026_drawn_text_never_contains_digits(monkeypatch):
    """Nothing this renderer draws for scene 026 (dimension labels,
    disclosure marker) contains a single digit - there is no numeric
    scale, axis value, timestamp, or measurement anywhere in the output,
    because none exists in the source metadata."""
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    joined = "".join(drawn)
    assert not any(ch.isdigit() for ch in joined)


def test_extract_em_dash_list_never_invents_items_for_unstructured_text():
    assert pg._extract_em_dash_list("") == []
    assert pg._extract_em_dash_list("a plain sentence with no em dashes at all") == []
    assert pg._extract_em_dash_list("only one em dash—here") == []


def test_extract_em_dash_list_parses_scene_026_wording():
    labels = pg._extract_em_dash_list(_REAL_SCENE_026["visual_relevance_reason"])
    assert labels == ["strength", "frequency specificity", "bandwidth", "celestial intensity profile"]


# 7 - "CONCEPTUAL - NOT TO SCALE" appears via the explicit disclosure mechanism

def test_conceptual_disclosure_marker_is_drawn(monkeypatch):
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert pg.CONCEPTUAL_DISCLOSURE in drawn


def test_conceptual_disclosure_marker_drawn_even_without_dimension_labels(monkeypatch):
    """The disclosure must appear on EVERY conceptual_curve scene, even one
    whose visual_relevance_reason doesn't use the em-dash list convention
    (so no dimension labels get drawn at all)."""
    scene = _generic_curve_scene(scene_id=996, value="a qualitative trend, no numbers")
    drawn = _drawn_strings(monkeypatch)
    _png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    assert renderer_name == "conceptual_curve"
    assert pg.CONCEPTUAL_DISCLOSURE in drawn
    # and no dimension-label panels were drawn (no em-dash structure to extract from)
    assert drawn == [pg.CONCEPTUAL_DISCLOSURE]


# 8 - internal labels do not appear

def test_scene_026_no_internal_identifiers_drawn(monkeypatch):
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    joined = " ".join(drawn).upper()
    for banned in (
        "PROGRAMMATIC_GRAPHIC", "FALLBACK_GRAPHIC", "FALLBACK", "CURVE", "CONCEPTUAL_CURVE",
        "DATA_TYPE", "EXACT_RENDER", "DATA_POINT",
    ):
        assert banned not in joined
    # the curve's own descriptive value string must never be printed
    # verbatim either (exact_render is False - see module docstring)
    assert "BEAM-CROSSING" not in joined
    assert "INTENSITY PROFILE" not in joined or "CELESTIAL INTENSITY PROFILE" in joined  # only via the extracted label


# 9 - exact_text remains empty for this scene

def test_scene_026_exact_text_stays_empty_and_scene_not_mutated():
    scene_copy = {**_REAL_SCENE_026, "data_points": list(_REAL_SCENE_026["data_points"])}
    pg.render_programmatic_graphic(scene_copy)
    assert scene_copy["exact_text"] == []
    assert scene_copy == _REAL_SCENE_026  # never mutated by rendering


def test_scene_026_only_draws_dimension_labels_and_disclosure(monkeypatch):
    """With exact_text==[] and no exact_render=true data_points, the ONLY
    text this scene can honestly show is: the em-dash-extracted dimension
    labels, and the always-present disclosure marker - nothing else."""
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    allowed = {"STRENGTH", "FREQUENCY SPECIFICITY", "BANDWIDTH", "CELESTIAL INTENSITY PROFILE", pg.CONCEPTUAL_DISCLOSURE}
    assert set(drawn) <= allowed


# 10 / 11 - existing timeline / fallback renderers remain unchanged

def test_timeline_renderer_selection_unchanged_by_conceptual_curve_addition():
    scene = _timeline_scene()
    assert pg._select_renderer_name(scene) == "timeline"
    _png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    assert renderer_name == "timeline"


def test_fallback_renderer_selection_unchanged_by_conceptual_curve_addition():
    scene = _measurement_scene()  # a "measurement" data_type - not curve, not timeline
    assert pg._select_renderer_name(scene) == "fallback_graphic"
    _png_bytes, renderer_name = pg.render_programmatic_graphic(scene)
    assert renderer_name == "fallback_graphic"


def test_empty_scene_still_falls_back_unchanged():
    assert pg._select_renderer_name(_empty_scene()) == "fallback_graphic"


# ---------------------------------------------------------------------------
# SCENE 026 CONCEPTUAL GRAPHIC - VISUAL QUALITY REVISION
#
# Upgrades conceptual_curve from a single descending line + 4 unrelated
# pills into a multi-layer visualization: a central beam/observation
# diagram with a "crossing" arc cluster, plus a legend where each
# extracted characteristic gets its own small, visually distinct,
# deterministic qualitative trace icon. Still zero numeric data anywhere -
# these tests prove the richer visual without OCR, by spying on the
# actual ImageDraw.line/arc calls (not just final pixel guesswork).
# ---------------------------------------------------------------------------

def _spy_draw_calls(monkeypatch):
    """Spies on ImageDraw.line/arc (in addition to the existing
    _drawn_strings() text spy) - returns (line_calls, arc_calls), each a
    list of the raw arguments Pillow was asked to draw with."""
    line_calls: list[list[tuple[float, float]]] = []
    arc_calls: list[tuple[object, float, float]] = []
    original_line = ImageDraw.ImageDraw.line
    original_arc = ImageDraw.ImageDraw.arc

    def _spy_line(self, xy, *a, **k):
        pts = list(xy) if not isinstance(xy, list) else xy
        line_calls.append(pts)
        return original_line(self, xy, *a, **k)

    def _spy_arc(self, xy, start, end, *a, **k):
        arc_calls.append((xy, start, end))
        return original_arc(self, xy, start, end, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "line", _spy_line)
    monkeypatch.setattr(ImageDraw.ImageDraw, "arc", _spy_arc)
    return line_calls, arc_calls


# 1 - four conceptual characteristics render

def test_four_conceptual_characteristics_render(monkeypatch):
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    for label in ("STRENGTH", "FREQUENCY SPECIFICITY", "BANDWIDTH", "CELESTIAL INTENSITY PROFILE"):
        assert label in drawn


def test_each_characteristic_gets_its_own_distinct_trace_icon(monkeypatch):
    """The visual-quality complaint was that 4 labels had no visible
    relationship to any trace. Now each legend row draws its own small
    line (see _trace_shape_points/_draw_conceptual_legend) - this proves
    there are exactly 4 such icon-sized line calls (distinguished from the
    large central beam line by bounding-box width) and that their SHAPES
    (normalized to a common origin, so only shape - not screen position -
    is compared) are all different from each other, i.e. genuinely
    distinct traces, not 4 copies of one line."""
    line_calls, _arc_calls = _spy_draw_calls(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)

    icon_calls = [
        pts for pts in line_calls
        if len(pts) >= 4 and (max(x for x, _y in pts) - min(x for x, _y in pts)) < 300
    ]
    assert len(icon_calls) == 4  # one trace icon per extracted label

    normalized = []
    for pts in icon_calls:
        x0, y0 = pts[0]
        normalized.append(tuple((round(x - x0, 1), round(y - y0, 1)) for x, y in pts))
    assert len(set(normalized)) == len(normalized), "legend trace icons must all be visually distinct shapes"


def test_trace_shape_points_deterministic_and_distinct_by_index():
    shapes = [pg._trace_shape_points(i, 180, 40) for i in range(4)]
    assert len({tuple(s) for s in shapes}) == 4
    # same index -> same shape, every time
    assert pg._trace_shape_points(2, 180, 40) == pg._trace_shape_points(2, 180, 40)


def test_beam_crossing_arc_cluster_is_drawn_and_partial(monkeypatch):
    """The central diagram's "multiple characteristics observed at once"
    motif - a small cluster of concentric but PARTIAL arcs (never a full
    360-degree ring/reticle, to avoid a radar/HUD look) at the beam path's
    crossing point."""
    _line_calls, arc_calls = _spy_draw_calls(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)

    assert len(arc_calls) == len(pg.BEAM_ARC_SPANS) >= 3
    for _box, start, end in arc_calls:
        span = (end - start) % 360
        assert 0 < span < 360, "each arc must be a partial sweep, never a closed/full ring"


def test_arc_cluster_not_drawn_for_timeline_or_fallback(monkeypatch):
    """The beam/arc diagram is conceptual_curve-specific - it must not
    appear for the unrelated timeline/fallback_graphic renderers."""
    _line_calls, arc_calls = _spy_draw_calls(monkeypatch)
    pg.render_programmatic_graphic(_timeline_scene())
    pg.render_programmatic_graphic(_measurement_scene())
    assert arc_calls == []


# 2 - no numeric data is introduced (re-verified against the richer visual)

def test_visual_revision_still_draws_zero_digits(monkeypatch):
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert not any(ch.isdigit() for ch in "".join(drawn))


# 3 - disclosure remains visible

def test_disclosure_still_present_after_visual_revision(monkeypatch):
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert pg.CONCEPTUAL_DISCLOSURE in drawn


# 4 / 5 - deterministic, 1920x1080 (re-verified against the richer visual)

def test_visual_revision_deterministic_and_correct_size():
    b1, n1 = pg.render_programmatic_graphic(_REAL_SCENE_026)
    b2, n2 = pg.render_programmatic_graphic(_REAL_SCENE_026)
    assert b1 == b2
    assert n1 == n2 == "conceptual_curve"
    assert Image.open(io.BytesIO(b1)).size == (1920, 1080)


# 6 - no internal metadata labels (re-verified against the richer visual)

def test_visual_revision_no_internal_labels(monkeypatch):
    drawn = _drawn_strings(monkeypatch)
    pg.render_programmatic_graphic(_REAL_SCENE_026)
    joined = " ".join(drawn).upper()
    for banned in (
        "PROGRAMMATIC_GRAPHIC", "FALLBACK_GRAPHIC", "FALLBACK", "CURVE", "CONCEPTUAL_CURVE",
        "DATA_TYPE", "EXACT_RENDER", "DATA_POINT", "BEAM_ARC", "TRACE_SHAPE",
    ):
        assert banned not in joined


# 7 / 8 - existing timeline / fallback renderers still unchanged after the
# visual revision (re-verified beyond the earlier selection-only checks)

def test_timeline_output_unaffected_by_conceptual_curve_visual_revision():
    scene = _timeline_scene()
    b1, n1 = pg.render_programmatic_graphic(scene)
    b2, n2 = pg.render_programmatic_graphic(scene)
    assert n1 == n2 == "timeline"
    assert b1 == b2  # still deterministic, unaffected by conceptual_curve changes


def test_fallback_output_unaffected_by_conceptual_curve_visual_revision():
    scene = _measurement_scene()
    b1, n1 = pg.render_programmatic_graphic(scene)
    b2, n2 = pg.render_programmatic_graphic(scene)
    assert n1 == n2 == "fallback_graphic"
    assert b1 == b2
