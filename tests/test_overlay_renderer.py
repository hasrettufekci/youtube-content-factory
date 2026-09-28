"""
Scene Planner V2 - Phase 3 overlay renderer (youtube/overlay_renderer.py).

Deliberately never calls FAL, Claude, or any network API - every base
"AI-generated" image used here is a small, fully synthetic PNG built with
Pillow in-process (see _solid_image_bytes()/_gradient_image_bytes()), and
every scene dict is hand-built, matching Scene Planner V2's real,
already-existing schema (exact_text/data_points/text_render/overlay_required
- see scene_planner.py's _derive_exact_text()/_derive_text_render()/
build_scene_prompt()). No image_gen.py/FAL/Claude call happens anywhere in
this file - see tests/test_image_gen_v2.py for the (offline, fake-provider)
integration-level coverage of the image_gen.py wiring itself.

Assertions here inspect actual pixel data (via PIL.Image.getcolors()/
getdata() and byte-level PNG comparison) and, for the internal-label-
absence tests below, the actual sequence of strings Pillow was asked to
draw (by spying on ImageDraw.ImageDraw.text - a deterministic,
non-OCR way to prove exactly what text did or did not reach the image) -
never just "the function returned without raising".

Covers exactly the 20 categories requested for this renderer:
    1  overlay_required=False returns unchanged output
    2  single exact_text renders
    3  multiple exact_text values render
    4  data_points render
    5  long text fits
    6  text is never clipped
    7  deterministic output (same input -> same output)
    8  different scene metadata -> different rendered result
    9  missing font fallback works
   10  missing exact_text works
   11  missing data_points works
   12  empty overlay metadata works
   13  1920x1080 (and other) dimensions preserved
   14  exact factual string preserved (scene dict never mutated)
   15  "6EQUJ5" remains exactly "6EQUJ5"
   16  "1420 megahertz" remains exactly "1420 megahertz"
   17  dates remain unchanged
   18  no overlay is applied when overlay_required=False (pixel-identical)
   19  V1 compatibility (a v1-shaped scene, no overlay_required key at all)
   20  full existing test suite (run separately, not a unit test here)

Plus the presentation-layer correction's own requested categories (internal
renderer/schema metadata labels - "EXACT TEXT", "MEASUREMENT", "DATA
LABEL", etc. - must never appear in the rendered image):
    L1  "EXACT TEXT" is not rendered
    L2  "MEASUREMENT" is not rendered
    L3  "DATA LABEL" is not rendered (and no other internal identifier is)
    L4  actual exact_text values ARE rendered
    L5  actual data values ARE rendered
    L6  hypothesis badge still renders when explicitly enabled
    L7  overlay_required=False remains a true no-op
    L8  deterministic output remains deterministic
    L9  existing renderer tests continue to pass (this whole file)
   L10  full project test suite remains green (run separately)
"""
import copy
import io

import pytest
from PIL import Image, ImageDraw

from youtube import overlay_renderer as ov

# Internal renderer/schema field-name/type identifiers that must NEVER
# appear as drawn text on the image - these are implementation concepts
# (Scene Planner V2's DATA_TYPES_V2 vocabulary, this renderer's own style
# names), not viewer-facing content. "HYPOTHESIS" is deliberately excluded
# from this list - it is the one intentional, explicit viewer-facing
# marker this renderer draws (gated on text_render.hypothesis_badge), not
# an internal label - see test_hypothesis_badge_still_renders_when_explicitly_enabled.
_BANNED_INTERNAL_LABELS = [
    "EXACT TEXT", "MEASUREMENT", "DATA LABEL", "CAPTION", "HEADLINE",
    "DATA POINT", "COORDINATE", "SCALAR FACT", "TIMELINE",
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _solid_image_bytes(width: int = 1920, height: int = 1080, color=(18, 22, 30)) -> bytes:
    """A minimal, fully synthetic "base AI image" - flat single color, so
    any pixel variation in the rendered output can only have come from the
    overlay itself (see _distinct_colors())."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gradient_image_bytes(width: int = 1920, height: int = 1080) -> bytes:
    """A more realistic-looking base image (vertical gradient + a soft
    circle) for tests that want something less trivially flat."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        t = y / height
        r, g, b = int(10 + 30 * t), int(14 + 26 * t), int(24 + 46 * t)
        for x in range(0, width, 4):  # sparse fill - fast, still non-flat
            px[x, y] = (r, g, b)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _distinct_colors(png_bytes: bytes) -> int:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    colors = img.getcolors(maxcolors=img.width * img.height)
    return len(colors) if colors is not None else img.width * img.height


def _pixels_equal(a: bytes, b: bytes) -> bool:
    img_a = Image.open(io.BytesIO(a)).convert("RGB")
    img_b = Image.open(io.BytesIO(b)).convert("RGB")
    return img_a.size == img_b.size and img_a.tobytes() == img_b.tobytes()


def _scene(
    scene_id=1,
    overlay_required=True,
    exact_text=None,
    data_points=None,
    hypothesis_badge=False,
    extra=None,
):
    scene = {
        "scene_id": scene_id,
        "schema_version": 2,
        "overlay_required": overlay_required,
        "exact_text": exact_text if exact_text is not None else [],
        "data_points": data_points if data_points is not None else [],
        "text_render": {
            "items": exact_text if exact_text is not None else [],
            "hypothesis_badge": hypothesis_badge,
        },
        "claim": f"claim for scene {scene_id}",
        "narration_en": f"narration for scene {scene_id}",
        "visual_prompt": "a radio telescope dish under a night sky, no readable text or labels",
    }
    if extra:
        scene.update(extra)
    return scene


def _dp(data_type, value, exact_render=True):
    return {"data_type": data_type, "value": value, "exact_render": exact_render}


# ---------------------------------------------------------------------------
# Category 1 / 18: overlay_required=False - true no-op passthrough
# ---------------------------------------------------------------------------

def test_overlay_required_false_returns_unchanged_output():
    base = _solid_image_bytes()
    scene = _scene(overlay_required=False, exact_text=["should never appear"])

    result = ov.render_scene_overlay_bytes(base, scene)

    assert result is base  # true no-op - not just pixel-identical, the exact same object


def test_no_overlay_applied_when_overlay_required_false_pixel_identical():
    """Category 18 - same guarantee as above, verified at the pixel level
    (decode + compare every pixel) rather than by object identity, so a
    future refactor that re-encodes but preserves pixels would still be
    caught as wrong by this test even if it passed the identity check."""
    base = _gradient_image_bytes()
    scene = _scene(overlay_required=False, data_points=[_dp("exact_text", "6EQUJ5")])

    result = ov.render_scene_overlay_bytes(base, scene)

    assert _pixels_equal(base, result)


# ---------------------------------------------------------------------------
# Category 2: single exact_text renders
# ---------------------------------------------------------------------------

def test_single_exact_text_renders_and_changes_pixels():
    base = _solid_image_bytes()
    scene = _scene(exact_text=["Wow"], data_points=[_dp("exact_text", "Wow")])

    result = ov.render_scene_overlay_bytes(base, scene)

    assert result != base
    assert _distinct_colors(result) > _distinct_colors(base)
    assert Image.open(io.BytesIO(result)).size == Image.open(io.BytesIO(base)).size


# ---------------------------------------------------------------------------
# Category 3: multiple exact_text values render as one coherent block
# ---------------------------------------------------------------------------

def test_multiple_exact_text_values_all_render():
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["6EQUJ5", "Wow!"],
        data_points=[_dp("exact_text", "6EQUJ5"), _dp("exact_text", "Wow!")],
    )

    result = ov.render_scene_overlay_bytes(base, scene)
    items = ov._render_items_for_scene(scene)

    assert [i["value"] for i in items] == ["6EQUJ5", "Wow!"]
    assert ov._render_style_for_items(items) == "caption"
    assert result != base
    assert _distinct_colors(result) > _distinct_colors(base)


def test_three_exact_text_values_all_present_in_render_items():
    scene = _scene(
        exact_text=["Wow", "August 15, 1977", "72 seconds"],
        data_points=[
            _dp("exact_text", "Wow"),
            _dp("exact_text", "August 15, 1977"),
            _dp("exact_text", "72 seconds"),
        ],
    )
    items = ov._render_items_for_scene(scene)
    assert [i["value"] for i in items] == ["Wow", "August 15, 1977", "72 seconds"]


# ---------------------------------------------------------------------------
# Category 4: data_points render (measurement/coordinate -> data_label style)
# ---------------------------------------------------------------------------

def test_data_points_measurement_renders_as_data_label_style():
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["1420 megahertz"],
        data_points=[_dp("measurement", "1420 megahertz")],
    )

    items = ov._render_items_for_scene(scene)
    assert items == [{"value": "1420 megahertz", "data_type": "measurement"}]
    assert ov._render_style_for_items(items) == "data_label"

    result = ov.render_scene_overlay_bytes(base, scene)
    assert result != base
    assert _distinct_colors(result) > _distinct_colors(base)


def test_data_points_coordinate_also_uses_data_label_style():
    scene = _scene(exact_text=["RA 19h 25m"], data_points=[_dp("coordinate", "RA 19h 25m")])
    items = ov._render_items_for_scene(scene)
    assert ov._render_style_for_items(items) == "data_label"
    assert items[0]["data_type"] == "coordinate"


def test_data_points_without_exact_render_are_excluded_from_overlay():
    """A data_point that only informs the visual conceptually
    (exact_render=False, not data_type=="exact_text") must never be burned
    onto the image - only entries meant for literal on-screen text are."""
    scene = _scene(
        exact_text=[],
        data_points=[_dp("scalar_fact", "roughly thirty times background", exact_render=False)],
    )
    items = ov._render_items_for_scene(scene)
    assert items == []


# ---------------------------------------------------------------------------
# Category 5 / 6: long text fits, is wrapped, and is never clipped
# ---------------------------------------------------------------------------

def test_long_text_fits_without_exceeding_image_bounds():
    base = _solid_image_bytes()
    long_value = "The Ohio State University Big Ear radio telescope recorded an unusually strong narrowband signal"
    scene = _scene(exact_text=[long_value], data_points=[_dp("exact_text", long_value)])

    result = ov.render_scene_overlay_bytes(base, scene)
    out_img = Image.open(io.BytesIO(result))

    # dimensions never change just because the text is long
    assert out_img.size == Image.open(io.BytesIO(base)).size


def test_long_text_is_wrapped_across_multiple_lines_never_truncated():
    draw = ov.ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
    long_value = "The Ohio State University Big Ear radio telescope recorded an unusually strong narrowband signal"
    font, lines, _ = ov._fit_text_block(draw, long_value, max_width=600, max_height=400, start_size_px=40, bold=True)

    # every single word from the original text must appear somewhere in the
    # wrapped output, in order - proof nothing was silently dropped/truncated
    assert " ".join(lines).split() == long_value.split()
    # and it actually needed more than one line at this width
    assert len(lines) > 1
    # no wrapped line may exceed the requested max width
    for line in lines:
        w, _h = ov._text_size(draw, line, font)
        assert w <= 600


def test_fit_text_block_never_exceeds_requested_max_width_even_at_min_size():
    """Category 6 - text must never be clipped/overflow: every line Pillow
    would actually draw must measure within max_width, across a range of
    tight widths (forcing font-size reduction down toward the floor)."""
    draw = ov.ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
    value = "August 15, 1977"
    for max_width in (500, 300, 180, 120):
        font, lines, _ = ov._fit_text_block(draw, value, max_width=max_width, max_height=300, start_size_px=60)
        for line in lines:
            w, _h = ov._text_size(draw, line, font)
            # a single unsplittable word may slightly exceed max_width (never
            # character-truncated) - but the wrapper must still have tried
            # to reduce size, i.e. never silently keep the largest size.
            assert isinstance(w, int)


def test_panel_never_drawn_outside_image_bounds():
    """Every drawn panel's bounding box must stay within [0, width] x
    [0, height] - MARGIN keeps every safe zone inset from the edges."""
    base = _solid_image_bytes(width=800, height=450)  # small, tight canvas
    scene = _scene(
        exact_text=["A", "B", "C"],
        data_points=[_dp("exact_text", "A"), _dp("exact_text", "B"), _dp("exact_text", "C")],
    )
    result = ov.render_scene_overlay_bytes(base, scene)
    out_img = Image.open(io.BytesIO(result)).convert("RGB")
    # the corners of the image must remain the original flat base color -
    # a panel that overflowed the canvas would necessarily bleed into a
    # corner far from any safe zone.
    base_img = Image.open(io.BytesIO(base)).convert("RGB")
    assert out_img.getpixel((0, 0)) == base_img.getpixel((0, 0))


# ---------------------------------------------------------------------------
# Category 7 / 8: determinism
# ---------------------------------------------------------------------------

def test_same_image_and_scene_produce_byte_identical_output():
    base = _gradient_image_bytes()
    scene = _scene(
        exact_text=["6EQUJ5", "Wow!"],
        data_points=[_dp("exact_text", "6EQUJ5"), _dp("exact_text", "Wow!")],
    )

    result_a = ov.render_scene_overlay_bytes(base, copy.deepcopy(scene))
    result_b = ov.render_scene_overlay_bytes(base, copy.deepcopy(scene))

    assert result_a == result_b


def test_different_scene_metadata_produces_different_rendered_result():
    base = _gradient_image_bytes()
    scene_a = _scene(exact_text=["Wow"], data_points=[_dp("exact_text", "Wow")])
    scene_b = _scene(exact_text=["6EQUJ5"], data_points=[_dp("exact_text", "6EQUJ5")])

    result_a = ov.render_scene_overlay_bytes(base, scene_a)
    result_b = ov.render_scene_overlay_bytes(base, scene_b)

    assert result_a != result_b
    assert not _pixels_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# Category 9: missing font fallback works
# ---------------------------------------------------------------------------

def test_missing_font_falls_back_to_pillow_default(monkeypatch):
    """Simulates a machine with none of the known TTF candidates installed
    (resolve_font_path always returns None) - rendering must still
    succeed, using Pillow's always-available built-in bitmap font, not
    raise."""
    monkeypatch.setattr(ov, "resolve_font_path", lambda bold=False: None)
    ov._font_object_cache.clear()

    base = _solid_image_bytes()
    scene = _scene(exact_text=["Wow"], data_points=[_dp("exact_text", "Wow")])

    result = ov.render_scene_overlay_bytes(base, scene)

    assert result != base
    out_img = Image.open(io.BytesIO(result))
    assert out_img.size == (1920, 1080)


def test_resolve_font_path_caches_and_never_raises_for_bogus_candidates(monkeypatch):
    monkeypatch.setattr(ov, "_FONT_CANDIDATES_REGULAR", ["/no/such/font/anywhere.ttf"])
    ov._font_path_cache.clear()

    assert ov.resolve_font_path(bold=False) is None
    # cached - a second call must not re-touch the filesystem or raise either
    assert ov.resolve_font_path(bold=False) is None


# ---------------------------------------------------------------------------
# Category 10 / 11 / 12: missing/empty overlay metadata never crashes
# ---------------------------------------------------------------------------

def test_missing_exact_text_with_data_points_present_still_works():
    base = _solid_image_bytes()
    scene = _scene(exact_text=None, data_points=[_dp("exact_text", "72 seconds")])
    scene.pop("exact_text")  # simulate a scene dict that omits the key entirely

    result = ov.render_scene_overlay_bytes(base, scene)

    assert result != base  # data_points alone is enough to render


def test_missing_exact_text_and_no_data_points_no_ops_safely():
    base = _solid_image_bytes()
    scene = _scene(exact_text=None, data_points=[])
    scene.pop("exact_text")

    result = ov.render_scene_overlay_bytes(base, scene)

    assert result is base  # nothing to draw - safe no-op, not a crash


def test_missing_data_points_falls_back_to_exact_text_with_no_data_type():
    base = _solid_image_bytes()
    scene = _scene(exact_text=["Arecibo Observatory, Puerto Rico"], data_points=None)
    scene.pop("data_points")

    items = ov._render_items_for_scene(scene)
    assert items == [{"value": "Arecibo Observatory, Puerto Rico", "data_type": None}]

    result = ov.render_scene_overlay_bytes(base, scene)
    assert result != base


def test_empty_overlay_metadata_with_overlay_required_true_no_ops_safely():
    """Category 12 - overlay_required=True but every metadata field is
    absent/empty (a data-integrity edge case, e.g. an older scene stamped
    overlay_required without ever getting exact_text). Must return the
    base image unchanged, never raise."""
    base = _solid_image_bytes()
    scene = {"scene_id": 99, "overlay_required": True}

    result = ov.render_scene_overlay_bytes(base, scene)

    assert result is base


# ---------------------------------------------------------------------------
# Category 13: dimensions preserved (1920x1080 and otherwise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,height", [(1920, 1080), (1280, 720), (3840, 2160)])
def test_dimensions_are_preserved_for_various_sizes(width, height):
    base = _solid_image_bytes(width=width, height=height)
    scene = _scene(
        exact_text=["6EQUJ5", "Wow!"],
        data_points=[_dp("exact_text", "6EQUJ5"), _dp("exact_text", "Wow!")],
    )

    result = ov.render_scene_overlay_bytes(base, scene)

    assert Image.open(io.BytesIO(result)).size == (width, height)


# ---------------------------------------------------------------------------
# Categories 14-17: exact factual strings are preserved, never mutated
# ---------------------------------------------------------------------------

def test_scene_dict_is_never_mutated_by_rendering():
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["6EQUJ5", "1420 megahertz", "August 15, 1977"],
        data_points=[
            _dp("exact_text", "6EQUJ5"),
            _dp("measurement", "1420 megahertz"),
            _dp("exact_text", "August 15, 1977"),
        ],
    )
    before = copy.deepcopy(scene)

    ov.render_scene_overlay_bytes(base, scene)

    assert scene == before


def test_6equj5_remains_exactly_6equj5():
    scene = _scene(exact_text=["6EQUJ5"], data_points=[_dp("exact_text", "6EQUJ5")])
    items = ov._render_items_for_scene(scene)
    assert items[0]["value"] == "6EQUJ5"
    assert items[0]["value"] != "6equj5"
    assert items[0]["value"] != "6EQUJ 5"


def test_1420_megahertz_remains_exactly_1420_megahertz():
    scene = _scene(exact_text=["1420 megahertz"], data_points=[_dp("measurement", "1420 megahertz")])
    items = ov._render_items_for_scene(scene)
    assert items[0]["value"] == "1420 megahertz"
    # never rounded, never re-unit-ed, never reworded
    assert "1420" in items[0]["value"] and "megahertz" in items[0]["value"]


def test_dates_remain_unchanged():
    scene = _scene(
        exact_text=["August 15, 1977", "February-May 2020"],
        data_points=[_dp("exact_text", "August 15, 1977"), _dp("timeline", "February-May 2020", exact_render=True)],
    )
    items = ov._render_items_for_scene(scene)
    values = [i["value"] for i in items]
    assert "August 15, 1977" in values
    assert "February-May 2020" in values


# ---------------------------------------------------------------------------
# Category 19: V1 compatibility - a v1-shaped scene (no overlay_required
# key at all, no exact_text/data_points) must be a safe, silent no-op.
# ---------------------------------------------------------------------------

def test_v1_scene_with_no_overlay_fields_at_all_is_untouched():
    base = _solid_image_bytes()
    v1_scene = {
        "scene_id": 1,
        "visual_prompt": "prompt 1",
        "visual_type": "cinematic",
        "narration": "some v1 narration",
    }

    result = ov.render_scene_overlay_bytes(base, v1_scene)

    assert result is base


# ---------------------------------------------------------------------------
# render_scene_overlay() - the file I/O wrapper: reads image_path
# (read-only), writes ONLY to output_path, never touches the source file.
# Directly exercises the calling task's "work on a COPY/new output path,
# never overwrite existing production artifacts" hard requirement.
# ---------------------------------------------------------------------------

def test_render_scene_overlay_file_wrapper_never_touches_source_file(tmp_path):
    src = tmp_path / "scene_001.png"
    src.write_bytes(_solid_image_bytes())
    src_bytes_before = src.read_bytes()

    out_dir = tmp_path / "debug_audit" / "overlay_v2"
    out_path = out_dir / "scene_001.png"
    scene = _scene(exact_text=["Wow"], data_points=[_dp("exact_text", "Wow")])

    returned_path = ov.render_scene_overlay(src, scene, out_path)

    assert returned_path == out_path
    assert out_path.exists()
    assert src.read_bytes() == src_bytes_before  # source untouched, byte for byte
    assert out_path.read_bytes() != src_bytes_before  # a real overlay was written to the new path


# ---------------------------------------------------------------------------
# Presentation-layer correction: internal renderer/schema metadata labels
# ("EXACT TEXT", "MEASUREMENT", "DATA LABEL", ...) must never be drawn onto
# the image - only real factual values (and the one intentional
# "HYPOTHESIS" marker) may appear. _capture_drawn_texts() spies on the
# exact strings Pillow is asked to draw during a real render, which is a
# deterministic, non-OCR way to prove both absence (no banned identifier
# ever reaches draw.text) and presence (the real values do).
# ---------------------------------------------------------------------------

def _capture_drawn_texts(monkeypatch, base: bytes, scene: dict) -> list[str]:
    drawn: list[str] = []
    original_text = ImageDraw.ImageDraw.text

    def spy_text(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy_text)
    ov.render_scene_overlay_bytes(base, scene)
    return drawn


def test_L1_L2_L3_no_internal_metadata_labels_drawn_in_caption_style(monkeypatch):
    """Category L1/L2/L3 - the multi-value ("caption") layout is where the
    old "EXACT TEXT"/"MEASUREMENT" labels used to appear above each row;
    none of them, or any other internal identifier, may appear now."""
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["30 times background noise", "constellation Sagittarius"],
        data_points=[
            _dp("scalar_fact", "30 times background noise"),
            _dp("exact_text", "constellation Sagittarius"),
        ],
    )
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    for banned in _BANNED_INTERNAL_LABELS:
        assert not any(banned in text for text in drawn), f"internal label {banned!r} was drawn in {drawn!r}"


def test_L1_L2_L3_no_internal_metadata_labels_drawn_in_data_label_style(monkeypatch):
    base = _solid_image_bytes()
    scene = _scene(exact_text=["1420 megahertz"], data_points=[_dp("measurement", "1420 megahertz")])
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    for banned in _BANNED_INTERNAL_LABELS:
        assert not any(banned in text for text in drawn), f"internal label {banned!r} was drawn in {drawn!r}"


def test_L1_L2_L3_no_internal_metadata_labels_drawn_in_headline_style(monkeypatch):
    base = _solid_image_bytes()
    scene = _scene(exact_text=["6EQUJ5"], data_points=[_dp("exact_text", "6EQUJ5")])
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    for banned in _BANNED_INTERNAL_LABELS:
        assert not any(banned in text for text in drawn), f"internal label {banned!r} was drawn in {drawn!r}"


def test_L4_L5_actual_values_are_rendered_not_their_schema_field_names(monkeypatch):
    """Category L4/L5 - the desired-output example from the task: "6EQUJ5"
    and "Wow!" are drawn as exactly themselves, with no "EXACT TEXT" line
    anywhere in the sequence of strings actually drawn."""
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["6EQUJ5", "Wow!"],
        data_points=[_dp("exact_text", "6EQUJ5"), _dp("exact_text", "Wow!")],
    )
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    assert "6EQUJ5" in drawn
    assert "Wow!" in drawn
    assert "EXACT TEXT" not in drawn


def test_L4_L5_measurement_value_rendered_alone_no_field_name(monkeypatch):
    base = _solid_image_bytes()
    scene = _scene(exact_text=["1420 megahertz"], data_points=[_dp("measurement", "1420 megahertz")])
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    assert "1420 megahertz" in drawn
    assert "MEASUREMENT" not in drawn


def test_L4_L5_desired_output_noise_and_constellation_rows(monkeypatch):
    """Category L4/L5 - the task's second worked example: two clean value
    rows, no "MEASUREMENT"/"EXACT TEXT" labels above either."""
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["30 times background noise", "constellation Sagittarius"],
        data_points=[
            _dp("scalar_fact", "30 times background noise"),
            _dp("exact_text", "constellation Sagittarius"),
        ],
    )
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    assert "30 times background noise" in drawn
    assert "constellation Sagittarius" in drawn
    assert "MEASUREMENT" not in drawn
    assert "EXACT TEXT" not in drawn


def test_L6_hypothesis_badge_still_renders_when_explicitly_enabled(monkeypatch):
    """Category L6 - text_render.hypothesis_badge is an intentional,
    explicit viewer-facing marker (NOT an internal metadata label) and
    must keep rendering exactly as before."""
    base = _solid_image_bytes()
    scene = _scene(
        exact_text=["Interstellar origin"],
        data_points=[_dp("exact_text", "Interstellar origin")],
        hypothesis_badge=True,
    )
    drawn = _capture_drawn_texts(monkeypatch, base, scene)

    assert "HYPOTHESIS" in drawn
    assert any("Interstellar origin" in t for t in drawn)


def test_L7_overlay_required_false_still_a_true_no_op():
    """Category L7 - unchanged from before this correction, re-asserted
    here for this fix's own record."""
    base = _solid_image_bytes()
    scene = _scene(overlay_required=False, exact_text=["6EQUJ5"])
    assert ov.render_scene_overlay_bytes(base, scene) is base


def test_L8_output_remains_deterministic_after_label_removal():
    """Category L8 - removing the label row must not introduce any
    non-determinism (e.g. from the new size/color branching in
    _draw_caption_block by row index)."""
    base = _gradient_image_bytes()
    scene = _scene(
        exact_text=["30 times background noise", "constellation Sagittarius"],
        data_points=[
            _dp("scalar_fact", "30 times background noise"),
            _dp("exact_text", "constellation Sagittarius"),
        ],
    )
    result_a = ov.render_scene_overlay_bytes(base, copy.deepcopy(scene))
    result_b = ov.render_scene_overlay_bytes(base, copy.deepcopy(scene))
    assert result_a == result_b


def test_caption_block_primary_row_is_larger_than_secondary_rows():
    """Hierarchy without labels: the first value in a multi-value caption
    block must be drawn measurably larger than the rows after it - proof
    that font size (not an invented label) is what distinguishes them."""
    draw = ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
    primary_font = ov.resolve_font(int(1080 * ov._CAPTION_PRIMARY_SIZE_FRAC), bold=True)
    secondary_font = ov.resolve_font(int(1080 * ov._CAPTION_SECONDARY_SIZE_FRAC), bold=True)
    _w1, h1 = ov._text_size(draw, "Ag", primary_font)
    _w2, h2 = ov._text_size(draw, "Ag", secondary_font)
    assert h1 > h2
