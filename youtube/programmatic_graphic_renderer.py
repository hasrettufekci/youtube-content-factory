"""
Scene Planner V2 - Phase 4: deterministic programmatic scientific graphic
renderer (spec follow-up to Phase 1-3 - see scene_planner.py's V2 section
banners and youtube/overlay_renderer.py's Phase 3 module docstring).

Replaces image_gen.py's diagnostic placeholder for
visual_strategy=="programmatic_graphic" scenes with a real, deterministic,
CPU-only graphic:

    scene metadata (data_points/exact_text/text_render)
        -> programmatic graphic renderer
        -> complete 1920x1080 PNG

No AI image model is involved at all for this strategy - not FAL, not any
other provider - see resolve_visual_strategy() (rule 2/3 in
scene_planner.py): a data_point whose data_type is one of
time_series/curve/spectrum/timeline is exactly the signal that a scene
needs a DRAWN chart/graphic, never a photographed/generated one, because
an image model cannot reliably render exact numbers. image_gen.py already
never calls provider.generate() for this strategy (see
_generate_image_bytes_for_scene()'s "programmatic_graphic" branch, which
returns before the provider call is reached) - this module is what that
branch now calls instead of writing a placeholder.

SCHEMA FINDING THIS MODULE IS BUILT AROUND (read before changing anything
here): Scene Planner V2's data_points schema
(SYSTEM_PROMPT_V2_CLASSIFICATION in scene_planner.py) shapes every data
point as {"data_type": ..., "value": "<the exact text/number/coordinate
STRING, verbatim>", "exact_render": bool} - "value" is always a single
STRING, never a numeric array/time series of samples. There is currently
no field anywhere in the schema that carries an actual plottable sequence
of (x, y) values, amplitudes, or sample points. This means a TRUE
quantitative chart - a real signal waveform, a real measurement-over-time
curve, a real bar-by-bar comparison of numbers - CANNOT be drawn from this
schema without inventing the missing sample data, which is exactly what
this module must never do (see NEVER INVENT SCIENTIFIC DATA below).

Consequently this module implements only the graphic types the CURRENT
schema can honestly support:
  - "timeline": 2+ data_points with data_type=="timeline" - plotted as
    labeled points along a horizontal line, in the exact order they
    appear in scene["data_points"] (assumed to already be narration/
    chronological order - never re-sorted or re-spaced by an invented
    date value). Point spacing is even/deterministic, not calendar-
    accurate, because no parseable numeric date exists in the schema to
    space accurately by - an evenly-spaced labeled timeline strip is an
    honest, standard documentary convention that asserts only ORDER, not
    precise intervals.
  - "conceptual_curve": a data_point with data_type=="curve" whose value
    is DESCRIPTIVE text (e.g. "celestial beam-crossing intensity
    profile"), not a bare number - real production scene 026 in
    storage/2026/09/the-wow-signal-6 is exactly this shape: {"data_type":
    "curve", "value": "celestial beam-crossing intensity profile",
    "exact_render": false}. exact_render is explicitly False here - same
    convention overlay_renderer.py already reads elsewhere in the schema
    (see its _render_items_for_scene()) as "this value describes intent/
    context, it is not meant to be printed on screen verbatim" - so this
    renderer never prints the curve's own value string. What it draws
    instead is a FIXED, deterministic, illustrative "beam encounter"
    shape (never derived from the value text or any numeric data - there
    is none) plus, only when scene["visual_relevance_reason"] uses the
    "...characteristics<EM-DASH>a, b, and c<EM-DASH>as..." structural
    list convention this scene's wording happens to use (see
    _extract_em_dash_list - a generic delimiter-based parse, not a
    scene-specific lookup), the enumerated qualitative dimension labels
    that sentence actually names. A restrained "CONCEPTUAL - NOT TO
    SCALE" marker is always drawn, since there is no dataset behind the
    shape. A future data_type=="curve" value that is instead a bare
    number/numeric list is deliberately NOT routed here (see
    _looks_purely_numeric()) - it would need its own honest quantitative
    renderer once the schema actually carries plottable samples.
  - "fallback_graphic": everything else, including empty/missing/
    malformed metadata. Draws whatever exact_text/data_point values ARE
    present (reusing overlay_renderer.py's own item-selection/drawing
    primitives - see below) on a restrained scientific-slate background;
    degrades all the way to a textless but still professional background
    when there is genuinely nothing safe to display. ALWAYS available -
    every scene this module is asked to render gets a complete, correct
    PNG, never a crash and never fabricated content.

A future schema change that adds real sample-array data (e.g. a
data_point shaped {"data_type": "time_series", "samples": [...]}) is what
would justify adding "signal_graph"/"measurement_chart"/"comparison_chart"
to the registry below - see _RENDERERS/_register(). Building any of those
today, against the schema as it actually exists, would require
fabricating the missing numbers, which this module refuses to do.

ONE SOURCE OF TRUTH FOR TEXT RENDERING: this module does not reimplement
font resolution, text measurement/wrapping/fitting, or the exact_text
drawing primitives - it imports and reuses them directly from
youtube/overlay_renderer.py (resolve_font, _text_size, _wrap_to_width,
_fit_text_block, _render_items_for_scene, _render_style_for_items,
_draw_data_label/_draw_headline/_draw_caption_block, and the shared design
tokens PANEL_FILL/PANEL_BORDER/DIVIDER/TEXT_PRIMARY/TEXT_SECONDARY/MARGIN/
CORNER_RADIUS). overlay_renderer.py's drawing primitives already take a
plain RGBA layer and a list of {"value","data_type"} items and know
nothing about where that layer came from - reusing them here (instead of
overlay_renderer.render_scene_overlay_bytes(), whose entrypoint gate is
scene["overlay_required"], which is only ever True for
"ai_generated_plus_overlay") means every exact_text-preservation/no-
internal-label guarantee already built and tested there (see
tests/test_overlay_renderer.py) applies here for free, with zero
duplicated logic.

Hard requirements this module satisfies structurally:
  - Pillow + stdlib only. No httpx/requests/network imports anywhere in
    this file - this module cannot make a network call even by accident.
  - Deterministic: no `random`, no wall-clock time in the drawing math
    (only font PATH resolution, delegated to overlay_renderer, is
    environment-dependent). Same scene dict in -> byte-identical PNG out.
  - Never fabricates a number, date, coordinate, or relationship not
    already present verbatim in scene["data_points"]/scene["exact_text"].
  - Never raises: a malformed/missing-data scene still gets a complete,
    professional PNG via the fallback_graphic path (see
    render_programmatic_graphic()'s try/except).
  - Internal renderer/schema identifiers (data_type names, this module's
    own renderer-type names) are never drawn as on-screen text - same
    rule overlay_renderer.py enforces for its own labels.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from core.logging_setup import get_logger
from youtube import overlay_renderer as ov

logger = get_logger("programmatic_graphic_renderer")

# ---------------------------------------------------------------------------
# Design tokens - deliberately the SAME dark documentary palette
# overlay_renderer.py uses (imported, not redefined) for panels/text, plus
# a small set of graphic-specific tokens (grid/axis/background) so a
# programmatic_graphic scene and an ai_generated_plus_overlay scene read as
# one consistent visual system across a finished video.
# ---------------------------------------------------------------------------
BACKGROUND_TOP = (10, 13, 19)
BACKGROUND_BOTTOM = (17, 22, 31)
GRID_LINE = (255, 255, 255, 14)
AXIS_LINE = (255, 255, 255, 70)
TICK_LINE = (255, 255, 255, 90)
GRID_SPACING_PX = 96

TIMELINE_MARGIN_X_FRAC = 0.10
TIMELINE_LABEL_GAP = 14
TIMELINE_TICK_HALF = 7

# conceptual_curve tokens
_EM_DASH = "—"
CONCEPTUAL_DISCLOSURE = "CONCEPTUAL — NOT TO SCALE"
CURVE_LINE_COLOR = (170, 190, 215)
CURVE_LINE_WIDTH = 3
CURVE_NODE_RADIUS = 5
_MAX_DIMENSION_LABELS = 6

# conceptual_curve - central beam/observation diagram
BEAM_AREA_TOP_FRAC = 0.10
BEAM_AREA_BOTTOM_FRAC = 0.46
BEAM_ARC_COLOR = (120, 145, 175, 90)
BEAM_ARC_BASE_RADIUS_FRAC = 0.05
BEAM_ARC_RADIUS_STEP_FRAC = 0.028
BEAM_ARC_SPANS = [(200, 340), (160, 320), (190, 300)]  # (start_deg, end_deg) - irregular, not full rings

# conceptual_curve - per-label qualitative trace legend
LEGEND_BLOCK_TOP_FRAC = 0.60
LEGEND_ROW_HEIGHT = 64
LEGEND_ROW_GAP = 16
LEGEND_ROW_PAD_X = 20
LEGEND_ROW_PAD_Y = 14
LEGEND_ICON_W = 180
LEGEND_ICON_GAP = 22

# Deterministic qualitative sparkline shapes for legend row icons, as
# (x_frac, y_frac) points within a unit box - selected by row index modulo
# len(_TRACE_SHAPES) (see _trace_shape_points). Purely a stylistic
# differentiator between rows - never derived from any real value.
_TRACE_SHAPES: list[list[tuple[float, float]]] = [
    [(0.00, 0.55), (0.30, 0.45), (0.55, 0.55), (0.80, 0.40), (1.00, 0.50)],  # gentle wave
    [(0.00, 0.70), (0.35, 0.70), (0.45, 0.30), (0.70, 0.30), (1.00, 0.30)],  # step up
    [(0.00, 0.30), (0.30, 0.30), (0.50, 0.75), (0.70, 0.75), (1.00, 0.35)],  # dip / valley
    [(0.00, 0.65), (0.25, 0.55), (0.50, 0.20), (0.75, 0.25), (1.00, 0.25)],  # rise then plateau
    [(0.00, 0.55), (0.20, 0.25), (0.40, 0.55), (0.60, 0.25), (0.80, 0.55), (1.00, 0.35)],  # double bump
    [(0.00, 0.35), (0.35, 0.35), (0.50, 0.60), (0.65, 0.35), (1.00, 0.35)],  # plateau with notch
]


def _build_background(width: int, height: int) -> Image.Image:
    """A restrained, deterministic "scientific slate" background: a dark
    vertical gradient plus a faint grid - the NASA/broadcast documentary
    look requested, with no photographic content (there is none to have -
    no AI image is generated for this strategy) and no randomness."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(BACKGROUND_TOP[0] + (BACKGROUND_BOTTOM[0] - BACKGROUND_TOP[0]) * t)
        g = int(BACKGROUND_TOP[1] + (BACKGROUND_BOTTOM[1] - BACKGROUND_TOP[1]) * t)
        b = int(BACKGROUND_TOP[2] + (BACKGROUND_BOTTOM[2] - BACKGROUND_TOP[2]) * t)
        for x in range(width):
            px[x, y] = (r, g, b)

    grid = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid)
    for x in range(0, width, GRID_SPACING_PX):
        gdraw.line([(x, 0), (x, height)], fill=GRID_LINE, width=1)
    for y in range(0, height, GRID_SPACING_PX):
        gdraw.line([(0, y), (width, y)], fill=GRID_LINE, width=1)

    return Image.alpha_composite(img.convert("RGBA"), grid).convert("RGB")


# ---------------------------------------------------------------------------
# Registry / dispatch - extensible by design (see module docstring for what
# would justify adding a new entry). Each renderer function takes the RGBA
# overlay layer to draw onto (same size as the final image) and the scene
# dict, and draws in place - it never returns/builds its own canvas, so
# every renderer composites onto the exact same background consistently.
# ---------------------------------------------------------------------------
_RENDERERS: dict[str, Callable[[Image.Image, dict], None]] = {}


def _register(name: str):
    def decorator(fn: Callable[[Image.Image, dict], None]):
        _RENDERERS[name] = fn
        return fn
    return decorator


def _sanitize_scene(scene: dict) -> dict:
    """Returns a shallow, defensively-cleaned copy of `scene` - never
    mutates the caller's dict (same invariant image_gen.py's overlay path
    already relies on) - with scene["data_points"] filtered down to actual
    dict entries only. Malformed data_points entries (None, a bare number,
    a bare string, ...) should never occur in a real, schema-validated
    scene (validate_prompts_v2() in scene_planner.py enforces the
    SYSTEM_PROMPT_V2_CLASSIFICATION shape before a scene is ever saved),
    but this renderer must still never crash on one - both for its own
    _timeline_items() above and for overlay_renderer._render_items_for_scene(),
    which this module's fallback_graphic renderer calls directly and which
    (as already-shipped, already-tested code this task does not modify)
    assumes well-formed dict entries."""
    data_points = scene.get("data_points")
    if isinstance(data_points, list):
        cleaned = [dp for dp in data_points if isinstance(dp, dict)]
        if cleaned != data_points:
            scene = {**scene, "data_points": cleaned}
    return scene


def _timeline_items(scene: dict) -> list[dict]:
    """The data_points this scene declares with data_type=="timeline" and a
    non-empty value - in their existing order (never re-sorted - see
    module docstring). Deliberately narrower than
    overlay_renderer._render_items_for_scene() (which also accepts a bare
    exact_render flag on any data_type): a timeline specifically needs
    data_type=="timeline" entries, since that is the one signal that a
    value represents a point in a chronological sequence rather than an
    isolated fact."""
    data_points = scene.get("data_points") or []
    items = []
    seen = set()
    if isinstance(data_points, list):
        for dp in data_points:
            if not isinstance(dp, dict):
                continue  # malformed entry (None/int/str/...) - skip, never crash
            if dp.get("data_type") != "timeline":
                continue
            value = str(dp.get("value", "")).strip()
            if value and value not in seen:
                seen.add(value)
                items.append(value)
    return items


def _looks_purely_numeric(value: str) -> bool:
    """True only if `value` is nothing but a number (or a bracket/comma-
    delimited list of numbers) - i.e. it would need to be PLOTTED as real
    data, not described qualitatively. Any real word/unit/letter content
    makes this False. Exists so a FUTURE schema that puts genuine numeric
    samples under data_type=="curve" is never silently routed into the
    conceptual/qualitative renderer below (see module docstring's
    FALLBACK guidance) - that would need its own honest quantitative
    renderer once real plottable data actually exists."""
    stripped = value.strip("[]() ")
    if not stripped:
        return False
    tokens = [t.strip() for t in stripped.replace(";", ",").split(",")]
    try:
        for t in tokens:
            float(t)
        return True
    except ValueError:
        return False


def _curve_items(scene: dict) -> list[str]:
    """Every data_point this scene declares with data_type=="curve" whose
    value is non-empty, DESCRIPTIVE text (not a bare number/numeric list -
    see _looks_purely_numeric) - the one structural signal (per the
    module's SCHEMA FINDING) that a scene needs a qualitative/conceptual
    curve diagram, not a plotted one. Purely structural - never checks
    scene_id/topic/slug, so any scene shaped this way (not only scene
    026) routes here the same way."""
    data_points = scene.get("data_points") or []
    items = []
    if isinstance(data_points, list):
        for dp in data_points:
            if not isinstance(dp, dict):
                continue
            if dp.get("data_type") != "curve":
                continue
            value = str(dp.get("value", "")).strip()
            if value and not _looks_purely_numeric(value):
                items.append(value)
    return items


def _extract_em_dash_list(text: str) -> list[str]:
    """Generic, scene-agnostic extraction of an enumerated list of
    concepts from a free-text description, when the text uses the
    "...characteristics<EM-DASH>a, b, and c<EM-DASH>as..." em-dash-
    delimited-list convention scene 026's visual_relevance_reason happens
    to use (a wording pattern Scene Planner V2's prompt can produce for
    any evidence-combining scene, not specific to this one topic/scene_id).
    Splits the clause between the FIRST pair of em-dashes on commas and a
    trailing "and" - deterministic, order-preserving, nothing invented.
    Returns [] whenever the text doesn't contain this exact structural
    shape (fewer than two em-dashes, or an empty clause between them) -
    never guesses at a list from unstructured prose, and a scene with a
    differently-worded visual_relevance_reason (or none at all) simply
    gets no dimension labels, not fabricated ones."""
    if not text or text.count(_EM_DASH) < 2:
        return []
    clause = text.split(_EM_DASH)[1].strip()
    if not clause:
        return []
    clause = re.sub(r"\band\b", ",", clause)
    items = [p.strip(" .") for p in clause.split(",")]
    items = [p for p in items if p]
    return items[:_MAX_DIMENSION_LABELS]


def _select_renderer_name(scene: dict) -> str:
    """Deterministic selection from STRUCTURAL facts only (which
    data_types are present and how many) - never from scene_id, topic, or
    narration content. Falls back to "fallback_graphic" whenever nothing
    more specific is honestly supportable by the data actually present -
    see the module docstring's SCHEMA FINDING for why "signal_graph" /
    "measurement_chart" / "comparison_chart" are not in this registry at
    all yet."""
    if len(_timeline_items(scene)) >= 2:
        return "timeline"
    if _curve_items(scene):
        return "conceptual_curve"
    return "fallback_graphic"


@_register("timeline")
def _draw_timeline(overlay: Image.Image, scene: dict) -> None:
    """Plots each timeline data_point's exact value string as a labeled
    point along one horizontal line - evenly spaced (not calendar-
    accurate; see module docstring), in narration order. Labels alternate
    above/below the line so adjacent dates never overlap regardless of
    string length."""
    width, height = overlay.size
    draw = ImageDraw.Draw(overlay)
    values = _timeline_items(scene)

    axis_y = height // 2
    x_start = int(width * TIMELINE_MARGIN_X_FRAC)
    x_end = width - int(width * TIMELINE_MARGIN_X_FRAC)
    draw.line([(x_start, axis_y), (x_end, axis_y)], fill=AXIS_LINE, width=2)

    n = len(values)
    step = (x_end - x_start) / (n - 1) if n > 1 else 0
    label_font = ov.resolve_font(int(height * 0.024), bold=True)
    max_label_width = int(width / max(n, 1)) - 12

    for i, value in enumerate(values):
        x = int(x_start + step * i)
        draw.line([(x, axis_y - TIMELINE_TICK_HALF), (x, axis_y + TIMELINE_TICK_HALF)], fill=TICK_LINE, width=2)
        draw.ellipse([x - 4, axis_y - 4, x + 4, axis_y + 4], fill=ov.TEXT_PRIMARY)

        font, lines, line_h = ov._fit_text_block(
            draw, value, max(max_label_width, 80), int(height * 0.12), int(height * 0.024), bold=True,
        )
        text_w = max((ov._text_size(draw, ln, font)[0] for ln in lines), default=0)
        above = (i % 2 == 0)
        ty = axis_y - TIMELINE_LABEL_GAP - line_h * len(lines) if above else axis_y + TIMELINE_LABEL_GAP
        for line in lines:
            lw, _lh = ov._text_size(draw, line, font)
            tx = min(max(x - lw // 2, 4), width - lw - 4)
            draw.text((tx, ty), line, font=font, fill=ov.TEXT_PRIMARY)
            ty += line_h


def _beam_diagram_points(width: int, height: int) -> list[tuple[int, int]]:
    """The central "beam encounter" path's five FIXED points, at fractions
    of the frame size within a top band of the frame (see
    BEAM_AREA_TOP_FRAC/BEAM_AREA_BOTTOM_FRAC) - never derived from any
    scene content, because there is no real sample data in the schema to
    derive an actual shape from (see module docstring)."""
    top = int(height * BEAM_AREA_TOP_FRAC)
    bottom = int(height * BEAM_AREA_BOTTOM_FRAC)
    span = bottom - top
    return [
        (int(width * 0.22), top + int(span * 0.05)),
        (int(width * 0.40), top + int(span * 0.45)),
        (int(width * 0.50), top + int(span * 0.62)),
        (int(width * 0.60), top + int(span * 0.62)),
        (int(width * 0.78), top + int(span * 0.95)),
    ]


def _draw_beam_crossing_arcs(draw: "ImageDraw.ImageDraw", cx: int, cy: int, height: int) -> None:
    """A small cluster of thin, irregular concentric arcs centered on the
    beam path's flat "crossing" segment - a restrained, deterministic
    visual motif for "several characteristics observed at once" (matching
    visual_relevance_reason's "overlaid evidence...present simultaneously"
    wording), deliberately partial arcs rather than full rings/tick marks
    so this reads as a documentary diagram, not a radar/HUD reticle."""
    for i, (start_deg, end_deg) in enumerate(BEAM_ARC_SPANS):
        r = int(height * BEAM_ARC_BASE_RADIUS_FRAC) + i * int(height * BEAM_ARC_RADIUS_STEP_FRAC)
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(box, start=start_deg, end=end_deg, fill=BEAM_ARC_COLOR, width=1)


def _draw_conceptual_beam(draw: "ImageDraw.ImageDraw", width: int, height: int) -> None:
    """Draws the central celestial-beam/observation diagram: the fixed
    five-point path (incoming beam -> crossing -> continuation) plus the
    crossing-point arc cluster. Purely illustrative - see module docstring
    for why nothing here is derived from real data."""
    points = _beam_diagram_points(width, height)
    draw.line(points, fill=CURVE_LINE_COLOR, width=CURVE_LINE_WIDTH, joint="curve")
    for x, y in (points[0], points[-1]):
        draw.ellipse(
            [x - CURVE_NODE_RADIUS, y - CURVE_NODE_RADIUS, x + CURVE_NODE_RADIUS, y + CURVE_NODE_RADIUS],
            fill=CURVE_LINE_COLOR,
        )
    crossing_x = (points[2][0] + points[3][0]) // 2
    crossing_y = (points[2][1] + points[3][1]) // 2
    _draw_beam_crossing_arcs(draw, crossing_x, crossing_y, height)


def _trace_shape_points(shape_index: int, box_w: float, box_h: float) -> list[tuple[float, float]]:
    """One of a small fixed set of qualitative sparkline-style shapes
    (as fractions of a box, converted to pixel points here), selected
    purely by shape_index % len(_TRACE_SHAPES) - a stylistic
    differentiator between legend rows so each labeled characteristic
    reads as visually distinct, never a plotted value (there is no data
    to plot - see module docstring). The same index always yields the
    same shape (deterministic), and this covers any number of legend rows
    generically - it is not tied to these 4 specific labels."""
    frac_points = _TRACE_SHAPES[shape_index % len(_TRACE_SHAPES)]
    return [(fx * box_w, fy * box_h) for fx, fy in frac_points]


def _draw_conceptual_legend(
    draw: "ImageDraw.ImageDraw", width: int, height: int, labels: list[str], font: "ImageFont.FreeTypeFont",
) -> None:
    """Draws one row per extracted dimension phrase: a small deterministic
    qualitative trace icon (see _trace_shape_points) immediately followed
    by the label text, inside a shared-width bordered panel (reusing
    ov._draw_panel() - the same panel styling overlay_renderer.py uses
    elsewhere) - giving each label a visible, distinct "trace" of its own
    rather than four unrelated pills with no visual relationship to the
    curve above them. Uppercasing the label is a pure typographic/label
    style choice (identical in spirit to the "HYPOTHESIS" badge's styling
    in overlay_renderer.py) - it does not alter which words were actually
    present in visual_relevance_reason."""
    upper_labels = [lbl.upper() for lbl in labels]
    text_sizes = [ov._text_size(draw, lbl, font) for lbl in upper_labels]
    max_text_w = max((w for w, _h in text_sizes), default=0)

    icon_h = LEGEND_ROW_HEIGHT - 2 * LEGEND_ROW_PAD_Y
    row_w = LEGEND_ROW_PAD_X + LEGEND_ICON_W + LEGEND_ICON_GAP + max_text_w + LEGEND_ROW_PAD_X
    total_h = len(labels) * LEGEND_ROW_HEIGHT + (len(labels) - 1) * LEGEND_ROW_GAP

    x = (width - row_w) // 2
    y = int(height * LEGEND_BLOCK_TOP_FRAC)
    # keep the whole legend block clear of the bottom-right disclosure marker
    max_y = height - ov.MARGIN - total_h
    y = min(y, max_y)

    for i, (label, (_tw, th)) in enumerate(zip(upper_labels, text_sizes)):
        row_y = y + i * (LEGEND_ROW_HEIGHT + LEGEND_ROW_GAP)
        box = (x, row_y, x + row_w, row_y + LEGEND_ROW_HEIGHT)
        ov._draw_panel(draw, box)

        icon_origin_x = x + LEGEND_ROW_PAD_X
        icon_origin_y = row_y + LEGEND_ROW_PAD_Y
        icon_pts = [
            (icon_origin_x + px, icon_origin_y + py)
            for px, py in _trace_shape_points(i, LEGEND_ICON_W, icon_h)
        ]
        draw.line(icon_pts, fill=CURVE_LINE_COLOR, width=2, joint="curve")

        text_x = x + LEGEND_ROW_PAD_X + LEGEND_ICON_W + LEGEND_ICON_GAP
        text_y = row_y + (LEGEND_ROW_HEIGHT - th) // 2
        draw.text((text_x, text_y), label, font=font, fill=ov.TEXT_PRIMARY)


def _draw_conceptual_disclosure(draw: "ImageDraw.ImageDraw", width: int, height: int) -> None:
    """Small, restrained bottom-right marker disclosing that the curve
    shape carries no real dataset - required whenever data_type=="curve"
    has no numeric samples behind it (see module docstring)."""
    font = ov.resolve_font(int(height * 0.016), bold=False)
    text_w, text_h = ov._text_size(draw, CONCEPTUAL_DISCLOSURE, font)
    x = width - ov.MARGIN - text_w
    y = height - ov.MARGIN - text_h
    draw.text((x, y), CONCEPTUAL_DISCLOSURE, font=font, fill=ov.TEXT_SECONDARY)


@_register("conceptual_curve")
def _draw_conceptual_curve(overlay: Image.Image, scene: dict) -> None:
    """Renders a restrained, multi-layer CONCEPTUAL EVIDENCE VISUALIZATION
    for a data_type=="curve" data point whose value is descriptive text,
    not a numeric sample array (see module docstring - this is exactly
    real production scene 026's shape). Two purely illustrative elements,
    neither ever derived from real data (there is none - see module
    docstring):
      - a central celestial-beam/observation diagram (_draw_conceptual_beam)
        with a small arc cluster at its "crossing" point, visually echoing
        visual_relevance_reason's "overlaid evidence...present
        simultaneously" wording;
      - when scene["visual_relevance_reason"] uses the structural
        em-dash-list convention (_extract_em_dash_list - generic, not a
        scene-026-specific lookup; genuinely absent, never fabricated, for
        a differently-worded scene), a legend of one row per extracted
        characteristic, each paired with its own small deterministic
        qualitative trace icon (_draw_conceptual_legend) so every label
        has a visible - if purely stylistic - "trace" of its own, instead
        of sitting on the diagram as unrelated pills.
    The curve's own value string (e.g. "celestial beam-crossing intensity
    profile") is deliberately never printed as on-screen text - its
    exact_render is False, the same "describes intent, not meant to be
    displayed verbatim" convention overlay_renderer.py already reads
    elsewhere in this schema (see _render_items_for_scene()). The
    "CONCEPTUAL - NOT TO SCALE" disclosure (_draw_conceptual_disclosure)
    is always drawn, since there is never a real dataset behind any of
    this."""
    width, height = overlay.size
    draw = ImageDraw.Draw(overlay)

    _draw_conceptual_beam(draw, width, height)

    labels = _extract_em_dash_list(str(scene.get("visual_relevance_reason") or ""))
    if labels:
        font = ov.resolve_font(int(height * 0.020), bold=True)
        _draw_conceptual_legend(draw, width, height, labels, font)

    _draw_conceptual_disclosure(draw, width, height)


@_register("fallback_graphic")
def _draw_fallback_graphic(overlay: Image.Image, scene: dict) -> None:
    """Always-safe path: draws whatever real exact_text/data_point values
    exist (via overlay_renderer's own item-selection + drawing primitives
    - see module docstring), placed exactly as overlay_renderer would
    place them on any other scene. When there are genuinely no renderable
    items (missing/empty metadata), draws nothing further - the
    scientific-slate background alone (already composited by the caller)
    is a complete, professional, textless graphic rather than a fabricated
    or placeholder-labeled one."""
    items = ov._render_items_for_scene(scene)
    if not items:
        return
    style = ov._render_style_for_items(items)
    if style == "data_label":
        ov._draw_data_label(overlay, items[0])
    elif style == "headline":
        ov._draw_headline(overlay, items[0])
    elif style == "caption":
        ov._draw_caption_block(overlay, items)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_programmatic_graphic(scene: dict, width: int = 1920, height: int = 1080) -> tuple[bytes, str]:
    """Builds the COMPLETE programmatic_graphic scene image from scene
    metadata alone - background + data - and returns (png_bytes,
    renderer_name_used). Deterministic: identical scene/width/height
    always produces byte-identical PNG bytes. Never raises: if the
    selected renderer fails for any reason (defensive only - the
    registered renderers above do not raise on the inputs
    _select_renderer_name() routes to them), this falls back to a clean
    background-only graphic rather than letting an image-generation stage
    fail the whole pipeline over a rendering bug."""
    scene = _sanitize_scene(scene)
    renderer_name = _select_renderer_name(scene)
    renderer_fn = _RENDERERS.get(renderer_name)
    if renderer_fn is None:
        # _select_renderer_name() returned a name with no registered
        # function (should never happen from the real selection logic
        # above - defensive only). Fall back and report the name of the
        # function that ACTUALLY ran, never the bogus/unrecognized one.
        renderer_name = "fallback_graphic"
        renderer_fn = _RENDERERS["fallback_graphic"]

    background = _build_background(width, height)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    try:
        renderer_fn(overlay, scene)
    except Exception:
        logger.exception(
            "[IMAGE][PROGRAMMATIC] renderer=%s raised while drawing - falling back to a clean "
            "background-only graphic rather than failing the scene", renderer_name,
        )
        renderer_name = "fallback_graphic"
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    composited = Image.alpha_composite(background.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    return buf.getvalue(), renderer_name


def render_programmatic_graphic_to_file(
    scene: dict, output_path: str | Path, width: int = 1920, height: int = 1080,
) -> tuple[Path, str]:
    """File-based wrapper: writes the rendered PNG to `output_path`
    (parent directories created as needed, atomic write) and returns
    (output_path, renderer_name_used). Never reads or writes anything
    under storage/ itself - the caller decides the path, exactly like
    overlay_renderer.render_scene_overlay()."""
    output_path = Path(output_path)
    png_bytes, renderer_name = render_programmatic_graphic(scene, width=width, height=height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_bytes(png_bytes)
    tmp_path.replace(output_path)
    return output_path, renderer_name
