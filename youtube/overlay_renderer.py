"""
Scene Planner V2 - Phase 3: deterministic overlay renderer (spec follow-up
to Phase 1/Phase 2 - see scene_planner.py's V2 section banners).

Replaces image_gen.py's [OVERLAY_PENDING] placeholder for
visual_strategy=="ai_generated_plus_overlay" scenes (and, generically, any
scene where overlay_required is truthy, whatever its visual_strategy) with
a real, deterministic, CPU-only compositing pass:

    AI-generated base image  +  Pillow-drawn factual text/numbers
        = the final scene image the assembler consumes.

Why a separate module instead of drawing inside image_gen.py: this is a
presentation-layer concern (font resolution, layout, text fitting,
drawing) with zero business logic - it never decides visual_strategy,
never calls an LLM or an image-generation API, and never touches
exact_text/data_points/narration/visual_prompt themselves. It only reads
those fields and burns their EXISTING values onto an EXISTING image.
Keeping it in its own module also keeps image_gen.py's provider-dispatch
responsibility unmixed with rendering responsibility - the same separation
scene_planner.py itself already draws between Phase 1 (classification),
Phase 2 (prompt assembly) and this Phase 3 (pixels).

Design principle (per the calling task's own framing):
    AI generates:    visual composition / background / atmosphere.
    Pillow generates: exact text, numbers, dates, scientific labels.
An image model is never asked to render exact factual text - see
scene_planner.py's SYSTEM_PROMPT_V2_CLASSIFICATION and
_scrub_exact_text_from_segments() for the upstream half of that same
invariant (exact_text is kept OUT of visual_prompt); this module is the
downstream half (exact_text is drawn deterministically instead).

Hard requirements this module satisfies structurally, not just by
intent - every one of these is a property of the code, not a comment:
  - Pillow only (PIL.Image/ImageDraw/ImageFont) - no other imaging lib.
  - Deterministic: no `random`, no wall-clock time, no environment-
    dependent branching in the drawing math itself (only font PATH
    resolution is environment-dependent, and that's cached and logged -
    see resolve_font_path()). Same (image bytes, scene dict) in ->
    byte-identical PNG out.
  - CPU-only, no network calls of any kind - this module does not import
    httpx or any HTTP client.
  - scene["exact_text"], scene["data_points"], scene["narration_en"] and
    scene["visual_prompt"] are read-only here - never mutated, never
    returned modified (the caller's scene dict is never touched; this
    module only ever returns new image bytes/files).
  - overlay_required is the single gate: False (or absent) -> the input
    image is returned/copied completely unchanged (not even re-encoded).

Schema this module reads (already produced by scene_planner.py - Phase
1/2's existing, real schema; nothing here invents a new scene field):
  - scene["overlay_required"]: bool (build_scene_prompt()) - the gate.
  - scene["exact_text"]: list[str] (_derive_exact_text()) - the literal
    strings that must render character-for-character.
  - scene["data_points"]: list[{"data_type", "value", "exact_render"}]
    (Scene Planner V2's own DATA_TYPES_V2 vocabulary) - each entry's
    data_type is used ONLY internally, to pick a render STYLE (see
    _render_style_for_items()) - it is never drawn as visible text. The
    viewer only ever sees the "value" strings themselves; "data_type"
    (measurement/coordinate/exact_text/...) is an implementation detail
    of this renderer and of Scene Planner V2's schema, not viewer-facing
    content - see _render_items_for_scene().
  - scene["text_render"]: {"items": [...], "hypothesis_badge": bool}
    (_derive_text_render()) - items is a redundant convenience mirror of
    exact_text (kept as the primary source of truth when data_points is
    unavailable/incomplete - see _render_items_for_scene());
    hypothesis_badge drives the small "HYPOTHESIS" corner tag - the one
    intentional, viewer-facing marker this module ever draws that isn't a
    literal exact_text/data_point value (it flags a scene as depicting an
    unproven idea, which the viewer genuinely needs to know).
  - scene["scene_id"]: used only for logging/determinism-by-content, not
    for layout (layout depends only on item count/data_type/position hint,
    never on scene_id - requirement: no scene_id-specific logic).

Presentation rule (production-quality correction): internal
renderer/schema identifiers - "EXACT TEXT", "MEASUREMENT", "DATA LABEL",
"CAPTION", "HEADLINE", "DATA POINT", or any other field/type name - are
NEVER drawn onto the image, however they get derived. Only two kinds of
text ever appear on screen: (1) a value string that already exists
verbatim in scene["exact_text"]/scene["data_points"], and (2) the literal
word "HYPOTHESIS", which is an intentional, explicit viewer-facing marker
gated on text_render.hypothesis_badge, not a schema-field label. Visual
hierarchy between multiple values on one panel (see _draw_caption_block())
comes only from font size/weight/spacing/dividers - never from an
invented or mechanically-derived semantic label.

No field in the current schema names a screen position, so
_default_position() below is a deterministic function of the RENDER
STYLE only (never of scene_id, topic, or any other scene-specific value)
- see its docstring. If a future caller starts stamping scene metadata
with an explicit position (e.g. scene["overlay_position"]), that value is
read and honored first - see render_scene_overlay_bytes().
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.logging_setup import get_logger

logger = get_logger("overlay_renderer")

# ---------------------------------------------------------------------------
# Design tokens - "premium science documentary" aesthetic: dark, restrained,
# translucent panels, a single muted accent color, no glow/animation/giant
# text. Every color is RGBA so panels can be alpha-composited onto the base
# image instead of drawn as flat opaque rectangles.
# ---------------------------------------------------------------------------
PANEL_FILL = (8, 11, 16, 196)
PANEL_BORDER = (255, 255, 255, 46)
DIVIDER = (255, 255, 255, 28)
TEXT_PRIMARY = (240, 241, 238, 255)
# A dimmer variant of TEXT_PRIMARY - NOT a label color (no label text is
# ever drawn). Used only to give a secondary row in a multi-value caption
# block slightly less visual weight than the primary row - hierarchy via
# opacity/tone, never via an invented semantic tag.
TEXT_SECONDARY = (196, 200, 204, 255)
BADGE_FILL = (94, 48, 18, 214)
BADGE_BORDER = (214, 150, 90, 90)
BADGE_TEXT = (247, 214, 178, 255)

MARGIN = 56                # safe distance from every image edge
PANEL_PADDING_X = 30
PANEL_PADDING_Y = 22
ROW_GAP = 16
CORNER_RADIUS = 10
LINE_SPACING = 1.22

# Font sizes are expressed as a fraction of image height so layout scales
# sanely to a non-1920x1080 input (still capped to sensible pixel bounds).
# Hierarchy between a caption block's rows is expressed purely through
# these two sizes (primary = first row, secondary = every row after it) -
# see _draw_caption_block().
_DATA_VALUE_SIZE_FRAC = 0.032
_HEADLINE_SIZE_FRAC = 0.046
_CAPTION_PRIMARY_SIZE_FRAC = 0.034
_CAPTION_SECONDARY_SIZE_FRAC = 0.026
_BADGE_SIZE_FRAC = 0.016
_MIN_FONT_PX = 14

# ---------------------------------------------------------------------------
# Font resolution - Windows is the primary production environment, but this
# must never hard-depend on one machine's exact font layout (the calling
# task's own requirement). Every candidate is checked with Path.is_file();
# the first that exists wins; the search result is cached per process
# (never re-globbing the filesystem on every scene). Pillow's built-in
# bitmap fallback (ImageFont.load_default) is the final, always-available
# fallback so this NEVER raises just because no TTF could be found.
# ---------------------------------------------------------------------------
_FONT_CANDIDATES_REGULAR = [
    # Windows
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
_FONT_CANDIDATES_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

_font_path_cache: dict[bool, str | None] = {}
_font_object_cache: dict[tuple[int, bool], "ImageFont.FreeTypeFont | ImageFont.ImageFont"] = {}


def resolve_font_path(bold: bool = False) -> str | None:
    """Returns the first existing TTF/OTF path from a sensible
    Windows/Linux/macOS search list, or None if none exist on this
    machine (caller falls back to Pillow's bitmap default font - see
    resolve_font()). Cached per process/per weight so this never re-stats
    the filesystem once resolved once."""
    if bold in _font_path_cache:
        return _font_path_cache[bold]

    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    found = None
    for candidate in candidates:
        try:
            if Path(candidate).is_file():
                found = candidate
                break
        except OSError:
            continue

    if found is None:
        logger.warning(
            "[IMAGE][OVERLAY][FONT] no %s TrueType font found in the known search paths - "
            "falling back to Pillow's built-in bitmap font (text will still render, just "
            "with a plainer typeface)",
            "bold" if bold else "regular",
        )
    _font_path_cache[bold] = found
    return found


def resolve_font(size_px: int, bold: bool = False) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Returns a usable Pillow font object for `size_px`, resolved via
    resolve_font_path() and cached per (size, weight) pair. Never raises -
    a missing/unreadable TTF falls back to the next candidate's failure
    being swallowed, and total absence of any TTF falls back to Pillow's
    always-available built-in font."""
    size_px = max(int(size_px), _MIN_FONT_PX)
    cache_key = (size_px, bold)
    if cache_key in _font_object_cache:
        return _font_object_cache[cache_key]

    font: "ImageFont.FreeTypeFont | ImageFont.ImageFont"
    path = resolve_font_path(bold=bold)
    font = None
    if path:
        try:
            font = ImageFont.truetype(path, size=size_px)
        except OSError:
            logger.warning("[IMAGE][OVERLAY][FONT] failed to load %s at size %d - falling back", path, size_px)
            font = None
    if font is None:
        try:
            font = ImageFont.load_default(size=size_px)
        except TypeError:
            # Older Pillow: load_default() takes no size argument at all.
            font = ImageFont.load_default()

    _font_object_cache[cache_key] = font
    return font


# ---------------------------------------------------------------------------
# Text measurement / wrapping / fitting
# ---------------------------------------------------------------------------

def _text_size(draw: "ImageDraw.ImageDraw", text: str, font) -> tuple[int, int]:
    if not text:
        return 0, 0
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _wrap_to_width(draw: "ImageDraw.ImageDraw", text: str, font, max_width: int) -> list[str]:
    """Deterministic greedy word-wrap - never truncates: every word from
    `text` appears in the output, split across as many lines as needed.
    A single word wider than max_width on its own is still kept whole
    (never character-truncated) - it simply overflows that one line
    slightly rather than losing factual content."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if _text_size(draw, trial, font)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_text_block(
    draw: "ImageDraw.ImageDraw",
    text: str,
    max_width: int,
    max_height: int,
    start_size_px: int,
    bold: bool = False,
    min_size_px: int = _MIN_FONT_PX,
) -> tuple["ImageFont.FreeTypeFont | ImageFont.ImageFont", list[str], int]:
    """Reduces font size in fixed, deterministic steps until the wrapped
    text fits within (max_width, max_height), or bottoms out at
    `min_size_px` - at which point the full (still-wrapped, never
    truncated) text is returned anyway, even if it slightly exceeds
    max_height, because factual text must never be cut off or dropped.
    Returns (font, wrapped_lines, line_height_px)."""
    size = max(int(start_size_px), min_size_px)
    font = resolve_font(size, bold=bold)
    lines = _wrap_to_width(draw, text, font, max_width)
    line_height = max(int(_text_size(draw, "Ag", font)[1] * LINE_SPACING), 1)

    while size > min_size_px:
        block_height = line_height * len(lines)
        widest = max((_text_size(draw, ln, font)[0] for ln in lines), default=0)
        if block_height <= max_height and widest <= max_width:
            return font, lines, line_height
        size = max(size - 2, min_size_px)
        font = resolve_font(size, bold=bold)
        lines = _wrap_to_width(draw, text, font, max_width)
        line_height = max(int(_text_size(draw, "Ag", font)[1] * LINE_SPACING), 1)

    return font, lines, line_height


# ---------------------------------------------------------------------------
# Render items - deterministically derived from scene metadata, never
# authored/invented here.
# ---------------------------------------------------------------------------

# data_type values that read as a compact "measurement/reading" rather
# than a bare quote/name/date - used ONLY to pick a render STYLE (see
# _render_style_for_items()); the data_type value itself is never drawn.
_DATA_STYLE_TYPES = {"measurement", "coordinate", "scalar_fact", "location"}


def _render_items_for_scene(scene: dict) -> list[dict]:
    """Builds the ordered list of {"value", "data_type"} items this scene
    must render, from data_points where available (data_type is carried
    along ONLY for internal style selection - see _render_style_for_items -
    and is never itself displayed), falling back to scene["exact_text"] /
    scene["text_render"]["items"] (data_type=None) when data_points is
    missing, empty, or doesn't line up 1:1 with exact_text (e.g.
    hand-built test scenes, or older data). De-duplicated by value,
    order-preserving - mirrors scene_planner.py's own
    _derive_exact_text() de-duplication so the overlay never repeats a
    value that was already de-duplicated upstream.

    Deliberately only includes entries that were actually meant to render
    as on-screen text (data_type=="exact_text" or exact_render truthy) -
    a data_point that only informs the visual conceptually
    (exact_render=False) is intentionally left off the image, per Scene
    Planner V2's own classification contract."""
    data_points = scene.get("data_points") or []
    items: list[dict] = []
    seen: set[str] = set()

    if isinstance(data_points, list):
        for dp in data_points:
            dp = dp or {}
            if not (dp.get("data_type") == "exact_text" or dp.get("exact_render")):
                continue
            value = str(dp.get("value", "")).strip()
            if value and value not in seen:
                seen.add(value)
                items.append({"value": value, "data_type": dp.get("data_type")})

    if items:
        return items

    # Fallback: no usable data_points - use exact_text / text_render.items
    # directly (already de-duplicated/order-preserving by construction -
    # see _derive_exact_text()/_derive_text_render()); no data_type is
    # available to attach, so style selection treats these as bare values.
    fallback_values = scene.get("exact_text")
    if not fallback_values:
        text_render = scene.get("text_render") or {}
        fallback_values = text_render.get("items")
    for value in (fallback_values or []):
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            items.append({"value": value, "data_type": None})

    return items


def _render_style_for_items(items: list[dict]) -> str:
    """Deterministic style selection from STRUCTURAL facts only (item
    count + each item's own internal data_type) - never from scene_id,
    topic, or any narration content, and never itself drawn as text:
      - 0 items: "none" (nothing to draw).
      - 1 item whose data_type is a measurement/reading-style category:
        compact "data_label" HUD readout.
      - 1 item otherwise (a bare quote/code/name/date): "headline" - one
        large, short callout.
      - 2+ items: "caption" - a stacked multi-row information block."""
    if not items:
        return "none"
    if len(items) == 1:
        if items[0]["data_type"] in _DATA_STYLE_TYPES:
            return "data_label"
        return "headline"
    return "caption"


def _default_position(style: str) -> str:
    """Deterministic default safe-zone for a given render style - a pure
    function of the style only (see _render_style_for_items() - never of
    scene_id/topic/narration), chosen so the block never sits over the
    vertical/horizontal center of the frame (where a generated image's
    primary subject typically sits - e.g. a telescope, a person, a
    building):
      - "data_label"  -> bottom-right corner (a HUD-style readout tucked
        into a corner, the conventional placement for a small data tag).
      - "headline"     -> bottom-left, within the lower third.
      - "caption"      -> lower-third, full width - a stacked list reads
        better as one wide band than a narrow corner block."""
    return {"data_label": "bottom-right", "headline": "bottom-left", "caption": "lower-third"}.get(style, "lower-third")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_panel(draw: "ImageDraw.ImageDraw", box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=CORNER_RADIUS, fill=PANEL_FILL, outline=PANEL_BORDER, width=1)


def _draw_hypothesis_badge(overlay: Image.Image, image_width: int) -> None:
    """Small, restrained top-right corner tag - the only place
    hypothesis_badge is drawn, entirely independent of whatever main
    block (headline/data_label/caption) is drawn elsewhere, per
    scene_planner.py's own hypothesis_badge field (set whenever
    evidence_level=="hypothesis" - see _derive_text_render())."""
    draw = ImageDraw.Draw(overlay)
    text = "HYPOTHESIS"
    font = resolve_font(max(int(overlay.height * _BADGE_SIZE_FRAC), _MIN_FONT_PX), bold=True)
    text_w, text_h = _text_size(draw, text, font)
    pad_x, pad_y = 18, 10
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    x1 = image_width - MARGIN - box_w
    y1 = MARGIN
    box = (x1, y1, x1 + box_w, y1 + box_h)
    draw.rounded_rectangle(box, radius=box_h // 2, fill=BADGE_FILL, outline=BADGE_BORDER, width=1)
    draw.text((x1 + pad_x, y1 + pad_y - 1), text, font=font, fill=BADGE_TEXT)


def _draw_data_label(overlay: Image.Image, item: dict) -> None:
    """Compact "HUD readout" block, bottom-right corner: the value alone,
    bold, in a tight panel - e.g. "1420 megahertz". No field-name/label
    text is drawn; the panel's small, corner-tucked, monospace-leaning
    presentation is itself what signals "this is a reading", not a
    caption above it."""
    width, height = overlay.size
    draw = ImageDraw.Draw(overlay)

    max_text_width = int(width * 0.34)
    value_font, value_lines, value_line_h = _fit_text_block(
        draw, item["value"], max_text_width, int(height * 0.22), int(height * _DATA_VALUE_SIZE_FRAC), bold=True,
    )
    value_w = max((_text_size(draw, ln, value_font)[0] for ln in value_lines), default=0)

    content_h = value_line_h * len(value_lines)
    panel_w = value_w + PANEL_PADDING_X * 2
    panel_h = content_h + PANEL_PADDING_Y * 2
    x1 = width - MARGIN - panel_w
    y1 = height - MARGIN - panel_h
    _draw_panel(draw, (x1, y1, x1 + panel_w, y1 + panel_h))

    cursor_y = y1 + PANEL_PADDING_Y
    for line in value_lines:
        draw.text((x1 + PANEL_PADDING_X, cursor_y), line, font=value_font, fill=TEXT_PRIMARY)
        cursor_y += value_line_h


def _draw_headline(overlay: Image.Image, item: dict) -> None:
    """One large, short callout, bottom-left within the lower third - a
    bare exact_text value with no structured data_type (a quote, a code,
    a name, a short date) presented as a single documentary-style title
    card line. No field-name/label text is drawn above it - the large,
    bold, standalone presentation is itself the hierarchy cue."""
    width, height = overlay.size
    draw = ImageDraw.Draw(overlay)

    max_text_width = int(width * 0.62)
    max_text_height = int(height * 0.22)

    headline_font, lines, line_h = _fit_text_block(
        draw, item["value"], max_text_width, max_text_height, int(height * _HEADLINE_SIZE_FRAC), bold=True,
    )
    content_w = max((_text_size(draw, ln, headline_font)[0] for ln in lines), default=0)
    content_h = line_h * len(lines)

    panel_w = content_w + PANEL_PADDING_X * 2
    panel_h = content_h + PANEL_PADDING_Y * 2
    x1 = MARGIN
    y1 = height - MARGIN - panel_h
    _draw_panel(draw, (x1, y1, x1 + panel_w, y1 + panel_h))

    cursor_y = y1 + PANEL_PADDING_Y
    for line in lines:
        draw.text((x1 + PANEL_PADDING_X, cursor_y), line, font=headline_font, fill=TEXT_PRIMARY)
        cursor_y += line_h


def _draw_caption_block(overlay: Image.Image, items: list[dict]) -> None:
    """Stacked multi-row information block, lower-third, full safe width -
    one row per value, separated by a thin divider. Used whenever a scene
    has 2+ exact-text values (e.g. "6EQUJ5" + "Wow!") so they read as one
    coherent panel rather than scattered independent strings.

    No field-name/label text is ever drawn - visual hierarchy between rows
    comes only from size/weight/color/dividers: the FIRST value (the
    scene's primary fact, in narration order - never re-ordered by
    "importance", which this renderer has no way to judge) is drawn larger
    and in full-brightness TEXT_PRIMARY; every subsequent value is drawn
    smaller, still bold, in the slightly dimmer TEXT_SECONDARY tone. A
    thin divider line separates each row from the next."""
    width, height = overlay.size
    draw = ImageDraw.Draw(overlay)

    max_text_width = width - MARGIN * 2 - PANEL_PADDING_X * 2

    rows = []
    content_h = 0
    for i, item in enumerate(items):
        is_primary = i == 0
        size_frac = _CAPTION_PRIMARY_SIZE_FRAC if is_primary else _CAPTION_SECONDARY_SIZE_FRAC
        color = TEXT_PRIMARY if is_primary else TEXT_SECONDARY
        value_font, value_lines, value_line_h = _fit_text_block(
            draw, item["value"], max_text_width, int(height * 0.14), int(height * size_frac), bold=True,
        )
        row_h = value_line_h * len(value_lines)
        rows.append((value_font, value_lines, value_line_h, color))
        content_h += row_h
    content_h += ROW_GAP * (len(rows) - 1) if rows else 0

    panel_w = width - MARGIN * 2
    panel_h = content_h + PANEL_PADDING_Y * 2
    x1 = MARGIN
    y1 = height - MARGIN - panel_h
    _draw_panel(draw, (x1, y1, x1 + panel_w, y1 + panel_h))

    cursor_y = y1 + PANEL_PADDING_Y
    for i, (value_font, value_lines, value_line_h, color) in enumerate(rows):
        if i > 0:
            draw.line(
                [(x1 + PANEL_PADDING_X, cursor_y - ROW_GAP // 2),
                 (x1 + panel_w - PANEL_PADDING_X, cursor_y - ROW_GAP // 2)],
                fill=DIVIDER, width=1,
            )
        for line in value_lines:
            draw.text((x1 + PANEL_PADDING_X, cursor_y), line, font=value_font, fill=color)
            cursor_y += value_line_h
        cursor_y += ROW_GAP


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_scene_overlay_bytes(image_bytes: bytes, scene: dict) -> bytes:
    """Pure, in-memory renderer - the primary entrypoint (render_scene_overlay()
    below is a thin file I/O wrapper around this). Never mutates `scene` or
    `image_bytes`; always returns a NEW bytes object except in the
    unchanged-passthrough case, where the exact same `image_bytes` object is
    returned (requirement: overlay_required=False returns the original
    image unchanged - not just pixel-identical, but a true no-op).

    Deterministic: identical (image_bytes, scene) always produces
    byte-identical PNG output - no randomness, no timestamps, no
    environment-dependent drawing decisions (only which font FILE is used
    depends on the machine; the drawing math itself does not)."""
    if not scene.get("overlay_required"):
        return image_bytes

    items = _render_items_for_scene(scene)
    hypothesis_badge = bool((scene.get("text_render") or {}).get("hypothesis_badge"))
    style = _render_style_for_items(items)

    if not items and not hypothesis_badge:
        # Nothing to draw (e.g. overlay_required somehow True but no
        # exact_text/data_points survived - missing-metadata edge case) -
        # still a no-op passthrough, never an error: a presentation layer
        # must not crash the pipeline over absent decoration.
        return image_bytes

    base = Image.open(io.BytesIO(image_bytes))
    base = base.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    scene_id = scene.get("scene_id")
    position = scene.get("overlay_position") or _default_position(style)

    if style == "data_label":
        _draw_data_label(overlay, items[0])
    elif style == "headline":
        _draw_headline(overlay, items[0])
    elif style == "caption":
        _draw_caption_block(overlay, items)

    if hypothesis_badge:
        _draw_hypothesis_badge(overlay, base.width)

    composited = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    logger.info(
        "[IMAGE][OVERLAY] scene=%s rendered style=%s position=%s exact_text=%d data_points=%d",
        f"{scene_id:03d}" if isinstance(scene_id, int) else scene_id,
        style, position, len(items), len(scene.get("data_points") or []),
    )

    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    return buf.getvalue()


def render_scene_overlay(image_path: str | Path, scene: dict, output_path: str | Path) -> Path:
    """File-based wrapper around render_scene_overlay_bytes(): reads
    `image_path` (read-only - never modified, never deleted), renders, and
    writes the result to `output_path` (parent directories created as
    needed). Returns `output_path`. Safe to call with output_path in a
    brand-new directory (e.g. debug_audit/overlay_v2/) so real production
    artifacts under storage/ are never at risk of being overwritten by a
    demo/audit run."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    image_bytes = image_path.read_bytes()
    rendered = render_scene_overlay_bytes(image_bytes, scene)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_bytes(rendered)
    tmp_path.replace(output_path)
    return output_path
