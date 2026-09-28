"""
Deterministic Text Renderer (Thumbnail System V1 - stage 3 of 4:
    STRATEGIST -> FAL BACKGROUND GENERATION -> TEXT OVERLAY -> QC).

Pillow only. Draws the strategist's exact headline/secondary_text strings
onto the stage-2 FAL background, producing the final
settings.THUMBNAIL_WIDTH x settings.THUMBNAIL_HEIGHT (1280x720) PNG. This
is the ONLY place in the thumbnail pipeline that draws viewer-facing text -
mirrors youtube/overlay_renderer.py's own design principle exactly ("AI
generates the visual composition; Pillow generates the exact text"), and
reuses that module's font-resolution and text-measurement/wrapping
primitives (resolve_font(), the internal text-size/greedy-wrap helpers)
rather than re-implementing font search or word-wrap from scratch - see the
import below. What IS new here (not reused from overlay_renderer.py,
because the requirement is genuinely different) is the fitting POLICY:
overlay_renderer's _fit_text_block() shrinks a font until it fits and, if
it still doesn't at the minimum size, draws it anyway (acceptable for a
small in-scene data caption). A thumbnail headline has no such fallback -
this task is explicit that a thumbnail must NEVER silently clip, overflow
the canvas, or shrink below a readable size; if wrapping and shrinking
within a safe font-size floor still doesn't fit, this module raises
ThumbnailRenderError instead of drawing anything - see _fit_or_fail().

POSITION MAPPING (semantic, never arbitrary pixel coordinates):
VALID_POSITIONS is the complete, fixed set of six positions this module
understands - top-left, top-right, bottom-left, bottom-right, center-left,
center-right. The strategist (youtube/thumbnail_strategist.py) only ever
supplies one of these strings; _position_anchor() is the single place that
maps a position name to actual pixel coordinates. There is no code path
anywhere in this module that accepts or draws at an LLM-supplied x/y pair.

TEXT FITTING (never silently truncates - see _fit_or_fail()):
  1. try line wrapping (attempted at every candidate font size)
  2. reduce font size in fixed steps, within a safe, still-readable floor
  3. if it still doesn't fit at the floor size: raise, never truncate,
     never draw text smaller than the readable floor, never let it clip.

DETERMINISM: no `random`, no wall-clock time, no environment-dependent
drawing decisions (only which font FILE gets used is machine-dependent,
exactly like overlay_renderer.py - see resolve_font()). Identical
(background_bytes, strategy) always produces byte-identical PNG output -
see tests/test_thumbnail_renderer.py's determinism test.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from core.config import settings
from core.logging_setup import get_logger
from youtube.overlay_renderer import resolve_font, _text_size, _wrap_to_width

logger = get_logger("thumbnail_renderer")

WIDTH = settings.THUMBNAIL_WIDTH
HEIGHT = settings.THUMBNAIL_HEIGHT

VALID_POSITIONS = (
    "top-left", "top-right", "bottom-left", "bottom-right", "center-left", "center-right",
)

MARGIN = 56
MAX_TEXT_WIDTH_FRAC = 0.60          # leaves negative space on the opposite side for the subject
HEADLINE_MAX_HEIGHT_FRAC = 0.42
SECONDARY_MAX_HEIGHT_FRAC = 0.16

HEADLINE_START_SIZE_FRAC = 0.145    # ~104px at 720 tall - "large, bold, dominant"
HEADLINE_MIN_SIZE_FRAC = 0.075      # ~54px - the readable floor; below this, fail rather than shrink further
SECONDARY_START_SIZE_FRAC = 0.060
SECONDARY_MIN_SIZE_FRAC = 0.032

FONT_SIZE_STEP = 2
STROKE_WIDTH_FRAC = 0.07            # relative to the font's own size
SHADOW_OFFSET = (4, 4)
SHADOW_COLOR = (0, 0, 0, 165)
HEADLINE_FILL = (255, 255, 255, 255)
HEADLINE_STROKE = (0, 0, 0, 255)
SECONDARY_FILL = (255, 214, 0, 255)   # accent color - visually distinct from + smaller than the headline
SECONDARY_STROKE = (0, 0, 0, 255)
LINE_SPACING = 1.15
BLOCK_GAP = 14                        # vertical gap between the headline block and the secondary block


class ThumbnailRenderError(RuntimeError):
    """Raised for a malformed/wrong-sized background, an invalid
    text_position, or text that cannot fit within its safe bounds even at
    the minimum readable font size. Never raised silently swallowed -
    every caller in this pipeline (youtube/thumbnail_demo.py) treats this
    as a hard pipeline failure."""


@dataclass
class ThumbnailRenderResult:
    image_bytes: bytes
    text_position: str
    headline_box: tuple[int, int, int, int]           # (x1, y1, x2, y2), pixel-absolute
    headline_font_size: int
    secondary_box: tuple[int, int, int, int] | None    # None when strategy.secondary_text is None
    secondary_font_size: int | None


def _validate_strategy_for_render(strategy: dict) -> None:
    if not strategy or not str(strategy.get("headline") or "").strip():
        raise ThumbnailRenderError("strategy has no non-empty headline to render")
    if strategy.get("text_position") not in VALID_POSITIONS:
        raise ThumbnailRenderError(
            f"strategy.text_position {strategy.get('text_position')!r} is not one of {VALID_POSITIONS} - "
            f"refusing to guess a pixel position"
        )


def _fit_or_fail(
    draw: "ImageDraw.ImageDraw", text: str, max_width: int, max_height: int,
    start_size_px: int, min_size_px: int, bold: bool = True,
) -> tuple:
    """Step 1 (wrap) is attempted at every candidate size below; step 2
    (shrink) is the loop; step 3 (fail) is the final raise if even
    min_size_px doesn't fit - see the module docstring. Returns
    (font, wrapped_lines, line_height_px)."""
    size = max(int(start_size_px), min_size_px)
    while True:
        font = resolve_font(size, bold=bold)
        lines = _wrap_to_width(draw, text, font, max_width)
        line_height = max(int(_text_size(draw, "Ag", font)[1] * LINE_SPACING), 1)
        block_height = line_height * len(lines)
        widest = max((_text_size(draw, ln, font)[0] for ln in lines), default=0)

        if block_height <= max_height and widest <= max_width:
            return font, lines, line_height

        if size <= min_size_px:
            raise ThumbnailRenderError(
                f"text {text!r} cannot fit within {max_width}x{max_height}px even at the minimum "
                f"readable font size ({min_size_px}px) after word-wrapping - refusing to silently "
                f"truncate the text or shrink it below a readable size"
            )
        size = max(size - FONT_SIZE_STEP, min_size_px)


def _position_anchor(position: str, block_w: int, block_h: int) -> tuple[int, int, str]:
    """The single place a semantic position name becomes pixel
    coordinates - see the module docstring's POSITION MAPPING section.
    Raises if `position` isn't one of the six fixed VALID_POSITIONS
    strings; there is no other way to influence text placement."""
    if position not in VALID_POSITIONS:
        raise ThumbnailRenderError(f"invalid text_position {position!r} - must be one of {VALID_POSITIONS}")

    vertical, horizontal = position.split("-")

    if horizontal == "left":
        x = MARGIN
        align = "left"
    else:
        x = WIDTH - MARGIN - block_w
        align = "right"

    if vertical == "top":
        y = MARGIN
    elif vertical == "bottom":
        y = HEIGHT - MARGIN - block_h
    else:  # center
        y = (HEIGHT - block_h) // 2

    return x, y, align


def _draw_text_block(
    draw: "ImageDraw.ImageDraw", lines: list[str], font, line_h: int,
    x: int, y: int, align: str, block_w: int, fill: tuple, stroke_fill: tuple,
) -> None:
    stroke_width = max(2, int(getattr(font, "size", 40) * STROKE_WIDTH_FRAC))
    cursor_y = y
    for line in lines:
        line_w = _text_size(draw, line, font)[0]
        line_x = x if align == "left" else x + block_w - line_w
        draw.text((line_x + SHADOW_OFFSET[0], cursor_y + SHADOW_OFFSET[1]), line, font=font, fill=SHADOW_COLOR)
        draw.text((line_x, cursor_y), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        cursor_y += line_h


def render_thumbnail(background_bytes: bytes, strategy: dict) -> ThumbnailRenderResult:
    """Pure, in-memory renderer - the primary entrypoint (render_thumbnail_bytes()
    and render_thumbnail_to_file() below are thin wrappers around this).
    Never mutates `strategy` or `background_bytes`. Deterministic: the
    same (background_bytes, strategy) always produces a byte-identical
    ThumbnailRenderResult.image_bytes."""
    _validate_strategy_for_render(strategy)

    try:
        base = Image.open(io.BytesIO(background_bytes))
        base.load()
    except Exception as e:
        raise ThumbnailRenderError(f"background image could not be decoded: {e}") from e
    base = base.convert("RGB")
    if base.size != (WIDTH, HEIGHT):
        raise ThumbnailRenderError(
            f"background is {base.size[0]}x{base.size[1]}, expected exactly {WIDTH}x{HEIGHT} - "
            f"this module trusts but verifies the upstream contract rather than re-normalizing it itself"
        )

    canvas = base.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    headline = strategy["headline"]
    secondary = strategy.get("secondary_text")
    position = strategy["text_position"]

    max_text_width = int(WIDTH * MAX_TEXT_WIDTH_FRAC)

    headline_font, headline_lines, headline_line_h = _fit_or_fail(
        draw, headline, max_text_width, int(HEIGHT * HEADLINE_MAX_HEIGHT_FRAC),
        int(HEIGHT * HEADLINE_START_SIZE_FRAC), int(HEIGHT * HEADLINE_MIN_SIZE_FRAC), bold=True,
    )
    headline_block_h = headline_line_h * len(headline_lines)
    headline_block_w = max((_text_size(draw, ln, headline_font)[0] for ln in headline_lines), default=0)

    secondary_font = secondary_lines = secondary_line_h = None
    secondary_block_h = secondary_block_w = 0
    if secondary:
        secondary_font, secondary_lines, secondary_line_h = _fit_or_fail(
            draw, secondary, max_text_width, int(HEIGHT * SECONDARY_MAX_HEIGHT_FRAC),
            int(HEIGHT * SECONDARY_START_SIZE_FRAC), int(HEIGHT * SECONDARY_MIN_SIZE_FRAC), bold=True,
        )
        secondary_block_h = secondary_line_h * len(secondary_lines)
        secondary_block_w = max((_text_size(draw, ln, secondary_font)[0] for ln in secondary_lines), default=0)

    total_block_h = headline_block_h + (BLOCK_GAP + secondary_block_h if secondary else 0)
    total_block_w = max(headline_block_w, secondary_block_w)

    x, y, align = _position_anchor(position, total_block_w, total_block_h)

    # Defense-in-depth: given max_text_width/max_height are always chosen
    # well within MARGIN-adjusted canvas bounds, this should be
    # unreachable in normal operation (see the module docstring) - but a
    # hard fail-loud check here means a future constant/logic change can
    # never silently ship clipped text.
    if x < 0 or y < 0 or x + total_block_w > WIDTH or y + total_block_h > HEIGHT:
        raise ThumbnailRenderError(
            f"computed text block ({total_block_w}x{total_block_h}) at position={position!r} "
            f"would clip the {WIDTH}x{HEIGHT} canvas - refusing to draw clipped text"
        )

    headline_box = (x, y, x + headline_block_w, y + headline_block_h)
    _draw_text_block(draw, headline_lines, headline_font, headline_line_h, x, y, align, total_block_w,
                      fill=HEADLINE_FILL, stroke_fill=HEADLINE_STROKE)

    secondary_box = None
    if secondary:
        sec_y = y + headline_block_h + BLOCK_GAP
        secondary_box = (x, sec_y, x + secondary_block_w, sec_y + secondary_block_h)
        _draw_text_block(draw, secondary_lines, secondary_font, secondary_line_h, x, sec_y, align, total_block_w,
                          fill=SECONDARY_FILL, stroke_fill=SECONDARY_STROKE)

    result_img = canvas.convert("RGB")
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")

    logger.info(
        "[THUMBNAIL_RENDERER] rendered headline=%r secondary=%r position=%s headline_font=%dpx",
        headline, secondary, position, getattr(headline_font, "size", -1),
    )

    return ThumbnailRenderResult(
        image_bytes=buf.getvalue(),
        text_position=position,
        headline_box=headline_box,
        headline_font_size=getattr(headline_font, "size", -1),
        secondary_box=secondary_box,
        secondary_font_size=getattr(secondary_font, "size", None) if secondary else None,
    )


def render_thumbnail_bytes(background_bytes: bytes, strategy: dict) -> bytes:
    """Convenience wrapper returning only the PNG bytes - mirrors
    overlay_renderer.render_scene_overlay_bytes()'s naming convention."""
    return render_thumbnail(background_bytes, strategy).image_bytes


def render_thumbnail_to_file(background_path: str | Path, strategy: dict, output_path: str | Path) -> Path:
    """File-based wrapper: reads `background_path` (read-only - never
    modified or deleted; raises a clear error if it doesn't exist rather
    than a bare FileNotFoundError traceback) and writes the rendered PNG
    atomically to `output_path` (parent directories created as needed)."""
    background_path = Path(background_path)
    output_path = Path(output_path)

    if not background_path.exists():
        raise ThumbnailRenderError(f"background image not found at {background_path}")

    result = render_thumbnail(background_path.read_bytes(), strategy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_bytes(result.image_bytes)
    tmp_path.replace(output_path)
    return output_path
