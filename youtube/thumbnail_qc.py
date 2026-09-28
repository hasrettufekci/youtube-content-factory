"""
Thumbnail QC (Thumbnail System V1 - stage 4 of 4:
    STRATEGIST -> FAL BACKGROUND GENERATION -> TEXT OVERLAY -> QC).

Validates a rendered thumbnail (and, as a separate advisory pass, the raw
FAL background it was built from) before it is treated as a usable
production artifact. Pillow only - no new dependency.

qc_thumbnail() is a HARD gate: any check in its `errors` list means the
thumbnail must not be used (mirrors youtube/quality_check.py's own
check()/hard-failure convention for the video pipeline). It deliberately
takes the renderer's own ThumbnailRenderResult (not just raw bytes) for
the "fully visible / not clipped" checks: youtube/thumbnail_renderer.py
already computes and reports the exact pixel box each text block occupies
(headline_box/secondary_box), and QC's job here is to independently verify
those reported boxes actually sit inside the canvas - a genuine second
opinion on the renderer's own geometry, not a re-implementation of its
wrapping/fitting math. (The renderer's own _fit_or_fail() already refuses
to produce a result at all if text can't fit - see
youtube/thumbnail_renderer.py - so by the time QC runs, a THUMBNAIL_BYTES
value existing at all already means fitting succeeded; QC's box check
catches a future regression in that geometry even if the renderer no
longer raises.)

check_background() is ADVISORY ONLY, not a hard gate - see its docstring
for why a reliable "is there text in this image" check needs OCR, which
this project does not (and this task explicitly says not to add) depend
on; it returns a best-effort heuristic score for the caller to log/decide
on, never a pass/fail QC error.

verify_deterministic() re-renders the SAME (background_bytes, strategy)
twice and confirms byte-identical output - the "deterministic output"
requirement, exercised directly rather than assumed.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter

from core.config import settings
from core.logging_setup import get_logger

logger = get_logger("thumbnail_qc")

WIDTH = settings.THUMBNAIL_WIDTH
HEIGHT = settings.THUMBNAIL_HEIGHT

MIN_FILE_SIZE_BYTES = 5_000          # guards against a near-empty/corrupt file
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024   # guards against something absurd for a 1280x720 PNG

# Best-effort, non-OCR "does this look like it has rendered text on it"
# score threshold - see check_background()'s docstring for exactly how
# unreliable this necessarily is without OCR (not a project dependency
# today, per this task's explicit constraint) and why it is advisory only.
TEXT_LIKELIHOOD_WARNING_THRESHOLD = 0.045


@dataclass
class QCResult:
    ok: bool
    checks: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _box_within_canvas(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 <= x2 <= WIDTH and 0 <= y1 <= y2 <= HEIGHT


def qc_thumbnail(render_result, strategy: dict, output_path: Path | str | None = None) -> QCResult:
    """`render_result` is a youtube.thumbnail_renderer.ThumbnailRenderResult
    (duck-typed here - only .image_bytes/.headline_box/.secondary_box are
    read - so this module has no import-time dependency on
    thumbnail_renderer, avoiding a cycle). `output_path`, when given, adds
    the "output file exists on disk and decodes" checks."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}

    image_bytes = render_result.image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        checks["png_decodes"] = True
    except Exception as e:
        return QCResult(ok=False, checks={"png_decodes": False}, warnings=[],
                         errors=[f"thumbnail PNG could not be decoded: {e}"])

    dims_ok = img.size == (WIDTH, HEIGHT)
    checks["dimensions_1280x720"] = dims_ok
    if not dims_ok:
        errors.append(f"thumbnail is {img.size[0]}x{img.size[1]}, expected exactly {WIDTH}x{HEIGHT}")

    headline = (strategy or {}).get("headline") or ""
    headline_exists = bool(str(headline).strip())
    checks["headline_exists"] = headline_exists
    if not headline_exists:
        errors.append("strategy has no headline")

    headline_visible = _box_within_canvas(render_result.headline_box)
    checks["headline_fully_visible"] = headline_visible
    if not headline_visible:
        errors.append(f"headline_box {render_result.headline_box} extends outside the {WIDTH}x{HEIGHT} canvas")

    if render_result.secondary_box is not None:
        secondary_visible = _box_within_canvas(render_result.secondary_box)
        checks["secondary_fully_visible"] = secondary_visible
        if not secondary_visible:
            errors.append(f"secondary_box {render_result.secondary_box} extends outside the {WIDTH}x{HEIGHT} canvas")
    else:
        checks["secondary_fully_visible"] = True  # nothing to check - not a failure

    size_bytes = len(image_bytes)
    size_ok = MIN_FILE_SIZE_BYTES <= size_bytes <= MAX_FILE_SIZE_BYTES
    checks["file_size_reasonable"] = size_ok
    checks["file_size_bytes"] = size_bytes
    if not size_ok:
        errors.append(
            f"thumbnail size {size_bytes} bytes is outside the reasonable range "
            f"[{MIN_FILE_SIZE_BYTES}, {MAX_FILE_SIZE_BYTES}]"
        )

    if output_path is not None:
        output_path = Path(output_path)
        exists = output_path.exists()
        checks["output_file_exists"] = exists
        if not exists:
            errors.append(f"output file does not exist at {output_path}")
        else:
            try:
                with Image.open(output_path) as on_disk:
                    on_disk.load()
                    checks["output_file_decodes"] = True
                    checks["output_file_matches_bytes"] = output_path.read_bytes() == image_bytes
                    if not checks["output_file_matches_bytes"]:
                        errors.append(f"on-disk file at {output_path} does not match the rendered bytes")
            except Exception as e:
                checks["output_file_decodes"] = False
                errors.append(f"on-disk file at {output_path} could not be decoded: {e}")

    return QCResult(ok=not errors, checks=checks, warnings=warnings, errors=errors)


def estimate_text_likelihood(image: "Image.Image") -> float:
    """Best-effort, non-OCR heuristic in [0, 1]: text tends to pack a high
    density of small, sharp, high-contrast edges into an image, unlike the
    smoother gradients of most photographic content. This is NOT a
    reliable text detector - pytesseract/OCR is not a dependency of this
    project (per this task's explicit "do not add OCR unless already
    present" constraint) - just a deterministic, comparative signal:
    see check_background()."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    strong_edge_pixels = sum(histogram[200:])  # near-white in the edge map = a strong edge
    return strong_edge_pixels / total


def check_background(background_bytes: bytes) -> dict:
    """Advisory-only pre-overlay check on the RAW FAL background (before
    any text is drawn on it) - never a hard QC failure (see the module
    docstring for why). Returns {"text_likelihood_score": float,
    "possible_text_detected": bool}; the caller (youtube/thumbnail_demo.py)
    logs a warning when possible_text_detected is True but does not fail
    the run on it alone."""
    img = Image.open(io.BytesIO(background_bytes)).convert("RGB")
    score = estimate_text_likelihood(img)
    return {
        "text_likelihood_score": round(score, 4),
        "possible_text_detected": score >= TEXT_LIKELIHOOD_WARNING_THRESHOLD,
    }


def verify_deterministic(background_bytes: bytes, strategy: dict) -> bool:
    """Renders the SAME (background_bytes, strategy) twice via
    youtube.thumbnail_renderer.render_thumbnail() and confirms
    byte-identical output - the "deterministic output" requirement,
    exercised directly. Local import avoids a module-level cycle with
    thumbnail_renderer (which does not import this module)."""
    from youtube.thumbnail_renderer import render_thumbnail

    first = render_thumbnail(background_bytes, strategy)
    second = render_thumbnail(background_bytes, strategy)
    is_deterministic = first.image_bytes == second.image_bytes
    if not is_deterministic:
        logger.error("[THUMBNAIL_QC] renderer produced non-deterministic output for identical inputs")
    return is_deterministic
