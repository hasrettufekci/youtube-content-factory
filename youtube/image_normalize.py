"""
Single source of truth for bringing an existing scene image to the
project's video frame size (settings.VIDEO_WIDTH x settings.VIDEO_HEIGHT,
1920x1080) before it is used as a base for overlay rendering or written
out as a final scene image.

Why this exists: the real production project's already-generated scene
images (storage/2026/09/the-wow-signal-6/scene_NNN.png) turned out to be
1920x1072, 8 pixels short of the 1920x1080 the assembler/final-render
pass requires. Stretching a 1920x1072 image to 1920x1080 would visibly
distort it (a ~0.7% vertical squeeze) for the sake of 8 pixels, and
cropping would discard real image content, so neither is acceptable.
Padding is the only option that satisfies "never distort, never crop
meaningful content" - this module is the ONE place that padding/
letterboxing logic lives, so it is never duplicated (and never drifts out
of sync) across youtube/overlay_renderer.py,
youtube/final_visual_render_demo.py, or youtube/image_gen.py.

Two supported shapes, both deterministic (no randomness, no wall-clock
time - identical input always produces identical output):

  1) Exact match on one axis, short on the other (the real-world case:
     1920x1072 -> 1920x1080). The image is kept at its NATIVE resolution
     - never resized/scaled - and padding is added on the short axis
     only, split as evenly as possible (floor on one side, the remainder
     on the other - for an 8px difference this is exactly 4px top + 4px
     bottom, the policy this task specifically asked for; the same floor/
     ceil rule generalizes correctly to any other even OR odd difference
     without a special case). The padding itself is not a flat/artificial
     color bar - it is the image's own edge row (or column), stretched to
     fill the pad area, so the frame reads as a natural continuation of
     the picture rather than a visible synthetic border - see
     _pad_axis()'s docstring.

  2) Any other size (a genuinely different aspect ratio, or an image
     larger than the target on the matching axis): scaled down (never up
     - never introduces blur/artifacts by upscaling) preserving aspect
     ratio to fit fully within the target box (never cropped), then the
     one remaining short axis (scale-to-fit only ever leaves a single
     axis short, never both - see _scale_to_fit() docstring) is padded
     with the same edge-extension technique as case 1.

An image that already IS exactly the target size is returned completely
unchanged - normalize_image_bytes_to_video_size() returns the exact same
bytes object (a true no-op, not just pixel-identical output), mirroring
the same "true no-op when there is nothing to do" convention
overlay_renderer.render_scene_overlay_bytes() already uses for
overlay_required=False.

This module never touches disk - normalize_image_to_video_size() takes
and returns a Pillow Image; normalize_image_bytes_to_video_size() takes
and returns bytes. Neither function has any notion of storage/ or any
other path - callers decide what to read and where to write, so this
module cannot modify production files even by accident.
"""
from __future__ import annotations

from PIL import Image

from core.config import settings

# Reuses the project's EXISTING video-frame configuration (settings.py) -
# no new configuration is introduced. These are simple aliases for
# readability at call sites in this module and its callers.
TARGET_WIDTH: int = settings.VIDEO_WIDTH
TARGET_HEIGHT: int = settings.VIDEO_HEIGHT


class ImageNormalizationError(RuntimeError):
    """Raised when the input cannot be normalized at all - e.g. the bytes
    given to normalize_image_bytes_to_video_size() are not a decodable
    image. Never raised for a merely differently-sized (but valid) image -
    that is the normal case this module exists to handle."""


def _pad_axis(image: Image.Image, axis: int, target: int) -> Image.Image:
    """Pads `image` along `axis` (0 = width/horizontal, 1 = height/
    vertical) up to `target`, split as evenly as possible: the smaller
    half goes first (top/left), the remainder goes last (bottom/right) -
    for an 8px difference this is exactly 4px + 4px; for an odd
    difference (e.g. 7px) it is 3px + 4px, deterministically, every time.

    The padding is NOT a flat/artificial color bar: each pad region is
    filled by taking the image's own single edge row (axis=1) or column
    (axis=0) and stretching it to fill the pad area (Image.resize() on a
    1px-thick strip has nothing to interpolate between, so every filter
    degenerates to pure replication of that one row/column - deterministic
    regardless of Pillow's resize implementation details). This reads as
    a natural continuation of the image's own edge/background rather than
    introducing a visible synthetic frame, per this task's explicit
    preference."""
    width, height = image.size
    current = width if axis == 0 else height
    diff = target - current
    if diff <= 0:
        return image

    pad_before = diff // 2
    pad_after = diff - pad_before

    if axis == 0:
        canvas = Image.new(image.mode, (target, height))
        canvas.paste(image, (pad_before, 0))
        if pad_before > 0:
            left_col = image.crop((0, 0, 1, height)).resize((pad_before, height))
            canvas.paste(left_col, (0, 0))
        if pad_after > 0:
            right_col = image.crop((width - 1, 0, width, height)).resize((pad_after, height))
            canvas.paste(right_col, (pad_before + width, 0))
    else:
        canvas = Image.new(image.mode, (width, target))
        canvas.paste(image, (0, pad_before))
        if pad_before > 0:
            top_row = image.crop((0, 0, width, 1)).resize((width, pad_before))
            canvas.paste(top_row, (0, 0))
        if pad_after > 0:
            bottom_row = image.crop((0, height - 1, width, height)).resize((width, pad_after))
            canvas.paste(bottom_row, (0, pad_before + height))

    return canvas


def _scale_to_fit(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Scales `image` down (never up) preserving aspect ratio so it fits
    entirely within (target_width, target_height) - never distorted
    (uniform scale factor on both axes) and never cropped (the whole
    image survives at the new, smaller size). Because the scale factor is
    min(target_width/width, target_height/height), one axis always comes
    out exactly equal to its target (zero padding needed there) and the
    other comes out <= its target (which _pad_axis() then pads) - scale-
    to-fit never leaves BOTH axes short, so the two _pad_axis() calls in
    normalize_image_to_video_size() never both do real work on the same
    image. If the image is already <= the target box on both axes
    (scale >= 1), it is left at its native resolution instead - this
    function only ever shrinks, never enlarges/upscales (upscaling would
    introduce blur/artifacts for no benefit)."""
    width, height = image.size
    scale = min(target_width / width, target_height / height, 1.0)
    if scale >= 1.0:
        return image
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return image.resize((new_width, new_height), Image.LANCZOS)


def normalize_image_to_video_size(
    image: Image.Image, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT,
) -> Image.Image:
    """Returns `image` resized/padded (never distorted, never cropped) to
    exactly (target_width, target_height). Deterministic: the same input
    image and target size always produce a pixel-identical result.

    Already the target size -> returned completely unchanged (the same
    object, not a copy) - a true no-op, mirroring
    overlay_renderer.render_scene_overlay_bytes()'s own no-op convention
    for overlay_required=False.

    Width already matches target (height differs) -> vertical padding
    only, image kept at native resolution (see _pad_axis()) - this is the
    real production shape (1920x1072 -> 1920x1080).
    Height already matches target (width differs) -> horizontal padding
    only, same native-resolution treatment, for the mirror case.
    Neither axis matches (or the image is larger than the target on the
    matching axis) -> scaled down preserving aspect ratio to fit inside
    the target box (see _scale_to_fit()), then the one remaining short
    axis is padded the same way."""
    width, height = image.size
    if (width, height) == (target_width, target_height):
        return image

    if width == target_width and height < target_height:
        return _pad_axis(image, axis=1, target=target_height)
    if height == target_height and width < target_width:
        return _pad_axis(image, axis=0, target=target_width)

    fitted = _scale_to_fit(image, target_width, target_height)
    fitted = _pad_axis(fitted, axis=0, target=target_width)
    fitted = _pad_axis(fitted, axis=1, target=target_height)
    return fitted


def normalize_image_bytes_to_video_size(
    image_bytes: bytes, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT,
) -> bytes:
    """Bytes-in/bytes-out wrapper around normalize_image_to_video_size().
    Returns the EXACT SAME bytes object, unchanged, when the decoded
    image is already (target_width, target_height) - a true byte-level
    no-op, not a re-encoded-but-pixel-identical copy (re-encoding can
    change file size/compression even for identical pixels, which would
    make an already-correct source image fail a byte-for-byte "unchanged"
    expectation for no reason). Raises ImageNormalizationError if
    `image_bytes` cannot be decoded as an image at all - never silently
    returns something wrong."""
    import io

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            original_size = img.size
            if original_size == (target_width, target_height):
                return image_bytes
            normalized = normalize_image_to_video_size(img, target_width, target_height)
    except ImageNormalizationError:
        raise
    except Exception as e:
        raise ImageNormalizationError(f"could not decode image bytes for normalization: {e}") from e

    buf = io.BytesIO()
    normalized.save(buf, format="PNG")
    return buf.getvalue()
