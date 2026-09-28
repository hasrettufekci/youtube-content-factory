"""
youtube/thumbnail_renderer.py tests.

Covers:
    5  deterministic position mapping (renderer side: full VALID_POSITIONS
       set, invalid position rejected)
    6  text fitting
    7  text never clipped
    8  1280x720 output
    9  malformed background
   10  missing background
   11  renderer determinism
plus headline-dominance (secondary text smaller than headline) and the
"never silently truncates - fails validation instead" fitting contract.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from youtube import thumbnail_renderer as tr

WIDTH, HEIGHT = tr.WIDTH, tr.HEIGHT


def _background_bytes(color=(20, 24, 32)) -> bytes:
    img = Image.new("RGB", (WIDTH, HEIGHT), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _strategy(headline="THE WOW! SIGNAL", secondary="AN ALIEN MESSAGE?", position="bottom-left") -> dict:
    return {"headline": headline, "secondary_text": secondary, "text_position": position}


# ---------------------------------------------------------------------------
# 8 - 1280x720 output
# ---------------------------------------------------------------------------

def test_output_is_exactly_1280x720():
    result = tr.render_thumbnail(_background_bytes(), _strategy())
    img = Image.open(io.BytesIO(result.image_bytes))
    assert img.size == (WIDTH, HEIGHT) == (1280, 720)


def test_output_is_valid_png():
    result = tr.render_thumbnail(_background_bytes(), _strategy())
    img = Image.open(io.BytesIO(result.image_bytes))
    img.load()  # raises if the PNG doesn't actually decode
    assert img.format == "PNG"


# ---------------------------------------------------------------------------
# 9 - malformed background
# ---------------------------------------------------------------------------

def test_malformed_background_raises():
    with pytest.raises(tr.ThumbnailRenderError, match="could not be decoded"):
        tr.render_thumbnail(b"not a real image at all", _strategy())


def test_wrong_sized_background_raises():
    img = Image.new("RGB", (800, 600), (10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with pytest.raises(tr.ThumbnailRenderError, match="expected exactly"):
        tr.render_thumbnail(buf.getvalue(), _strategy())


# ---------------------------------------------------------------------------
# 10 - missing background (file-based wrapper)
# ---------------------------------------------------------------------------

def test_missing_background_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(tr.ThumbnailRenderError, match="not found"):
        tr.render_thumbnail_to_file(missing, _strategy(), tmp_path / "out.png")


def test_render_thumbnail_to_file_writes_file(tmp_path):
    bg_path = tmp_path / "background.png"
    bg_path.write_bytes(_background_bytes())
    out_path = tmp_path / "thumbnail.png"

    result_path = tr.render_thumbnail_to_file(bg_path, _strategy(), out_path)

    assert result_path == out_path
    assert out_path.exists()
    img = Image.open(out_path)
    assert img.size == (WIDTH, HEIGHT)


# ---------------------------------------------------------------------------
# 5 - deterministic position mapping (full set + invalid rejection)
# ---------------------------------------------------------------------------

def test_all_six_valid_positions_produce_a_box_within_canvas():
    assert len(tr.VALID_POSITIONS) == 6
    for position in tr.VALID_POSITIONS:
        result = tr.render_thumbnail(_background_bytes(), _strategy(position=position))
        x1, y1, x2, y2 = result.headline_box
        assert 0 <= x1 <= x2 <= WIDTH
        assert 0 <= y1 <= y2 <= HEIGHT
        assert result.text_position == position


def test_left_positions_anchor_near_left_margin():
    result = tr.render_thumbnail(_background_bytes(), _strategy(position="bottom-left"))
    assert result.headline_box[0] == tr.MARGIN


def test_right_positions_anchor_near_right_margin():
    result = tr.render_thumbnail(_background_bytes(), _strategy(position="bottom-right"))
    assert result.headline_box[2] == WIDTH - tr.MARGIN


def test_top_positions_anchor_near_top_margin():
    result = tr.render_thumbnail(_background_bytes(), _strategy(position="top-left"))
    assert result.headline_box[1] == tr.MARGIN


def test_bottom_positions_anchor_near_bottom_margin():
    result = tr.render_thumbnail(_background_bytes(), _strategy(position="bottom-left"))
    total_h = result.headline_box[3] - result.headline_box[1]
    if result.secondary_box:
        total_h = result.secondary_box[3] - result.headline_box[1]
    assert result.headline_box[1] + total_h <= HEIGHT - tr.MARGIN + 1  # allow rounding


def test_invalid_text_position_is_rejected_never_an_arbitrary_pixel_coordinate():
    with pytest.raises(tr.ThumbnailRenderError, match="text_position"):
        tr.render_thumbnail(_background_bytes(), _strategy(position="dead-center"))


def test_no_position_api_accepts_raw_pixel_coordinates():
    """The public render_thumbnail() signature takes only (background_bytes,
    strategy) - there is no x/y parameter anywhere an LLM-supplied pixel
    coordinate could enter."""
    import inspect
    sig = inspect.signature(tr.render_thumbnail)
    assert list(sig.parameters) == ["background_bytes", "strategy"]


# ---------------------------------------------------------------------------
# 6 - text fitting (wrap, then shrink)
# ---------------------------------------------------------------------------

def test_long_headline_gets_wrapped_onto_multiple_lines():
    long_headline = "A VERY LONG HEADLINE THAT MUST WRAP"
    result = tr.render_thumbnail(_background_bytes(), _strategy(headline=long_headline, secondary=None))
    # multi-line means block height is a multiple of a single line's height,
    # detectable via a taller-than-one-line box for a headline this long
    box_h = result.headline_box[3] - result.headline_box[1]
    assert box_h > int(HEIGHT * tr.HEADLINE_START_SIZE_FRAC)  # taller than a single line at the start size


def test_headline_font_shrinks_when_needed_to_fit():
    short = tr.render_thumbnail(_background_bytes(), _strategy(headline="SHORT", secondary=None))
    longer = "A REASONABLY LONG DOCUMENTARY STYLE HEADLINE PHRASE"
    long_result = tr.render_thumbnail(_background_bytes(), _strategy(headline=longer, secondary=None))
    assert long_result.headline_font_size <= short.headline_font_size


# ---------------------------------------------------------------------------
# 7 - text never clipped; impossible fit fails validation, never truncates
# ---------------------------------------------------------------------------

def test_headline_box_never_exceeds_canvas_for_a_realistic_range_of_headlines():
    for headline in ["X", "TWO WORDS", "THE WOW! SIGNAL", "A REASONABLY LONG DOCUMENTARY HEADLINE HERE"]:
        result = tr.render_thumbnail(_background_bytes(), _strategy(headline=headline, secondary=None))
        x1, y1, x2, y2 = result.headline_box
        assert x1 >= 0 and y1 >= 0 and x2 <= WIDTH and y2 <= HEIGHT


def test_impossible_to_fit_headline_fails_validation_rather_than_truncating():
    """Many long words with no natural break that can't fit even at the
    minimum readable font size must raise, never silently drop/truncate
    words."""
    impossible = " ".join(["PNEUMONOULTRAMICROSCOPICSILICOVOLCANOCONIOSIS"] * 20)
    with pytest.raises(tr.ThumbnailRenderError, match="cannot fit"):
        tr.render_thumbnail(_background_bytes(), _strategy(headline=impossible, secondary=None))


def test_missing_headline_raises_before_any_drawing():
    with pytest.raises(tr.ThumbnailRenderError, match="headline"):
        tr.render_thumbnail(_background_bytes(), _strategy(headline="", secondary=None))


# ---------------------------------------------------------------------------
# 11 - renderer determinism
# ---------------------------------------------------------------------------

def test_renderer_is_byte_identical_across_repeated_calls():
    background = _background_bytes()
    strategy = _strategy()
    first = tr.render_thumbnail(background, strategy)
    second = tr.render_thumbnail(background, strategy)
    assert first.image_bytes == second.image_bytes
    assert first.headline_box == second.headline_box
    assert first.secondary_box == second.secondary_box


def test_renderer_bytes_wrapper_is_deterministic():
    background = _background_bytes()
    strategy = _strategy()
    assert tr.render_thumbnail_bytes(background, strategy) == tr.render_thumbnail_bytes(background, strategy)


# ---------------------------------------------------------------------------
# Headline dominance / secondary text handling
# ---------------------------------------------------------------------------

def test_secondary_text_is_smaller_than_headline():
    result = tr.render_thumbnail(_background_bytes(), _strategy())
    assert result.secondary_font_size is not None
    assert result.secondary_font_size < result.headline_font_size


def test_no_secondary_block_when_strategy_has_none():
    result = tr.render_thumbnail(_background_bytes(), _strategy(secondary=None))
    assert result.secondary_box is None
    assert result.secondary_font_size is None
