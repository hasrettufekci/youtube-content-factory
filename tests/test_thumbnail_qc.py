"""
youtube/thumbnail_qc.py tests.

Covers:
   12  QC success
   13  QC failure
plus the background text-likelihood advisory heuristic and
verify_deterministic().
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from youtube import thumbnail_qc as qc
from youtube import thumbnail_renderer as tr


def _background_bytes(color=(20, 24, 32), width=None, height=None) -> bytes:
    width = width or tr.WIDTH
    height = height or tr.HEIGHT
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _strategy(headline="THE WOW! SIGNAL", secondary="AN ALIEN MESSAGE?", position="bottom-left") -> dict:
    return {"headline": headline, "secondary_text": secondary, "text_position": position}


# ---------------------------------------------------------------------------
# 12 - QC success
# ---------------------------------------------------------------------------

def test_qc_success_all_checks_pass():
    strategy = _strategy()
    render_result = tr.render_thumbnail(_background_bytes(), strategy)

    result = qc.qc_thumbnail(render_result, strategy)

    assert result.ok is True
    assert result.errors == []
    assert result.checks["png_decodes"] is True
    assert result.checks["dimensions_1280x720"] is True
    assert result.checks["headline_exists"] is True
    assert result.checks["headline_fully_visible"] is True
    assert result.checks["secondary_fully_visible"] is True
    assert result.checks["file_size_reasonable"] is True


def test_qc_success_with_output_file_checks(tmp_path):
    strategy = _strategy()
    render_result = tr.render_thumbnail(_background_bytes(), strategy)
    out_path = tmp_path / "thumbnail.png"
    out_path.write_bytes(render_result.image_bytes)

    result = qc.qc_thumbnail(render_result, strategy, output_path=out_path)

    assert result.ok is True
    assert result.checks["output_file_exists"] is True
    assert result.checks["output_file_decodes"] is True
    assert result.checks["output_file_matches_bytes"] is True


def test_qc_success_without_secondary_text():
    strategy = _strategy(secondary=None)
    render_result = tr.render_thumbnail(_background_bytes(), strategy)

    result = qc.qc_thumbnail(render_result, strategy)

    assert result.ok is True
    assert result.checks["secondary_fully_visible"] is True  # nothing to check - not a failure


# ---------------------------------------------------------------------------
# 13 - QC failure
# ---------------------------------------------------------------------------

def test_qc_failure_wrong_dimensions():
    class _FakeResult:
        image_bytes = _background_bytes(width=800, height=600)
        headline_box = (0, 0, 10, 10)
        secondary_box = None

    result = qc.qc_thumbnail(_FakeResult(), _strategy())

    assert result.ok is False
    assert result.checks["dimensions_1280x720"] is False
    assert any("800x600" in e for e in result.errors)


def test_qc_failure_missing_headline():
    strategy_without_headline = _strategy(headline="", secondary=None)
    # bypass the renderer (which itself would refuse an empty headline) to
    # exercise QC's own independent headline_exists check directly
    class _FakeResult:
        image_bytes = _background_bytes()
        headline_box = (0, 0, 10, 10)
        secondary_box = None

    result = qc.qc_thumbnail(_FakeResult(), strategy_without_headline)

    assert result.ok is False
    assert result.checks["headline_exists"] is False


def test_qc_failure_box_outside_canvas():
    class _FakeResult:
        image_bytes = _background_bytes()
        headline_box = (0, 0, tr.WIDTH + 50, 100)  # x2 beyond the canvas
        secondary_box = None

    result = qc.qc_thumbnail(_FakeResult(), _strategy())

    assert result.ok is False
    assert result.checks["headline_fully_visible"] is False


def test_qc_failure_secondary_box_outside_canvas():
    class _FakeResult:
        image_bytes = _background_bytes()
        headline_box = (0, 0, 100, 100)
        secondary_box = (0, tr.HEIGHT - 5, 100, tr.HEIGHT + 50)  # y2 beyond the canvas

    result = qc.qc_thumbnail(_FakeResult(), _strategy())

    assert result.ok is False
    assert result.checks["secondary_fully_visible"] is False


def test_qc_failure_undecodable_bytes():
    class _FakeResult:
        image_bytes = b"not a real png"
        headline_box = (0, 0, 10, 10)
        secondary_box = None

    result = qc.qc_thumbnail(_FakeResult(), _strategy())

    assert result.ok is False
    assert result.checks["png_decodes"] is False


def test_qc_failure_output_file_missing(tmp_path):
    strategy = _strategy()
    render_result = tr.render_thumbnail(_background_bytes(), strategy)
    missing_path = tmp_path / "never_written.png"

    result = qc.qc_thumbnail(render_result, strategy, output_path=missing_path)

    assert result.ok is False
    assert result.checks["output_file_exists"] is False


def test_qc_failure_file_size_too_small():
    # a genuinely valid, decodable PNG - just a 1x1 pixel, so its file size
    # is trivially under MIN_FILE_SIZE_BYTES without conflating this check
    # with the separate png_decodes check.
    tiny_buf = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(tiny_buf, format="PNG")

    class _FakeResult:
        image_bytes = tiny_buf.getvalue()
        headline_box = (0, 0, 10, 10)
        secondary_box = None

    result = qc.qc_thumbnail(_FakeResult(), _strategy())

    assert result.checks["png_decodes"] is True
    assert result.checks["file_size_reasonable"] is False
    assert result.ok is False


# ---------------------------------------------------------------------------
# Background text-likelihood advisory check
# ---------------------------------------------------------------------------

def test_plain_background_scores_low_text_likelihood():
    plain = _background_bytes()
    check = qc.check_background(plain)
    assert check["possible_text_detected"] is False


def test_text_heavy_background_scores_higher_than_plain():
    plain_img = Image.new("RGB", (tr.WIDTH, tr.HEIGHT), (30, 30, 30))
    text_img = plain_img.copy()
    draw = ImageDraw.Draw(text_img)
    font = ImageFont.load_default()
    for y in range(0, tr.HEIGHT, 18):
        draw.text((10, y), "SAMPLE TEXT WORDS HERE " * 6, font=font, fill=(255, 255, 255))

    plain_buf, text_buf = io.BytesIO(), io.BytesIO()
    plain_img.save(plain_buf, format="PNG")
    text_img.save(text_buf, format="PNG")

    plain_score = qc.check_background(plain_buf.getvalue())["text_likelihood_score"]
    text_score = qc.check_background(text_buf.getvalue())["text_likelihood_score"]
    assert text_score > plain_score


# ---------------------------------------------------------------------------
# verify_deterministic()
# ---------------------------------------------------------------------------

def test_verify_deterministic_true_for_identical_inputs():
    background = _background_bytes()
    strategy = _strategy()
    assert qc.verify_deterministic(background, strategy) is True
