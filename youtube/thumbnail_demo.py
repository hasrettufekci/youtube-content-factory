"""
YOUTUBE_CONTENT_FACTORY - SAFE THUMBNAIL DEMO (Thumbnail System V1).

Wires the four thumbnail stages together against an EXISTING project's
already-generated artifacts:

    STRATEGIST (thumbnail_strategist.py)
        -> FAL BACKGROUND GENERATION (thumbnail_generator.py)
        -> DETERMINISTIC TEXT OVERLAY (thumbnail_renderer.py)
        -> QC (thumbnail_qc.py)

Reads ONLY metadata_en.json / script_en.json / research.json from an
existing storage/<year>/<month>/<slug>/ project (read-only - nothing under
storage/ is ever written, and the existing production thumbnail.png there
is never opened, let alone overwritten). Writes exclusively under a
brand-new output directory this module creates itself:

    debug_audit/thumbnail_v1/<slug>/
        strategy.json
        background.png
        thumbnail.png

COST/API SAFETY: exactly ONE image-provider generation call is made per
run (see thumbnail_generator.generate_background_bytes()) - never calls
Claude, OpenAI/TTS, or the research stage at all (this module imports none
of youtube.llm_client, youtube.researcher, youtube.tts - see
tests/test_thumbnail_demo.py's static check). `--fixture` (or
fixture=True) swaps in thumbnail_generator's fully offline
FixtureImageProvider instead of the real FAL provider - see
thumbnail_generator.py's module docstring - for safe/free demo runs and
all automated tests; the live default path (fixture=False) calls
youtube.image_gen.get_provider() exactly as production image generation
does, so running this for real requires a real FAL_KEY exactly like any
other stage.

Usage (against the real production project - e.g. on the Windows machine):

    python -m youtube.thumbnail_demo --slug the-wow-signal-6            # real FAL call
    python -m youtube.thumbnail_demo --slug the-wow-signal-6 --fixture  # offline, deterministic
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from core.config import settings
from core.logging_setup import get_logger
from youtube import thumbnail_generator, thumbnail_qc, thumbnail_renderer, thumbnail_strategist

logger = get_logger("thumbnail_demo")

OUTPUT_ROOT = Path("debug_audit") / "thumbnail_v1"


class ThumbnailDemoError(RuntimeError):
    """Raised for any precondition failure (no script.json) or a hard
    failure from any of the four stages (strategist/generator/renderer
    error, or a failed QC gate). Always carries a clear, human-readable
    reason."""


@dataclass
class ThumbnailPipelineResult:
    strategy: dict
    background_path: Path
    thumbnail_path: Path
    strategy_path: Path
    qc: thumbnail_qc.QCResult
    background_check: dict
    deterministic: bool


def _find_project_dir(slug: str) -> Path | None:
    """Read-only search under storage/<year>/<month>/<slug> - mirrors
    youtube/final_visual_render_demo.py's and
    youtube/final_video_assembly_demo.py's own lookup exactly (no DB
    dependency, never writes or modifies anything it finds)."""
    storage_root = settings.STORAGE_DIR
    if not storage_root.exists():
        return None
    matches = sorted(storage_root.glob(f"*/*/{slug}"))
    return matches[0] if matches else None


def _load_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_bytes_atomic(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _write_json_atomic(dest: Path, data: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def run_thumbnail_pipeline(
    project_dir: Path, out_dir: Path, fixture: bool = False, language: str = "en",
) -> ThumbnailPipelineResult:
    """Runs all four stages against `project_dir` (read-only) and writes
    strategy.json / background.png / thumbnail.png under `out_dir`. Raises
    ThumbnailDemoError on any precondition failure, stage failure, or a
    failed QC/determinism check - never returns a result for a thumbnail
    that didn't pass QC."""
    project_dir = Path(project_dir)
    out_dir = Path(out_dir)

    # Safety: this pass must never write inside the source project
    # directory (which is where the EXISTING production thumbnail.png
    # lives - see storage/project_store.py's ProjectPaths.thumbnail_png)
    # or anywhere under storage/ at all.
    if out_dir.resolve() == project_dir.resolve():
        raise ThumbnailDemoError("out_dir must not be the same as the source project directory")
    try:
        out_dir.resolve().relative_to(settings.STORAGE_DIR.resolve())
        raise ThumbnailDemoError(f"out_dir must not be inside {settings.STORAGE_DIR} - refusing to write to storage/")
    except ValueError:
        pass  # out_dir is NOT inside STORAGE_DIR - this is the expected/required case

    script = _load_json_or_none(project_dir / f"script_{language}.json")
    if not script:
        raise ThumbnailDemoError(
            f"no script_{language}.json found at {project_dir} - run script_generator.py first"
        )
    research = _load_json_or_none(project_dir / "research.json")
    metadata = _load_json_or_none(project_dir / f"metadata_{language}.json")

    try:
        strategy = thumbnail_strategist.build_thumbnail_strategy(script, research=research, metadata=metadata)
    except thumbnail_strategist.ThumbnailStrategistError as e:
        raise ThumbnailDemoError(f"strategist failed: {e}") from e

    provider_label = "FIXTURE (deterministic, offline, no network)" if fixture else f"LIVE ({settings.IMAGE_PROVIDER})"
    logger.info("[THUMBNAIL_DEMO] background provider: %s - exactly one generation call will be made", provider_label)
    try:
        background_bytes = thumbnail_generator.generate_background_bytes(strategy, fixture=fixture)
    except thumbnail_generator.ThumbnailBackgroundError as e:
        raise ThumbnailDemoError(f"background generation failed: {e}") from e

    background_check = thumbnail_qc.check_background(background_bytes)
    if background_check["possible_text_detected"]:
        logger.warning(
            "[THUMBNAIL_DEMO] background may contain readable text (advisory heuristic score=%.4f, "
            "threshold=%.4f) - not a hard failure; no OCR dependency is available in this project for "
            "a reliable check",
            background_check["text_likelihood_score"], thumbnail_qc.TEXT_LIKELIHOOD_WARNING_THRESHOLD,
        )

    try:
        render_result = thumbnail_renderer.render_thumbnail(background_bytes, strategy)
    except thumbnail_renderer.ThumbnailRenderError as e:
        raise ThumbnailDemoError(f"text rendering failed: {e}") from e

    strategy_path = out_dir / "strategy.json"
    background_path = out_dir / "background.png"
    thumbnail_path = out_dir / "thumbnail.png"
    _write_json_atomic(strategy_path, strategy)
    _write_bytes_atomic(background_path, background_bytes)
    _write_bytes_atomic(thumbnail_path, render_result.image_bytes)

    qc_result = thumbnail_qc.qc_thumbnail(render_result, strategy, output_path=thumbnail_path)
    deterministic = thumbnail_qc.verify_deterministic(background_bytes, strategy)

    if not qc_result.ok:
        raise ThumbnailDemoError(f"QC failed: {qc_result.errors}")
    if not deterministic:
        raise ThumbnailDemoError("renderer produced non-deterministic output across two identical calls")

    logger.info(
        "[THUMBNAIL_DEMO] OK headline=%r position=%s -> %s", strategy["headline"], strategy["text_position"], thumbnail_path,
    )
    return ThumbnailPipelineResult(
        strategy=strategy, background_path=background_path, thumbnail_path=thumbnail_path,
        strategy_path=strategy_path, qc=qc_result, background_check=background_check, deterministic=deterministic,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SAFE THUMBNAIL DEMO: run the Thumbnail System V1 pipeline (strategist -> FAL background -> "
                     "deterministic text overlay -> QC) against an existing project's script/research/metadata. "
                     "Read-only against storage/ - writes only under debug_audit/thumbnail_v1/<slug>/.",
    )
    parser.add_argument("--slug", help="Project slug under storage/<year>/<month>/<slug> (e.g. the-wow-signal-6)")
    parser.add_argument("--project-dir", help="Explicit source project directory (overrides --slug)")
    parser.add_argument("--out-dir", default=None,
                         help="Output directory (default: debug_audit/thumbnail_v1/<slug>) - never storage/")
    parser.add_argument("--fixture", action="store_true",
                         help="Use the deterministic offline FixtureImageProvider instead of calling FAL - "
                              "makes zero network calls, zero cost")
    parser.add_argument("--language", default="en")
    args = parser.parse_args(argv)

    if args.project_dir:
        project_dir = Path(args.project_dir)
        slug_name = args.slug or project_dir.name
    elif args.slug:
        project_dir = _find_project_dir(args.slug)
        if project_dir is None:
            print(f"FAILED: no project found for slug={args.slug!r} under {settings.STORAGE_DIR}", file=sys.stderr)
            return 1
        slug_name = args.slug
    else:
        print("FAILED: one of --slug or --project-dir is required", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT / slug_name

    logger.info("[THUMBNAIL_DEMO] project=%s -> %s (fixture=%s)", project_dir, out_dir, args.fixture)
    try:
        result = run_thumbnail_pipeline(project_dir, out_dir, fixture=args.fixture, language=args.language)
    except ThumbnailDemoError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print(
        f"OK - headline={result.strategy['headline']!r} secondary_text={result.strategy['secondary_text']!r} "
        f"position={result.strategy['text_position']} -> {result.thumbnail_path}"
    )
    print(f"strategy: {result.strategy_path}")
    print(f"background: {result.background_path} (text_likelihood={result.background_check['text_likelihood_score']})")
    print(f"QC: ok={result.qc.ok} checks={result.qc.checks}")
    print(f"deterministic: {result.deterministic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
