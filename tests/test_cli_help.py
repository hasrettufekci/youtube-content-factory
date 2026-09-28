"""
CLI wiring (spec Round 2 section 13): every entrypoint must print clean
--help output and exit 0 - critically, WITHOUT requiring any API key to be
set (core/config.py's lazy require() design is what makes this possible;
this test is what actually proves it holds, rather than just documenting
the intent in a docstring somewhere).

Run as real subprocesses (not just `import` + call main()) so this also
catches import-time crashes, not just argparse wiring bugs.
"""
import os
import subprocess
import sys

import pytest

import pipeline as pipeline_module

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (module_args, expected substring in --help output)
ENTRYPOINTS = [
    (["pipeline.py"], "Run the YOUTUBE_CONTENT_FACTORY pipeline"),
    (["-m", "youtube.researcher"], "Research a documentary topic"),
    (["-m", "youtube.script_generator"], "Generate a documentary script"),
    (["-m", "youtube.scene_planner"], "Plan scenes"),
    (["-m", "youtube.image_gen"], "Generate scene images"),
    (["-m", "youtube.tts"], "Generate narration audio"),
    (["-m", "youtube.assemble"], "Assemble the final video"),
    (["-m", "youtube.thumbnail"], "Generate the video thumbnail"),
    (["-m", "youtube.metadata"], "Build YouTube metadata"),
    (["-m", "youtube.quality_check"], "Run post-assembly quality checks"),
    (["-m", "youtube.uploader"], "Upload the final video to YouTube"),
    (["telegram_bot.py"], "Telegram approval bot"),
]


def _run_help(args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args, "--help"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("args,expected_substring", ENTRYPOINTS, ids=[a[-1] for a, _ in ENTRYPOINTS])
def test_entrypoint_help_exits_zero_without_any_api_keys_set(args, expected_substring):
    # Deliberately strip every API-key-shaped env var: --help must never
    # require a real key, per core/config.py's lazy require() design (keys
    # are only checked at the moment a provider call actually happens).
    env = {k: v for k, v in os.environ.items() if not any(
        s in k for s in ("API_KEY", "FAL_KEY", "TELEGRAM_BOT_TOKEN", "YOUTUBE_CLIENT", "YOUTUBE_REFRESH_TOKEN")
    )}
    result = _run_help(args, env)

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert expected_substring in result.stdout
    assert "Traceback" not in result.stderr


def test_pipeline_english_only_flag_is_recognized_by_argparse():
    """Direct regression test for the reported symptom: `pipeline.py: error:
    unrecognized arguments: --english-only`.

    Deliberately NOT a subprocess test: `python pipeline.py --topic ...
    --english-only` is a real, fully-formed invocation, so a subprocess
    running it doesn't stop at argument parsing - it sails past argparse
    and straight into produce_video(), i.e. the real pipeline (research,
    LLM, FAL, TTS, YouTube, Telegram...). That's exactly what happened
    when this was first written as a subprocess test: it "passed" in the
    sense of not raising an argparse error, but only because it hung
    running real production code until the test's own subprocess timeout.

    Instead this calls pipeline.build_arg_parser() - the same
    ArgumentParser main() builds and uses, extracted verbatim so it can be
    exercised on its own - directly, in-process. parser.parse_args(...)
    only parses; it never touches main()'s try/except execution block,
    so there is no pipeline stage, no API call, and no subprocess
    involved at all. This still pins down the original symptom precisely:
    if --english-only were ever removed or misspelled again, parse_args()
    would raise SystemExit(2) with "unrecognized arguments" on stderr,
    exactly like it did against the real CLI.
    """
    parser = pipeline_module.build_arg_parser()
    args = parser.parse_args(["--topic", "CLI Flag Regression Topic", "--english-only"])
    assert args.english_only is True


def test_no_help_output_ever_contains_api_key_looking_strings():
    """Belt-and-suspenders: even if a key happened to be set in the
    environment this test runs in, --help output must never echo it back -
    argparse renders defaults sometimes, so this guards against a future
    `default=settings.SOME_API_KEY` mistake making it into --help text."""
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = "sk-ant-definitely-a-secret-value-12345"
    env["FAL_KEY"] = "fal-definitely-a-secret-value-67890"
    for args, _ in ENTRYPOINTS:
        result = _run_help(args, env)
        assert "sk-ant-definitely-a-secret-value" not in result.stdout
        assert "fal-definitely-a-secret-value" not in result.stdout
