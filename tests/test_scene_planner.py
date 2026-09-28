import importlib

import pytest

from core.config import settings
from core.state import PipelineState
from storage import project_store as ps
from youtube import scene_planner as sp


def _fake_narration(word_count: int) -> str:
    sentences = []
    words_left = word_count
    i = 0
    while words_left > 0:
        n = min(12, words_left)
        sentences.append(" ".join(["word"] * n) + ".")
        words_left -= n
        i += 1
    return " ".join(sentences)


def _long_sentence_narration(word_count: int, words_per_sentence: int = 60) -> str:
    """Reproduces the Round 1 bug scenario: a script written in a small
    number of LONG sentences (documentary-essay style) instead of many short
    spoken-style ones, which used to make target scene counts unreachable."""
    sentences = []
    words_left = word_count
    while words_left > 0:
        n = min(words_per_sentence, words_left)
        # commas partway through so clause-splitting has something to work with
        words = ["word"] * n
        if n > 20:
            words[n // 3] += ","
            words[2 * n // 3] += ","
        sentences.append(" ".join(words) + ".")
        words_left -= n
    return " ".join(sentences)


def test_target_scene_count_is_clamped_to_25_40():
    assert sp._target_scene_count(100) == sp.MIN_SCENES
    assert sp._target_scene_count(100_000) == sp.MAX_SCENES
    mid = sp._target_scene_count(1400)
    assert sp.MIN_SCENES <= mid <= sp.MAX_SCENES


def test_split_into_chunks_respects_target_count_and_preserves_words():
    narration = _fake_narration(1400)
    target = sp._target_scene_count(1400)
    chunks = sp._split_into_chunks(narration, target)

    assert len(chunks) <= target  # greedy grouping may under-shoot slightly, never over
    original_words = narration.split()
    rebuilt_words = " ".join(chunks).split()
    assert original_words == rebuilt_words  # no words lost or reordered


def test_split_into_chunks_handles_short_narration():
    chunks = sp._split_into_chunks("Just one short sentence.", 30)
    assert chunks == ["Just one short sentence."]


def test_duration_hint_uses_words_per_second_rate():
    words = 30
    chunk = " ".join(["word"] * words) + "."
    expected = round(words / sp.WORDS_PER_SECOND, 1)
    assert expected == 12.0


def test_long_sentence_narration_falls_back_to_clause_splitting_to_reach_target():
    # Round 1 bug: 1251 words of long (~60-word) sentences produced only 20
    # scenes against a target of 28, because chunking could only cut at
    # sentence boundaries and there weren't enough sentences. Same shape of
    # input here (1260 words, ~60 words/sentence -> only ~21 sentences).
    narration = _long_sentence_narration(1260, words_per_sentence=60)
    sentence_count = len(sp._split_sentences(narration))
    target = sp._target_scene_count(1260)
    assert sentence_count < target  # confirms this input reproduces the shortfall

    chunks = sp._split_into_chunks(narration, target)
    assert len(chunks) > sentence_count  # clause-splitting produced more scenes than sentences alone could
    assert " ".join(chunks).split() == narration.split()  # still lossless


def test_split_into_chunks_never_exceeds_target_even_with_clause_splitting():
    narration = _long_sentence_narration(1260, words_per_sentence=60)
    target = sp._target_scene_count(1260)
    chunks = sp._split_into_chunks(narration, target)
    assert len(chunks) <= target


def _uneven_sentence_narration(word_counts: list[int]) -> str:
    """One sentence per entry in `word_counts` - lets a test control the
    exact sentence-length distribution, unlike `_fake_narration` (uniform
    ~12-word sentences) or `_long_sentence_narration` (uniform ~60-word
    sentences)."""
    return " ".join(" ".join(["word"] * n) + "." for n in word_counts)


def test_split_into_chunks_reaches_exact_target_with_uneven_sentence_lengths():
    # Real-world regression (found during the Round 2 full-scale E2E run):
    # 32 real sentences of wildly uneven length (a genuine 1193-word
    # narration) against a target of 27 chunks produced only 18 chunks, the
    # last one an 411-word outlier, because the greedy grouper's fit
    # heuristic ("does accumulating one more unit land closer to
    # target_words_per_chunk?") kept deferring a close exactly at the
    # tight feasibility boundary - where units remaining == chunks still
    # needed, so every remaining unit MUST become its own chunk from that
    # point on. One deferral there permanently breaks feasibility for
    # every unit after it, silently collapsing the rest of the narration
    # into the final chunk. This reproduces that shape (more sentences
    # than the target, uneven lengths, several short ones in a row) with
    # synthetic data instead of depending on a real script.
    word_counts = [
        28, 37, 3, 31, 30, 37, 7, 15, 64, 63, 51, 44, 8, 55, 22, 44, 34, 44,
        44, 17, 43, 61, 4, 54, 38, 42, 40, 51, 80, 34, 39, 29,
    ]
    narration = _uneven_sentence_narration(word_counts)
    sentence_count = len(sp._split_sentences(narration))
    assert sentence_count == len(word_counts) == 32

    target = 27
    chunks = sp._split_into_chunks(narration, target)

    # Enough sentence-level units existed (32 >= 27) to hit the target
    # exactly - the old bug undershot to 18.
    assert len(chunks) == target
    # Still lossless: every word survives, in order, none duplicated.
    assert " ".join(chunks).split() == narration.split()
    # No pathological outlier chunk: the real bug's final chunk (411 words)
    # was ~9x the target_words_per_chunk (~44). Every chunk here should be
    # within a sane multiple of the average.
    avg = sum(word_counts) / target
    assert all(len(c.split()) <= avg * 3 for c in chunks)


def test_split_into_chunks_hits_target_exactly_whenever_enough_units_exist():
    # General invariant, not just the one reproduced scenario above:
    # whenever at least `target_count` sentence-level units are available,
    # the greedy grouper must be able to reach target_count exactly (each
    # unit can always become its own chunk in the worst case) - it should
    # never silently undershoot.
    import random
    rng = random.Random(1234)
    for target in (25, 27, 30, 35):
        word_counts = [rng.randint(3, 90) for _ in range(target + rng.randint(1, 8))]
        narration = _uneven_sentence_narration(word_counts)
        chunks = sp._split_into_chunks(narration, target)
        assert len(chunks) == target
        assert " ".join(chunks).split() == narration.split()


def test_scene_bounds_are_read_from_config_not_hard_coded():
    assert sp.MIN_SCENES == settings.SCENE_MIN_COUNT
    assert sp.MAX_SCENES == settings.SCENE_MAX_COUNT
    assert sp.WORDS_PER_SCENE_TARGET == settings.SCENE_WORDS_PER_SCENE_TARGET


def test_scene_bounds_respect_env_override(monkeypatch):
    # Deliberately does NOT importlib.reload(core.config): that would swap
    # the module-level `settings` singleton for a brand-new object.
    # Every other module already imported at test-collection time (tts.py,
    # image_gen.py, uploader.py, ...) did `from core.config import
    # settings`, which copies today's object reference into their own
    # namespace - reloading core.config would silently orphan every one of
    # those references for the rest of the pytest session (while
    # core.config.require(), whose __globals__ IS core.config's module
    # dict, would keep following the newest object), causing
    # monkeypatch.setattr(settings, ...) in unrelated, later-running test
    # files to patch an object require() no longer reads from. Patching
    # attributes directly on the one shared singleton instead - reverted
    # automatically by monkeypatch - avoids that split entirely, and still
    # proves the thing this test cares about: scene_planner's bounds are
    # DERIVED from settings, not hard-coded.
    original_min = settings.SCENE_MIN_COUNT
    original_max = settings.SCENE_MAX_COUNT
    original_target = settings.SCENE_WORDS_PER_SCENE_TARGET

    monkeypatch.setattr(settings, "SCENE_MIN_COUNT", 10)
    monkeypatch.setattr(settings, "SCENE_MAX_COUNT", 15)
    monkeypatch.setattr(settings, "SCENE_WORDS_PER_SCENE_TARGET", 50)

    importlib.reload(sp)
    try:
        assert sp.MIN_SCENES == 10
        assert sp.MAX_SCENES == 15
        assert sp.WORDS_PER_SCENE_TARGET == 50
        assert sp._target_scene_count(100_000) == 15
    finally:
        # Restore the real values on the (never-replaced) settings object
        # *before* reloading scene_planner again, so the module-level
        # constants it captures here are correct for every test that runs
        # after this one - monkeypatch's own teardown fires after this
        # function returns, too late to trigger another reload itself.
        monkeypatch.setattr(settings, "SCENE_MIN_COUNT", original_min)
        monkeypatch.setattr(settings, "SCENE_MAX_COUNT", original_max)
        monkeypatch.setattr(settings, "SCENE_WORDS_PER_SCENE_TARGET", original_target)
        importlib.reload(sp)


def test_target_scene_count_consistent_with_typical_documentary_duration():
    # 1200-1600 words (spec's 8-10min script target) should land inside 25-40
    # with the default words-per-scene pacing, i.e. neither an absurdly low
    # scene count (static video) nor an excessive one (unnecessary image cost).
    for words in (1200, 1400, 1600):
        count = sp._target_scene_count(words)
        assert sp.MIN_SCENES <= count <= sp.MAX_SCENES


def test_run_and_save_records_failed_state_when_script_json_missing(isolated_storage):
    """Regression: this precondition check used to raise BEFORE
    db.start_run()/the try block, so it never went through
    save_failure() - the video was left stuck with no FAILED state
    recorded and no diagnostic trail in state.json."""
    paths, _ = ps.create_project("Missing Script JSON Topic")
    # script.json deliberately not written

    with pytest.raises(RuntimeError, match="no script.json found"):
        sp.run_and_save(paths)

    assert ps.load_state(paths) == PipelineState.FAILED
    data = ps.load_json(paths.state_json)
    assert data["failed_stage"] == "scene_plan"
