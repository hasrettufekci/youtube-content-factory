"""
YouTube metadata (spec section 22; extended for multilingual production in
Round 3 section 15): title, description (with research sources listed,
when available), tags, category, language.

Deliberately deterministic/templated (no extra LLM call) - the script
already has a good (and, for a localization, already professionally
localized - see youtube/localizer.py) title, and the research already has
the sources; this just assembles them into what the YouTube Data API's
`videos.insert` expects (see uploader.py), which keeps this stage free,
fast, and simple to unit test.

Multilingual: `run_and_save(paths, language=...)` reads that language's own
script (script_en.json for the master, script_<lang>.json for a
localization - both share the same title/hook/sections/full_narration
shape, see youtube/localizer.py) and writes metadata_<lang>.json. Research
sources (URLs/titles) are carried over as-is for every language - a cited
source isn't "translated", the same documents/records back every language's
video. Tag keywords and the fixed boilerplate sentences (disclaimer,
subscribe CTA) ARE localized per language via the small tables below - the
Turkish metadata describes the exact same story as the English metadata,
just properly worded in Turkish, not machine-translated at call time.

Standalone test:
    python -m youtube.metadata --video-id 1
"""
from __future__ import annotations

import argparse
import sys

from core.config import settings
from core.logging_setup import get_logger
from storage import db, project_store as ps

logger = get_logger("metadata")

MAX_TAGS = 15

# Per-language stopwords (for filtering the topic string down to keyword
# tags) and a short fixed set of documentary-genre tags. Adding a future
# localization language: add one entry to each dict below - build_metadata()
# and _keywords_from_topic() need no other changes. A language with no
# entry here falls back to the English tables (still produces valid, if
# English-flavored, tags rather than crashing).
_STOPWORDS_BY_LANGUAGE: dict[str, set[str]] = {
    "en": {"the", "a", "an", "of", "in", "on", "at", "to", "is", "are", "who", "where", "no", "nobody"},
    "tr": {"bir", "bu", "şu", "ve", "ile", "de", "da", "ki", "mi", "mı", "mu", "mü", "en", "çok", "ne", "neden", "nasıl"},
}
_TAG_SUFFIX_BY_LANGUAGE: dict[str, list[str]] = {
    "en": ["documentary", "mystery", "true story", "explained"],
    "tr": ["belgesel", "gizem", "gerçek hikaye", "açıklandı"],
}

# Fixed boilerplate sentences, per language - naturally worded, not
# machine-translated at call time (see module docstring).
_BOILERPLATE_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "en": {
        "sources_heading": "Sources / further reading:",
        "disclaimer": (
            "This video is a factual documentary production. Every claim is based on the "
            "research sources above; anything genuinely unverified is presented as such in "
            "the video itself."
        ),
        "subscribe_cta": (
            "Subscribe for more documentaries on mysteries, history, science, and the "
            "extraordinary stories that don't get told often enough."
        ),
    },
    "tr": {
        "sources_heading": "Kaynaklar / daha fazla okuma:",
        "disclaimer": (
            "Bu video, gerçek olaylara dayanan bir belgesel yapımıdır. Her iddia yukarıdaki "
            "araştırma kaynaklarına dayanmaktadır; doğrulanamayan noktalar videoda da "
            "belirsiz/tartışmalı olarak sunulmuştur."
        ),
        "subscribe_cta": (
            "Gizemler, tarih, bilim ve yeterince anlatılmamış olağanüstü hikayeler hakkında "
            "daha fazla belgesel için abone olmayı unutmayın."
        ),
    },
}


def _keywords_from_topic(topic: str, category: str | None, language: str = "en") -> list[str]:
    stopwords = _STOPWORDS_BY_LANGUAGE.get(language, _STOPWORDS_BY_LANGUAGE["en"])
    suffix = _TAG_SUFFIX_BY_LANGUAGE.get(language, _TAG_SUFFIX_BY_LANGUAGE["en"])
    words = [w.strip(".,!?\"'").lower() for w in topic.split()]
    keywords = [w for w in words if w and w not in stopwords and len(w) > 2]
    tags = list(dict.fromkeys(keywords))  # dedupe, keep order
    if category:
        tags.insert(0, category.lower())
    tags += suffix
    seen = set()
    unique = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:MAX_TAGS]


def build_metadata(script: dict, research: dict, category: str | None, language: str = "en") -> dict:
    title = script["title"].strip()
    if len(title) > 100:
        title = title[:97] + "..."

    boilerplate = _BOILERPLATE_BY_LANGUAGE.get(language, _BOILERPLATE_BY_LANGUAGE["en"])
    sources = research.get("sources", [])
    source_lines = "\n".join(
        f"- {s.get('title') or s.get('url')}: {s.get('url')}" for s in sources if s.get("url")
    )

    description_parts = [
        script.get("hook", "").strip(),
        "",
        (script["sections"][1]["narration"] if len(script.get("sections", [])) > 1 else "").strip(),
        "",
    ]
    if source_lines:
        description_parts += [boilerplate["sources_heading"], source_lines, ""]
    description_parts += [
        boilerplate["disclaimer"],
        "",
        boilerplate["subscribe_cta"],
    ]
    description = "\n".join(description_parts).strip()
    if len(description) > 4900:
        description = description[:4890] + "..."

    # research.json is shared/language-independent (spec Round 3 section
    # 10) and never carries a localized "topic" field, so the master
    # language keeps deriving tags from research["topic"] exactly as
    # before; a localization derives them from its own (already-localized)
    # title instead, since that's the only Turkish-language text available
    # to extract Turkish keyword tags from.
    topic_source = research.get("topic", title) if language == settings.LANGUAGE_MASTER else title

    return {
        "title": title,
        "description": description,
        "tags": _keywords_from_topic(topic_source, category, language=language),
        "category_id": settings.YOUTUBE_CATEGORY_ID,
        "language": language,
        "privacy_status": settings.YOUTUBE_UPLOAD_PRIVACY,
        "source_count": len(sources),
    }


def run_and_save(paths: ps.ProjectPaths, category: str | None = None, language: str | None = None) -> dict:
    # Unlike every other stage's run_and_save(), this one previously had no
    # db.start_run()/db.finish_run() run tracking and no save_failure() on
    # error at all - a bug in build_metadata() (or a missing upstream
    # script.json/research.json) would raise straight out with no FAILED
    # state ever recorded and no run history, unlike identical failures in
    # every neighboring stage. Brought in line with the rest of the
    # pipeline here.
    language = language or settings.LANGUAGE_MASTER
    is_master = language == settings.LANGUAGE_MASTER
    stage_label = "metadata" if is_master else f"metadata_{language}"
    run_id = db.start_run(paths.video_id, stage_label)
    try:
        script_path = paths.script_en_json if is_master else paths.script_localized_json(language)
        script = ps.load_json(script_path)
        research = ps.load_json(paths.research_json)
        if not script or not research:
            raise RuntimeError("[METADATA] script.json and research.json are both required")

        metadata = build_metadata(script, research, category=category, language=language)
        dest = paths.metadata_en_json if is_master else paths.metadata_json_for(language)
        ps.save_json(dest, metadata)
        logger.info("[METADATA] built metadata: language=%s title=%r tags=%d sources=%d",
                    language, metadata["title"], len(metadata["tags"]), metadata["source_count"])
        db.finish_run(run_id, "success")
        return metadata
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, stage_label, str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build YouTube metadata (standalone test).")
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--language", default="en")
    args = parser.parse_args(argv)

    paths = ps.get_project(args.video_id)
    if not paths:
        print(f"FAILED: no project with video_id={args.video_id}", file=sys.stderr)
        return 1

    try:
        result = run_and_save(paths, category=args.category, language=args.language)
        print(f"OK - metadata saved to {paths.metadata_json}")
        print(result["title"])
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
