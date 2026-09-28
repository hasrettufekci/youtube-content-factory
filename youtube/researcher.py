"""
Research stage (spec section 10).

Produces a structured research result for a topic:

    {
        "topic": "...",
        "summary": "...",
        "facts": ["..."],
        "timeline": [{"date": "...", "event": "..."}],
        "sources": [{"title","url","publisher","published_at","relevance"}],
        "uncertainties": ["..."],
        "search_notes": ["..."]
    }

Uses Claude with the server-side web_search tool so facts are grounded in
real, live sources rather than the model's own unverified recollection.
Every source Claude cites is merged into "sources" (deduplicated by URL),
and the system prompt explicitly forbids inventing sources, quotes, dates
or events - if something can't be verified it must go in "uncertainties"
instead of being stated as fact (spec section 7: no fabricated info).

Standalone test:
    python -m youtube.researcher --topic "The Dyatlov Pass Incident"
"""
from __future__ import annotations

import argparse
import sys

from core.config import settings
from core.logging_setup import get_logger
from core.state import PipelineState
from storage import db, project_store as ps
from youtube import llm_client

logger = get_logger("research")

SYSTEM_PROMPT = """You are the lead researcher for a factual YouTube documentary channel
covering mysteries, unexplained events, history, science, and extraordinary human stories.

Your job: research the given topic using web search and return ONLY a single valid JSON
object (no markdown, no code fences, no commentary before or after it) with this exact
shape:

{
  "topic": "concise version of the topic",
  "summary": "2-4 paragraph neutral overview of what is reliably known",
  "facts": ["short, verifiable factual statement", "..."],
  "timeline": [{"date": "YYYY or YYYY-MM-DD or approximate", "event": "what happened"}],
  "sources": [{"title": "...", "url": "...", "publisher": "...", "published_at": "...", "relevance": "why this source matters"}],
  "uncertainties": ["things that are disputed, unverified, or genuinely unknown"],
  "search_notes": ["short notes on what you searched for and what you found or could not find"]
}

CRITICAL RULES:
- Never invent a source, quote, date, statistic, or event. If you cannot verify something
  with a real source you found, put it in "uncertainties" instead of "facts".
- Every non-trivial factual claim in "facts" or "timeline" should be something you could
  point to at least one source for.
- Prefer primary sources, established news organizations, academic/scientific publications,
  museums, government/archival records over blogs, forums, or unsourced listicles.
- If the topic as given is factually overstated or misleading (e.g. an absolute claim that
  isn't actually true), note that plainly in "uncertainties" and reflect the accurate
  picture in "summary" - do not silently go along with a false premise.
- "sources" must list the real URLs you actually found via search, not placeholders.
- Output ONLY the JSON object. No prose outside it."""


def _chat_json_research(user_prompt: str) -> tuple[dict, list[dict]]:
    """Calls llm_client.chat_json with retry-on-truncation/retry-on-bad-JSON
    (production incident: a real web-searched topic's full research payload
    got cut off by Claude's own max_tokens before the JSON finished, which
    json.loads then reported as a generic parse error). Each attempt is a
    completely fresh request - we never try to salvage or continue from a
    prior response's partial text, per spec: incomplete research data must
    never be silently accepted. Only raises (never returns a dict built
    from partial content) once every attempt has failed."""
    attempts = max(1, settings.CLAUDE_RESEARCH_MAX_ATTEMPTS)
    last_error: llm_client.ClaudeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return llm_client.chat_json(
                SYSTEM_PROMPT,
                user_prompt,
                max_tokens=settings.CLAUDE_RESEARCH_MAX_TOKENS,
                use_web_search=True,
            )
        except llm_client.ClaudeTruncatedResponseError as e:
            last_error = e
            logger.warning(
                "[RESEARCH] attempt %d/%d: Claude response truncated (max_tokens=%d) - %s",
                attempt, attempts, settings.CLAUDE_RESEARCH_MAX_TOKENS,
                "retrying with a fresh request" if attempt < attempts else "giving up",
            )
        except llm_client.ClaudeError as e:
            last_error = e
            logger.warning(
                "[RESEARCH] attempt %d/%d: Claude returned unparseable JSON (%s) - %s",
                attempt, attempts, e,
                "retrying with a fresh request" if attempt < attempts else "giving up",
            )
    raise llm_client.ClaudeError(
        f"[RESEARCH] gave up after {attempts} attempt(s), all truncated or unparseable: {last_error}"
    ) from last_error


def research_topic(topic: str, category: str | None = None) -> dict:
    logger.info("[RESEARCH] starting topic=%r category=%r", topic, category)

    user_prompt = f"Research this documentary topic:\n\n\"{topic}\""
    if category:
        user_prompt += f"\n\nContent category: {category}"

    data, cited_sources = _chat_json_research(user_prompt)

    for key in ("topic", "summary", "facts", "sources"):
        if key not in data:
            raise RuntimeError(f"[RESEARCH] Claude output missing required field: {key}")

    data.setdefault("timeline", [])
    data.setdefault("uncertainties", [])
    data.setdefault("search_notes", [])

    # Merge citations the API actually returned with whatever the model
    # self-reported, de-duplicated by URL - belt and suspenders so we never
    # end up with fewer real sources than were actually consulted.
    seen = {s.get("url") for s in data["sources"] if s.get("url")}
    for src in cited_sources:
        if src.get("url") and src["url"] not in seen:
            data["sources"].append(src)
            seen.add(src["url"])

    if not data["sources"]:
        logger.warning("[RESEARCH] no sources found for topic=%r - treat facts as unverified", topic)

    logger.info(
        "[RESEARCH] done topic=%r facts=%d sources=%d uncertainties=%d",
        topic, len(data["facts"]), len(data["sources"]), len(data["uncertainties"]),
    )
    return data


def run_and_save(paths: ps.ProjectPaths, topic: str, category: str | None = None) -> dict:
    ps.save_state(paths, PipelineState.RESEARCHING)
    run_id = db.start_run(paths.video_id, "research")
    try:
        data = research_topic(topic, category=category)
        ps.save_json(paths.research_json, data)
        db.insert_research_sources(paths.video_id, data["sources"])
        ps.save_state(paths, PipelineState.RESEARCHED)
        db.finish_run(run_id, "success")
        return data
    except Exception as e:
        db.finish_run(run_id, "failed", error=str(e))
        ps.save_failure(paths, "research", str(e))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research a documentary topic (standalone test).")
    parser.add_argument("--topic", required=True, help="Topic to research")
    parser.add_argument("--category", default=None, help="Content category (e.g. Mysteries)")
    args = parser.parse_args(argv)

    try:
        paths, created = ps.create_project(args.topic, category=args.category)
        logger.info(
            "[RESEARCH] project %s (%s) at %s",
            paths.slug, "new" if created else "reused (duplicate topic)", paths.root,
        )
        run_and_save(paths, args.topic, category=args.category)
        print(f"OK - research saved to {paths.research_json}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
