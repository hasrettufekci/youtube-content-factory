"""
Topic scout (spec section 11).

MVP: manual topic entry only.

    python -m youtube.topic_scout --topic "The mysterious place where nobody is allowed to go"

Registers the topic as a scored candidate (storage/factory.db `topics`
table) without yet creating a video project - that happens when
pipeline.py actually produces it (or via --produce below, for convenience).

Phase 2 (future, spec section 37 PHASE 2): automatic candidate discovery
(web/YouTube trend research -> N candidates -> topic_scorer -> best pick)
would live in a `discover_candidates()` function added here; the DB schema
and `top_candidate()` below are already shaped to support that without
changes to topic_scorer.py or pipeline.py.
"""
from __future__ import annotations

import argparse
import json

from core.logging_setup import get_logger
from storage import db, project_store as ps
from youtube import topic_scorer

logger = get_logger("topic_scout")


def add_candidate(topic: str, category: str | None = None, use_llm_scoring: bool = False) -> dict:
    scored = topic_scorer.score_topic(topic, category=category, use_llm=use_llm_scoring)
    normalized = ps.normalize_topic(topic)
    db.upsert_topic_candidate(
        topic=topic,
        normalized_topic=normalized,
        category=category,
        score=scored["overall"],
        score_json=json.dumps(scored["scores"]),
    )
    logger.info("[TOPIC_SCOUT] candidate added topic=%r overall=%s", topic, scored["overall"])
    return scored


def top_candidate() -> dict | None:
    rows = db.top_topic_candidates(limit=1)
    if not rows:
        return None
    row = rows[0]
    return {
        "topic": row["topic"],
        "category": row["category"],
        "score": row["score"],
        "scores": json.loads(row["score_json"]) if row["score_json"] else {},
    }


def list_candidates(limit: int = 20) -> list[dict]:
    rows = db.top_topic_candidates(limit=limit)
    return [
        {
            "topic": r["topic"],
            "category": r["category"],
            "score": r["score"],
            "scores": json.loads(r["score_json"]) if r["score_json"] else {},
        }
        for r in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add/inspect candidate documentary topics.")
    parser.add_argument("--topic", help="Manually add a topic candidate")
    parser.add_argument("--category", default=None)
    parser.add_argument("--llm-score", action="store_true", help="Use Claude to score instead of the offline heuristic")
    parser.add_argument("--list", action="store_true", help="List current candidates by score")
    args = parser.parse_args(argv)

    if args.topic:
        scored = add_candidate(args.topic, category=args.category, use_llm_scoring=args.llm_score)
        print(json.dumps(scored, indent=2, ensure_ascii=False))

    if args.list or not args.topic:
        print("\nTop candidates:")
        for c in list_candidates():
            print(f"  [{c['score']:>5}] {c['topic']}  ({c['category'] or 'uncategorized'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
