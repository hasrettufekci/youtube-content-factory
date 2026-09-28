"""
Thin Anthropic Claude wrapper shared by researcher.py, script_generator.py,
topic_scorer.py and metadata.py. Not tied to any single call shape - just
"send system+user, get text back" and "...get parsed JSON back", plus an
optional server-side web-search tool for researcher.py.

Kept deliberately provider-light (raw httpx, no SDK dependency) so the
project stays "Python + requests-like client + APIs" per spec section 38
(no over-engineering).
"""
from __future__ import annotations

import json
import re
import time

import httpx

from core.config import settings, require
from core.logging_setup import get_logger

logger = get_logger("claude")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class ClaudeError(RuntimeError):
    pass


class ClaudeTruncatedResponseError(ClaudeError):
    """Claude's own `stop_reason` says the response was cut off by hitting
    max_tokens before it finished generating - the returned text is
    provably incomplete (not just malformed), so it must never be handed to
    a JSON parser. A subclass of ClaudeError so any existing `except
    ClaudeError:` still catches it; callers that want to distinguish
    "definitely truncated, a fresh request is worth retrying" from "the
    model produced something and it happened not to parse" can catch this
    specifically (see youtube/researcher.py)."""


def _headers() -> dict:
    return {
        "x-api-key": require("ANTHROPIC_API_KEY"),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _post_with_retries(payload: dict, timeout: float | None = None) -> dict:
    """POSTs to the Anthropic Messages API with retry-on-transient-failure.

    `timeout` (httpx per-request timeout, seconds) defaults to
    `settings.CLAUDE_TIMEOUT_SECONDS` (config/env-overridable via
    CLAUDE_TIMEOUT_SECONDS, 240s by default) when not given explicitly - read
    from `settings` at call time (not baked into the function signature) so a
    test or caller can override it without reloading this module. Passing an
    explicit `timeout` still works exactly as before and takes precedence
    over the config value, preserving backward compatibility for any
    existing caller that relies on it. Retry count/backoff and which status
    codes are retryable are unchanged by this."""
    resolved_timeout = timeout if timeout is not None else settings.CLAUDE_TIMEOUT_SECONDS
    logger.debug("[CLAUDE] request timeout=%ss (max_retries=%d)", resolved_timeout, settings.CLAUDE_MAX_RETRIES)

    last_error: Exception | None = None
    for attempt in range(1, settings.CLAUDE_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=resolved_timeout) as client:
                resp = client.post(ANTHROPIC_URL, headers=_headers(), json=payload)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise ClaudeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError, ClaudeError) as e:
            last_error = e
            wait = min(2 ** attempt, 20)
            logger.warning(
                "[CLAUDE] attempt %d/%d failed (%s), retrying in %ss",
                attempt, settings.CLAUDE_MAX_RETRIES, e, wait,
            )
            time.sleep(wait)
        except httpx.HTTPStatusError as e:
            # 4xx other than 429 - not retryable (bad request, auth, etc.)
            raise ClaudeError(f"[CLAUDE] request failed: {e.response.status_code} {e.response.text[:500]}") from e
    raise ClaudeError(f"[CLAUDE] gave up after {settings.CLAUDE_MAX_RETRIES} attempts: {last_error}")


def _extract_text(data: dict) -> str:
    blocks = data.get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


def _extract_citations(data: dict) -> list[dict]:
    """Best-effort extraction of web-search citations/results, if the
    web_search server tool was used. Anthropic's web search tool attaches
    `citations` to text blocks and/or emits `web_search_tool_result` blocks;
    we tolerate either shape (or its absence) since exact schema can evolve."""
    sources: list[dict] = []
    seen_urls: set[str] = set()

    def add(url: str | None, title: str | None = None, published: str | None = None):
        if not url or url in seen_urls:
            return
        seen_urls.add(url)
        sources.append({"title": title or url, "url": url, "published_at": published})

    for block in data.get("content", []):
        for citation in block.get("citations", []) or []:
            add(citation.get("url"), citation.get("title"), citation.get("published_date"))
        if block.get("type") == "web_search_tool_result":
            content = block.get("content", [])
            if isinstance(content, list):
                for item in content:
                    add(item.get("url"), item.get("title"), item.get("page_age"))
    return sources


def _chat_raw(
    system: str,
    user: str,
    max_tokens: int = 2000,
    model: str | None = None,
    use_web_search: bool = False,
) -> tuple[str, list[dict], str | None]:
    """Does the actual API call. Returns (text, sources, stop_reason) - the
    one place that knows the raw Anthropic response shape. `chat()` wraps
    this and drops stop_reason to keep its existing public contract;
    `chat_json()` uses stop_reason directly to detect truncation before
    ever attempting to parse the text as JSON."""
    payload = {
        "model": model or settings.CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if use_web_search and settings.CLAUDE_WEB_SEARCH_ENABLED:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}]

    data = _post_with_retries(payload)
    text = _extract_text(data)
    if not text:
        raise ClaudeError(f"[CLAUDE] empty response: {json.dumps(data)[:500]}")
    sources = _extract_citations(data) if use_web_search else []
    return text, sources, data.get("stop_reason")


def chat(
    system: str,
    user: str,
    max_tokens: int = 2000,
    model: str | None = None,
    use_web_search: bool = False,
) -> tuple[str, list[dict]]:
    """Returns (text, sources). `sources` is only non-empty when
    use_web_search=True and the model actually performed a search."""
    text, sources, _stop_reason = _chat_raw(
        system, user, max_tokens=max_tokens, model=model, use_web_search=use_web_search,
    )
    return text, sources


def _extract_json_text(text: str) -> str:
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if not match:
        raise ClaudeError(f"[CLAUDE] no JSON found in response:\n{text[:800]}")
    return match.group(0)


def chat_json(
    system: str,
    user: str,
    max_tokens: int = 2000,
    model: str | None = None,
    use_web_search: bool = False,
) -> tuple[dict, list[dict]]:
    text, sources, stop_reason = _chat_raw(
        system, user, max_tokens=max_tokens, model=model, use_web_search=use_web_search,
    )
    if stop_reason == "max_tokens":
        # The response was cut off mid-generation - `text` is provably
        # incomplete (this is exactly the production symptom: JSON that
        # parses fine right up until it just stops, e.g. mid-word inside a
        # string). Never hand this to json.loads and never return it as if
        # it were real data - raise a distinguishable error so a caller
        # that wants to retry (a fresh request, not a reparse) can do so.
        raise ClaudeTruncatedResponseError(
            f"[CLAUDE] response truncated (stop_reason=max_tokens, max_tokens={max_tokens}) "
            f"before JSON was complete; text ended with: ...{text[-300:]}"
        )
    json_text = _extract_json_text(text)
    try:
        return json.loads(json_text), sources
    except json.JSONDecodeError as e:
        raise ClaudeError(f"[CLAUDE] invalid JSON returned: {e}\n\n{text[:800]}") from e
