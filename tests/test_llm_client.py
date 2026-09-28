"""
Phase 2.1 - Scene Planner LLM request reliability tests.

Covers only the `_post_with_retries()` timeout-configurability fix:
  - CLAUDE_TIMEOUT_SECONDS (config/env) drives the default httpx timeout
    used for every Claude API call, instead of the old hard-coded 120s.
  - An explicit `timeout=` argument still overrides the config value
    (backward compatibility for any existing caller).
  - Existing retry behavior (429/5xx retried, transient transport/timeout
    errors retried, other 4xx NOT retried, CLAUDE_MAX_RETRIES respected) is
    completely unchanged by this fix.
  - chat()/chat_json() (the public API other modules use) keep working
    end-to-end through the same scripted-client harness.

No real network access, no real API key, no real Anthropic call anywhere in
this file - httpx.Client is replaced with a small scripted fake per test.
"""
from __future__ import annotations

import httpx
import pytest

from core.config import settings
from youtube import llm_client


class _FakeResponse:
    """Stands in for an httpx.Response - only the surface _post_with_retries
    actually touches (status_code, text, json(), raise_for_status())."""

    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", llm_client.ANTHROPIC_URL)
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)

    def json(self) -> dict:
        return self._json_data


def _scripted_client(actions: list):
    """Builds a fake replacement for httpx.Client. `actions` is a list
    consumed one-per-call to .post(): each entry is either an Exception
    instance (raised) or a response-like object (returned). Also records
    every `timeout=` kwarg the fake was constructed with, in call order, so
    tests can assert exactly what timeout each attempt used."""
    captured_timeouts: list[float | None] = []

    class _ScriptedClient:
        def __init__(self, timeout=None, *args, **kwargs):
            captured_timeouts.append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def post(self, *args, **kwargs):
            action = actions.pop(0)
            if isinstance(action, Exception):
                raise action
            return action

    return _ScriptedClient, captured_timeouts


def _ok_response(text: str = "ok") -> _FakeResponse:
    return _FakeResponse(status_code=200, json_data={"content": [{"type": "text", "text": text}]})


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    # _post_with_retries()/_headers() call require("ANTHROPIC_API_KEY") -
    # give every test in this file a harmless fake key so that check never
    # gets in the way of what's actually being tested here.
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-test-fake-key-not-real")


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Retry backoff uses time.sleep(); never actually wait in this test file.
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_args, **_kwargs: None)


# ---------------------------------------------------------------------------
# Timeout configurability (the actual Phase 2.1 fix)
# ---------------------------------------------------------------------------

def test_post_with_retries_uses_config_timeout_when_not_given_explicitly(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_TIMEOUT_SECONDS", 77.0)
    client_cls, timeouts = _scripted_client([_ok_response()])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    result = llm_client._post_with_retries({"model": "x"})

    assert timeouts == [77.0]
    assert result == {"content": [{"type": "text", "text": "ok"}]}


def test_post_with_retries_default_config_timeout_is_240(monkeypatch):
    # Uses the real (unpatched) settings.CLAUDE_TIMEOUT_SECONDS to confirm
    # the wiring reaches all the way from config to the actual httpx call.
    client_cls, timeouts = _scripted_client([_ok_response()])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    llm_client._post_with_retries({"model": "x"})

    assert timeouts == [settings.CLAUDE_TIMEOUT_SECONDS] == [240.0]


def test_post_with_retries_explicit_timeout_overrides_config(monkeypatch):
    # Backward compatibility: a caller that still passes timeout= explicitly
    # gets exactly that value, not the config default.
    monkeypatch.setattr(settings, "CLAUDE_TIMEOUT_SECONDS", 77.0)
    client_cls, timeouts = _scripted_client([_ok_response()])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    llm_client._post_with_retries({"model": "x"}, timeout=15.0)

    assert timeouts == [15.0]


def test_post_with_retries_uses_configured_timeout_on_every_retry_attempt(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "CLAUDE_TIMEOUT_SECONDS", 55.0)
    request = httpx.Request("POST", llm_client.ANTHROPIC_URL)
    timeout_error = httpx.ReadTimeout("simulated read timeout", request=request)
    client_cls, timeouts = _scripted_client([timeout_error, _ok_response()])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    result = llm_client._post_with_retries({"model": "x"})

    assert result == {"content": [{"type": "text", "text": "ok"}]}
    # Same configured timeout used on both the failed and the retried attempt.
    assert timeouts == [55.0, 55.0]


# ---------------------------------------------------------------------------
# Existing retry behavior - unchanged by this fix
# ---------------------------------------------------------------------------

def test_post_with_retries_retries_on_read_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_MAX_RETRIES", 3)
    request = httpx.Request("POST", llm_client.ANTHROPIC_URL)
    timeout_error = httpx.ReadTimeout("simulated read timeout", request=request)
    client_cls, timeouts = _scripted_client([timeout_error, _ok_response("recovered")])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    result = llm_client._post_with_retries({"model": "x"})

    assert result == {"content": [{"type": "text", "text": "recovered"}]}
    assert len(timeouts) == 2


def test_post_with_retries_retries_on_429(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_MAX_RETRIES", 3)
    rate_limited = _FakeResponse(status_code=429, text="rate limited")
    client_cls, timeouts = _scripted_client([rate_limited, _ok_response()])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    result = llm_client._post_with_retries({"model": "x"})

    assert result == {"content": [{"type": "text", "text": "ok"}]}
    assert len(timeouts) == 2


def test_post_with_retries_retries_on_5xx(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_MAX_RETRIES", 3)
    server_error = _FakeResponse(status_code=503, text="upstream overloaded")
    client_cls, timeouts = _scripted_client([server_error, _ok_response()])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    result = llm_client._post_with_retries({"model": "x"})

    assert result == {"content": [{"type": "text", "text": "ok"}]}
    assert len(timeouts) == 2


def test_post_with_retries_does_not_retry_non_429_4xx(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_MAX_RETRIES", 3)
    bad_request = _FakeResponse(status_code=400, text="bad request")
    client_cls, timeouts = _scripted_client([bad_request])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    with pytest.raises(llm_client.ClaudeError, match="400"):
        llm_client._post_with_retries({"model": "x"})

    assert len(timeouts) == 1  # no retry attempted


def test_post_with_retries_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_MAX_RETRIES", 2)
    request = httpx.Request("POST", llm_client.ANTHROPIC_URL)
    timeout_error = httpx.ReadTimeout("simulated read timeout", request=request)
    client_cls, timeouts = _scripted_client([timeout_error, timeout_error])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    with pytest.raises(llm_client.ClaudeError, match="gave up after 2 attempts"):
        llm_client._post_with_retries({"model": "x"})

    assert len(timeouts) == 2


# ---------------------------------------------------------------------------
# Backward compatibility: the public chat()/chat_json() API still works
# end-to-end through the same scripted-client harness.
# ---------------------------------------------------------------------------

def test_chat_still_works_and_uses_config_timeout(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_TIMEOUT_SECONDS", 200.0)
    client_cls, timeouts = _scripted_client([_ok_response("hello there")])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    text, sources = llm_client.chat("system prompt", "user prompt", max_tokens=10)

    assert text == "hello there"
    assert sources == []
    assert timeouts == [200.0]


def test_chat_json_still_works_and_uses_config_timeout(monkeypatch):
    monkeypatch.setattr(settings, "CLAUDE_TIMEOUT_SECONDS", 200.0)
    response = _FakeResponse(status_code=200, json_data={"content": [{"type": "text", "text": '{"a": 1}'}]})
    client_cls, timeouts = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    data, sources = llm_client.chat_json("system prompt", "user prompt", max_tokens=10)

    assert data == {"a": 1}
    assert sources == []
    assert timeouts == [200.0]


# ---------------------------------------------------------------------------
# Research JSON robustness fix: truncated (max_tokens) responses must never
# reach json.loads, fenced JSON must still parse, and a genuinely malformed
# (but NOT truncated) response must still raise a plain ClaudeError - not be
# misclassified as truncation. Production incident: a real web-searched
# research call for "The Ocean Sound Scientists Still Can't Explain" was cut
# off mid-JSON by Claude's own max_tokens limit.
# ---------------------------------------------------------------------------

def _response_with_stop_reason(text: str, stop_reason: str | None) -> _FakeResponse:
    content = {"content": [{"type": "text", "text": text}]}
    if stop_reason is not None:
        content["stop_reason"] = stop_reason
    return _FakeResponse(status_code=200, json_data=content)


def test_chat_json_parses_markdown_fenced_json(monkeypatch):
    fenced = '```json\n{"a": 1, "b": [1, 2]}\n```'
    response = _response_with_stop_reason(fenced, "end_turn")
    client_cls, _ = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    data, sources = llm_client.chat_json("system prompt", "user prompt", max_tokens=10)

    assert data == {"a": 1, "b": [1, 2]}
    assert sources == []


def test_chat_json_raises_truncated_error_when_stop_reason_is_max_tokens(monkeypatch):
    # Mirrors the real production symptom: valid-looking JSON that simply
    # stops mid-string because the model ran out of output budget.
    truncated_text = (
        '{"topic": "x", "summary": "y", "facts": ["a"], "sources": [], '
        '"uncertainties": ["others remain officially unre'
    )
    response = _response_with_stop_reason(truncated_text, "max_tokens")
    client_cls, _ = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    with pytest.raises(llm_client.ClaudeTruncatedResponseError, match="max_tokens"):
        llm_client.chat_json("system prompt", "user prompt", max_tokens=4000)


def test_claude_truncated_response_error_is_still_a_claude_error(monkeypatch):
    # Any existing `except ClaudeError:` elsewhere in the codebase must
    # keep catching this - it's a subclass, not a parallel error type.
    response = _response_with_stop_reason('{"a": 1', "max_tokens")
    client_cls, _ = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    with pytest.raises(llm_client.ClaudeError):
        llm_client.chat_json("system prompt", "user prompt", max_tokens=4000)


def test_chat_json_truncated_response_never_reaches_json_loads(monkeypatch):
    # If this were handed to json.loads it would raise json.JSONDecodeError
    # (wrapped as a plain ClaudeError) instead of the distinguishable
    # truncation error - confirms the stop_reason check happens BEFORE any
    # parse attempt, not as a side effect of a parse failure.
    response = _response_with_stop_reason('{"a": 1}', "max_tokens")  # syntactically valid JSON, but still truncated
    client_cls, _ = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    with pytest.raises(llm_client.ClaudeTruncatedResponseError):
        llm_client.chat_json("system prompt", "user prompt", max_tokens=4000)


def test_chat_json_malformed_but_not_truncated_raises_plain_claude_error(monkeypatch):
    # stop_reason "end_turn" (a normal, complete stop) but the text itself
    # is simply broken JSON - must NOT be misclassified as a truncation.
    malformed_text = '{"a": 1, "b": }'
    response = _response_with_stop_reason(malformed_text, "end_turn")
    client_cls, _ = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    with pytest.raises(llm_client.ClaudeError) as exc_info:
        llm_client.chat_json("system prompt", "user prompt", max_tokens=4000)
    assert not isinstance(exc_info.value, llm_client.ClaudeTruncatedResponseError)


def test_chat_json_still_works_when_stop_reason_is_absent(monkeypatch):
    # A response shape with no stop_reason field at all must not be
    # misclassified as truncated - absence must never mean "max_tokens".
    response = _response_with_stop_reason('{"a": 1}', stop_reason=None)
    client_cls, _ = _scripted_client([response])
    monkeypatch.setattr(llm_client.httpx, "Client", client_cls)

    data, sources = llm_client.chat_json("system prompt", "user prompt", max_tokens=10)
    assert data == {"a": 1}
