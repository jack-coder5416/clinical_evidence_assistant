import json
import pytest

from src.llm import openrouter
from src.llm.openrouter import LLMError


class FakeResp:
    def __init__(self, status=200, json_data=None, lines=None, headers=None):
        self.status_code, self._json, self._lines = status, json_data or {}, lines or []
        self.headers, self.text = headers or {}, json.dumps(json_data or {})
    def json(self): return self._json
    def iter_lines(self, decode_unicode=True): return iter(self._lines)


@pytest.fixture(autouse=True)
def _key_and_no_sleep(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)


def sse(*events):
    out = []
    for e in events:
        out += [f"data: {json.dumps(e)}" if e != "[DONE]" else "data: [DONE]", ""]
    return [": OPENROUTER PROCESSING", ""] + out


def test_chat_returns_content(monkeypatch):
    monkeypatch.setattr(openrouter.requests, "post", lambda *a, **k: FakeResp(
        json_data={"choices": [{"message": {"content": "hello [1]"}}]}))
    assert openrouter.chat([{"role": "user", "content": "x"}]) == "hello [1]"


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = []
    def post(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            return FakeResp(429, {"error": {"message": "rate limited"}})
        return FakeResp(json_data={"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(openrouter.requests, "post", post)
    assert openrouter.chat([{"role": "user", "content": "x"}]) == "ok" and len(calls) == 3


def test_gives_clear_error_after_repeated_429(monkeypatch):
    monkeypatch.setattr(openrouter.requests, "post",
                        lambda *a, **k: FakeResp(429, {"error": {"message": "rate limited"}}))
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        openrouter.chat([{"role": "user", "content": "x"}])


def test_non_retryable_error_fails_fast(monkeypatch):
    calls = []
    monkeypatch.setattr(openrouter.requests, "post",
                        lambda *a, **k: (calls.append(1), FakeResp(401, {"error": {"message": "bad key"}}))[1])
    with pytest.raises(LLMError, match="bad key"):
        openrouter.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_system_role_rejected_falls_back_to_folded_prompt(monkeypatch):
    seen = []
    def post(url, headers, json, **k):
        seen.append(json["messages"])
        if json["messages"][0]["role"] == "system":
            return FakeResp(400, {"error": {"message": "Developer instruction is not enabled"}})
        return FakeResp(json_data={"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(openrouter.requests, "post", post)
    msgs = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "Q"}]
    assert openrouter.chat(msgs) == "ok"
    assert seen[1][0]["role"] == "user" and seen[1][0]["content"].startswith("RULES")


def test_reasoning_only_response_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(openrouter.requests, "post", lambda *a, **k: FakeResp(
        json_data={"choices": [{"message": {"content": None, "reasoning": "thinking..."}}]}))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        openrouter.chat([{"role": "user", "content": "x"}])


def test_stream_skips_reasoning_and_comments(monkeypatch):
    lines = sse(
        {"choices": [{"delta": {"reasoning": "let me think"}}]},
        {"choices": [{"delta": {"content": "Consent "}}]},
        {"choices": [{"delta": {"content": "is required [1]."}}]},
        "[DONE]")
    monkeypatch.setattr(openrouter.requests, "post", lambda *a, **k: FakeResp(lines=lines))
    assert "".join(openrouter.stream_chat([{"role": "user", "content": "x"}])) == "Consent is required [1]."


def test_stream_reasoning_only_raises(monkeypatch):
    lines = sse({"choices": [{"delta": {"reasoning": "hmm"}}]}, "[DONE]")
    monkeypatch.setattr(openrouter.requests, "post", lambda *a, **k: FakeResp(lines=lines))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        list(openrouter.stream_chat([{"role": "user", "content": "x"}]))


def test_missing_key_message(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", None)
    with pytest.raises(LLMError, match=".env"):
        openrouter.chat([{"role": "user", "content": "x"}])
