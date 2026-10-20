"""OpenRouter chat client: retries, streaming, and clear errors."""
from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from src.config.settings import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MODEL_NAME,
    OPENROUTER_API_KEY,
)

URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """Raised with a message that is safe to show directly in the UI."""


def _payload(messages, stream: bool, temperature, max_tokens) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "stream": stream,
    }


def _error_text(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error", resp.text)
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    except ValueError:
        return resp.text[:300]


def _fold_system(messages: list[dict]) -> list[dict]:
    """Some models reject the system role; merge it into the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0]["content"], list(messages[1:])
    for m in rest:
        if m["role"] == "user":
            rest[rest.index(m)] = {"role": "user", "content": f"{system}\n\n{m['content']}"}
            break
    return rest


def _post(messages, stream, temperature, max_tokens) -> requests.Response:
    if not OPENROUTER_API_KEY:
        raise LLMError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    folded = False
    last = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        retry_after = 0.0
        try:
            resp = requests.post(
                URL,
                headers=headers,
                json=_payload(messages, stream, temperature, max_tokens),
                timeout=LLM_TIMEOUT,
                stream=stream,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            last = f"network error: {e}"
        else:
            if resp.status_code == 200:
                return resp
            last = f"{resp.status_code}: {_error_text(resp)}"
            # Models that reject a system prompt: retry once with it folded in.
            if resp.status_code == 400 and not folded and messages[0]["role"] == "system":
                messages, folded = _fold_system(messages), True
                continue
            if resp.status_code not in _RETRY_STATUS:
                raise LLMError(f"OpenRouter error {last}")
            try:
                retry_after = float(resp.headers.get("Retry-After", 0))
            except ValueError:
                pass
        if attempt < _MAX_ATTEMPTS:
            time.sleep(min(max(retry_after, 1.5 * 2 ** (attempt - 1)), 10))
    raise LLMError(f"OpenRouter request failed after {_MAX_ATTEMPTS} attempts ({last}).")


def chat(messages, temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Single, non-streaming completion."""
    data = _post(messages, False, temperature, max_tokens).json()
    if "error" in data:
        raise LLMError(f"OpenRouter error: {data['error']}")
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected OpenRouter response: {str(data)[:300]}")
    content = message.get("content")
    if not content or not str(content).strip():
        if message.get("reasoning"):
            raise LLMError(
                "The model spent its whole token budget reasoning and produced no answer. "
                "Increase LLM_MAX_TOKENS in .env or choose a model with lighter reasoning."
            )
        raise LLMError("The model returned an empty response. Try again.")
    return str(content)


def stream_chat(messages, temperature: float | None = None, max_tokens: int | None = None) -> Iterator[str]:
    """Yield answer text as it is generated.

    Reasoning-model 'thinking' deltas are deliberately skipped; only the final
    answer content is yielded.
    """
    resp = _post(messages, True, temperature, max_tokens)
    got_any = False
    saw_reasoning = False
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue  # blank keep-alives and ": OPENROUTER PROCESSING" comments
        body = raw[5:].strip()
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
        except ValueError:
            continue
        if "error" in chunk:
            raise LLMError(f"OpenRouter stream error: {chunk['error']}")
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("reasoning"):
                saw_reasoning = True
            text = delta.get("content")
            if text:
                got_any = True
                yield text
    if not got_any:
        raise LLMError(
            "The model spent its whole token budget reasoning and produced no answer. "
            "Increase LLM_MAX_TOKENS in .env."
            if saw_reasoning
            else "The model returned an empty response. Try again."
        )


# Backwards-compatible with the old single-string API.
def generate_answer(prompt: str) -> str:
    return chat([{"role": "user", "content": prompt}])
