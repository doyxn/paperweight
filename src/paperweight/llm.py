"""Minimal LLM clients (Anthropic Messages API and OpenAI Chat Completions)."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from paperweight.config import ConfigError, Settings

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 16000) -> str: ...


class AnthropicClient:
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str | None = None, client: httpx.Client | None = None):
        self.api_key = api_key
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.http = client or httpx.Client(timeout=httpx.Timeout(600.0))

    def complete(self, system: str, user: str, max_tokens: int = 16000) -> str:
        response = self.http.post(
            self.url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        if response.status_code >= 400:
            raise LLMError(f"Anthropic API error {response.status_code}: {response.text[:500]}")
        blocks = response.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class OpenAIClient:
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str | None = None, client: httpx.Client | None = None):
        self.api_key = api_key
        self.model = model or DEFAULT_OPENAI_MODEL
        self.http = client or httpx.Client(timeout=httpx.Timeout(600.0))

    def complete(self, system: str, user: str, max_tokens: int = 16000) -> str:
        response = self.http.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if response.status_code >= 400:
            raise LLMError(f"OpenAI API error {response.status_code}: {response.text[:500]}")
        return response.json()["choices"][0]["message"]["content"] or ""


def make_llm(settings: Settings) -> LLMClient:
    provider = (settings.llm_provider or "").lower()
    if provider in ("", "anthropic") and settings.anthropic_api_key:
        return AnthropicClient(settings.anthropic_api_key, settings.llm_model)
    if provider in ("", "openai") and settings.openai_api_key:
        return OpenAIClient(settings.openai_api_key, settings.llm_model)
    raise ConfigError("set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the spec stage")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object in an LLM reply, tolerating code fences and prose."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
    raise LLMError("LLM reply did not contain a JSON object")
