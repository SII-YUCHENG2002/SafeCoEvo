"""Minimal configurable OpenAI-compatible client."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class ChatResult:
    content: str
    reasoning_content: str | None
    model: str
    usage: dict[str, Any]
    latency_seconds: float
    raw: dict[str, Any]


class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        temperature: float = 0,
        max_tokens: int = 4096,
        timeout_seconds: int = 60,
        extra_body: dict[str, Any] | None = None,
        user_agent: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.extra_body = dict(extra_body or {})
        self.user_agent = user_agent.strip()

    @property
    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        return value

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> ChatResult:
        model = overrides.pop("model", self.model)
        token_limit = overrides.pop("max_tokens", self.max_tokens)
        token_field = "max_completion_tokens" if model.lower().startswith("gpt-") else "max_tokens"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": overrides.pop("temperature", self.temperature),
            token_field: token_limit,
        }
        payload.update(self.extra_body)
        payload.update(overrides)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.user_agent:
            headers["User-Agent"] = self.user_agent

        start = time.perf_counter()
        for token_field_attempt in range(2):
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8", errors="replace"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                unsupported_token_field = (
                    exc.code == 400
                    and "unsupported_parameter" in body.lower()
                    and token_field in body
                )
                if unsupported_token_field and token_field_attempt == 0:
                    alternate = (
                        "max_tokens" if token_field == "max_completion_tokens" else "max_completion_tokens"
                    )
                    payload[alternate] = payload.pop(token_field)
                    token_field = alternate
                    continue
                raise RuntimeError(f"Chat request failed with HTTP {exc.code}: {body}") from exc
        latency = time.perf_counter() - start

        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ChatResult(
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content"),
            model=raw.get("model") or payload["model"],
            usage=raw.get("usage") or {},
            latency_seconds=latency,
            raw=raw,
        )
