"""The LLM client interface, its OpenRouter and fake implementations, and the request cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    seed: int | None = None
    response_schema: dict[str, Any] | None = None

    def cache_key(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    parsed: Any = None
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


def _parse_if_schema(request: LLMRequest, text: str) -> Any:
    if request.response_schema is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class FakeLLMClient:
    """Scriptable client for tests: `responder` maps a request to text or a JSON-able object.

    Every request is recorded in `calls`. Token counts are word counts so they are non-zero.
    """

    def __init__(self, responder: Callable[[LLMRequest], str | dict | list]):
        self.responder = responder
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        out = self.responder(request)
        text = out if isinstance(out, str) else json.dumps(out)
        prompt_words = sum(len(m["content"].split()) for m in request.messages)
        usage = Usage(prompt_tokens=prompt_words, completion_tokens=len(text.split()))
        return LLMResponse(text=text, parsed=_parse_if_schema(request, text), usage=usage)


class OpenRouterClient:
    """OpenRouter chat-completions client. Reads the key from OPENROUTER_API_KEY."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None, http: httpx.Client | None = None, timeout: float = 120.0):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._http = http or httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {key}"}

    def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "usage": {"include": True},
        }
        if request.seed is not None:
            body["seed"] = request.seed
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": request.response_schema},
            }
        resp = self._http.post(self.URL, json=body, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens", 0)),
            completion_tokens=int(u.get("completion_tokens", 0)),
            cost=float(u.get("cost", 0.0) or 0.0),
        )
        return LLMResponse(text=text, parsed=_parse_if_schema(request, text), usage=usage)


class CachedClient:
    """Wraps a client; responses are stored on disk keyed by the hash of the full request."""

    def __init__(self, inner: LLMClient, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, request: LLMRequest) -> LLMResponse:
        path = self.cache_dir / f"{request.cache_key()}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            return LLMResponse(text=cached["text"], parsed=cached["parsed"], usage=Usage(**cached["usage"]))
        response = self.inner.complete(request)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": asdict(request), **asdict(response)}, ensure_ascii=False))
        tmp.replace(path)
        return response
