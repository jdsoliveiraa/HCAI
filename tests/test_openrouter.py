"""The OpenRouter adapter, driven through an in-process transport: no request leaves the machine."""

import json

import httpx
import pytest

from hcai.llm import LLMRequest, OpenRouterClient

SCHEMA = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}


def test_sends_structured_request_with_env_key_and_parses_object_and_usage(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"score": 61}'}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9, "cost": 0.00042},
        })

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = OpenRouterClient(http=httpx.Client(transport=httpx.MockTransport(handler)))

    response = client.complete(LLMRequest(
        model="vendor/model", messages=[{"role": "user", "content": "hi"}], temperature=0.0, seed=42,
        response_schema=SCHEMA,
    ))

    assert response.parsed == {"score": 61}
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.cost) == (120, 9, 0.00042)
    body = json.loads(seen[0].content)
    assert seen[0].headers["Authorization"] == "Bearer sk-test"
    assert (body["model"], body["temperature"], body["seed"]) == ("vendor/model", 0.0, 42)
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient()
