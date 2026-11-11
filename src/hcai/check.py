"""Provider checks (`hcai models`, `hcai check`): what a provider offers and how each configured model behaves.

`probe_model` sends a few tiny requests, uncached, to see whether a model:
- answers on the OpenAI-style chat endpoint (`endpoint_ok`) and with a JSON-schema response format, as the
  pipeline's agents need (`json_schema_ok`);
- gives the identical answer to identical requests at temperature 0 with the seed (`deterministic`);
- honours temperature: answers vary at temperature 1 (`temperature_honoured`);
- honours the seed: at temperature 1 the same seed gives the same answer (`seed_honoured`; None when the
  answers never vary, as then the seed cannot be told apart from an ignored temperature).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hcai.llm import LLMClient, LLMRequest

ENDPOINT_PROMPT = "Reply with the single word: ok"
JSON_PROMPT = 'Answer with JSON: {"word": "<any English word>"}.'
JSON_SCHEMA = {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"],
               "additionalProperties": False}
VARIED_PROMPT = ("Invent a name for a fictional town and describe it in one sentence of at most 25 words. "
                 "Be imaginative.")


@dataclass(frozen=True)
class ProbeResult:
    model: str
    provider: str
    listed: bool | None  # in the provider's /models listing (None: the listing failed)
    endpoint_ok: bool
    json_schema_ok: bool | None = None
    deterministic: bool | None = None  # temperature 0 + seed: identical answers
    temperature_honoured: bool | None = None
    seed_honoured: bool | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


def probe_model(client: LLMClient, model: str, *, provider: str = "", listed: bool | None = None,
                repeats: int = 3, seed: int = 42) -> ProbeResult:
    """Probe `model` through `client` with `3 * repeats + 2` tiny calls (none cached: see the module doc)."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to compare answers")
    tally = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def ask(prompt: str, temperature: float, seed: int | None, schema: dict | None = None):
        tally["calls"] += 1
        response = client.complete(LLMRequest(model=model, messages=[{"role": "user", "content": prompt}],
                                              temperature=temperature, seed=seed, response_schema=schema))
        tally["prompt_tokens"] += response.usage.prompt_tokens
        tally["completion_tokens"] += response.usage.completion_tokens
        return response

    def result(**kw) -> ProbeResult:
        return ProbeResult(model=model, provider=provider, listed=listed, **tally, **kw)

    try:
        endpoint_ok = bool(ask(ENDPOINT_PROMPT, 0.0, seed).text.strip())
    except Exception as e:  # noqa: BLE001 - reported, not raised: the other models are still probed
        return result(endpoint_ok=False, error=f"{type(e).__name__}: {e}")
    if not endpoint_ok:
        return result(endpoint_ok=False, error="empty answer")
    errors: list[str] = []

    def attempt(what: str, fn: Callable[[], bool | None]) -> bool | None:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{what}: {type(e).__name__}: {e}")
            return None

    def json_ok() -> bool:
        try:
            parsed = ask(JSON_PROMPT, 0.0, seed, JSON_SCHEMA).parsed
        except Exception as e:  # noqa: BLE001 - a provider refusing response_format (often HTTP 400) is a "no"
            errors.append(f"json_schema: {type(e).__name__}: {e}")
            return False
        return isinstance(parsed, dict) and isinstance(parsed.get("word"), str)

    def texts(temperature: float, seeds: list[int]) -> list[str]:
        return [ask(VARIED_PROMPT, temperature, s).text.strip() for s in seeds]

    json_schema_ok = attempt("json_schema", json_ok)
    t0 = attempt("temperature 0", lambda: len(set(texts(0.0, [seed] * repeats))) == 1)
    same = attempt("temperature 1, same seed", lambda: len(set(texts(1.0, [seed] * repeats))) == 1)
    differ = attempt("temperature 1, other seeds",
                     lambda: len(set(texts(1.0, [seed + k for k in range(1, repeats + 1)]))) > 1)
    temperature_honoured = None if same is None or differ is None else (differ or not same)
    seed_honoured = same if temperature_honoured else None
    return result(endpoint_ok=True, json_schema_ok=json_schema_ok, deterministic=t0,
                  temperature_honoured=temperature_honoured, seed_honoured=seed_honoured, error="; ".join(errors))


def _mark(value: bool | None) -> str:
    return "?" if value is None else ("yes" if value else "NO")


def format_probes(results: list[ProbeResult]) -> str:
    header = ["model", "provider", "listed", "chat", "json_schema", "t=0 deterministic", "temperature", "seed",
              "calls", "tokens"]
    rows = [[r.model, r.provider, _mark(r.listed), _mark(r.endpoint_ok), _mark(r.json_schema_ok),
             _mark(r.deterministic), _mark(r.temperature_honoured), _mark(r.seed_honoured), str(r.calls),
             str(r.prompt_tokens + r.completion_tokens)] for r in results]
    widths = [max(len(x) for x in col) for col in zip(header, *rows)]
    lines = ["  ".join(x.ljust(w) for x, w in zip(line, widths)) for line in [header, *rows]]
    lines += [f"{r.model}: {r.error}" for r in results if r.error]
    return "\n".join(lines)
