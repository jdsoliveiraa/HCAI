"""The Evaluator agent: scores a CV against a job ad, 0-100, with a rationale."""

from __future__ import annotations

from dataclasses import dataclass

from hcai.llm import LLMClient, LLMRequest, Usage

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a recruitment Evaluator. Score how well the candidate's CV fits the job ad "
    "on a 0-100 scale and give a short rationale. Respond only with JSON matching the schema."
)


class MalformedOutput(ValueError):
    pass


@dataclass(frozen=True)
class Evaluation:
    score: int
    rationale: str
    usage: Usage
    cached: bool


def evaluate(client: LLMClient, *, model: str, job_ad: str, cv: str, temperature: float, seed: int | None) -> Evaluation:
    request = LLMRequest(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"## Job ad\n{job_ad}\n\n## CV\n{cv}"},
        ],
        temperature=temperature,
        seed=seed,
        response_schema=SCHEMA,
    )
    response = client.complete(request)
    p = response.parsed
    if not isinstance(p, dict) or not isinstance(p.get("score"), int) or not isinstance(p.get("rationale"), str):
        raise MalformedOutput(f"unparseable Evaluator output: {response.text[:200]!r}")
    if not 0 <= p["score"] <= 100:
        raise MalformedOutput(f"score out of range: {p['score']}")
    return Evaluation(score=p["score"], rationale=p["rationale"], usage=response.usage, cached=response.cached)
