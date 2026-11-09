"""The hiring crew: Screener -> Evaluator -> Ranker as CrewAI agents.

Every LLM call the crew makes goes through our `LLMClient` (and so through the request cache and the
fake client in tests) via the `ClientLLM` adapter. Each agent has a fixed response schema; its output
is validated here and returned as an `AgentOutput`, with `parsed=None` and an error when malformed.
"""

from __future__ import annotations

import os

# Before importing CrewAI: no telemetry or tracing leaves the machine, and CrewAI's local kickoff
# store uses one app-data folder for this project instead of one per working-directory name.
for _key, _value in {
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_STORAGE_DIR": "hcai",
}.items():
    os.environ.setdefault(_key, _value)

import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

from crewai import Agent, BaseLLM, Crew, Process, Task  # noqa: E402
from crewai.tasks.conditional_task import ConditionalTask  # noqa: E402
from crewai.tasks.task_output import TaskOutput  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from hcai.evaluator import SCHEMA as EVALUATOR_SCHEMA  # noqa: E402
from hcai.llm import LLMClient, LLMRequest, LLMResponse, Usage  # noqa: E402

SCREENER_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
    "additionalProperties": False,
}


def ranker_schema(n_candidates: int) -> dict[str, Any]:
    """The ranking is a list of candidate numbers (1..n), best first."""
    return {
        "type": "object",
        "properties": {
            "ranking": {"type": "array", "items": {"type": "integer", "enum": list(range(1, n_candidates + 1))}},
            "rationale": {"type": "string"},
        },
        "required": ["ranking", "rationale"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class AgentOutput:
    parsed: dict[str, Any] | None  # validated output; None when missing
    usage: Usage
    cached: bool
    error: str = ""


class ClientLLM(BaseLLM):
    """CrewAI LLM adapter: sends CrewAI's messages through an `LLMClient` with a fixed response schema."""

    _client: Any = PrivateAttr()
    _schema: dict[str, Any] = PrivateAttr()
    _temperature: float = PrivateAttr()
    _responses: list[LLMResponse] = PrivateAttr(default_factory=list)

    def __init__(self, client: LLMClient, *, model: str, temperature: float, seed: int | None, schema: dict[str, Any]):
        super().__init__(model=model, temperature=temperature, seed=seed)
        self._client = client
        self._schema = schema
        self._temperature = temperature
        self._responses = []

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
             from_task: Any = None, from_agent: Any = None, response_model: Any = None, **kwargs: Any) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        request = LLMRequest(
            model=self.model,
            messages=[{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
            temperature=self._temperature,
            seed=self.seed,
            response_schema=self._schema,
        )
        response = self._client.complete(request)
        self._responses.append(response)
        return response.text

    def supports_function_calling(self) -> bool:
        return False

    def get_input_tokens_count(self, messages: Any, tools: Any = None, response_model: Any = None) -> int:
        """CrewAI's context-window preflight; the base version warns on every call. Real counts come from the API."""
        text = messages if isinstance(messages, str) else "".join(str(m.get("content", "")) for m in messages)
        return len(text) // 4 + 1

    def output(self, validate: Any, error: str = "") -> AgentOutput:
        """This agent's result: `validate(parsed)` returns an error message, or "" when the output is valid."""
        usage = Usage(
            prompt_tokens=sum(r.usage.prompt_tokens for r in self._responses),
            completion_tokens=sum(r.usage.completion_tokens for r in self._responses),
            cost=sum(r.usage.cost for r in self._responses),
        )
        cached = bool(self._responses) and all(r.cached for r in self._responses)
        last = self._responses[-1] if self._responses else None
        if not error:
            error = "no response" if last is None else validate(last.parsed)
        if error:
            snippet = f": {last.text[:200]!r}" if last is not None else ""
            return AgentOutput(parsed=None, usage=usage, cached=cached, error=error + snippet)
        assert last is not None
        return AgentOutput(parsed=last.parsed, usage=usage, cached=cached)


def _agent(role: str, goal: str, backstory: str, llm: ClientLLM) -> Agent:
    return Agent(role=role, goal=goal, backstory=backstory, llm=llm, allow_delegation=False, verbose=False,
                 max_retry_limit=0)


def _check_screening(p: Any) -> str:
    if not isinstance(p, dict) or not isinstance(p.get("passed"), bool) or not isinstance(p.get("reason"), str):
        return "unparseable Screener output"
    return ""


def _check_evaluation(p: Any) -> str:
    if not isinstance(p, dict) or not isinstance(p.get("score"), int) or not isinstance(p.get("rationale"), str):
        return "unparseable Evaluator output"
    if not 0 <= p["score"] <= 100:
        return f"score out of range: {p['score']}"
    return ""


def _passed(output: TaskOutput) -> bool:
    """ConditionalTask gate: the Evaluator runs only when the Screener returned a valid pass."""
    try:
        p = json.loads(output.raw)
    except json.JSONDecodeError:
        return False
    return _check_screening(p) == "" and p["passed"] is True


def screen_and_evaluate(client: LLMClient, *, model: str, job_ad: str, cv: str, temperature: float,
                        seed: int | None) -> tuple[AgentOutput, AgentOutput | None]:
    """Run the Screener, then the Evaluator only on a CV the Screener passed (`None` otherwise)."""
    screener_llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=SCREENER_SCHEMA)
    evaluator_llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=EVALUATOR_SCHEMA)
    screener = _agent(
        "Screener", "Decide whether each candidate meets every mandatory requirement of the job ad.",
        "You screen CVs for a recruitment team. You check only the mandatory requirements.", screener_llm,
    )
    evaluator = _agent(
        "Evaluator", "Score how well each candidate fits the job ad on a 0-100 scale.",
        "You assess CVs for a recruitment team against the mandatory and desirable requirements.", evaluator_llm,
    )
    documents = f"## Job ad\n{job_ad}\n\n## CV\n{cv}"
    screen = Task(
        description=(
            "Decide whether the candidate meets ALL mandatory requirements of the job ad (pass) or not "
            f"(reject), and give a one-sentence reason.\n\n{documents}"
        ),
        expected_output='JSON: {"passed": true|false, "reason": "..."}',
        agent=screener,
    )
    evaluate = ConditionalTask(
        description=(
            "Score how well the candidate's CV fits the job ad from 0 to 100 and give a short rationale."
            f"\n\n{documents}"
        ),
        expected_output='JSON: {"score": 0-100, "rationale": "..."}',
        agent=evaluator,
        context=[],  # the Evaluator judges the CV itself, not the Screener's reason
        condition=_passed,
    )
    crew = Crew(agents=[screener, evaluator], tasks=[screen, evaluate], process=Process.sequential, verbose=False)
    try:
        crew.kickoff()
    except Exception as e:  # noqa: BLE001 - any crew failure is recorded as missing data, not a crash
        failed = f"crew failed: {type(e).__name__}: {e}"
        screening = screener_llm.output(_check_screening, error="" if screener_llm._responses else failed)
        return screening, (evaluator_llm.output(_check_evaluation, error=failed) if evaluator_llm._responses else None)
    screening = screener_llm.output(_check_screening)
    if not evaluator_llm._responses:
        return screening, None
    return screening, evaluator_llm.output(_check_evaluation)


def rank(client: LLMClient, *, model: str, job_ad: str, cvs: list[str], temperature: float,
         seed: int | None) -> AgentOutput:
    """Rank the CVs, shown in the given order as Candidate 1..n. `parsed["ranking"]` lists numbers, best first."""
    n = len(cvs)
    llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=ranker_schema(n))
    ranker = _agent(
        "Ranker", "Order a pool of screened-in candidates from best to worst fit and shortlist the top 3.",
        "You build shortlists for a recruitment team.", llm,
    )
    candidates = "\n\n".join(f"### Candidate {i}\n{cv}" for i, cv in enumerate(cvs, start=1))
    task = Task(
        description=(
            f"Rank all {n} candidates from best to worst fit for the job ad, by candidate number. "
            "The first three form the shortlist. Give a short rationale."
            f"\n\n## Job ad\n{job_ad}\n\n## Candidates\n{candidates}"
        ),
        expected_output='JSON: {"ranking": [candidate numbers, best first], "rationale": "..."}',
        agent=ranker,
    )

    def check(p: Any) -> str:
        if not isinstance(p, dict) or not isinstance(p.get("ranking"), list) or not isinstance(p.get("rationale"), str):
            return "unparseable Ranker output"
        if sorted(p["ranking"]) != list(range(1, n + 1)):
            return f"ranking is not a permutation of 1..{n}: {p['ranking']}"
        return ""

    try:
        Crew(agents=[ranker], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    except Exception as e:  # noqa: BLE001
        return llm.output(check, error=f"crew failed: {type(e).__name__}: {e}")
    return llm.output(check)
