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
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from typing import Any  # noqa: E402

from crewai import Agent, BaseLLM, Crew, Process, Task  # noqa: E402
from crewai.tasks.conditional_task import ConditionalTask  # noqa: E402
from crewai.tasks.task_output import TaskOutput  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from hcai import runlog  # noqa: E402
from hcai.llm import LLMClient, LLMRequest, LLMResponse, Usage, add_usage  # noqa: E402

SCREENER_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
    "additionalProperties": False,
}

EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}


RUBRIC_POINTS = 10  # each criterion is scored 0..RUBRIC_POINTS


def rubric_schema(n_criteria: int) -> dict[str, Any]:
    """C2/C3 Evaluator: one score per job criterion (numbered 1..n), each with the CV text quoted as evidence."""
    item = {
        "type": "object",
        "properties": {
            "criterion": {"type": "integer", "enum": list(range(1, n_criteria + 1))},
            "score": {"type": "integer", "minimum": 0, "maximum": RUBRIC_POINTS},
            "quote": {"type": "string"},
        },
        "required": ["criterion", "score", "quote"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "criteria": {"type": "array", "items": item, "minItems": n_criteria, "maxItems": n_criteria},
            "rationale": {"type": "string"},
        },
        "required": ["criteria", "rationale"],
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
    usage: Usage  # summed over every attempt
    cached: bool
    error: str = ""
    attempts: int = 0  # LLM calls made for this agent, retries included


Validator = Callable[[Any], str]  # parsed output -> error message, or "" when valid


@dataclass(frozen=True)
class Retry:
    """Retry policy for an agent's LLM call.

    A malformed answer (unparseable, or failing the agent's validation) is asked again with a note naming
    the problem appended to the prompt; the note gives the retry its own cache key, so a rerun replays the
    same attempts. A call that raises is repeated as is, after `backoff` seconds (doubled each time).

    A transient failure (`ProviderError.transient`: HTTP 5xx, a timeout, a dropped connection) waits
    `transient_backoff[k]` seconds instead before its (k+1)-th repeat (the last value once they run out), and
    may be repeated up to `max(retries, len(transient_backoff))` times, so a short outage is ridden out. The
    requests are the same either way: only the waits and the number of repeats of a raised call differ.
    """

    retries: int = 2
    backoff: float = 0.0
    sleep: Callable[[float], None] = time.sleep
    transient_backoff: tuple[float, ...] = ()


def _is_transient(error: BaseException) -> bool:
    return bool(getattr(error, "transient", False))


class ClientLLM(BaseLLM):
    """CrewAI LLM adapter: sends CrewAI's messages through an `LLMClient` with a fixed response schema.

    Each `call` validates the answer and retries it under the `Retry` policy, so the crew only ever sees
    the last attempt's text.
    """

    _client: Any = PrivateAttr()
    _schema: dict[str, Any] = PrivateAttr()
    _temperature: float = PrivateAttr()
    _validate: Validator = PrivateAttr()
    _retry: Retry = PrivateAttr()
    _responses: list[LLMResponse] = PrivateAttr(default_factory=list)
    _attempts: int = PrivateAttr(default=0)
    _last_error: str = PrivateAttr(default="")
    _agent: str = PrivateAttr(default="")

    def __init__(self, client: LLMClient, *, model: str, temperature: float, seed: int | None,
                 schema: dict[str, Any], validate: Validator, retry: Retry, agent: str = ""):
        super().__init__(model=model, temperature=temperature, seed=seed)
        self._client = client
        self._schema = schema
        self._temperature = temperature
        self._validate = validate
        self._retry = retry
        self._responses = []
        self._attempts = 0
        self._last_error = ""
        self._agent = agent  # its name in the run logs (`hcai.runlog`); not part of any request

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
             from_task: Any = None, from_agent: Any = None, response_model: Any = None, **kwargs: Any) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        base = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        note = ""
        failures = 0  # raised calls so far (the doubling `backoff`)
        transient = 0  # transient ones among them (`transient_backoff`)
        schedule = self._retry.transient_backoff
        attempt = 0
        while True:
            attempt += 1
            request = self._request(base, note)
            self._attempts += 1
            logged = runlog.Attempt(self._agent or "agent", attempt, request)
            try:
                response = self._client.complete(request)
            except Exception as e:  # noqa: BLE001 - provider/network failure: retried, then the crew fails
                self._last_error = f"call failed: {type(e).__name__}: {e}"
                if schedule and _is_transient(e):
                    if attempt > max(self._retry.retries, len(schedule)):
                        logged.failed(e)
                        raise
                    wait = schedule[min(transient, len(schedule) - 1)]
                    logged.failed(e, retry=True, wait=wait)
                    self._retry.sleep(wait)
                    transient += 1
                else:
                    if attempt > self._retry.retries:
                        logged.failed(e)
                        raise
                    wait = self._retry.backoff * 2 ** failures if self._retry.backoff else 0.0
                    logged.failed(e, retry=True, wait=wait)
                    if self._retry.backoff:
                        self._retry.sleep(wait)
                failures += 1
                continue
            except BaseException as e:  # a clean stop (ProviderExhausted) or an interrupt: logged, then raised
                logged.failed(e)
                raise
            self._responses.append(response)
            try:
                self._last_error = self._validate(response.parsed)
            except Exception as e:  # noqa: BLE001 - an answer the validator chokes on is malformed
                self._last_error = f"invalid output: {type(e).__name__}: {e}"
            final = not self._last_error or attempt > self._retry.retries
            logged.answered(response, error=self._last_error, retry=not final)
            if final:
                break
            note = (f"\n\n(Attempt {attempt + 1}. The previous answer was rejected: {self._last_error}. "
                    "Respond only with JSON matching the required schema.)")
        return self._responses[-1].text if self._responses else ""

    def _request(self, base: list[dict[str, str]], note: str) -> LLMRequest:
        messages = [dict(m) for m in base]
        if note:
            messages[-1]["content"] += note
        return LLMRequest(model=self.model, messages=messages, temperature=self._temperature, seed=self.seed,
                          response_schema=self._schema)

    def supports_function_calling(self) -> bool:
        return False

    def get_input_tokens_count(self, messages: Any, tools: Any = None, response_model: Any = None) -> int:
        """CrewAI's context-window preflight; the base version warns on every call. Real counts come from the API."""
        text = messages if isinstance(messages, str) else "".join(str(m.get("content", "")) for m in messages)
        return len(text) // 4 + 1

    @property
    def called(self) -> bool:
        """Whether the crew has called this agent's LLM."""
        return self._attempts > 0

    def output(self, error: str = "") -> AgentOutput:
        """This agent's result: the last answer if it is valid, else `parsed=None` and the error."""
        usage = add_usage([r.usage for r in self._responses])
        cached = bool(self._responses) and all(r.cached for r in self._responses)
        last = self._responses[-1] if self._responses else None
        if not error:
            error = "no response" if not self._attempts else self._last_error
        if error:
            snippet = f": {last.text[:200]!r}" if last is not None else ""
            return AgentOutput(parsed=None, usage=usage, cached=cached, error=error + snippet, attempts=self._attempts)
        assert last is not None
        return AgentOutput(parsed=last.parsed, usage=usage, cached=cached, attempts=self._attempts)


def _agent(role: str, goal: str, backstory: str, llm: ClientLLM) -> Agent:
    return Agent(role=role, goal=goal, backstory=backstory, llm=llm, allow_delegation=False, verbose=False,
                 max_retry_limit=0)


def _check_screening(p: Any) -> str:
    if not isinstance(p, dict) or not isinstance(p.get("passed"), bool) or not isinstance(p.get("reason"), str):
        return "unparseable Screener output"
    return ""


def _check_evaluation(p: Any) -> str:
    if not isinstance(p, dict) or type(p.get("score")) is not int or not isinstance(p.get("rationale"), str):
        return "unparseable Evaluator output"
    if not 0 <= p["score"] <= 100:
        return f"score out of range: {p['score']}"
    return ""


def _rubric_checker(n: int) -> Validator:
    def check(p: Any) -> str:
        if not isinstance(p, dict) or not isinstance(p.get("criteria"), list) or not isinstance(p.get("rationale"), str):
            return "unparseable rubric output"
        items = p["criteria"]
        if not all(isinstance(c, dict) and type(c.get("criterion")) is int and type(c.get("score")) is int
                   and isinstance(c.get("quote"), str) for c in items):
            return "unparseable rubric criterion"
        if sorted(c["criterion"] for c in items) != list(range(1, n + 1)):
            return f"criteria are not exactly 1..{n}: {[c['criterion'] for c in items]}"
        for c in items:
            if not 0 <= c["score"] <= RUBRIC_POINTS:
                return f"criterion {c['criterion']} score out of range: {c['score']}"
            if c["score"] > 0 and not c["quote"].strip():
                return f"criterion {c['criterion']} has a score but no quoted evidence"
        return ""

    return check


def rubric_result(parsed: dict[str, Any], criteria: list[str]) -> dict[str, Any]:
    """A validated rubric answer in criterion order, with the criterion texts and the 0-100 total.

    The total is computed here (sum of the criterion scores over the maximum, rounded half to even),
    not asked of the model, so it always follows the rubric.
    """
    by_id = {c["criterion"]: c for c in parsed["criteria"]}
    items: list[dict[str, Any]] = [dict(criterion=text, score=by_id[i]["score"], quote=by_id[i]["quote"])
                                   for i, text in enumerate(criteria, start=1)]
    total = round(100 * sum(int(c["score"]) for c in items) / (RUBRIC_POINTS * len(items)))
    return dict(score=total, rationale=parsed["rationale"], criteria=items)


def _passed(output: TaskOutput) -> bool:
    """ConditionalTask gate: the Evaluator runs only when the Screener returned a valid pass."""
    try:
        p = json.loads(output.raw)
    except json.JSONDecodeError:
        return False
    return _check_screening(p) == "" and p["passed"] is True


def screen_and_evaluate(client: LLMClient, *, model: str, job_ad: str, cv: str, temperature: float,
                        seed: int | None, retry: Retry = Retry(),
                        criteria: list[str] | None = None) -> tuple[AgentOutput, AgentOutput | None]:
    """Run the Screener, then the Evaluator only on a CV the Screener passed (`None` otherwise).

    With `criteria` (conditions C2/C3) the Evaluator is the rubric Evaluator: it scores each criterion
    0..RUBRIC_POINTS and quotes the CV text behind each score; its parsed output is `rubric_result`
    (`score` is the 0-100 total, `criteria` the per-criterion scores and quotes).
    """
    screener_llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=SCREENER_SCHEMA,
                             validate=_check_screening, retry=retry, agent="screener")
    if criteria:
        evaluator_llm = ClientLLM(client, model=model, temperature=temperature, seed=seed,
                                  schema=rubric_schema(len(criteria)), validate=_rubric_checker(len(criteria)),
                                  retry=retry, agent="evaluator")
    else:
        evaluator_llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=EVALUATOR_SCHEMA,
                                  validate=_check_evaluation, retry=retry, agent="evaluator")
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
    if criteria:
        listed = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))
        evaluate_description = (
            f"Score the candidate's CV on each job criterion below separately, from 0 to {RUBRIC_POINTS}. For "
            "every criterion, quote verbatim the CV text that supports the score; a criterion the CV gives no "
            "evidence for scores 0 with an empty quote. Give a short overall rationale."
            f"\n\n## Criteria\n{listed}\n\n{documents}"
        )
        evaluate_output = ('JSON: {"criteria": [{"criterion": number, "score": 0-' f"{RUBRIC_POINTS}"
                           ', "quote": "verbatim CV text"}, ...], "rationale": "..."}')
    else:
        evaluate_description = (
            "Score how well the candidate's CV fits the job ad from 0 to 100 and give a short rationale."
            f"\n\n{documents}"
        )
        evaluate_output = 'JSON: {"score": 0-100, "rationale": "..."}'
    evaluate = ConditionalTask(
        description=evaluate_description,
        expected_output=evaluate_output,
        agent=evaluator,
        context=[],  # the Evaluator judges the CV itself, not the Screener's reason
        condition=_passed,
    )
    crew = Crew(agents=[screener, evaluator], tasks=[screen, evaluate], process=Process.sequential, verbose=False)
    try:
        crew.kickoff()
    except Exception as e:  # noqa: BLE001 - any crew failure is recorded as missing data, not a crash
        failed = f"crew failed: {type(e).__name__}: {e}"
        screening = screener_llm.output(error="" if screener_llm.called else failed)
        return screening, (evaluator_llm.output(error=failed) if evaluator_llm.called else None)
    screening = screener_llm.output()
    if not evaluator_llm.called:
        return screening, None
    evaluation = evaluator_llm.output()
    if criteria and evaluation.parsed is not None:
        evaluation = replace(evaluation, parsed=rubric_result(evaluation.parsed, criteria))
    return screening, evaluation


def rank(client: LLMClient, *, model: str, job_ad: str, cvs: list[str], temperature: float,
         seed: int | None, retry: Retry = Retry()) -> AgentOutput:
    """Rank the CVs, shown in the given order as Candidate 1..n. `parsed["ranking"]` lists numbers, best first."""
    n = len(cvs)
    def check_ranking(p: Any) -> str:
        if not isinstance(p, dict) or not isinstance(p.get("ranking"), list) or not isinstance(p.get("rationale"), str):
            return "unparseable Ranker output"
        if any(type(x) is not int for x in p["ranking"]) or sorted(p["ranking"]) != list(range(1, n + 1)):
            return f"ranking is not a permutation of 1..{n}: {p['ranking']}"
        return ""

    llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=ranker_schema(n),
                    validate=check_ranking, retry=retry, agent="ranker")
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

    try:
        Crew(agents=[ranker], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    except Exception as e:  # noqa: BLE001 - recorded as missing data, not a crash
        return llm.output(error=f"crew failed: {type(e).__name__}: {e}")
    return llm.output()
