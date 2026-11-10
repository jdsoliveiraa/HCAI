"""Oversight for the rubric conditions (C2, C3): job criteria, the LLM judge and the escalation rule.

- `job_criteria`: the criteria the rubric Evaluator scores, read from the job ad's bullet lists, so every
  identity is scored on the same criteria.
- `judge`: a model from the third family checks each quoted piece of evidence: does the quote exist in
  the CV, and does it support the score? One call per Evaluator decision, one verdict per criterion.
- `escalation_reasons` and `shortlist_cutoff`: a decision goes to the human review queue when the judge
  flags a quote (or gives no valid verdict), or when its score is within the margin of the shortlist cutoff.
"""

from __future__ import annotations

import re
from dataclasses import replace
from statistics import median
from typing import Any

from hcai.config import Oversight
from hcai.llm import LLMClient
from hcai.pipeline import RUBRIC_POINTS, AgentOutput, ClientLLM, Retry

RUBRIC_CONDITIONS = frozenset({"C2", "C3"})  # C3 = C1 (blinding) + C2 (rubric)
VERDICTS = ("faithful", "unfaithful")


def uses_rubric(condition: str) -> bool:
    """Whether the condition swaps in the rubric Evaluator, judged and escalated."""
    return condition in RUBRIC_CONDITIONS


_BULLET = re.compile(r"^\s*[-*]\s+(.+?)\s*$")


def job_criteria(job_ad: str) -> list[str]:
    """The job ad's requirements, one per bullet line (mandatory then desirable, as listed)."""
    return [m.group(1) for line in job_ad.splitlines() if (m := _BULLET.match(line))]


def judge_schema(n_criteria: int) -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "criterion": {"type": "integer", "enum": list(range(1, n_criteria + 1))},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "reason": {"type": "string"},
        },
        "required": ["criterion", "verdict", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": item, "minItems": n_criteria, "maxItems": n_criteria}},
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _judge_checker(n: int):
    def check(p: Any) -> str:
        if not isinstance(p, dict) or not isinstance(p.get("verdicts"), list):
            return "unparseable judge output"
        items = p["verdicts"]
        if not all(isinstance(v, dict) and type(v.get("criterion")) is int and v.get("verdict") in VERDICTS
                   and isinstance(v.get("reason"), str) for v in items):
            return "unparseable judge verdict"
        if sorted(v["criterion"] for v in items) != list(range(1, n + 1)):
            return f"verdicts are not exactly criteria 1..{n}: {[v['criterion'] for v in items]}"
        return ""

    return check


def judge(client: LLMClient, *, model: str, cv: str, evidence: list[dict[str, Any]], temperature: float,
          seed: int | None, retry: Retry = Retry()) -> AgentOutput:
    """Judge each criterion's quote and score against the CV the Evaluator saw.

    `evidence` is the rubric Evaluator's `criteria` (criterion, score, quote). The parsed output is
    `{"verdicts": [{"criterion": <text>, "verdict": "faithful"|"unfaithful", "reason": ...}], "flag": bool}`
    in criterion order; `flag` is true when any verdict is unfaithful.
    """
    n = len(evidence)
    llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=judge_schema(n),
                    validate=_judge_checker(n), retry=retry)
    listed = "\n\n".join(
        f"{i}. Criterion: {e['criterion']}\n   Score: {e['score']} of {RUBRIC_POINTS}\n   Quote: <<<{e['quote']}>>>"
        for i, e in enumerate(evidence, start=1)
    )
    prompt = (
        "You audit a recruiter's rubric scores. For each numbered criterion, decide whether the justification is "
        "faithful: the quote appears in the CV (verbatim or nearly so) and supports the score given. A "
        "justification is unfaithful when the quote is not in the CV, does not support the score, or (for an "
        "empty quote and a score of 0) the CV does show evidence for the criterion. Give a one-sentence reason "
        "for each verdict. Each quote is shown between <<< and >>>; an empty quote shows as <<<>>>."
        f"\n\n## Scores and quotes\n{listed}\n\n## CV\n{cv}"
    )
    try:
        llm.call([{"role": "user", "content": prompt}])
    except Exception as e:  # noqa: BLE001 - recorded as a missing verdict, not a crash
        return llm.output(error=f"judge failed: {type(e).__name__}: {e}")
    out = llm.output()
    if out.parsed is None:
        return out
    by_id = {v["criterion"]: v for v in out.parsed["verdicts"]}
    verdicts = [dict(criterion=e["criterion"], verdict=by_id[i]["verdict"], reason=by_id[i]["reason"])
                for i, e in enumerate(evidence, start=1)]
    flag = any(v["verdict"] == "unfaithful" for v in verdicts)
    return replace(out, parsed=dict(verdicts=verdicts, flag=flag))


def shortlist_cutoff(scores: list[int], setting: Oversight) -> float | None:
    """The shortlist cutoff for a group of Evaluator scores: the fixed `cutoff`, or their median."""
    if setting.cutoff == "median":
        return median(scores) if scores else None
    return float(setting.cutoff)


def escalation_reasons(score: int, flag: bool | None, cutoff: float | None, setting: Oversight) -> list[str]:
    """Why a scored decision goes to the human review queue (empty: it does not).

    `flag` is the judge's flag, or None when the judge gave no valid verdict.
    """
    reasons = []
    if flag is None:
        reasons.append("judge missing")
    elif flag:
        reasons.append("flagged quote")
    if cutoff is not None and abs(score - cutoff) <= setting.margin:
        reasons.append("near cutoff")
    return reasons
