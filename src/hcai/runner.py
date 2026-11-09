"""Experiment runner: expands the design into jobs, runs the agents, writes the tidy results table."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from string import Template

from hcai.config import Design, Identity
from hcai.evaluator import MalformedOutput, evaluate
from hcai.llm import CachedClient, LLMClient


@dataclass(frozen=True)
class ResultRow:
    base_cv: str
    identity: str
    gender: str
    origin: str
    name: str
    language: str
    condition: str
    model: str
    agent: str
    score: int | None
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    status: str  # "ok" or "missing"
    error: str


def render_cv(template: str, identity: Identity, name: str) -> str:
    return Template(template).substitute(name=name, nationality=identity.nationality)


def _base_cvs(design: Design, role: str, quality: str, language: str) -> list[tuple[str, str]]:
    folder = design.corpus_dir / role / quality
    files = sorted(folder.glob(f"*.{language}.md"))[: design.base_cvs_per_role]
    return [(f"{role}/{quality}/{f.name.removesuffix(f'.{language}.md')}", f.read_text()) for f in files]


def run_experiment(design: Design, client: LLMClient, cache_dir: str | Path) -> list[ResultRow]:
    cached = CachedClient(client, cache_dir)
    rows: list[ResultRow] = []
    for role in design.roles:
        for language in design.languages:
            job_ad = (design.corpus_dir / role / f"job_ad.{language}.md").read_text()
            for quality in design.quality_levels:
                for base_id, template in _base_cvs(design, role, quality, language):
                    for identity in design.identities:
                        name = identity.names[0]
                        cv = render_cv(template, identity, name)
                        for condition in design.conditions:
                            for model in design.models:
                                rows.append(_run_evaluator(cached, design, job_ad, cv, base_id, identity, name, language, condition, model))
    return rows


def _run_evaluator(client, design, job_ad, cv, base_id, identity, name, language, condition, model) -> ResultRow:
    key = dict(base_cv=base_id, identity=identity.id, gender=identity.gender, origin=identity.origin, name=name,
               language=language, condition=condition, model=model, agent="evaluator")
    try:
        ev = evaluate(client, model=model, job_ad=job_ad, cv=cv, temperature=design.temperature, seed=design.seed)
    except MalformedOutput as e:
        return ResultRow(**key, score=None, rationale="", prompt_tokens=0, completion_tokens=0, cost=0.0, status="missing", error=str(e))
    return ResultRow(**key, score=ev.score, rationale=ev.rationale, prompt_tokens=ev.usage.prompt_tokens,
                     completion_tokens=ev.usage.completion_tokens, cost=ev.usage.cost, status="ok", error="")


_INT_FIELDS = {"prompt_tokens", "completion_tokens"}


def write_results(rows: list[ResultRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(ResultRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_results(path: str | Path) -> list[ResultRow]:
    with Path(path).open(newline="") as f:
        return [_parse_row(raw) for raw in csv.DictReader(f)]


def _parse_row(raw: dict[str, str]) -> ResultRow:
    values: dict = dict(raw)
    values["score"] = int(raw["score"]) if raw["score"] else None
    for k in _INT_FIELDS:
        values[k] = int(raw[k])
    values["cost"] = float(raw["cost"])
    return ResultRow(**values)
