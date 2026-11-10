"""Experiment runner: expands the design into CVs and Ranker pools, runs the crew, writes the tidy results table."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from itertools import zip_longest
from pathlib import Path
from random import Random
from string import Template
from typing import Any

from hcai.config import Design, Identity
from hcai.llm import CachedClient, LLMClient, MeteredClient
from hcai.pipeline import AgentOutput, Retry, rank, screen_and_evaluate
from hcai.stability import stability_report


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
    agent: str  # "screener", "evaluator" or "ranker"
    score: int | None
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    cost: float  # cost of the original call, also when this run served it from the cache
    cached: bool
    status: str  # "ok", "missing", or "rejected" (screened out upstream: not scored or ranked)
    error: str
    screen_pass: bool | None = None  # Screener rows
    pool: str = ""  # Ranker rows: the pool id
    pool_kind: str = ""  # Ranker rows: "equal" or "mixed" quality
    input_position: int | None = None  # Ranker rows: 1-based position of this CV in the Ranker prompt
    rank: int | None = None  # Ranker rows: 1 = best
    top3: bool | None = None  # Ranker rows: on the shortlist
    order_seed: int | None = None  # Ranker rows: seed of the pool's input-order shuffle
    attempts: int = 0  # LLM calls behind this row, retries included (on the row that carries the tokens)
    repeat: int = 0  # 0 in the results; k >= 1 for the k-th re-run of the call in the stability sample
    # A Ranker call covers a whole pool: its tokens and cost sit on the input_position == 1 row only.


@dataclass(frozen=True)
class Variant:
    """One rendered CV: a base CV with one identity and name."""

    base_cv: str
    quality: str
    identity: Identity
    name: str
    text: str


@dataclass(frozen=True)
class Pool:
    id: str
    kind: str  # "equal" or "mixed"
    members: list[Variant]


def render_cv(template: str, identity: Identity, name: str) -> str:
    return Template(template).substitute(name=name, nationality=identity.nationality)


def _base_cvs(design: Design, role: str, quality: str, language: str) -> list[tuple[str, str]]:
    folder = design.corpus_dir / role / quality
    files = sorted(folder.glob(f"*.{language}.md"))[: design.base_cvs_per_role]
    return [(f"{role}/{quality}/{f.name.removesuffix(f'.{language}.md')}", f.read_text()) for f in files]


def stable_seed(*parts: object) -> int:
    """A seed derived from its parts, identical across runs and machines (unlike `hash`)."""
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "big")


def build_pools(design: Design, role: str, language: str, variants: list[Variant]) -> list[Pool]:
    pools: list[Pool] = []
    by_base: dict[str, list[Variant]] = {}
    for v in variants:
        by_base.setdefault(v.base_cv, []).append(v)
    if design.pools.equal_quality:
        pools += [Pool(id=base, kind="equal", members=members) for base, members in by_base.items()]
    for k in range(1, design.pools.mixed_quality + 1):
        rng = Random(stable_seed(design.seed, "mixed", role, language, k))
        # Draw distinct base CVs round-robin over the quality levels, each with the next identity.
        per_quality = [[b for b, m in by_base.items() if m[0].quality == q] for q in design.quality_levels]
        for bases in per_quality:
            rng.shuffle(bases)
        drawn = [b for group in zip_longest(*per_quality) for b in group if b is not None][: design.pools.mixed_size]
        identities = list(design.identities)
        rng.shuffle(identities)
        members = [
            next(v for v in by_base[b] if v.identity == identities[i % len(identities)]) for i, b in enumerate(drawn)
        ]
        pools.append(Pool(id=f"{role}/mixed{k:02d}", kind="mixed", members=members))
    return pools


@dataclass(frozen=True)
class Run:
    rows: list[ResultRow]  # the results table
    repeats: list[ResultRow]  # stability sample: re-runs of sampled calls, `repeat` = 1..k
    totals: dict[str, Any]
    stability: list[dict[str, Any]]


def execute(design: Design, client: LLMClient, out_dir: str | Path, cache_dir: str | Path) -> Run:
    """Run the design and write its outputs to `out_dir`.

    `results.csv` is the results table and `run_summary.json` its token, call and cost totals. With a
    stability sample (`[run] repeat_fraction`), `repeats.csv` holds the re-runs and `stability.json`
    their agreement with the original answers.
    """
    out_dir = Path(out_dir)
    meter = MeteredClient(client)
    rows, repeats = _run(design, meter, cache_dir)
    totals = run_totals(rows, repeats, meter)
    if design.pilot_of is not None:
        totals["pilot"] = _pilot_estimate(design, design.pilot_of, rows + repeats)
    report = stability_report(rows, repeats)
    write_results(rows, out_dir / "results.csv")
    (out_dir / "run_summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    if repeats:
        write_results(repeats, out_dir / "repeats.csv")
        (out_dir / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    return Run(rows=rows, repeats=repeats, totals=totals, stability=report)


def _pilot_estimate(pilot: Design, full: Design, rows: list[ResultRow]) -> dict[str, float]:
    """Scale the pilot's cost up to the full design: Screener and Evaluator by CV variants, Ranker by ranked CVs.

    Rough: it assumes the full run screens in as many CVs as the pilot did and costs the same per CV.
    """
    cv_scale = full.size / pilot.size if pilot.size else 0.0
    ranker_scale = full.ranked_slots / pilot.ranked_slots if pilot.ranked_slots else 0.0
    crew = sum(r.cost for r in rows if r.agent != "ranker")
    ranker = sum(r.cost for r in rows if r.agent == "ranker")
    return dict(cv_scale=round(cv_scale, 4), ranker_scale=round(ranker_scale, 4),
                estimated_full_cost=round(crew * cv_scale + ranker * ranker_scale, 4))


def run_totals(rows: list[ResultRow], repeats: list[ResultRow], meter: MeteredClient) -> dict[str, Any]:
    """Token, call and cost totals of a run, overall and per model and agent, stability sample included.

    - `attempts`, `prompt_tokens`, `completion_tokens`, `cost_total`: what the outputs took to produce,
      retries included (cache hits count their original tokens and cost).
    - `calls`, `cost_spent`: what this run sent to the provider and paid (measured below the cache);
      `cache_hits` = attempts - calls.
    - `rows` and `status` count the results table; `repeat_sample` tallies the stability sample on its own.
    """

    def tally(subset: list[ResultRow]) -> dict[str, Any]:
        return dict(
            attempts=sum(r.attempts for r in subset),
            prompt_tokens=sum(r.prompt_tokens for r in subset),
            completion_tokens=sum(r.completion_tokens for r in subset),
            cost_total=round(sum(r.cost for r in subset), 6),
        )

    def spent(models: list[str], subset: list[ResultRow]) -> dict[str, Any]:
        calls = sum(meter.calls.get(m, 0) for m in models)
        return dict(calls=calls, cache_hits=sum(r.attempts for r in subset) - calls,
                    cost_spent=round(sum(meter.cost.get(m, 0.0) for m in models), 6))

    status: dict[str, int] = {}
    for r in rows:
        status[r.status] = status.get(r.status, 0) + 1
    every = rows + repeats
    models = list(dict.fromkeys(r.model for r in every))
    by_model = {}
    for m in models:
        subset = [r for r in every if r.model == m]
        by_model[m] = {**tally(subset), **spent([m], subset)}
    return dict(
        rows=len(rows), status=status, **tally(every), **spent(models, every), by_model=by_model,
        by_agent={a: tally([r for r in every if r.agent == a]) for a in dict.fromkeys(r.agent for r in every)},
        repeat_sample=tally(repeats),
    )


def run_experiment(design: Design, client: LLMClient, cache_dir: str | Path) -> list[ResultRow]:
    """The results table of the design, without the stability sample (see `execute`)."""
    return _run(replace(design, run=replace(design.run, repeat_fraction=0.0)), client, cache_dir)[0]


def _run(design: Design, client: LLMClient, cache_dir: str | Path) -> tuple[list[ResultRow], list[ResultRow]]:
    cached = CachedClient(client, cache_dir)
    # Each repeat is the identical request, cached in its own namespace: paid once, replayed on reruns.
    repeat_clients = tuple(CachedClient(client, Path(cache_dir) / "repeats" / f"r{k}")
                           for k in range(1, design.run.repeats + 1)) if design.run.repeat_fraction > 0 else ()
    rows: list[ResultRow] = []
    repeats: list[ResultRow] = []
    for role in design.roles:
        for language in design.languages:
            job_ad = (design.corpus_dir / role / f"job_ad.{language}.md").read_text()
            variants = [
                Variant(base_id, quality, identity, identity.names[0], render_cv(template, identity, identity.names[0]))
                for quality in design.quality_levels
                for base_id, template in _base_cvs(design, role, quality, language)
                for identity in design.identities
            ]
            pools = build_pools(design, role, language, variants)
            for condition in design.conditions:
                for model in design.models:
                    ctx = _Context(cached, design, job_ad, language, condition, model)
                    passed: dict[tuple[str, str], bool | None] = {}
                    for v in variants:
                        screener_row, evaluator_row = _screen_and_evaluate(ctx, v)
                        rows += [screener_row, evaluator_row]
                        passed[(v.base_cv, v.identity.id)] = screener_row.screen_pass
                        if ctx.sampled(v.base_cv, v.identity.id, v.name):
                            for k, again in enumerate(repeat_clients, start=1):
                                repeats += [replace(r, repeat=k) for r in _screen_and_evaluate(ctx.using(again), v)]
                    for pool in pools:
                        rows += _rank_pool(ctx, pool, passed)
                        if ctx.sampled(pool.id):
                            for k, again in enumerate(repeat_clients, start=1):
                                repeats += [replace(r, repeat=k) for r in _rank_pool(ctx.using(again), pool, passed)]
    return rows, repeats


@dataclass(frozen=True)
class _Context:
    client: LLMClient
    design: Design
    job_ad: str
    language: str
    condition: str
    model: str

    def using(self, client: LLMClient) -> _Context:
        return replace(self, client=client)

    def sampled(self, *item: str) -> bool:
        """Whether this CV variant or pool, under this language, condition and model, is in the stability sample."""
        fraction = self.design.run.repeat_fraction
        draw = stable_seed(self.design.seed, "repeat", *item, self.language, self.condition, self.model)
        return fraction > 0 and draw / 2**32 < fraction

    @property
    def retry(self) -> Retry:
        return Retry(retries=self.design.run.retries, backoff=self.design.run.retry_backoff)

    def key(self, v: Variant, agent: str) -> dict[str, Any]:
        return dict(base_cv=v.base_cv, identity=v.identity.id, gender=v.identity.gender, origin=v.identity.origin,
                    name=v.name, language=self.language, condition=self.condition, model=self.model, agent=agent)


def _usage(out: AgentOutput) -> dict[str, Any]:
    return dict(prompt_tokens=out.usage.prompt_tokens, completion_tokens=out.usage.completion_tokens,
                cost=out.usage.cost, cached=out.cached, attempts=out.attempts)


_NO_CALL: dict[str, Any] = dict(prompt_tokens=0, completion_tokens=0, cost=0.0, cached=False, attempts=0)


def _screen_and_evaluate(ctx: _Context, v: Variant) -> tuple[ResultRow, ResultRow]:
    screening, evaluation = screen_and_evaluate(
        ctx.client, model=ctx.model, job_ad=ctx.job_ad, cv=v.text,
        temperature=ctx.design.temperature, seed=ctx.design.seed, retry=ctx.retry,
    )
    s = screening.parsed
    screener_row = ResultRow(
        **ctx.key(v, "screener"), score=None, rationale=s["reason"] if s else "", **_usage(screening),
        status="ok" if s else "missing", error=screening.error, screen_pass=s["passed"] if s else None,
    )
    if evaluation is None:
        status, error = ("rejected", "") if s else ("missing", "no screening decision")
        evaluator_row = ResultRow(**ctx.key(v, "evaluator"), score=None, rationale="", **_NO_CALL, status=status, error=error)
    else:
        e = evaluation.parsed
        evaluator_row = ResultRow(
            **ctx.key(v, "evaluator"), score=e["score"] if e else None, rationale=e["rationale"] if e else "",
            **_usage(evaluation), status="ok" if e else "missing", error=evaluation.error,
        )
    return screener_row, evaluator_row


def _rank_pool(ctx: _Context, pool: Pool, passed: dict[tuple[str, str], bool | None]) -> list[ResultRow]:
    """Rank the pool's screened-in CVs in a shuffled, recorded order; screened-out CVs get 'rejected' rows."""
    pool_cols: dict[str, Any] = dict(pool=pool.id, pool_kind=pool.kind)
    eligible = [v for v in pool.members if passed[(v.base_cv, v.identity.id)] is True]
    seed = stable_seed(ctx.design.seed, pool.id, ctx.language, ctx.condition, ctx.model)
    order = list(eligible)
    Random(seed).shuffle(order)
    out = rank(ctx.client, model=ctx.model, job_ad=ctx.job_ad, cvs=[v.text for v in order],
               temperature=ctx.design.temperature, seed=ctx.design.seed, retry=ctx.retry) if order else None

    rows: list[ResultRow] = []
    ranks = {label: r for r, label in enumerate(out.parsed["ranking"], start=1)} if out and out.parsed else {}
    for position, v in enumerate(order, start=1):
        assert out is not None
        position_rank = ranks.get(position)
        rows.append(ResultRow(
            **ctx.key(v, "ranker"), score=None, rationale=out.parsed["rationale"] if out.parsed else "",
            **(_usage(out) if position == 1 else {**_NO_CALL, "cached": out.cached}),
            status="ok" if out.parsed else "missing", error=out.error, **pool_cols, input_position=position,
            rank=position_rank, top3=None if position_rank is None else position_rank <= 3, order_seed=seed,
        ))
    for v in pool.members:
        if v in eligible:
            continue
        rejected = passed[(v.base_cv, v.identity.id)] is False
        rows.append(ResultRow(
            **ctx.key(v, "ranker"), score=None, rationale="", **_NO_CALL,
            status="rejected" if rejected else "missing", error="" if rejected else "no screening decision",
            **pool_cols, order_seed=seed,
        ))
    return rows


_INT_FIELDS = {"prompt_tokens", "completion_tokens", "attempts", "repeat"}
_OPTIONAL_INT_FIELDS = {"score", "input_position", "rank", "order_seed"}
_OPTIONAL_BOOL_FIELDS = {"screen_pass", "top3"}


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
    for k in _OPTIONAL_INT_FIELDS:
        values[k] = int(raw[k]) if raw.get(k) else None
    for k in _OPTIONAL_BOOL_FIELDS:
        values[k] = raw[k] == "True" if raw.get(k) else None
    for k in _INT_FIELDS:
        values[k] = int(raw[k])
    values["cost"] = float(raw["cost"])
    values["cached"] = raw["cached"] == "True"
    return ResultRow(**values)
