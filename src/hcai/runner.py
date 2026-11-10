"""Experiment runner: expands the design into CVs and Ranker pools, runs the crew, writes the tidy results table."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from itertools import zip_longest
from pathlib import Path
from random import Random
from typing import Any

from hcai.blinding import BLINDING_CONDITIONS, blind_variant, proxy_leak_summary, proxy_leaks
from hcai.config import Design
from hcai.llm import CachedClient, LLMClient, MeteredClient
from hcai.oversight import escalation_reasons, job_criteria, judge, shortlist_cutoff, uses_rubric
from hcai.pipeline import AgentOutput, Retry, rank, screen_and_evaluate
from hcai.stability import stability_report
from hcai.variants import Variant, build_variants


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
    # "screener", "evaluator", "ranker", "blinder" (C1/C3; its rationale is the blinded CV), or "judge"
    # (C2/C3: the oversight judge)
    agent: str
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
    # Rubric conditions (C2, C3), see `hcai.oversight`:
    evidence: str = ""  # Evaluator rows: JSON list of {criterion, score, quote}; `score` is their 0-100 total
    judge_verdicts: str = ""  # judge rows, copied onto their Evaluator row: JSON list of {criterion, verdict, reason}
    judge_flag: bool | None = None  # judge and Evaluator rows: any unfaithful justification (None: no verdict)
    escalated: bool | None = None  # Evaluator rows: sent to the human review queue
    # Evaluator rows: "flagged quote", "judge missing" and/or "near cutoff", or "evaluator missing"; "; "-joined
    escalation_reason: str = ""
    judge_model: str = ""  # judge rows: the model that judged (`model` stays the pipeline model it judged)
    # A Ranker call covers a whole pool: its tokens and cost sit on the input_position == 1 row only.


@dataclass(frozen=True)
class Pool:
    id: str
    kind: str  # "equal" or "mixed"
    members: list[Variant]


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
        identities = [i.id for i in design.identities]
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
    proxy_leaks: list[dict[str, Any]] = field(default_factory=list)  # C1: what survives blinding (summary)


def execute(design: Design, client: LLMClient, out_dir: str | Path, cache_dir: str | Path) -> Run:
    """Run the design and write its outputs to `out_dir`.

    `results.csv` is the results table and `run_summary.json` its token, call and cost totals. With a
    stability sample (`[run] repeat_fraction`), `repeats.csv` holds the re-runs and `stability.json`
    their agreement with the original answers. With condition C1 (or C3), `proxy_leaks.csv` holds what
    survived blinding per blinded CV and `proxy_leaks.json` its summary per model x condition x language.
    With a rubric condition (C2, C3), `review_queue.csv` holds the escalated Evaluator rows, counted under
    `oversight` in the totals.
    """
    out_dir = Path(out_dir)
    meter = MeteredClient(client)
    rows, repeats = _run(design, meter, cache_dir)
    totals = run_totals(rows, repeats, meter)
    if design.pilot_of is not None:
        totals["pilot"] = _pilot_estimate(design, design.pilot_of, rows)
    report = stability_report(rows, repeats)
    write_results(rows, out_dir / "results.csv")
    if any(uses_rubric(c) for c in design.conditions):
        write_results([r for r in rows if r.escalated], out_dir / "review_queue.csv")
    (out_dir / "run_summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    if repeats:
        write_results(repeats, out_dir / "repeats.csv")
        (out_dir / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    leaks = proxy_leaks(design, rows)
    leak_summary = proxy_leak_summary(leaks)
    if leaks:
        _write_table([asdict(x) for x in leaks], out_dir / "proxy_leaks.csv")
        (out_dir / "proxy_leaks.json").write_text(json.dumps(leak_summary, indent=2) + "\n")
    return Run(rows=rows, repeats=repeats, totals=totals, stability=report, proxy_leaks=leak_summary)


def _write_table(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _pilot_estimate(pilot: Design, full: Design, rows: list[ResultRow]) -> dict[str, float]:
    """Scale the pilot's cost up to the full design: Screener and Evaluator by CV variants, Ranker by ranked CVs.

    Rough: it assumes the full run screens in as many CVs as the pilot did and costs the same per CV. The
    stability sample is added as its expected share (`repeat_fraction x repeats`) of the scaled cost, not
    scaled from the pilot's own few, randomly drawn repeats.
    """
    cv_scale = full.size / pilot.size if pilot.size else 0.0
    ranker_scale = full.ranked_slots / pilot.ranked_slots if pilot.ranked_slots else 0.0
    crew = sum(r.cost for r in rows if r.agent != "ranker")
    ranker = sum(r.cost for r in rows if r.agent == "ranker")
    scaled = crew * cv_scale + ranker * ranker_scale
    repeat_share = full.run.repeat_fraction * full.run.repeats
    return dict(cv_scale=round(cv_scale, 4), ranker_scale=round(ranker_scale, 4),
                estimated_full_cost=round(scaled * (1 + repeat_share), 4))


def run_totals(rows: list[ResultRow], repeats: list[ResultRow], meter: MeteredClient) -> dict[str, Any]:
    """Token, call and cost totals of a run, overall and per model and agent, stability sample included.

    - `attempts`, `prompt_tokens`, `completion_tokens`, `cost_total`: what the outputs took to produce,
      retries included (cache hits count their original tokens and cost).
    - `calls`, `cost_spent`: what this run sent to the provider and paid (measured below the cache);
      `cache_hits` = attempts - calls.
    - `rows` and `status` count the results table; `repeat_sample` tallies the stability sample on its own.
    - Judge rows are billed to their `judge_model`, not the pipeline model they judged.
    - `oversight` (rubric conditions only): per condition and model, the Evaluator decisions (screened in),
      how many the judge gave a verdict on, flagged, were near the cutoff, had no verdict or no valid
      Evaluator answer, and were escalated to the human review queue.
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
    models = list(dict.fromkeys(_billed_model(r) for r in every))
    by_model = {}
    for m in models:
        subset = [r for r in every if _billed_model(r) == m]
        by_model[m] = {**tally(subset), **spent([m], subset)}
    totals = dict(
        rows=len(rows), status=status, **tally(every), **spent(models, every), by_model=by_model,
        by_agent={a: tally([r for r in every if r.agent == a]) for a in dict.fromkeys(r.agent for r in every)},
        repeat_sample=tally(repeats),
    )
    if any(r.escalated is not None for r in rows):
        totals["oversight"] = oversight_counts(rows)
    return totals


def _billed_model(r: ResultRow) -> str:
    return r.judge_model or r.model


def oversight_counts(rows: list[ResultRow]) -> dict[str, dict[str, dict[str, int]]]:
    """Per condition and model: the escalation tally of the rubric Evaluator decisions (see `run_totals`)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for r in rows:
        if r.agent != "evaluator" or r.escalated is None:
            continue
        c = out.setdefault(r.condition, {}).setdefault(r.model, dict(
            decisions=0, judged=0, flagged=0, near_cutoff=0, judge_missing=0, evaluator_missing=0, escalated=0))
        c["decisions"] += 1
        c["judged"] += r.judge_flag is not None
        c["flagged"] += r.judge_flag is True
        c["near_cutoff"] += "near cutoff" in r.escalation_reason
        c["judge_missing"] += "judge missing" in r.escalation_reason
        c["evaluator_missing"] += r.status == "missing"
        c["escalated"] += r.escalated
    return out


def run_experiment(design: Design, client: LLMClient, cache_dir: str | Path) -> list[ResultRow]:
    """The results table of the design, without the stability sample (see `execute`)."""
    return _run(replace(design, run=replace(design.run, repeat_fraction=0.0)), client, cache_dir)[0]


def _run(design: Design, client: LLMClient, cache_dir: str | Path) -> tuple[list[ResultRow], list[ResultRow]]:
    """Run the crew on every variant of the design: Screener and Evaluator per CV, then the Ranker per pool,
    plus the stability sample's repeats.

    All variants are rendered and pass the demographic-only diff check, and all job ads are read, before
    any LLM call: a bad template, variant, identity or missing file raises without spending anything.
    """
    variants = build_variants(design)
    job_ads = {  # read up front, so a missing job ad also fails before any LLM call
        (role, language): (design.corpus_dir / role / f"job_ad.{language}.md").read_text()
        for role in design.roles for language in design.languages
    }
    criteria: dict[tuple[str, str], list[str]] = {}
    if any(uses_rubric(c) for c in design.conditions):  # also checked before any LLM call
        if design.judge_model is None:
            raise ValueError("a rubric condition (C2/C3) needs a judge: set [oversight] judge_model or [generator] model")
        for key, ad in job_ads.items():
            criteria[key] = job_criteria(ad)
            if not criteria[key]:
                raise ValueError(f"job ad {key} lists no criteria (bullet lines) for the rubric Evaluator")
    cached = CachedClient(client, cache_dir)
    # Each repeat is the identical request, cached in its own namespace: paid once, replayed on reruns.
    repeat_clients = tuple(CachedClient(client, Path(cache_dir) / "repeats" / f"r{k}")
                           for k in range(1, design.run.repeats + 1)) if design.run.repeat_fraction > 0 else ()
    rows: list[ResultRow] = []
    repeats: list[ResultRow] = []
    for role in design.roles:
        for language in design.languages:
            job_ad = job_ads[(role, language)]
            group = [v for v in variants if v.role == role and v.language == language]
            pools = build_pools(design, role, language, group)
            for condition in design.conditions:
                for model in design.models:
                    rubric = criteria.get((role, language)) if uses_rubric(condition) else None
                    ctx = _Context(cached, design, job_ad, language, condition, model, rubric)
                    crew_group, crew_pools, unblinded = group, pools, {}
                    if condition in BLINDING_CONDITIONS:  # C1, C3: the crew sees the Blinder's text only
                        blinder_rows, crew_group, unblinded = _blind_group(ctx, group)
                        rows += blinder_rows
                        by_id = {v.id: v for v in crew_group}
                        crew_pools = [replace(p, members=[by_id[m.id] for m in p.members]) for p in pools]
                    passed: dict[str, bool | None] = {}
                    block: list[ResultRow] = []
                    for v in crew_group:  # under C1/C3, `v.text` is the blinded text the crew and the judge see
                        if v.id in unblinded:
                            block += _not_run(ctx, v, unblinded[v.id])
                            passed[v.id] = None
                            continue
                        screener_row, evaluator_row = _screen_and_evaluate(ctx, v)
                        block += [screener_row, evaluator_row]
                        if rubric and evaluator_row.status == "ok":
                            block.append(_judge(ctx, v, evaluator_row))
                        passed[v.id] = screener_row.screen_pass
                        if ctx.sampled(v.id):
                            for k, again in enumerate(repeat_clients, start=1):
                                repeats += [replace(r, repeat=k) for r in _screen_and_evaluate(ctx.using(again), v)]
                    rows += _escalate(block, design) if rubric else block
                    for pool in crew_pools:
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
    criteria: list[str] | None = None  # rubric conditions: the job criteria the Evaluator scores

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
        return dict(base_cv=v.base_cv, identity=v.identity, gender=v.gender, origin=v.origin,
                    name=v.name, language=v.language, condition=self.condition, model=self.model, agent=agent)


def _usage(out: AgentOutput) -> dict[str, Any]:
    return dict(prompt_tokens=out.usage.prompt_tokens, completion_tokens=out.usage.completion_tokens,
                cost=out.usage.cost, cached=out.cached, attempts=out.attempts)


_NO_CALL: dict[str, Any] = dict(prompt_tokens=0, completion_tokens=0, cost=0.0, cached=False, attempts=0)


def _screen_and_evaluate(ctx: _Context, v: Variant) -> tuple[ResultRow, ResultRow]:
    screening, evaluation = screen_and_evaluate(
        ctx.client, model=ctx.model, job_ad=ctx.job_ad, cv=v.text,
        temperature=ctx.design.temperature, seed=ctx.design.seed, retry=ctx.retry, criteria=ctx.criteria,
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
            evidence=json.dumps(e["criteria"], ensure_ascii=False) if e and "criteria" in e else "",
        )
    return screener_row, evaluator_row


def _judge(ctx: _Context, v: Variant, evaluator_row: ResultRow) -> ResultRow:
    """The judge's verdicts on a rubric Evaluator row's quotes, as a judge row (billed to the judge model)."""
    judge_model = ctx.design.judge_model
    assert judge_model is not None  # checked in `_run` before any call
    out = judge(ctx.client, model=judge_model, cv=v.text, evidence=json.loads(evaluator_row.evidence),
                temperature=ctx.design.temperature, seed=ctx.design.seed, retry=ctx.retry)
    j = out.parsed
    return ResultRow(
        **ctx.key(v, "judge"), score=None, rationale="", **_usage(out), status="ok" if j else "missing",
        error=out.error, judge_verdicts=json.dumps(j["verdicts"], ensure_ascii=False) if j else "",
        judge_flag=j["flag"] if j else None, judge_model=judge_model,
    )


def _escalate(block: list[ResultRow], design: Design) -> list[ResultRow]:
    """Apply the escalation rule to the Evaluator rows of one role x language x condition x model.

    The judge's flag and verdicts are copied onto its Evaluator row, so the review queue is self-contained;
    the cutoff is the fixed one or the median of the block's valid scores. An Evaluator that gave no valid
    answer for a screened-in CV is escalated too ("evaluator missing"); screened-out CVs are not decisions.
    """
    judged = {(r.base_cv, r.identity): r for r in block if r.agent == "judge"}
    scores = [r.score for r in block if r.agent == "evaluator" and r.status == "ok" and r.score is not None]
    cutoff = shortlist_cutoff(scores, design.oversight)
    out = []
    for r in block:
        if r.agent == "evaluator" and r.status == "ok" and r.score is not None:
            j = judged.get((r.base_cv, r.identity))
            flag = j.judge_flag if j else None
            reasons = escalation_reasons(r.score, flag, cutoff, design.oversight)
            r = replace(r, judge_flag=flag, judge_verdicts=j.judge_verdicts if j else "", escalated=bool(reasons),
                        escalation_reason="; ".join(reasons))
        elif r.agent == "evaluator" and r.status == "missing":
            r = replace(r, escalated=True, escalation_reason="evaluator missing")
        out.append(r)
    return out


def _blind_group(ctx: _Context, group: list[Variant]) -> tuple[list[ResultRow], list[Variant], dict[str, str]]:
    """C1: blind every CV of the group. Returns the Blinder rows, the variants with the blinded text (the
    original variant where blinding failed), and, per failed variant id, the error."""
    rows: list[ResultRow] = []
    blinded: list[Variant] = []
    failed: dict[str, str] = {}
    for v in group:
        out, text = blind_variant(ctx.client, ctx.design, v, model=ctx.model, retry=ctx.retry)
        rows.append(ResultRow(
            **ctx.key(v, "blinder"), score=None, rationale=out.parsed["blinded_cv"] if out.parsed else "", **_usage(out),
            status="ok" if text is not None else "missing", error=out.error,
        ))
        if text is None:
            failed[v.id] = f"blinding failed: {out.error}"
            blinded.append(v)
        else:
            blinded.append(replace(v, text=text))
    return rows, blinded, failed


def _not_run(ctx: _Context, v: Variant, error: str) -> list[ResultRow]:
    """Screener and Evaluator rows of a CV the crew could not be shown (its blinding failed): missing, no call."""
    return [ResultRow(**ctx.key(v, agent), score=None, rationale="", **_NO_CALL, status="missing", error=error)
            for agent in ("screener", "evaluator")]


def _rank_pool(ctx: _Context, pool: Pool, passed: dict[str, bool | None]) -> list[ResultRow]:
    """Rank the pool's screened-in CVs in a shuffled, recorded order; screened-out CVs get 'rejected' rows."""
    pool_cols: dict[str, Any] = dict(pool=pool.id, pool_kind=pool.kind)
    eligible = [v for v in pool.members if passed[v.id] is True]
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
        rejected = passed[v.id] is False
        rows.append(ResultRow(
            **ctx.key(v, "ranker"), score=None, rationale="", **_NO_CALL,
            status="rejected" if rejected else "missing", error="" if rejected else "no screening decision",
            **pool_cols, order_seed=seed,
        ))
    return rows


_INT_FIELDS = {"prompt_tokens", "completion_tokens", "attempts", "repeat"}
_OPTIONAL_INT_FIELDS = {"score", "input_position", "rank", "order_seed"}
_OPTIONAL_BOOL_FIELDS = {"screen_pass", "top3", "judge_flag", "escalated"}


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
        values[k] = int(raw.get(k) or 0)  # `attempts` and `repeat` are absent from tables written before #6
    values["cost"] = float(raw["cost"])
    values["cached"] = raw["cached"] == "True"
    return ResultRow(**values)
