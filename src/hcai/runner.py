"""Experiment runner: expands the design into CVs and Ranker pools, runs the crew, writes the tidy results table."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from itertools import zip_longest
from pathlib import Path
from random import Random
from collections.abc import Callable
from typing import Any

from hcai import runlog
from hcai.blinding import BLINDING_CONDITIONS, blind_variant, proxy_leak_summary, proxy_leaks
from hcai.concurrency import Task, run_tasks
from hcai.config import Design
from hcai.llm import (
    BillingGuard, BillingStop, CachedClient, FailureGuard, LLMClient, MeteredClient, ProviderExhausted, StopGate,
    TooManyFailures, stop_scope,
)
from hcai.oversight import escalation_reasons, job_criteria, judge, shortlist_cutoff, uses_rubric
from hcai.pipeline import AgentOutput, Retry, rank, screen_and_evaluate
from hcai.stability import stability_report
from hcai.variants import Variant, build_variants

logger = logging.getLogger("hcai.runner")

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
    # cost of the original call, also when this run served it from the cache; None (blank in the CSV) when the
    # provider reported no price (a subscription): the tokens are still counted
    cost: float | None
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
    # Evaluator rows: "flagged quote", "judge missing" and/or "near cutoff" ("; "-joined); or, for a CV with
    # no Evaluator score, "evaluator missing", "screener missing" or "blinding failed"
    escalation_reason: str = ""
    judge_model: str = ""  # judge rows: the model that judged (`model` stays the pipeline model it judged)
    # OpenCode Go: dollars this row's calls took from the Go allowance (also when served from the cache), and
    # what billed them ("go", or e.g. "balance" for Extra Usage; "+"-joined if the attempts differ). Blank when
    # the provider reported neither (another provider, or a cache entry from before these fields).
    allowance: float | None = None
    billing_source: str = ""
    # A Ranker call covers a whole pool: its tokens and cost sit on the input_position == 1 row only.
    # Not a column (never written or read): (cache key, served from the cache) of each answer behind the row, so a
    # concurrent run can set `cached` as the sequential run would (`_sequential_cache_flags`).
    calls: tuple[tuple[str, bool], ...] = field(default=(), compare=False, repr=False)


# The columns of the results tables: every `ResultRow` field but `calls`.
COLUMNS = [f.name for f in fields(ResultRow) if f.name != "calls"]


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


def execute(design: Design, client: LLMClient, out_dir: str | Path, cache_dir: str | Path,
            log: Callable[[str], None] | None = None, config_path: str | Path | None = None) -> Run:
    """Run the design and write its outputs to `out_dir`.

    `results.csv` is the results table and `run_summary.json` its token, call and cost totals. With a
    stability sample (`[run] repeat_fraction`), `repeats.csv` holds the re-runs and `stability.json`
    their agreement with the original answers. With condition C1 (or C3), `proxy_leaks.csv` holds what
    survived blinding per blinded CV and `proxy_leaks.json` its summary per model x condition x language.
    With a rubric condition (C2, C3), `review_queue.csv` holds the escalated Evaluator rows, counted under
    `oversight` in the totals.

    The tables of paid answers (`results.csv`, `review_queue.csv`, `repeats.csv`) are written as soon as the
    calls are done, before any summary is computed, so a failing summary step leaves them on disk. After each
    role x language x condition x model block, one progress line goes to `log` (the `hcai.runner` logger by
    default).

    Run logs (`hcai.runlog`): the start lists the config (path, sha256, git commit), corpus, models, providers,
    timeouts and guards; every agent attempt is one line in the log and one record appended to
    `out_dir/calls.jsonl`; every `[run] heartbeat_s` a heartbeat line says where the run is and rewrites
    `progress.json` (status "running"), which a stop turns into "stopped" and a finished run deletes.
    """
    out_dir = Path(out_dir)
    meter = MeteredClient(client)
    progress = out_dir / "progress.json"
    started = time.monotonic()
    say: Callable[[str], None] = log or logger.info
    warn: Callable[[str], None] = (lambda m: log("WARNING: " + m)) if log else logger.warning

    def on_block(i: int, n: int, label: str, rows: list[ResultRow], repeats: list[ResultRow]) -> None:
        say(_progress_line(i, n, label, rows, repeats, meter, time.monotonic() - started))

    track = runlog.RunLog(out_dir, heartbeat_s=design.run.heartbeat_s, stats=lambda: _stats(meter))
    _log_start(track, design, out_dir, cache_dir, config_path)
    failure: BaseException | None = None
    snapshot: dict[str, Any] = {}
    rows: list[ResultRow] = []
    repeats: list[ResultRow] = []
    _log_previous_stop(progress)
    with track:
        track.write_progress("running")
        try:
            rows, repeats = _run(design, meter, cache_dir, on_block=on_block, track=track)
        except BaseException as e:  # noqa: BLE001 - recorded below, after the heartbeat has stopped, then raised
            failure = e
            try:
                snapshot = track.snapshot()
            except Exception:  # noqa: BLE001 - the stop must still be recorded
                snapshot = {}
    if failure is not None:  # the heartbeat thread has stopped: nothing overwrites what is written here
        if isinstance(failure, ProviderExhausted):
            _record_stop(progress, failure, meter, cache_dir, snapshot)
            logger.error(f"run {track.run_id} stopped ({failure.kind}): {failure}; {track.rows_done} of "
                         f"{track.rows_total} planned rows done. Every completed call is cached: rerun the same "
                         f"command to resume (progress in {progress})")
        else:
            what = "interrupted" if isinstance(failure, KeyboardInterrupt) else "crashed"
            logger.error(f"run {track.run_id} {what}: {type(failure).__name__}: {failure}")
            logger.debug("traceback", exc_info=failure)  # the CLI prints it, or names the error, on stderr
            track.write_progress(what, snapshot, reason=f"{type(failure).__name__}: {failure}")
        raise failure
    progress.unlink(missing_ok=True)
    write_results(rows, out_dir / "results.csv")
    if any(uses_rubric(c) for c in design.conditions):
        write_results([r for r in rows if r.escalated], out_dir / "review_queue.csv")
    if repeats:
        write_results(repeats, out_dir / "repeats.csv")
    totals = run_totals(rows, repeats, meter, design)
    if design.pilot_of is not None:
        totals["pilot"] = _pilot_estimate(design, design.pilot_of, rows)
    (out_dir / "run_summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    _log_end(track, totals, out_dir, time.monotonic() - started)
    if totals["missing"]["above_target"]:
        warn(f"{totals['missing']['missing']} of {totals['missing']['rows']} rows are missing "
            f"({100 * totals['missing']['share']:.2f}%), above [run] missing_target = "
            f"{100 * totals['missing']['target']:.2f}%; failed calls are not cached, so a rerun asks them again")
    report = stability_report(rows, repeats)
    if repeats:
        (out_dir / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    leaks = proxy_leaks(design, rows)
    leak_summary = proxy_leak_summary(leaks)
    if leaks:
        _write_table([asdict(x) for x in leaks], out_dir / "proxy_leaks.csv")
        (out_dir / "proxy_leaks.json").write_text(json.dumps(leak_summary, indent=2) + "\n")
    return Run(rows=rows, repeats=repeats, totals=totals, stability=report, proxy_leaks=leak_summary)


def _log_previous_stop(progress: Path) -> None:
    """A rerun replaces the last run's `progress.json`: keep what stopped it in the log first."""
    try:
        old = json.loads(progress.read_text())
    except (OSError, ValueError):
        return
    if isinstance(old, dict) and old.get("status") != "running":
        logger.info(f"previous run {old.get('run_id', '?')} ended {old.get('status')}"
                    f"{' (' + str(old['stop']) + ')' if old.get('stop') else ''}: {str(old.get('reason', ''))[:300]}; "
                    f"{old.get('rows_done', '?')} of {old.get('rows_total', '?')} rows were done")


def _stats(meter: MeteredClient) -> dict[str, Any]:
    """This run's calls and spending so far, for the heartbeat (read from its thread: copies only)."""
    return dict(calls=sum(dict(meter.calls).values()), allowance_spent=_sum(dict(meter.allowance)),
                balance_spent=_sum(dict(meter.balance)), failed_calls=sum(dict(meter.failed).values()),
                timed_out_calls=sum(dict(meter.timed_out).values()))


def _corpus_version(design: Design) -> str:
    try:
        return str(json.loads((design.corpus_dir / "manifest.json").read_text())["version"])
    except (OSError, ValueError, KeyError, TypeError):
        return design.corpus_dir.name


def _log_start(track: runlog.RunLog, design: Design, out_dir: Path, cache_dir: str | Path,
               config_path: str | Path | None) -> None:
    """What the run is about to do, in the log: config, code, corpus, models, providers, timeouts and guards.

    Provider base URLs only: never a key, a header or the session id.
    """
    try:
        r = design.run
        logger.info(f"run {track.run_id} starting{' (pilot slice)' if design.pilot_of is not None else ''}: out "
                    f"{out_dir}, cache {cache_dir}; run.log and calls.jsonl are appended to")
        where = Path(config_path).resolve().parent if config_path is not None else Path.cwd()
        commit, dirty = runlog.git_state(where)
        code = (f"git {commit} ({'dirty: tracked files differ' if dirty else 'clean'})" if commit
                else "git commit unknown")
        if config_path is not None:
            logger.info(f"config {Path(config_path).resolve()} sha256 {runlog.file_sha256(config_path)}; {code}")
        else:
            logger.info(f"config: none given; {code}")
        logger.info(f"corpus {_corpus_version(design)} at {design.corpus_dir}; roles {design.roles}, languages "
                    f"{design.languages}, conditions {design.conditions}, {len(design.identities)} identities, "
                    f"temperature {design.temperature}, seed {design.seed}")
        judge = design.judge_model if any(uses_rubric(c) for c in design.conditions) else None
        models = list(dict.fromkeys(design.models + ([judge] if judge else [])))
        providers = "; ".join(f"{m} -> {p.name} {p.base_url} ({p.billing})"
                              for m in models for p in [design.provider_for(m)])
        logger.info(f"models {design.models}" + (f", judge {judge}" if judge else "") + f"; providers: {providers}")
        logger.info(f"timeouts: request_timeout {r.request_timeout:g}s, connect "
                    f"{min(20.0, float(r.request_timeout)):g}s; retries {r.retries}, retry_backoff "
                    f"{r.retry_backoff:g}s, transient_backoff {list(r.transient_backoff)}; rate limits: backoff "
                    f"{r.rate_limit_backoff:g}s up to {r.rate_limit_max_backoff:g}s, max wait {r.rate_limit_max_wait:g}s")
        logger.info(f"guards: allowed_billing {list(r.allowed_billing)}, max_consecutive_failures "
                    f"{r.max_consecutive_failures}, missing_target {r.missing_target:g}; stability sample "
                    f"{r.repeat_fraction:g} x {r.repeats}; heartbeat every {r.heartbeat_s:g}s")
    except Exception:  # noqa: BLE001 - the start lines must never stop a run
        logger.debug("could not log the run's start", exc_info=True)


def _log_end(track: runlog.RunLog, totals: dict[str, Any], out_dir: Path, elapsed: float) -> None:
    try:
        m = totals["missing"]
        h, rest = divmod(int(elapsed), 3600)
        status = ", ".join(f"{k} {v}" for k, v in sorted(totals["status"].items()))
        logger.info(f"run {track.run_id} done in {h}:{rest // 60:02d}:{rest % 60:02d}: {totals['rows']} rows "
                    f"({status}); missing {m['missing']} of {m['rows']} due ({100 * m['share']:.2f}%, target "
                    f"{100 * m['target']:.2f}%); {totals['calls']} calls, {totals['cache_hits']} cache hits, "
                    f"{totals['failed_calls']} failed, {totals['timed_out_calls']} timed out; allowance "
                    f"${totals['allowance_spent']:.4f}, balance ${totals['balance_spent']:.4f} this run -> "
                    f"{out_dir / 'results.csv'}")
    except Exception:  # noqa: BLE001
        logger.debug("could not log the run's end", exc_info=True)


def _sum(by_model: dict[str, float]) -> float:
    return round(sum(by_model.values()), 6)


def _sources(meter: MeteredClient) -> dict[str, int]:
    out: dict[str, int] = {}
    for by_source in meter.billing.values():
        for source, n in by_source.items():
            out[source] = out.get(source, 0) + n
    return dict(sorted(out.items()))


def _progress_line(i: int, n: int, label: str, rows: list[ResultRow], repeats: list[ResultRow],
                   meter: MeteredClient, elapsed: float) -> str:
    """One stderr line after a role x language x condition x model block: where the run is and what it spent."""
    calls = sum(meter.calls.values())
    attempts = sum(r.attempts for r in rows + repeats)
    missing = sum(r.status == "missing" for r in rows)
    sources = ", ".join(f"{s} x{k}" for s, k in _sources(meter).items()) or "none"
    h, rest = divmod(int(elapsed), 3600)
    return (f"[block {i}/{n}] {label}: {calls} calls, {attempts - calls} cache hits, {missing} of {len(rows)} rows "
            f"missing so far; allowance ${_sum(meter.allowance):.4f}, balance ${_sum(meter.balance):.4f}, "
            f"billing {sources}; elapsed {h}:{rest // 60:02d}:{rest % 60:02d}")


def _record_stop(path: Path, error: ProviderExhausted, meter: MeteredClient, cache_dir: str | Path,
                 snapshot: dict[str, Any] | None = None) -> None:
    """Write `progress.json` for a run the provider stopped: what it did before stopping and how to resume.

    No results table is written for a stopped run; every call it completed is in the cache, so rerunning the
    same command replays them for free and carries on from the first call not yet made. `stop` says what
    stopped it: "limit" (a rate or usage limit, no credits, a refused key), "billing" (a call billed outside
    `[run] allowed_billing`) or "failures" (`[run] max_consecutive_failures` provider errors in a row).
    `snapshot` adds the run log's last progress numbers (rows done, cache hits, block, ...).
    """
    extra: dict[str, Any] = {}
    if isinstance(error, BillingStop):
        extra = dict(model=error.model, billing_source=error.billing_source)
    elif isinstance(error, TooManyFailures):
        extra = dict(model=error.model, failures_in_a_row=error.failures)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = dict(
        status="stopped", stop=error.kind, **extra, reason=str(error),
        stopped_at=datetime.now(UTC).isoformat(timespec="seconds"),
        calls_this_run=dict(meter.calls), billing_sources_this_run=_sources(meter),
        allowance_spent=_sum(meter.allowance), balance_spent=_sum(meter.balance),
        failed_calls=dict(meter.failed), timed_out_calls=dict(meter.timed_out), cache_dir=str(cache_dir),
        resume="rerun the same command: completed calls are replayed from the cache, none is repeated",
    )
    data = {**{k: v for k, v in (snapshot or {}).items() if k not in stop}, **stop}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def _write_table(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _pilot_estimate(pilot: Design, full: Design, rows: list[ResultRow]) -> dict[str, Any]:
    """Scale the pilot's cost up to the full design, per agent and condition.

    Per condition, the Screener, Evaluator, Blinder and judge scale by CV variants and the Ranker by ranked
    CVs. A full-design condition the pilot left out borrows the pilot's mean per-CV cost of that agent
    under the conditions where it does the same work: the Blinder under a blinding condition (C1, C3), the
    judge and the rubric Evaluator under a rubric one (C2, C3), the plain Evaluator under C0 or C1, the
    Screener and Ranker under any. An agent with no such condition in the pilot is listed in `unestimated`
    ("blinder C1") and left out of `estimated_full_cost`, which is then a lower bound.

    The prompt and completion tokens are scaled the same way (`estimated_full_tokens`): what a subscription's
    usage budget is spent on. When the provider gave no price for some pilot calls, `estimated_full_cost` is
    None (unknown). On OpenCode Go, `estimated_full_allowance` scales the dollars taken from the Go allowance
    the same way; it is None unless every pilot call that answered reported its allowance.

    Rough: it assumes the full run screens in as many CVs as the pilot did and costs the same per CV. The
    stability sample is added as its expected share (`repeat_fraction x repeats`) of the scaled cost of the
    agents it repeats (Screener, Evaluator, Ranker), not scaled from the pilot's own few, randomly drawn repeats.
    """
    cv_scale = full.size / pilot.size if pilot.size else 0.0
    ranker_scale = full.ranked_slots / pilot.ranked_slots if pilot.ranked_slots else 0.0

    def units(design: Design, agent: str) -> float:  # per condition
        total = design.ranked_slots if agent == "ranker" else design.size
        return total / len(design.conditions)

    def scale(value: Callable[[ResultRow], float]) -> tuple[dict[str, float], list[str]]:
        totals: dict[tuple[str, str], float] = {}
        for r in rows:
            totals[(r.agent, r.condition)] = totals.get((r.agent, r.condition), 0.0) + value(r)
        by_agent: dict[str, float] = {}
        unestimated: list[str] = []
        for agent in _ESTIMATED_AGENTS:
            for condition in full.conditions:
                if not _runs_agent(agent, condition):
                    continue
                like = [condition] if condition in pilot.conditions else [
                    c for c in pilot.conditions if _runs_agent(agent, c) and _same_work(agent, c, condition)]
                pilot_units = units(pilot, agent)
                if not like or not pilot_units:
                    unestimated.append(f"{agent} {condition}")
                    continue
                per_unit = sum(totals.get((agent, c), 0.0) for c in like) / len(like) / pilot_units
                by_agent[agent] = by_agent.get(agent, 0.0) + per_unit * units(full, agent)
        return by_agent, unestimated

    repeat_share = full.run.repeat_fraction * full.run.repeats

    def total(by_agent: dict[str, float]) -> float:
        return sum(c * (1 + repeat_share if a in _REPEATED_AGENTS else 1) for a, c in by_agent.items())

    by_agent, unestimated = scale(lambda r: r.cost or 0.0)
    tokens = {kind: round(total(scale(lambda r, k=kind: float(getattr(r, f"{k}_tokens")))[0]))
              for kind in ("prompt", "completion")}
    unpriced = any(r.cost is None for r in rows)
    # a row with tokens but no allowance was billed without one (another provider, an older cache entry); a row
    # with neither had no answer at all (every attempt failed), so it took nothing
    no_allowance = (not any(r.allowance is not None for r in rows)
                    or any(r.allowance is None and (r.prompt_tokens or r.completion_tokens) for r in rows))
    allowance = None if no_allowance else round(total(scale(lambda r: r.allowance or 0.0)[0]), 4)
    return dict(cv_scale=round(cv_scale, 4), ranker_scale=round(ranker_scale, 4),
                # None when the pilot's provider reported no price for some calls (a subscription)
                estimated_full_cost=None if unpriced else round(total(by_agent), 4),
                # None unless every pilot call reported what it took from the Go allowance (OpenCode Go)
                estimated_full_allowance=allowance,
                by_agent={} if unpriced else {a: round(c, 4) for a, c in by_agent.items()},
                estimated_full_tokens=tokens, unestimated=sorted(unestimated))


_ESTIMATED_AGENTS = ("screener", "evaluator", "ranker", "blinder", "judge")
_REPEATED_AGENTS = frozenset({"screener", "evaluator", "ranker"})  # what the stability sample re-runs


def _runs_agent(agent: str, condition: str) -> bool:
    if agent == "blinder":
        return condition in BLINDING_CONDITIONS
    if agent == "judge":
        return uses_rubric(condition)
    return True


def _same_work(agent: str, a: str, b: str) -> bool:
    """Whether `agent` does the same work, at about the same cost per CV, under conditions `a` and `b`."""
    return agent != "evaluator" or uses_rubric(a) == uses_rubric(b)


def run_totals(rows: list[ResultRow], repeats: list[ResultRow], meter: MeteredClient,
               design: Design | None = None) -> dict[str, Any]:
    """Token, call and cost totals of a run, overall and per model and agent, stability sample included.

    - `attempts`, `prompt_tokens`, `completion_tokens`, `cost_total`: what the outputs took to produce,
      retries included (cache hits count their original tokens and cost).
    - `calls`, `cost_spent`: what this run sent to the provider and paid (measured below the cache);
      `cache_hits` = attempts - calls.
    - Costs sum the known prices only: `unpriced_attempts` and `unpriced_calls` count the calls the provider
      gave no price for (their cost is unknown, e.g. on a subscription; their tokens are counted). With the
      design, each model also names its `provider` and `billing` ("metered" or "subscription").
    - OpenCode Go: `allowance_total` and `allowance_spent` are the dollars taken from the Go allowance by the
      outputs and by this run's calls; `balance_spent` what this run's calls billed elsewhere (Extra Usage);
      `billing_sources` this run's calls per billing source; `not_go_calls` counts this run's calls not
      billed to "go", and `not_go_attempts` the attempts of output rows with any such call (a row mixing
      "go" and another source counts all its attempts). A call naming no billing source ("unknown") is
      counted at its allowance and flagged. Rows and calls without the
      `usage.opencode` block (another provider, older cache entries) add nothing to these.
    - `missing_session_errors`: calls the provider refused for a missing or rejected session header (HTTP 400
      MissingSessionID; a warning is printed the first time). Their rows are missing like any failed call.
    - `failed_calls`, `timed_out_calls`: this run's calls that raised a provider error, and those among them that
      got no answer within `[run] request_timeout`.
    - `allowed_billing` (with the design): the billing sources calls may be paid from; `not_allowed_billing_calls`
      counts this run's calls billed elsewhere (the one that stopped a run is in its `progress.json` instead).
    - `missing`: the share of missing rows (of those not `rejected`) overall and per model x condition x language (`by_cell`), the
      `target` (`[run] missing_target`) and whether the overall share is `above_target`.
    - `rows` and `status` count the results table; `repeat_sample` tallies the stability sample on its own.
    - Judge rows are billed to their `judge_model`, not the pipeline model they judged.
    - `oversight` (rubric conditions only): per condition and model, the Evaluator decisions (every CV not
      screened out), how many the judge gave a verdict on, flagged, were near the cutoff, had no verdict,
      had no Evaluator score (`evaluator_missing`: the Evaluator, Screener or Blinder failed; the reason says
      which), and were escalated to the human review queue.
    """

    def tally(subset: list[ResultRow]) -> dict[str, Any]:
        return dict(
            attempts=sum(r.attempts for r in subset),
            prompt_tokens=sum(r.prompt_tokens for r in subset),
            completion_tokens=sum(r.completion_tokens for r in subset),
            cost_total=round(sum(r.cost or 0.0 for r in subset), 6),
            unpriced_attempts=sum(r.attempts for r in subset if r.cost is None),
            allowance_total=round(sum(r.allowance or 0.0 for r in subset), 6),
            not_go_attempts=sum(r.attempts for r in subset if r.billing_source not in ("", "go")),
        )

    def spent(models: list[str], subset: list[ResultRow]) -> dict[str, Any]:
        calls = sum(meter.calls.get(m, 0) for m in models)
        sources: dict[str, int] = {}
        for m in models:
            for source, n in meter.billing.get(m, {}).items():
                sources[source] = sources.get(source, 0) + n
        return dict(calls=calls, cache_hits=sum(r.attempts for r in subset) - calls,
                    cost_spent=round(sum(meter.cost.get(m, 0.0) for m in models), 6),
                    unpriced_calls=sum(meter.unpriced.get(m, 0) for m in models),
                    allowance_spent=round(sum(meter.allowance.get(m, 0.0) for m in models), 6),
                    balance_spent=round(sum(meter.balance.get(m, 0.0) for m in models), 6),
                    billing_sources=sources,
                    not_go_calls=sum(n for s, n in sources.items() if s != "go"),
                    not_allowed_billing_calls=sum(n for s, n in sources.items() if allowed and s not in allowed),
                    missing_session_errors=sum(meter.missing_session.get(m, 0) for m in models),
                    failed_calls=sum(meter.failed.get(m, 0) for m in models),
                    timed_out_calls=sum(meter.timed_out.get(m, 0) for m in models))

    allowed = design.run.allowed_billing if design is not None else ()

    status: dict[str, int] = {}
    for r in rows:
        status[r.status] = status.get(r.status, 0) + 1
    every = rows + repeats
    models = list(dict.fromkeys(_billed_model(r) for r in every))
    by_model = {}
    for m in models:
        subset = [r for r in every if _billed_model(r) == m]
        by_model[m] = {**tally(subset), **spent([m], subset)}
        if design is not None:
            provider = design.provider_for(m)
            by_model[m].update(provider=provider.name, billing=provider.billing)
    totals: dict[str, Any] = dict(
        rows=len(rows), status=status, **tally(every), **spent(models, every), by_model=by_model,
        by_agent={a: tally([r for r in every if r.agent == a]) for a in dict.fromkeys(r.agent for r in every)},
        repeat_sample=tally(repeats),
    )
    if design is not None:
        totals["allowed_billing"] = list(allowed)
    totals["missing"] = missing_counts(rows, design.run.missing_target if design is not None else 0.02)
    if any(r.escalated is not None for r in rows):
        totals["oversight"] = oversight_counts(rows)
    return totals


def missing_counts(rows: list[ResultRow], target: float) -> dict[str, Any]:
    """The missing rows of the results table: overall and per model x condition x language ("model/C0/en").

    `rows` counts the rows due an answer (all but `rejected`, the CVs screened out upstream), and `share` is
    `missing / rows`.
    """
    def share(subset: list[ResultRow]) -> dict[str, Any]:
        missing = sum(r.status == "missing" for r in subset)
        due = sum(r.status != "rejected" for r in subset)  # screened-out rows were never due an answer
        return dict(rows=due, missing=missing, share=round(missing / due, 6) if due else 0.0)

    cells: dict[str, list[ResultRow]] = {}
    for r in rows:
        cells.setdefault(f"{r.model}/{r.condition}/{r.language}", []).append(r)
    overall = share(rows)
    return dict(**overall, target=target, above_target=overall["share"] > target,
                by_cell={k: share(v) for k, v in cells.items()})


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


def _run(design: Design, client: LLMClient, cache_dir: str | Path,
         on_block: Callable[[int, int, str, list[ResultRow], list[ResultRow]], None] | None = None,
         track: runlog.RunLog | None = None) -> tuple[list[ResultRow], list[ResultRow]]:
    """Run the crew on every variant of the design: Screener and Evaluator per CV, then the Ranker per pool,
    plus the stability sample's repeats.

    All variants are rendered and pass the demographic-only diff check, and all job ads are read, before
    any LLM call: a bad template, variant, identity or missing file raises without spending anything.

    Below the cache, `FailureGuard` stops the run after `[run] max_consecutive_failures` provider errors in a
    row on one model; above it, `BillingGuard` stops it on the first fresh answer billed outside
    `[run] allowed_billing` (that answer cached first). `on_block(i, n, label, rows, repeats)` is called after
    each role x language x condition x model block. `track` (the run log) is told the plan, each block, the CV
    or pool each call is for, and the rows as they are added.
    """
    track = track if track is not None else runlog.RunLog(None)
    variants = build_variants(design)
    job_ads = {  # read up front, so a missing job ad also fails before any LLM call
        (role, language): (design.corpus_dir / role / f"job_ad.{language}.md").read_text()
        for role in design.roles for language in design.languages
    }
    criteria: dict[tuple[str, str], list[str]] = {}
    if any(uses_rubric(c) for c in design.conditions):  # also checked before any LLM call
        if design.judge_model is None:
            raise ValueError("a rubric condition (C2/C3) needs a judge: set [oversight] judge_model or [generator] model")
        if design.judge_model in design.models:
            raise ValueError(f"the judge {design.judge_model!r} is an audited pipeline model; it must be a third family")
        for key, ad in job_ads.items():
            criteria[key] = job_criteria(ad)
            if not criteria[key]:
                raise ValueError(f"job ad {key} lists no criteria (bullet lines) for the rubric Evaluator")
    guarded = FailureGuard(client, design.run.max_consecutive_failures)
    allowed = design.run.allowed_billing
    # `StopGate` on top: once a worker stops the run (or Ctrl-C), the others make no new call.
    cached = StopGate(BillingGuard(CachedClient(guarded, cache_dir), allowed))
    # Each repeat is the identical request, cached in its own namespace: paid once, replayed on reruns.
    repeat_clients = tuple(StopGate(BillingGuard(CachedClient(guarded, Path(cache_dir) / "repeats" / f"r{k}"), allowed))
                           for k in range(1, design.run.repeats + 1)) if design.run.repeat_fraction > 0 else ()
    rows: list[ResultRow] = []
    repeats: list[ResultRow] = []
    blocks = len(design.roles) * len(design.languages) * len(design.conditions) * len(design.models)
    groups = {(role, language): [v for v in variants if v.role == role and v.language == language]
              for role in design.roles for language in design.languages}
    all_pools = {key: build_pools(design, key[0], key[1], group) for key, group in groups.items()}
    track.plan(sum(len(groups[key]) * _planned_per_cv(condition, criteria.get(key))
                   + sum(len(p.members) for p in all_pools[key])
                   for key in groups for condition in design.conditions for _ in design.models),
               blocks=blocks, variants=len(variants), pools=sum(len(p) for p in all_pools.values()))
    done = 0
    with stop_scope(threading.Event()) as stop:
        for role in design.roles:
            for language in design.languages:
                job_ad = job_ads[(role, language)]
                group = groups[(role, language)]
                pools = all_pools[(role, language)]
                for condition in design.conditions:
                    for model in design.models:
                        track.block(done + 1, blocks, role=role, language=language, condition=condition, model=model)
                        rubric = criteria.get((role, language)) if uses_rubric(condition) else None
                        ctx = _Context(cached, design, job_ad, language, condition, model, rubric, track)
                        block_rows, block_repeats = _run_block(ctx, group, pools, repeat_clients, stop)
                        rows += block_rows
                        repeats += block_repeats
                        done += 1
                        if on_block is not None:
                            on_block(done, blocks, f"{role}/{language}/{condition}/{model}", rows, repeats)
    return rows, repeats


def _run_block(ctx: _Context, group: list[Variant], pools: list[Pool], repeat_clients: tuple[LLMClient, ...],
               stop: threading.Event) -> tuple[list[ResultRow], list[ResultRow]]:
    """One role x language x condition x model block: its result rows and stability repeats, in sequential order.

    The work is split into tasks run on up to `[run] concurrency` workers (`hcai.concurrency.run_tasks`): per CV
    variant the Blinder (C1, C3), then the crew (Screener, Evaluator, judge, and the CV's repeats), and per Ranker
    pool the Ranker once every member's crew is done (it needs their screening decisions). Every task builds the
    same requests whatever runs beside it, and the rows are put together in the sequential order: the Blinder rows,
    the CV rows (escalated together under a rubric condition: the cutoff is the median of the whole block), then the
    pool rows. With concurrency 1 the tasks run in that order, as a plain loop.
    """
    track = ctx.track
    blinding = ctx.condition in BLINDING_CONDITIONS
    per_cv = _planned_per_cv(ctx.condition, ctx.criteria) - blinding
    crew_variant: dict[str, Variant] = {}  # under C1/C3, the variant with the blinded text the crew and judge see
    unblinded: dict[str, str] = {}  # per variant whose blinding failed, the error
    passed: dict[str, bool | None] = {}

    def blind(v: Variant) -> ResultRow:
        row, crew_v, error = _blind_one(ctx, v)
        crew_variant[v.id] = crew_v
        if error is not None:
            unblinded[v.id] = error
        return row

    def crew(v: Variant) -> tuple[list[ResultRow], list[ResultRow]]:
        v = crew_variant.get(v.id, v)
        reps: list[ResultRow] = []
        with track.item(**_cv_fields(v)):
            if v.id in unblinded:
                cv_rows = _not_run(ctx, v, unblinded[v.id])
                passed[v.id] = None
            else:
                screener_row, evaluator_row = _screen_and_evaluate(ctx, v)
                cv_rows = [screener_row, evaluator_row]
                if ctx.criteria and evaluator_row.status == "ok":
                    cv_rows.append(_judge(ctx, v, evaluator_row))
                passed[v.id] = screener_row.screen_pass
                if ctx.sampled(v.id):
                    for k, again in enumerate(repeat_clients, start=1):
                        with track.item(repeat=k):
                            reps += [replace(r, repeat=k) for r in _screen_and_evaluate(ctx.using(again), v)]
            track.done(cv_rows, per_cv)
        return cv_rows, reps

    def ranker(pool: Pool) -> tuple[list[ResultRow], list[ResultRow]]:
        if blinding:
            pool = replace(pool, members=[crew_variant[m.id] for m in pool.members])
        reps: list[ResultRow] = []
        with track.item(pool=pool.id, pool_kind=pool.kind):
            pool_rows = _rank_pool(ctx, pool, passed)
            track.done(pool_rows, len(pool.members))
            if ctx.sampled(pool.id):
                for k, again in enumerate(repeat_clients, start=1):
                    with track.item(repeat=k):
                        reps += [replace(r, repeat=k) for r in _rank_pool(ctx.using(again), pool, passed)]
        return pool_rows, reps

    # Ready Ranker pools first, then crews, then Blinders: a pool's long call starts as soon as it can.
    tasks: list[Task] = []
    blind_at: dict[str, int] = {}
    if blinding:
        for v in group:
            blind_at[v.id] = len(tasks)
            tasks.append(Task(partial(blind, v), priority=2))
    crew_at: dict[str, int] = {}
    for v in group:
        crew_at[v.id] = len(tasks)
        tasks.append(Task(partial(crew, v), deps=(blind_at[v.id],) if blinding else (), priority=1))
    pool_at: list[int] = []
    for pool in pools:
        pool_at.append(len(tasks))
        tasks.append(Task(partial(ranker, pool), deps=tuple(sorted({crew_at[m.id] for m in pool.members})),
                          priority=0))
    results = run_tasks(tasks, ctx.design.run.concurrency, stop)

    rows: list[ResultRow] = [results[blind_at[v.id]] for v in group] if blinding else []
    block = [r for v in group for r in results[crew_at[v.id]][0]]
    repeats = [r for v in group for r in results[crew_at[v.id]][1]]
    rows += _escalate(block, ctx.design) if ctx.criteria else block
    for i in pool_at:
        rows += results[i][0]
        repeats += results[i][1]
    if ctx.design.run.concurrency > 1:
        rows = _sequential_cache_flags(rows)
        fixed = list(repeats)  # each repeat namespace on its own, rows kept in their sequential places
        for k in dict.fromkeys(r.repeat for r in repeats):
            at = [i for i, r in enumerate(repeats) if r.repeat == k]
            for i, r in zip(at, _sequential_cache_flags([repeats[i] for i in at]), strict=True):
                fixed[i] = r
        repeats = fixed
    return rows, repeats


def _sequential_cache_flags(rows: list[ResultRow]) -> list[ResultRow]:
    """`rows` (in sequential order, one cache namespace) with `cached` as a sequential run would set it.

    Two CVs of a block can send the same request (under C1/C3 two identities' blinded CVs can be the same text).
    `CachedClient` makes that call once, but under concurrency either CV may be the one that makes it: here the
    fresh answer goes to the first of them in sequential order, and the others are cache hits. Only `cached`
    changes: the answer, tokens, cost and attempts are those of the one cached call either way. The other rows of a
    Ranker pool take the flag of its first row, as they share its call.
    """
    fresh = {key for r in rows for key, cached in r.calls if not cached}
    seen: set[str] = set()
    pool_flag: dict[str, bool] = {}
    out = []
    for r in rows:
        if r.calls and not (r.agent == "ranker" and r.input_position not in (None, 1)):
            calls = tuple((key, not (key in fresh and key not in seen)) for key, _ in r.calls)
            seen.update(key for key, _ in r.calls)
            r = replace(r, calls=calls, cached=all(c for _, c in calls))
            if r.agent == "ranker":
                pool_flag[r.pool] = r.cached
        elif r.agent == "ranker" and r.pool in pool_flag and r.calls:
            r = replace(r, cached=pool_flag[r.pool])
        out.append(r)
    return out


def _planned_per_cv(condition: str, rubric: list[str] | None) -> int:
    """Result rows per CV variant under `condition`: Screener and Evaluator, plus the Blinder (C1, C3) and the judge
    (C2, C3; an upper bound, as a judge row only follows a valid Evaluator score)."""
    return 2 + (condition in BLINDING_CONDITIONS) + (bool(rubric) and uses_rubric(condition))


def _cv_fields(v: Variant) -> dict[str, Any]:
    return dict(variant=v.id, base_cv=v.base_cv, identity=v.identity)


@dataclass(frozen=True)
class _Context:
    client: LLMClient
    design: Design
    job_ad: str
    language: str
    condition: str
    model: str
    criteria: list[str] | None = None  # rubric conditions: the job criteria the Evaluator scores
    track: runlog.RunLog = field(default_factory=lambda: runlog.RunLog(None))  # the run log

    def using(self, client: LLMClient) -> _Context:
        return replace(self, client=client)

    def sampled(self, *item: str) -> bool:
        """Whether this CV variant or pool, under this language, condition and model, is in the stability sample."""
        fraction = self.design.run.repeat_fraction
        draw = stable_seed(self.design.seed, "repeat", *item, self.language, self.condition, self.model)
        return fraction > 0 and draw / 2**32 < fraction

    @property
    def retry(self) -> Retry:
        run = self.design.run
        return Retry(retries=run.retries, backoff=run.retry_backoff, transient_backoff=run.transient_backoff)

    def key(self, v: Variant, agent: str) -> dict[str, Any]:
        return dict(base_cv=v.base_cv, identity=v.identity, gender=v.gender, origin=v.origin,
                    name=v.name, language=v.language, condition=self.condition, model=self.model, agent=agent)


def _usage(out: AgentOutput) -> dict[str, Any]:
    return dict(prompt_tokens=out.usage.prompt_tokens, completion_tokens=out.usage.completion_tokens,
                cost=out.usage.cost, cached=out.cached, attempts=out.attempts,
                allowance=out.usage.allowance, billing_source=out.usage.billing_source or "", calls=out.calls)


_NO_CALL: dict[str, Any] = dict(prompt_tokens=0, completion_tokens=0, cost=0.0, cached=False, attempts=0,
                                allowance=None, billing_source="")


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
    answer is escalated too, and so is a CV whose screening or blinding failed (no decision was made; the
    reason says which agent failed); screened-out CVs are not decisions.
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
            r = replace(r, escalated=True, escalation_reason=_missing_reason(r))
        out.append(r)
    return out


def _missing_reason(evaluator_row: ResultRow) -> str:
    if evaluator_row.error.startswith("blinding failed"):
        return "blinding failed"
    if evaluator_row.error == "no screening decision":
        return "screener missing"
    return "evaluator missing"


def _blind_one(ctx: _Context, v: Variant) -> tuple[ResultRow, Variant, str | None]:
    """C1, C3: blind one CV. Returns its Blinder row, the variant the crew sees (with the blinded text, or the
    original where blinding failed) and, if it failed, the error."""
    with ctx.track.item(**_cv_fields(v)):
        out, text = blind_variant(ctx.client, ctx.design, v, model=ctx.model, retry=ctx.retry)
        row = ResultRow(
            **ctx.key(v, "blinder"), score=None, rationale=out.parsed["blinded_cv"] if out.parsed else "",
            **_usage(out), status="ok" if text is not None else "missing", error=out.error,
        )
        ctx.track.done([row], 1)
    if text is None:
        return row, v, f"blinding failed: {out.error}"
    return row, replace(v, text=text), None


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
            **(_usage(out) if position == 1 else {**_NO_CALL, "cached": out.cached, "calls": out.calls}),
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
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in asdict(row).items() if k != "calls"})


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
    values["cost"] = float(raw["cost"]) if raw.get("cost") else None  # blank: the provider gave no price
    values["allowance"] = float(raw["allowance"]) if raw.get("allowance") else None  # absent before #11 Part B
    if "billing_source" not in raw and values["cost"] == 0 and (values["prompt_tokens"] or values["completion_tokens"]):
        values["cost"] = None  # a table from before #11 Part B: OpenCode Go's 0 was misread as a price
    values["cached"] = raw["cached"] == "True"
    return ResultRow(**values)
