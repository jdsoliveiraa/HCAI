"""Command line: `hcai run <config.toml>` runs the design and prints the paired score-gap report.

`hcai corpus <config.toml>` generates the job ads and base CVs with the generator model and freezes them
(in English, translated into the design's other languages); `--translate` adds the design's other languages
to an already frozen corpus without regenerating it; `--recheck` re-validates a frozen corpus with today's checks
(e.g. the English-language check) and regenerates only the failing documents and their translations. If documents stay invalid after `[generator] max_attempts`,
the command names them and exits 1, keeping what succeeded; rerunning it resumes with only the failing ones.
With condition C1 in the design it also prints what survived blinding (`hcai.blinding`).
`hcai variants <config.toml>` renders every identity x language variant and runs the demographic-only diff check.
`hcai models <config.toml>` lists the models of the config's providers; `hcai check <config.toml>` also probes
each configured model (chat endpoint, JSON schema, temperature and seed determinism), see `hcai.check`.
`hcai names <config.toml>` runs the name manipulation check: does each name of the bank signal its intended
gender and origin to every pipeline model (`hcai.namecheck`, ADRs 0009, 0011)? It reports failing names and changes nothing.
`hcai site <config.toml>` writes the static project site (`hcai.site`, ADR 0010): the realism review page and the
project's navigation hub. It reads the repository only and never calls an LLM.

Each model is served by its provider (`hcai.providers`: `opencode-go` by default, `openrouter` as the fallback);
API keys come from the environment or a `.env` file in the working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from hcai.blinding import fake_blinding
from hcai.check import ProbeResult, format_probes, probe_model
from hcai.config import Design, load_design, pilot_design
from hcai.corpus import (
    CorpusFrozen, GenerationFailed, fake_generator_response, generate_corpus, recheck_corpus, translate_corpus,
)
from hcai.llm import CachedClient, FakeLLMClient, LLMClient, LLMRequest, MeteredClient, ProviderExhausted
from hcai.namecheck import (
    DEFAULT_MODE, DEFAULT_ORIGIN_MINIMUM, DEFAULT_REASKS, DEFAULT_THRESHOLD, MODES, PROBABILITY, NameCheck,
    NameCheckAborted, ProbabilityCheck, ProbabilityRule, ValidAnswersOnly, format_check, perceive, score,
    score_probabilities, write_outputs,
)
from hcai.names import NameBankError
from hcai.oversight import uses_rubric
from hcai.providers import MissingKey, ProviderRouter, http_client, load_dotenv, provider_client
from hcai.report import format_proxy_leaks, format_report, format_stability, score_gaps
from hcai.runner import execute
from hcai.variants import IdentityError, TemplateError, VariantCheckError, build_variants, write_variants


def _go_billing_summary(totals: dict[str, Any]) -> str:
    """The OpenCode Go lines of the run summary: allowance spent per model and overall, and calls billed elsewhere."""
    models = {m: t for m, t in totals["by_model"].items()
              if t["billing_sources"] or t["allowance_total"] or t["not_go_attempts"]}
    if not models:
        return ""
    out = (f"\nGo allowance: ${totals['allowance_spent']:.4f} spent this run "
           f"(the outputs took ${totals['allowance_total']:.4f} in all)")
    for m, t in models.items():
        out += f"\n  {m}: ${t['allowance_spent']:.4f} this run (${t['allowance_total']:.4f} in all)"
    if totals["not_go_calls"] or totals["not_go_attempts"]:
        elsewhere = ", ".join(f"{s} x{n}" for s, n in sorted(totals["billing_sources"].items()) if s != "go")
        out += (f"\nWARNING: {totals['not_go_calls']} calls this run ({totals['not_go_attempts']} attempts in all) "
                f"were not billed to the Go allowance" + (f" ({elsewhere})" if elsewhere else "")
                + f"; ${totals['balance_spent']:.4f} billed elsewhere (e.g. the pay-as-you-go balance) this run")
        if totals["billing_sources"].get("unknown"):
            out += (f"; {totals['billing_sources']['unknown']} calls named no billing source "
                    "(counted at their allowance)")
    return out


def _missing_summary(missing: dict[str, Any]) -> str:
    """Missing rows overall and per model x condition x language, flagged above `[run] missing_target`."""
    out = (f"\nmissing: {missing['missing']} of {missing['rows']} rows ({100 * missing['share']:.2f}%; target "
           f"under {100 * missing['target']:.2f}%)" + (" ABOVE TARGET" if missing["above_target"] else ""))
    for cell, c in missing["by_cell"].items():
        out += f"\n  {cell}: {c['missing']} of {c['rows']} ({100 * c['share']:.2f}%)"
    return out


def run_command(
    config: str | Path, out_dir: str | Path, cache_dir: str | Path, client: LLMClient, pilot: bool = False,
) -> str:
    design = load_design(config)
    if pilot:
        design = pilot_design(design)
    run = execute(design, client, out_dir=out_dir, cache_dir=cache_dir)
    totals = run.totals
    status = totals["status"]
    reference = design.identities[0].id
    summary = (
        f"\n\n{totals['rows']} rows ({status.get('missing', 0)} missing, {status.get('rejected', 0)} rejected); "
        f"{totals['calls']} LLM calls, {totals['cache_hits']} from cache; "
        f"{totals['prompt_tokens']} prompt + {totals['completion_tokens']} completion tokens; "
        f"spent ${totals['cost_spent']:.4f} this run (the outputs cost ${totals['cost_total']:.4f} in all)"
    )
    if totals["unpriced_attempts"]:
        billing = sorted({m.get("billing", "") for m in totals["by_model"].values() if m.get("unpriced_attempts")} - {""})
        summary += (f"; cost unknown for {totals['unpriced_calls']} calls this run "
                    f"({totals['unpriced_attempts']} in all): no price reported"
                    + (f" ({', '.join(billing)})" if billing else ""))
    summary += f" -> {Path(out_dir) / 'results.csv'}, {Path(out_dir) / 'run_summary.json'}"
    summary += _missing_summary(totals["missing"])
    if totals.get("timed_out_calls"):
        summary += (f"\n{totals['timed_out_calls']} calls this run timed out ([run] request_timeout = "
                    f"{design.run.request_timeout:g}s)")
    summary += _go_billing_summary(totals)
    if "pilot" in totals:
        p = totals["pilot"]
        tokens = p["estimated_full_tokens"]
        cost = ("unknown (no price reported)" if p["estimated_full_cost"] is None
                else f"${p['estimated_full_cost']:.2f}")
        if p.get("estimated_full_allowance") is not None:
            cost += f" (${p['estimated_full_allowance']:.2f} of it from the Go allowance)"
        summary += (f"\npilot: the full design has {p['cv_scale']:g}x these CVs and {p['ranker_scale']:g}x the "
                    f"Ranker load; estimated full-run cost {cost}, {tokens['prompt']} prompt + "
                    f"{tokens['completion']} completion tokens (before caching)")
        if p["unestimated"]:
            summary += (f"; a lower bound: the pilot ran no condition like {', '.join(p['unestimated'])}, "
                        "so their cost is not included")
    for condition, by_model in totals.get("oversight", {}).items():
        for model, c in by_model.items():
            summary += (f"\noversight {condition} {model}: {c['escalated']} of {c['decisions']} Evaluator decisions "
                        f"escalated ({c['flagged']} flagged by the judge, {c['near_cutoff']} near the cutoff, "
                        f"{c['judge_missing']} without a verdict, {c['evaluator_missing']} without a score)"
                        f" -> {Path(out_dir) / 'review_queue.csv'}")
    if run.stability:
        summary += "\n\n" + format_stability(run.stability) + f"\n-> {Path(out_dir) / 'stability.json'}"
    if run.proxy_leaks:
        summary += "\n\n" + format_proxy_leaks(run.proxy_leaks) + f"\n-> {Path(out_dir) / 'proxy_leaks.csv'}"
    return format_report(score_gaps(run.rows, reference=reference), reference=reference) + summary


def corpus_command(
    config: str | Path, cache_dir: str | Path, client: LLMClient, force: bool = False, fake: bool = False,
    translate: bool = False, recheck: bool = False,
) -> str:
    design = load_design(config)
    if fake:  # a fake corpus must never take the place of (or freeze over) the real one
        design = replace(design, corpus_dir=design.corpus_dir.with_name(design.corpus_dir.name + "-fake"))
    prefix = ""
    if recheck:
        result = recheck_corpus(design, client, cache_dir=cache_dir)
        corpus = result.corpus
        prefix = ("recheck: every frozen document passes; nothing regenerated\n"
                  if not result.regenerated and not result.retranslated else
                  f"recheck: regenerated {len(result.regenerated)} English document(s)"
                  + (f" ({', '.join(result.regenerated)})" if result.regenerated else "")
                  + (f", retranslated {', '.join(result.retranslated)}" if result.retranslated else "") + "\n")
    else:
        make = translate_corpus if translate else generate_corpus
        corpus = make(design, client, cache_dir=cache_dir, force=force)
    borderline = sum(cv.borderline for cv in corpus.base_cvs)
    return prefix + (
        f"corpus {corpus.version}: {len(corpus.job_ads)} job ads, {len(corpus.base_cvs)} base CVs "
        f"({borderline} borderline) in {', '.join(corpus.languages)}, generator {corpus.generator.model}"
        f" -> {design.corpus_dir}"
    )


def variants_command(config: str | Path, out_dir: str | Path) -> str:
    """Render, check and write every identity x language variant of the design's base CVs."""
    design = load_design(config)
    variants = build_variants(design)
    manifest = write_variants(variants, out_dir)
    base_cvs = len({v.base_cv for v in variants})
    return (
        f"{len(variants)} variants ({base_cvs} base CVs x {len(design.identities)} identities"
        f" x {len(design.languages)} languages), demographic-only diff check passed -> {manifest}"
    )


def run_models(design: Design) -> list[str]:
    """The models `hcai run` calls: the pipeline models, and the judge under a rubric condition."""
    judge = [design.judge_model] if design.judge_model and any(uses_rubric(c) for c in design.conditions) else []
    return list(dict.fromkeys(design.models + judge))


def make_client(design: Design, models: list[str]) -> ProviderRouter:
    """The real client for `models`: each sent to its provider, rate limits waited out per `[run]`.

    Every call waits `[run] request_timeout` seconds for the answer.
    Fails with `MissingKey` before any call if a key those models need is not set.
    """
    r = design.run
    return provider_client(design.provider_for, models, backoff=r.rate_limit_backoff,
                           max_backoff=r.rate_limit_max_backoff, max_wait=r.rate_limit_max_wait,
                           request_timeout=r.request_timeout)


def models_command(config: str | Path, provider: str | None = None, http=None) -> str:
    """The model ids offered by `provider`, or by each provider the config's models use."""
    design = load_design(config)
    if provider is not None and provider not in design.providers:
        raise ValueError(f"unknown provider {provider!r} (known: {', '.join(sorted(design.providers))})")
    names = [provider] if provider else list(dict.fromkeys(design.provider_for(m).name for m in design.all_models))
    out = []
    for name in names:
        p = design.providers[name]
        ids = http_client(p, http=http, request_timeout=design.run.request_timeout).list_models()
        out.append(f"{name} ({p.base_url}): {len(ids)} models\n" + "\n".join(f"  {i}" for i in ids))
    return "\n".join(out)


def check_command(config: str | Path, client: LLMClient, listings: dict[str, list[str] | None],
                  models: list[str] | None = None, repeats: int = 3) -> list[ProbeResult]:
    """Probe each configured model (or `models`) through `client`, uncached; `listings` are the provider's ids."""
    design = load_design(config)
    results = []
    for model in models or design.all_models:
        provider = design.provider_for(model).name
        ids = listings.get(provider)
        results.append(probe_model(client, model, provider=provider, listed=None if ids is None else model in ids,
                                   repeats=repeats, seed=design.seed if design.seed is not None else 42))
    return results


def names_command(config: str | Path, client: LLMClient, *, out_dir: str | Path, cache_dir: str | Path,
                  repeats: int = 5, threshold: float = DEFAULT_THRESHOLD, temperature: float | None = None,
                  mode: str = DEFAULT_MODE, origin_minimum: float = DEFAULT_ORIGIN_MINIMUM,
                  reasks: int = DEFAULT_REASKS) -> tuple[str, bool]:
    """Run the name manipulation check through `client` (cached in `cache_dir`); (report, every name passes).

    `mode` is "probability" (ADR 0011: `threshold` is the minimum mean p(intended gender), `origin_minimum` the
    minimum mean p(intended origin) of a non-Portuguese name, and an all-hedge answer is asked again up to
    `reasks` times) or "choice" (ADR 0009: `threshold` is the minimum gender and origin hit rate; no re-asks).
    The last line counts every ask: the answers and the re-asks are the LLM calls plus those from the cache."""
    design = load_design(config)
    rule = ProbabilityRule(threshold, origin_minimum) if mode == PROBABILITY else None  # validated before any call
    metered = MeteredClient(client)
    # a missing answer (unparseable, or outside the options) is not cached, so a rerun asks it again
    cached = CachedClient(ValidAnswersOnly(metered, mode), cache_dir, cache_unparseable=False)
    perceptions = perceive(design, cached, repeats=repeats, temperature=temperature, mode=mode, reasks=reasks)
    check: NameCheck | ProbabilityCheck = (score_probabilities(perceptions, rule, max_reasks=reasks)
                                           if rule is not None else score(perceptions, threshold))
    paths = write_outputs(check, out_dir)
    calls = sum(metered.calls.values())
    missing = sum(p.missing for p in check.perceptions)
    reasked = [e for p in check.perceptions for e in p.reasked]  # the all-hedge answers that were asked again
    from_cache = sum(p.cached for p in [*check.perceptions, *reasked])
    asked = f"{len(check.perceptions)} answers ({missing} missing)"
    if mode == PROBABILITY:
        asked += f" and {len(reasked)} re-asks (up to {reasks} per all-hedge answer)"
    report = (format_check(check)
              + f"\n\n{asked}, {calls} LLM calls, {from_cache} from cache -> " + ", ".join(str(p) for p in paths))
    return report, not check.failing


def _fake_response(request: LLMRequest) -> dict:
    props = (request.response_schema or {}).get("properties", {})
    if set(props) == {"gender", "origin"}:  # name manipulation check: no signal at all
        if props["gender"].get("type") == "object":  # probability mode: all on "unsure" and "Other or unsure"
            return {q: {o: float(i == len(props[q]["required"]) - 1) for i, o in enumerate(props[q]["required"])}
                    for q in ("gender", "origin")}
        return {"gender": "unsure", "origin": props["origin"]["enum"][-1]}
    if "blinded_cv" in props:
        return fake_blinding(request)
    if "passed" in props:
        return {"passed": True, "reason": "fake LLM: everyone passes, no LLM was called"}
    if "ranking" in props:
        return {"ranking": props["ranking"]["items"]["enum"], "rationale": "fake LLM: order as shown"}
    if "criteria" in props:  # rubric Evaluator (C2/C3): middling scores, quoting nothing in particular
        n = len(props["criteria"]["items"]["properties"]["criterion"]["enum"])
        return {"criteria": [{"criterion": i, "score": 5, "quote": "fake LLM quote"} for i in range(1, n + 1)],
                "rationale": "fake LLM: fixed rubric, no LLM was called"}
    if "verdicts" in props:  # oversight judge
        n = len(props["verdicts"]["items"]["properties"]["criterion"]["enum"])
        return {"verdicts": [{"criterion": i, "verdict": "faithful", "reason": "fake LLM judge"}
                             for i in range(1, n + 1)]}
    return {"score": 50, "rationale": "fake LLM: fixed score, no LLM was called"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hcai")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a design config and report paired score gaps")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, default=None,
                     help="output directory (default: results, or results/pilot with --pilot)")
    run.add_argument("--pilot", action="store_true",
                     help="run only the config's [pilot] slice and estimate the full run's cost")
    run.add_argument(
        "--cache", type=Path, default=None,
        help="LLM request cache (default: .cache/llm, or .cache/llm-fake with --fake-llm)",
    )
    run.add_argument("--fake-llm", action="store_true", help="smoke test: a fake LLM returns a fixed score, no API calls")
    gen = sub.add_parser("corpus", help="generate and freeze the job ads and base CVs of a design config")
    gen.add_argument("config", type=Path)
    gen.add_argument("--force", action="store_true", help="overwrite the frozen corpus version (or its translations)")
    gen.add_argument(
        "--translate", action="store_true",
        help="add the design's non-English languages to the frozen corpus, leaving its English files unchanged",
    )
    gen.add_argument(
        "--recheck", action="store_true",
        help="re-validate the frozen corpus and regenerate only the failing documents and their translations",
    )
    gen.add_argument(
        "--cache", type=Path, default=None,
        help="LLM request cache (default: .cache/llm, or .cache/llm-fake with --fake-llm)",
    )
    gen.add_argument("--fake-llm", action="store_true", help="smoke test: a fake generator writes <corpus_dir>-fake, no API calls")
    variants = sub.add_parser("variants", help="render the identity x language CV variants and check them")
    variants.add_argument("config", type=Path)
    variants.add_argument("--out", type=Path, default=Path("data/variants"), help="output directory (default: data/variants)")
    models = sub.add_parser("models", help="list the models offered by the config's providers")
    models.add_argument("config", type=Path)
    models.add_argument("--provider", default=None, help="only this provider (default: every provider the config uses)")
    check = sub.add_parser("check", help="probe each configured model: chat endpoint, JSON schema, temperature, seed")
    check.add_argument("config", type=Path)
    check.add_argument("--model", action="append", default=None, help="probe only this model (repeatable)")
    check.add_argument("--repeats", type=int, default=3, help="identical requests per determinism probe (default 3)")
    check.add_argument("--out", type=Path, default=None, help="also write the results as JSON to this file")
    names = sub.add_parser("names", help="name manipulation check: does each name signal its gender and origin?")
    names.add_argument("config", type=Path)
    names.add_argument("--repeats", type=int, default=5, help="answers per name and model (default 5)")
    names.add_argument("--mode", choices=MODES, default=DEFAULT_MODE,
                       help="probability: a probability per option (default); choice: one gender and the single "
                            "most likely origin (ADRs 0011, 0009)")
    names.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="probability mode: minimum mean p(intended gender); choice mode: minimum gender and origin "
                            f"hit rate (default {DEFAULT_THRESHOLD})")
    names.add_argument("--origin-min", type=float, default=DEFAULT_ORIGIN_MINIMUM,
                       help="probability mode: minimum mean p(intended origin) of a non-Portuguese name, which must "
                            "also be the highest non-Portuguese origin, both scored without the hedge options "
                            f"(default {DEFAULT_ORIGIN_MINIMUM})")
    names.add_argument("--reasks", type=int, default=DEFAULT_REASKS,
                       help="probability mode: times an answer with everything on a hedge option (unsure, Other or "
                            "unsure) is asked again as a new attempt before it stays missing; 0: never "
                            f"(default {DEFAULT_REASKS})")
    names.add_argument("--temperature", type=float, default=None,
                       help="sampling temperature (default: the design's [design] temperature)")
    names.add_argument("--out", type=Path, default=Path("results/names"), help="output directory (default: results/names)")
    names.add_argument(
        "--cache", type=Path, default=None,
        help="LLM request cache (default: .cache/llm, or .cache/llm-fake with --fake-llm)",
    )
    names.add_argument("--fake-llm", action="store_true", help="smoke test: a fake LLM perceives nothing, no API calls")
    site = sub.add_parser("site", help="write the static project site (realism review + project hub); no API calls")
    site.add_argument("config", type=Path)
    site.add_argument("--out", type=Path, default=Path("site"), help="output directory (default: site)")
    args = parser.parse_args(argv)
    load_dotenv(".env")  # keys from a gitignored .env in the working directory; the environment wins

    try:
        return _dispatch(args)
    except (MissingKey, FileNotFoundError, NameBankError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ProviderExhausted as e:
        if args.command in ("check", "models"):  # probes and listings are never cached
            print(f"stopped: {e}. Try again once the provider's limit has reset.", file=sys.stderr)
            return 3
        out = getattr(args, "out", None) or Path("results/pilot" if getattr(args, "pilot", False) else "results")
        where = f"; progress in {out / 'progress.json'}" if args.command == "run" else ""
        print(f"stopped: {e}{where}. Every completed call is cached: rerun the same command to resume.",
              file=sys.stderr)
        return 3


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "site":
        from hcai.site import build_site  # the markdown dependency is only needed here

        try:
            pages = build_site(args.config, args.out)
        except (ValueError, FileExistsError, TemplateError, IdentityError) as e:
            print(f"site: {e}", file=sys.stderr)
            return 1
        print(f"{len(pages)} pages -> {args.out / 'index.html'}")
        return 0

    if args.command == "models":
        try:
            print(models_command(args.config, provider=args.provider))
        except Exception as e:  # noqa: BLE001 - config errors and provider/HTTP errors (already redacted)
            print(f"models: {e}", file=sys.stderr)
            return 1
        return 0

    if args.command == "check":
        design = load_design(args.config)
        targets = args.model or design.all_models
        client = make_client(design, targets)
        listings: dict[str, list[str] | None] = {}
        for name in dict.fromkeys(design.provider_for(m).name for m in targets):
            try:
                listings[name] = http_client(design.providers[name],
                                             request_timeout=design.run.request_timeout).list_models()
            except Exception as e:  # noqa: BLE001 - the probes still run
                print(f"check: could not list {name}'s models: {type(e).__name__}: {e}", file=sys.stderr)
                listings[name] = None
        results = check_command(args.config, client, listings, models=targets, repeats=args.repeats)
        print(format_probes(results))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps([r.__dict__ for r in results], indent=2) + "\n")
            print(f"-> {args.out}")
        return 0 if all(r.endpoint_ok for r in results) else 2

    if args.command == "corpus":
        if args.recheck and (args.force or args.translate):
            print("error: --recheck cannot be combined with --force or --translate", file=sys.stderr)
            return 1
        if args.fake_llm:
            client = FakeLLMClient(fake_generator_response)
        else:
            design = load_design(args.config)
            client = make_client(design, [design.generator.model] if design.generator else [])
        cache = args.cache or Path(".cache/llm-fake" if args.fake_llm else ".cache/llm")
        try:
            print(corpus_command(args.config, cache_dir=cache, client=client, force=args.force, fake=args.fake_llm,
                                 translate=args.translate, recheck=args.recheck))
        except (CorpusFrozen, GenerationFailed, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.command == "names":
        if (args.repeats < 1 or args.reasks < 0 or not 0.0 <= args.threshold <= 1.0
                or not 0.0 <= args.origin_min <= 1.0):
            print("error: --repeats must be >= 1, --reasks >= 0 and --threshold and --origin-min between 0 and 1",
                  file=sys.stderr)
            return 1
        cache = args.cache or Path(".cache/llm-fake" if args.fake_llm else ".cache/llm")
        try:
            if args.fake_llm:
                client = FakeLLMClient(_fake_response)
            else:
                design = load_design(args.config)
                client = make_client(design, design.models)
            report, all_pass = names_command(args.config, client, out_dir=args.out, cache_dir=cache,
                                             repeats=args.repeats, threshold=args.threshold,
                                             temperature=args.temperature, mode=args.mode,
                                             origin_minimum=args.origin_min, reasks=args.reasks)
        except (ValueError, NameCheckAborted) as e:  # a config error (e.g. an origin with no option), or a model
            # whose calls keep failing (e.g. a wrong model id): answers so far are cached, so a rerun resumes
            print(f"names: {e}", file=sys.stderr)
            return 1
        print(report)
        return 0 if all_pass else 2

    if args.command == "variants":
        try:
            print(variants_command(args.config, out_dir=args.out))
        except (TemplateError, IdentityError, VariantCheckError, FileNotFoundError) as e:
            print(f"variants: {e}", file=sys.stderr)
            return 1
        return 0

    # Fake responses are keyed like real ones, so they must never share a cache with real runs.
    client: LLMClient = FakeLLMClient(_fake_response) if args.fake_llm else make_client(
        design := load_design(args.config), run_models(pilot_design(design) if args.pilot else design))
    cache = args.cache or Path(".cache/llm-fake" if args.fake_llm else ".cache/llm")
    out = args.out or Path("results/pilot" if args.pilot else "results")
    try:
        print(run_command(args.config, out_dir=out, cache_dir=cache, client=client, pilot=args.pilot))
    except (TemplateError, IdentityError, VariantCheckError, FileNotFoundError) as e:  # raised before any LLM call
        print(f"run: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
