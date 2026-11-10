"""Command line: `hcai run <config.toml>` runs the design and prints the paired score-gap report.

`hcai corpus <config.toml>` generates the job ads and base CVs with the generator model and freezes them
(in English, translated into the design's other languages); `--translate` adds the design's other languages
to an already frozen corpus without regenerating it.
With condition C1 in the design it also prints what survived blinding (`hcai.blinding`).
`hcai variants <config.toml>` renders every identity x language variant and runs the demographic-only diff check.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from hcai.blinding import fake_blinding
from hcai.config import load_design, pilot_design
from hcai.corpus import CorpusFrozen, GenerationFailed, fake_generator_response, generate_corpus, translate_corpus
from hcai.llm import FakeLLMClient, LLMClient, LLMRequest, OpenRouterClient
from hcai.report import format_proxy_leaks, format_report, format_stability, score_gaps
from hcai.runner import execute
from hcai.variants import IdentityError, TemplateError, VariantCheckError, build_variants, write_variants


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
        f" -> {Path(out_dir) / 'results.csv'}, {Path(out_dir) / 'run_summary.json'}"
    )
    if "pilot" in totals:
        p = totals["pilot"]
        summary += (f"\npilot: the full design has {p['cv_scale']:g}x these CVs and {p['ranker_scale']:g}x the "
                    f"Ranker load; estimated full-run cost ${p['estimated_full_cost']:.2f} (before caching)")
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
    translate: bool = False,
) -> str:
    design = load_design(config)
    if fake:  # a fake corpus must never take the place of (or freeze over) the real one
        design = replace(design, corpus_dir=design.corpus_dir.with_name(design.corpus_dir.name + "-fake"))
    make = translate_corpus if translate else generate_corpus
    corpus = make(design, client, cache_dir=cache_dir, force=force)
    borderline = sum(cv.borderline for cv in corpus.base_cvs)
    return (
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


def _fake_response(request: LLMRequest) -> dict:
    props = (request.response_schema or {}).get("properties", {})
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
        "--cache", type=Path, default=None,
        help="LLM request cache (default: .cache/llm, or .cache/llm-fake with --fake-llm)",
    )
    gen.add_argument("--fake-llm", action="store_true", help="smoke test: a fake generator writes <corpus_dir>-fake, no API calls")
    variants = sub.add_parser("variants", help="render the identity x language CV variants and check them")
    variants.add_argument("config", type=Path)
    variants.add_argument("--out", type=Path, default=Path("data/variants"), help="output directory (default: data/variants)")
    args = parser.parse_args(argv)

    if args.command == "corpus":
        client = FakeLLMClient(fake_generator_response) if args.fake_llm else OpenRouterClient()
        cache = args.cache or Path(".cache/llm-fake" if args.fake_llm else ".cache/llm")
        try:
            print(corpus_command(args.config, cache_dir=cache, client=client, force=args.force, fake=args.fake_llm,
                                 translate=args.translate))
        except (CorpusFrozen, GenerationFailed, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.command == "variants":
        try:
            print(variants_command(args.config, out_dir=args.out))
        except (TemplateError, IdentityError, VariantCheckError, FileNotFoundError) as e:
            print(f"variants: {e}", file=sys.stderr)
            return 1
        return 0

    # Fake responses are keyed like real ones, so they must never share a cache with real runs.
    client: LLMClient = FakeLLMClient(_fake_response) if args.fake_llm else OpenRouterClient()
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
