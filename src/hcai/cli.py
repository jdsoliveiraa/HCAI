"""Command line: `hcai run <config.toml>` runs the design and prints the paired score-gap report.

`hcai corpus <config.toml>` generates the job ads and base CVs with the generator model and freezes them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from hcai.config import load_design, pilot_design
from hcai.corpus import CorpusFrozen, fake_generator_response, generate_corpus
from hcai.llm import FakeLLMClient, LLMClient, LLMRequest, OpenRouterClient
from hcai.report import format_report, score_gaps
from hcai.runner import execute


def run_command(
    config: str | Path, out_dir: str | Path, cache_dir: str | Path, client: LLMClient, pilot: bool = False,
) -> str:
    design = load_design(config)
    if pilot:
        design = pilot_design(design)
    run = execute(design, client, out_dir=out_dir, cache_dir=cache_dir)
    t = run.totals
    reference = design.identities[0].id
    summary = (
        f"\n\n{t['rows']} rows ({t['status'].get('missing', 0)} missing, {t['status'].get('rejected', 0)} rejected); "
        f"{t['calls']} LLM calls, {t['cache_hits']} from cache; "
        f"{t['prompt_tokens']} prompt + {t['completion_tokens']} completion tokens; "
        f"spent ${t['cost_spent']:.4f} this run (results cost ${t['cost_total']:.4f} in all)"
        f" -> {Path(out_dir) / 'results.csv'}, {Path(out_dir) / 'run_summary.json'}"
    )
    if "pilot" in t:
        summary += (f"\npilot: the full design is {t['pilot']['full_design_scale']:g}x this slice, "
                    f"estimated full-run cost ${t['pilot']['estimated_full_cost']:.2f} (before caching)")
    return format_report(score_gaps(run.rows, reference=reference), reference=reference) + summary


def corpus_command(
    config: str | Path, cache_dir: str | Path, client: LLMClient, force: bool = False, fake: bool = False,
) -> str:
    design = load_design(config)
    if fake:  # a fake corpus must never take the place of (or freeze over) the real one
        design = replace(design, corpus_dir=design.corpus_dir.with_name(design.corpus_dir.name + "-fake"))
    corpus = generate_corpus(design, client, cache_dir=cache_dir, force=force)
    borderline = sum(cv.borderline for cv in corpus.base_cvs)
    return (
        f"corpus {corpus.version}: {len(corpus.job_ads)} job ads, {len(corpus.base_cvs)} base CVs "
        f"({borderline} borderline), generator {corpus.generator.model} -> {design.corpus_dir}"
    )


def _fake_response(request: LLMRequest) -> dict:
    props = (request.response_schema or {}).get("properties", {})
    if "passed" in props:
        return {"passed": True, "reason": "fake LLM: everyone passes, no LLM was called"}
    if "ranking" in props:
        return {"ranking": props["ranking"]["items"]["enum"], "rationale": "fake LLM: order as shown"}
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
    gen.add_argument("--force", action="store_true", help="overwrite the frozen corpus version")
    gen.add_argument(
        "--cache", type=Path, default=None,
        help="LLM request cache (default: .cache/llm, or .cache/llm-fake with --fake-llm)",
    )
    gen.add_argument("--fake-llm", action="store_true", help="smoke test: a fake generator writes <corpus_dir>-fake, no API calls")
    args = parser.parse_args(argv)

    if args.command == "corpus":
        client = FakeLLMClient(fake_generator_response) if args.fake_llm else OpenRouterClient()
        cache = args.cache or Path(".cache/llm-fake" if args.fake_llm else ".cache/llm")
        try:
            print(corpus_command(args.config, cache_dir=cache, client=client, force=args.force, fake=args.fake_llm))
        except CorpusFrozen as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    # Fake responses are keyed like real ones, so they must never share a cache with real runs.
    client: LLMClient = FakeLLMClient(_fake_response) if args.fake_llm else OpenRouterClient()
    cache = args.cache or Path(".cache/llm-fake" if args.fake_llm else ".cache/llm")
    out = args.out or Path("results/pilot" if args.pilot else "results")
    print(run_command(args.config, out_dir=out, cache_dir=cache, client=client, pilot=args.pilot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
