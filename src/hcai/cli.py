"""Command line: `hcai run <config.toml>` runs the design and prints the paired score-gap report.

`hcai corpus <config.toml>` generates the job ads and base CVs with the generator model and freezes them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from hcai.config import load_design
from hcai.corpus import CorpusFrozen, fake_generator_response, generate_corpus
from hcai.llm import FakeLLMClient, LLMClient, LLMRequest, OpenRouterClient
from hcai.report import format_report, score_gaps
from hcai.runner import run_experiment, write_results


def run_command(config: str | Path, out_dir: str | Path, cache_dir: str | Path, client: LLMClient) -> str:
    design = load_design(config)
    rows = run_experiment(design, client, cache_dir=cache_dir)
    write_results(rows, Path(out_dir) / "results.csv")
    missing = sum(r.status != "ok" for r in rows)
    spent = sum(r.cost for r in rows if not r.cached)
    hits = sum(r.cached for r in rows)
    reference = design.identities[0].id
    summary = (
        f"\n\n{len(rows)} rows ({missing} missing, {hits} from cache), spent ${spent:.4f} this run"
        f" -> {Path(out_dir) / 'results.csv'}"
    )
    return format_report(score_gaps(rows, reference=reference), reference=reference) + summary


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
    run.add_argument("--out", type=Path, default=Path("results"), help="output directory (default: results)")
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
    print(run_command(args.config, out_dir=args.out, cache_dir=cache, client=client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
