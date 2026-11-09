"""Command line: `hcai run <config.toml>` runs the design and prints the paired score-gap report."""

from __future__ import annotations

import argparse
from pathlib import Path

from hcai.config import load_design
from hcai.llm import FakeLLMClient, LLMClient, LLMRequest, OpenRouterClient
from hcai.report import format_report, score_gaps
from hcai.runner import run_experiment, write_results


def run_command(config: str | Path, out_dir: str | Path, cache_dir: str | Path, client: LLMClient) -> str:
    design = load_design(config)
    rows = run_experiment(design, client, cache_dir=cache_dir)
    write_results(rows, Path(out_dir) / "results.csv")
    missing = sum(r.status != "ok" for r in rows)
    cost = sum(r.cost for r in rows)
    reference = design.identities[0].id
    summary = f"\n\n{len(rows)} rows ({missing} missing), cost ${cost:.4f} -> {Path(out_dir) / 'results.csv'}"
    return format_report(score_gaps(rows, reference=reference), reference=reference) + summary


def _dry_run_response(request: LLMRequest) -> dict:
    return {"score": 50, "rationale": "dry run: fixed score, no LLM was called"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hcai")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a design config and report paired score gaps")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, default=Path("results"), help="output directory (default: results)")
    run.add_argument(
        "--cache", type=Path, default=None,
        help="LLM request cache (default: .cache/llm, or .cache/llm-dry-run with --dry-run)",
    )
    run.add_argument("--dry-run", action="store_true", help="use a fake LLM that returns a fixed score")
    args = parser.parse_args(argv)

    # Dry-run responses are keyed like real ones, so they must never share a cache with real runs.
    client: LLMClient = FakeLLMClient(_dry_run_response) if args.dry_run else OpenRouterClient()
    cache = args.cache or Path(".cache/llm-dry-run" if args.dry_run else ".cache/llm")
    print(run_command(args.config, out_dir=args.out, cache_dir=cache, client=client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
