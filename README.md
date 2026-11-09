# HCAI: auditing demographic bias in a multi-agent LLM hiring pipeline

Human-Centred AI course project (University of Coimbra, 2026/27). The proposal is in `proposal/`; the implementation spec is GitHub issue #1.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
export OPENROUTER_API_KEY=sk-or-...   # only needed for real runs
```

## Running the skeleton

`configs/skeleton.toml` is a tiny design config: 1 role, 1 hand-written base CV, 2 identities, EN, condition C0. It renders each identity variant of the base CV and runs it through the hiring crew, writes the tidy results table and prints the paired score gap between the identities.

The crew is a sequential [CrewAI](https://docs.crewai.com/) pipeline of three agents, each with structured output:

1. **Screener:** pass/reject against the job ad's mandatory requirements, with a reason.
2. **Evaluator:** 0–100 score and rationale, only for CVs the Screener passed.
3. **Ranker:** orders a pool of screened-in CVs and shortlists the top 3. Pools come from the `[pools]` config section: equal-quality pools (all identities of one base CV) and optional mixed-quality pools (distinct base CVs across the quality levels). The CV order in each Ranker prompt is shuffled with a recorded seed.

CrewAI calls the LLM through the project's own client (`hcai.pipeline.ClientLLM`), so every call is cached and the fake client works in tests. CrewAI telemetry and tracing are switched off. See `docs/adr/0001-ranker-sees-screened-in-cv-texts-only.md` for what the Ranker sees.

```sh
uv run hcai run configs/skeleton.toml                # real run through OpenRouter
uv run hcai run configs/skeleton.toml --fake-llm     # smoke test: fake LLM, fixed score, no API key needed
```

Options: `--out DIR` (default `results/`) and `--cache DIR` (default `.cache/llm`; `--fake-llm` uses `.cache/llm-fake` so fake responses never reach a real run).

- **Results table:** `results/results.csv`, one row per base CV × identity × language × condition × model × agent (`screener`, `evaluator`, `ranker`; Ranker rows also per pool), with the outcome (`screen_pass`, `score`, or `rank`/`top3`), rationale, token counts, cost, a `cached` flag and an `ok`/`missing`/`rejected` status. Rejected CVs are not scored or ranked. Ranker rows add `pool`, `pool_kind`, `input_position` (1-based position in the Ranker prompt) and `order_seed`; a Ranker call's tokens and cost sit on its `input_position == 1` row. The printed summary shows what this run actually spent (cache hits cost nothing).
- **Cache:** every LLM call is cached on disk, keyed by the hash of the full request (model, messages, temperature, seed, schema). A rerun with the same config makes no calls and costs nothing. Unparseable structured outputs are not cached, so a rerun retries them. Delete `.cache/llm` to force fresh calls.
- **Corpus layout:** base CVs are `data/corpus/<role>/<quality>/<id>.<lang>.md` templates with `$name` and `$nationality` placeholders; job ads are `data/corpus/<role>/job_ad.<lang>.md`.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
