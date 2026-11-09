# HCAI: auditing demographic bias in a multi-agent LLM hiring pipeline

Human-Centred AI course project (University of Coimbra, 2026/27). The proposal is in `proposal/`; the implementation spec is GitHub issue #1.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
export OPENROUTER_API_KEY=sk-or-...   # only needed for real runs
```

## Running the skeleton

`configs/skeleton.toml` is a tiny design config: 1 role, 1 hand-written base CV, 2 identities, EN, condition C0. It renders each identity variant of the base CV, sends each to the Evaluator agent (0–100 score and rationale, structured output), writes the tidy results table and prints the paired score gap between the identities.

```sh
uv run hcai run configs/skeleton.toml                # real run through OpenRouter
uv run hcai run configs/skeleton.toml --dry-run      # fake LLM, fixed score, no API key needed
```

Options: `--out DIR` (default `results/`) and `--cache DIR` (default `.cache/llm`; dry runs use `.cache/llm-dry-run` so fake responses never reach a real run).

- **Results table:** `results/results.csv`, one row per base CV × identity × language × condition × model × agent, with the score, rationale, token counts, cost and an `ok`/`missing` status.
- **Cache:** every LLM call is cached on disk, keyed by the hash of the full request (model, messages, temperature, seed, schema). A rerun with the same config makes no calls and costs nothing. Delete `.cache/llm` to force fresh calls.
- **Corpus layout:** base CVs are `data/corpus/<role>/<quality>/<id>.<lang>.md` templates with `$name` and `$nationality` placeholders; job ads are `data/corpus/<role>/job_ad.<lang>.md`.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
