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
uv run hcai run configs/skeleton.toml --fake-llm     # smoke test: fake LLM, fixed score, no API key needed
```

Options: `--out DIR` (default `results/`) and `--cache DIR` (default `.cache/llm`; `--fake-llm` uses `.cache/llm-fake` so fake responses never reach a real run).

- **Results table:** `results/results.csv`, one row per base CV × identity × language × condition × model × agent, with the score, rationale, token counts, cost, a `cached` flag and an `ok`/`missing` status. The printed summary shows what this run actually spent (cache hits cost nothing).
- **Cache:** every LLM call is cached on disk, keyed by the hash of the full request (model, messages, temperature, seed, schema). A rerun with the same config makes no calls and costs nothing. Unparseable structured outputs are not cached, so a rerun retries them. Delete `.cache/llm` to force fresh calls.
- **Corpus layout:** base CVs are `data/corpus/<role>/<quality>/<id>.<lang>.md` templates with `$name` and `$nationality` placeholders; job ads are `data/corpus/<role>/job_ad.<lang>.md`.

## Generating the corpus

`configs/core.toml` is the core design: 3 roles (software engineer, nurse, accountant) × 3 quality levels (strong, average, weak) × `base_cvs_per_role` base CVs per role and quality level. Its `[generator]` section sets the generator model (a third model family, not one of the audited pipeline `models`), its temperature, and how many borderline base CVs to write per role and quality level.

```sh
uv run hcai corpus configs/core.toml              # real generation through OpenRouter
uv run hcai corpus configs/core.toml --fake-llm   # smoke test: fake generator, writes data/corpus/v1-fake
```

- **Job ads:** one per role, with explicit mandatory and desirable requirements.
- **Base CVs:** written from fixed per-quality templates (levers: years above the minimum, desirable requirements met, quantified achievements, certifications) and identity-free: the name and nationality are `$name` / `$nationality` placeholders in a fixed header that also states the right to work in Portugal and the language proficiency. Each CV must quote, for every mandatory requirement, evidence from its own text; answers that don't are regenerated (up to 3 attempts, then generation fails and nothing is written).
- **Borderline CVs** meet every mandatory requirement but only just; the last N CVs of a role × quality level are borderline, per `[generator] borderline`, and carry a note saying which requirement is borderline.
- **Frozen and versioned:** the corpus is written to `corpus_dir` (e.g. `data/corpus/v1`) with a `manifest.json` holding each CV's stable id (`<role>/<quality>/cvNN`), its ground-truth quality level and borderline tag. Running again refuses to overwrite it; pass `--force`, or point `corpus_dir` to a new version. Each CV is also stored as structured JSON next to its `.en.md` template.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
