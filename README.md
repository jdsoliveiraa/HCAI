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
- **Corpus layout:** base CVs are `data/corpus/<role>/<quality>/<id>.<lang>.md` slot templates (see *Identity and language variants* below); job ads are `data/corpus/<role>/job_ad.<lang>.md`.

## Identity and language variants

Each base CV is rendered into 8 identity variants (2 genders × 4 origins: Portuguese, Brazilian, PALOP represented by Cape Verde, Eastern European represented by Ukraine) in every language of the design (EN, PT-PT). Only demographic cues change: the name, the nationality, pronouns and PT-PT grammatical gender. Every variant also states the right to work in Portugal and native-level (C2) Portuguese.

```sh
uv run hcai variants configs/variants.toml               # render + check, write data/variants/
uv run hcai run configs/variants.toml --fake-llm         # evaluate the same 16 variants
```

- **Name bank:** `data/names/name_bank.toml` has 4 full names per gender × origin, the nationality per language, and the public frequency sources they come from. Each base CV × identity gets one name, picked by a stable hash, so a name stays the same across languages. Set `name_bank = "..."` in a config's `[design]` to use every gender × origin of the bank as the identities.
- **Templates:** base CVs are slot templates (`<id>.en.md`, `<id>.pt.md`) with `$name`, `Nationality: $nationality` (`Nacionalidade: $nationality` in PT-PT), `$work_eligibility` (the standard right-to-work and proficiency lines) and `{{masc|fem}}` for whole gendered words, e.g. `{{He|She}}`, `{{Engenheiro|Engenheira}}`, `{{licenciado|licenciada}}`. See `src/hcai/variants.py`.
- **Diff check:** within each base CV × language, the variants are compared after masking the name, the nationality value, pronouns and the gendered pairs. Any other difference, a gendered form of the wrong gender, or a missing right-to-work / proficiency line fails the check. Templates are also rejected if they use an EN pronoun or a common PT-PT gendered word (Engenheiro, licenciada, ...) outside a slot; write `{{X|X}}` for one that does not refer to the candidate. `hcai variants` then exits with status 1 without writing anything, and `hcai run` stops before its first LLM call.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
