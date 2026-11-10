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

Options: `--out DIR` (default `results/`, or `results/pilot/` with `--pilot`), `--cache DIR` (default `.cache/llm`; `--fake-llm` uses `.cache/llm-fake` so fake responses never reach a real run) and `--pilot`.

- **Pilot:** `uv run hcai run configs/core.toml --pilot` runs only the config's `[pilot]` slice (by default the first role and one base CV per quality level, under every identity, language, condition and model) without editing the design, and prints an estimate of the full run's cost scaled from the slice (Screener and Evaluator by CV variants, Ranker by CVs ranked; rough, as it assumes the same screen-in rate). Pilot calls share the cache, so the full run reuses its Screener, Evaluator and equal-quality Ranker calls; mixed-quality pools are drawn from fewer base CVs in the pilot, so the full run pays for those again.
- **Retries and missing data:** each agent call is validated (parseable JSON, score in 0–100, ranking a permutation). A malformed answer is asked again with a note naming the problem appended to the prompt (a new request, so it has its own cache key); a call that raises (HTTP error, timeout) is repeated after `retry_backoff` seconds (default 2), doubling. After `[run] retries` retries the agent's rows get status `missing` and the error; the `attempts` column counts the calls behind each row.
- **Cost and tokens:** `results/run_summary.json` holds the run's totals: `attempts`, prompt and completion tokens and `cost_total` (what the outputs took to produce, retries and stability sample included; cache hits count their original cost) and `calls`, `cache_hits` and `cost_spent` (what this run sent to OpenRouter and paid, measured below the cache), overall, per model and per agent, plus the stability sample on its own.
- **Determinism and stability:** every call uses the design's `temperature` (0) and `seed` (sent to OpenRouter, honoured where the provider supports it). With `[run] repeat_fraction > 0`, that share of CV variants and Ranker pools per condition and model (drawn by seed) is re-run `repeats` times with the identical request, cached in `.cache/llm/repeats/r<k>` so reruns stay free. The re-runs go to `repeats.csv` (`repeat` = 1..k) and `stability.json` reports, per model × condition × language × agent, the share of repeats that match the original answer, and for scores and ranks the mean absolute difference and mean SD.

- **Results table:** `results/results.csv`, one row per base CV × identity × language × condition × model × agent (`screener`, `evaluator`, `ranker`; Ranker rows also per pool), with the outcome (`screen_pass`, `score`, or `rank`/`top3`), rationale, token counts, cost, a `cached` flag, the number of `attempts` and an `ok`/`missing`/`rejected` status. Rejected CVs are not scored or ranked. Ranker rows add `pool`, `pool_kind`, `input_position` (1-based position in the Ranker prompt) and `order_seed`; a Ranker call's tokens and cost sit on its `input_position == 1` row. The printed summary shows what this run actually spent (cache hits cost nothing).
- **Cache:** every LLM call is cached on disk, keyed by the hash of the full request (model, messages, temperature, seed, schema). A rerun with the same config makes no calls, costs nothing and replays the same answers, malformed ones included; raise `[run] retries` to ask those again (see `docs/adr/0002-cache-malformed-answers-retry-with-a-note.md`). Delete `.cache/llm` to force fresh calls.
- **Corpus layout:** base CVs are `data/corpus/<role>/<quality>/<id>.<lang>.md` slot templates (see *Identity and language variants* below); job ads are `data/corpus/<role>/job_ad.<lang>.md`.

## Identity and language variants

Each base CV is rendered into 8 identity variants (2 genders × 4 origins: Portuguese, Brazilian, PALOP represented by Cape Verde, Eastern European represented by Ukraine) in every language of the design (EN, PT-PT). Only demographic cues change: the name, the nationality, pronouns and PT-PT grammatical gender. Every variant also states the right to work in Portugal and native-level (C2) Portuguese.

```sh
uv run hcai variants configs/variants.toml               # render + check, write data/variants/
uv run hcai run configs/variants.toml --fake-llm         # run the same 16 variants through the crew
```

- **Name bank:** `data/names/name_bank.toml` has 4 full names per gender × origin, the nationality per language, and the public frequency sources they come from. Each base CV × identity gets one name, picked by a stable hash, so a name stays the same across languages. Set `name_bank = "..."` in a config's `[design]` to use every gender × origin of the bank as the identities.
- **Templates:** base CVs are slot templates (`<id>.en.md`, `<id>.pt.md`) with `$name`, `Nationality: $nationality` (`Nacionalidade: $nationality` in PT-PT), `$work_eligibility` (the standard right-to-work and proficiency lines) and `{{masc|fem}}` for whole gendered words, e.g. `{{He|She}}`, `{{Engenheiro|Engenheira}}`, `{{licenciado|licenciada}}`. See `src/hcai/variants.py`.
- **Diff check:** within each base CV × language, the variants are compared after masking the name, the nationality value, pronouns and the gendered pairs. Any other difference, a gendered form of the wrong gender, or a missing right-to-work / proficiency line fails the check. Templates are also rejected if they use an EN pronoun or a common PT-PT gendered word (Engenheiro, licenciada, ...) outside a slot; write `{{X|X}}` for one that does not refer to the candidate. All-caps acronyms (`MS Excel`) are not pronouns. The PT-PT word list is not exhaustive, so slot every gendered word yourself; an alternation of only an article (`{{o|a}}`) is rejected, slot the whole phrase (`{{o engenheiro|a engenheira}}`). Every identity needs a nationality in every design language. `hcai variants` then exits with status 1 without writing anything, and `hcai run` stops before its first LLM call.
- **Gender cue strength:** generated base CVs (`hcai corpus`) are pronoun-free by instruction, so in EN their only gender cue is the name; hand-written templates such as the `data_analyst` one also use `{{He|She}}`. Keep this in mind when comparing gender gaps across corpora.

## Generating the corpus

`configs/core.toml` is the core design: 3 roles (software engineer, nurse, accountant) × 3 quality levels (strong, average, weak) × `base_cvs_per_role` base CVs per role and quality level. Its `[generator]` section sets the generator model (a third model family; generation refuses a model that is also one of the audited pipeline `models`), its temperature, and how many borderline base CVs to write per role and quality level.

```sh
uv run hcai corpus configs/core.toml              # real generation through OpenRouter
uv run hcai corpus configs/core.toml --fake-llm   # smoke test: fake generator, writes data/corpus/v1-fake
```

- **Job ads:** one per role, with explicit mandatory and desirable requirements.
- **Base CVs:** written from fixed per-quality templates (levers: years above the minimum, desirable requirements met, quantified achievements, certifications) and identity-free: each is a slot template for the variant renderer above, with `$name`, `Nationality: $nationality` and `$work_eligibility` in a fixed header. An answer that breaks the slot rules (e.g. a pronoun outside a slot) is regenerated. `core.toml` takes its 8 identities from the name bank; the generator writes EN base CVs only, so PT-PT for the core corpus still needs a PT version of each base CV. Each CV must quote, for every mandatory requirement, evidence from its own body (or cite `header` for a requirement only the fixed header meets, such as fluent Portuguese); the check is that the quote exists, not that it satisfies the minimum, so the planned hand review still matters. Answers that fail it are regenerated (up to 3 attempts, then generation fails and nothing is written).
- **Borderline CVs** meet every mandatory requirement but only just; the last N CVs of a role × quality level are borderline, per `[generator] borderline`, and carry a note saying which requirement is borderline.
- **Frozen and versioned:** the corpus is written to `corpus_dir` (e.g. `data/corpus/v1`) with a `manifest.json` holding each CV's stable id (`<role>/<quality>/cvNN`), its ground-truth quality level and borderline tag. Running again refuses to overwrite it; pass `--force`, or point `corpus_dir` to a new version. Each CV is also stored as structured JSON next to its `.en.md` template.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
