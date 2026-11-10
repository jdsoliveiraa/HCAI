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
- **Base CVs:** written from fixed per-quality templates (levers: years above the minimum, desirable requirements met, quantified achievements, certifications) and identity-free: each is a slot template for the variant renderer above, with `$name`, `Nationality: $nationality` and `$work_eligibility` in a fixed header. An answer that breaks the slot rules (e.g. a pronoun outside a slot) is regenerated. `core.toml` takes its 8 identities from the name bank. Each CV must quote, for every mandatory requirement, evidence from its own body (or cite `header` for a requirement only the fixed header meets, such as fluent Portuguese); the check is that the quote exists, not that it satisfies the minimum, so the planned hand review still matters. Answers that fail it are regenerated (up to 3 attempts, then generation fails and nothing is written).
- **PT-PT:** for every design language other than EN (`core.toml`: `["en", "pt"]`), the same generator model translates each job ad and base CV (at temperature 0) into European Portuguese, written next to the EN one with the same stable id and quality level (`job_ad.pt.md`, `cvNN.pt.md`, `cvNN.pt.json`; the PT-PT header and section headings are fixed). A translation is regenerated unless it keeps the original's structure (jobs, bullets, degrees, certifications, skills) and every number (dates, years, percentages, requirement minimums), puts every gendered word in a `{{masc|fem}}` slot and passes the diff check for both genders (a differing word of a slot may not reappear outside it), and uses no common PT-BR term. The PT-BR check (`src/hcai/ptpt.py`: e.g. usuário, equipe, gerenciamento, ensino médio, registro, "está fazendo") also rejects any PT-PT template in `hcai variants` and `hcai run`; it is a list of common terms, not a dialect classifier, so the hand review still matters. To add PT-PT to a corpus already frozen in EN only, run `hcai corpus <config> --translate`: it writes the PT-PT files and records them, and the translating model (`translated_by`), in the manifest without touching the EN files. `--force` rewrites existing translations, but cached answers are reused: point `--cache` elsewhere or delete it to get fresh translations.
- **Borderline CVs** meet every mandatory requirement but only just; the last N CVs of a role × quality level are borderline, per `[generator] borderline`, and carry a note saying which requirement is borderline.
- **Frozen and versioned:** the corpus is written to `corpus_dir` (e.g. `data/corpus/v1`) with a `manifest.json` holding each CV's stable id (`<role>/<quality>/cvNN`), its ground-truth quality level and borderline tag. Running again refuses to overwrite it; pass `--force`, or point `corpus_dir` to a new version. Each CV is also stored as structured JSON next to its `.en.md` template.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
