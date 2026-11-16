# HCAI: auditing demographic bias in a multi-agent LLM hiring pipeline

Human-Centred AI course project (University of Coimbra, 2026/27). The proposal is in `proposal/`; the implementation spec is GitHub issue #1.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env   # then fill in the keys (only needed for real runs; .env is gitignored)
```

### Providers: OpenCode Go (default) and OpenRouter (fallback)

Each model is served by a **provider**, an OpenAI-compatible endpoint with its own API key:

| provider | base URL | key variable | billing |
|---|---|---|---|
| `opencode-go` (default) | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` | subscription: each call reports the dollars it took from the Go allowance |
| `openrouter` (fallback) | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | metered: price per call |

Keys come from the environment or from a `.env` file in the working directory. A variable already set in the environment wins over the file. Keys are only sent in the Authorization header and are never logged or written into results or the cache. Every call sends `User-Agent: hcai/<version>`. OpenCode Go also gets an `x-opencode-session` header with one random id per run (it refuses calls without one); like the key, it is not part of the cache key and is never written anywhere. If the provider still refuses calls for a missing or rejected session (HTTP 400 `MissingSessionID`), the rows are recorded as missing, one warning is printed per run, and `run_summary.json` counts them in `missing_session_errors`. A `[providers.<name>]` table can add more headers with a `headers` subtable. Only the keys of the providers a command uses are needed, and a missing key fails before any call.

A config's `[model_providers]` table names each model's provider (pipeline models, generator and judge). A model left out uses `opencode-go`. **To switch a model to OpenRouter, add or change its one line:**

```toml
[model_providers]
"kimi-k2" = "opencode-go"
"openai/gpt-4o-mini" = "openrouter"
```

Model ids are each provider's own (see `hcai models`). So a switch may also need the id changed in `[design] models`, and a new id counts as a new model in the results. A `[providers.<name>]` table overrides a built-in provider's `base_url`, `key_env` or `billing`, or adds a new one. The placeholder ids in `configs/` are OpenRouter ids until the pilot (#11 Part B) pins OpenCode Go ones. See `docs/adr/0005-providers-rate-limits-stop-and-resume.md`.

```sh
uv run hcai models configs/core.toml                       # list the models of each provider the config uses
uv run hcai models configs/core.toml --provider opencode-go
uv run hcai check configs/core.toml --out results/check.json   # probe every configured model
```

`hcai check` lists each provider's models and flags configured ids it does not offer. It then probes each configured model with 3 × `--repeats` + 2 tiny uncached requests (`--model` limits it to some models). It reports whether the model answers on the OpenAI-style chat endpoint and with a JSON-schema response format (as the agents need), whether identical requests at temperature 0 with the seed give identical answers, whether temperature is honoured (answers vary at temperature 1), and whether the seed is honoured (the same seed at temperature 1 gives the same answer; `?` when the answers never vary). It exits with code 2 if a model fails the endpoint probe.

**Rate and usage limits.** OpenCode Go's limit is a usage budget over a rolling window (about $12 per 5 h). When a provider answers with a rate or usage limit (HTTP 429), the call waits: `Retry-After` if given, else `[run] rate_limit_backoff` seconds (default 30), doubling up to `rate_limit_max_backoff` (900). It then repeats the identical call. These waits are not retries and never make a row `missing`. If a call would wait more than `rate_limit_max_wait` seconds in all (default 21600, 6 h), or the provider has no credits left (HTTP 402), the run **stops cleanly**: it writes `progress.json` to the output directory (the reason and the calls made), writes no results table, and exits with code 3. Every completed call is in the cache, so **rerunning the same command resumes** with no repeated calls.

## Running the skeleton

`configs/skeleton.toml` is a tiny design config: 1 role, 1 hand-written base CV, 2 identities, EN, condition C0. It renders each identity variant of the base CV and runs it through the hiring crew, writes the tidy results table and prints the paired score gap between the identities.

The crew is a sequential [CrewAI](https://docs.crewai.com/) pipeline of three agents, each with structured output:

1. **Screener:** pass/reject against the job ad's mandatory requirements, with a reason.
2. **Evaluator:** 0–100 score and rationale, only for CVs the Screener passed.
3. **Ranker:** orders a pool of screened-in CVs and shortlists the top 3. Pools come from the `[pools]` config section: equal-quality pools (all identities of one base CV) and optional mixed-quality pools (distinct base CVs across the quality levels). The CV order in each Ranker prompt is shuffled with a recorded seed.

CrewAI calls the LLM through the project's own client (`hcai.pipeline.ClientLLM`), so every call is cached and the fake client works in tests. CrewAI telemetry and tracing are switched off. See `docs/adr/0001-ranker-sees-screened-in-cv-texts-only.md` for what the Ranker sees.

```sh
uv run hcai run configs/skeleton.toml                # real run (each model through its provider)
uv run hcai run configs/skeleton.toml --fake-llm     # smoke test: fake LLM, fixed score, no API key needed
```

Options: `--out DIR` (default `results/`, or `results/pilot/` with `--pilot`), `--cache DIR` (default `.cache/llm`; `--fake-llm` uses `.cache/llm-fake` so fake responses never reach a real run) and `--pilot`.

- **Pilot:** `uv run hcai run configs/core.toml --pilot` runs only the config's `[pilot]` slice (by default the first role and one base CV per quality level, under every identity, language, condition and model) without editing the design (a slice of N identities is spread over origins and genders), and prints an estimate of the full run's cost scaled from the slice, per agent and condition (Screener, Evaluator, Blinder and judge by CV variants, Ranker by CVs ranked; rough, as it assumes the same screen-in rate). A condition the slice leaves out (`[pilot] conditions = N`) borrows the cost of a like one: the Blinder from C1/C3, the judge and rubric Evaluator from C2/C3, the plain Evaluator from C0/C1, the Screener and Ranker from any. An agent with no like condition in the pilot is listed under `unestimated` in `run_summary.json` and the printed estimate is marked as a lower bound. The stability share is added to the Screener, Evaluator and Ranker only, the agents it repeats. Pilot calls share the cache, so the full run reuses its Screener, Evaluator and equal-quality Ranker calls; mixed-quality pools are drawn from fewer base CVs in the pilot, so the full run pays for those again.
- **Retries and missing data:** each agent call is validated (parseable JSON, score in 0–100, ranking a permutation). A malformed answer is asked again with a note naming the problem appended to the prompt (a new request, so it has its own cache key); a call that raises (HTTP error, timeout) is repeated after `retry_backoff` seconds (default 2), doubling. After `[run] retries` retries the agent's rows get status `missing` and the error; the `attempts` column counts the calls behind each row.
- **Cost and tokens:** `results/run_summary.json` holds the run's totals: `attempts`, prompt and completion tokens and `cost_total` (what the outputs took to produce, retries and stability sample included; cache hits count their original cost) and `calls`, `cache_hits` and `cost_spent` (what this run sent to the providers and paid, measured below the cache), overall, per model and per agent, plus the stability sample on its own. Each model names its `provider` and `billing`. When a provider reports no price (or a subscription provider reports 0 without OpenCode Go's `usage.opencode` block), the tokens are still counted, the row's `cost` in `results.csv` is blank (unknown), and the costs sum the known prices only: `unpriced_attempts` and `unpriced_calls` count the calls without a price, and the printed summary says the cost is unknown. A pilot's estimate then gives the full run's tokens (`estimated_full_tokens`), and `estimated_full_cost` is `null`.
- **OpenCode Go allowance:** an OpenCode Go answer reports `usage.cost = 0` but, in `usage.opencode`, the dollars the call took from the Go allowance (`allowance`), its `billing_source` (`go`, or e.g. `balance` once Extra Usage bills the pay-as-you-go balance) and its `list_cost`. A call's `cost` is what it took from its billing source: the allowance when billed to `go`, else its price (the reported `cost`, or `list_cost` when that is 0). The cache keeps all three, and `results.csv` adds `allowance` and `billing_source` per row. `run_summary.json` gives, overall and per model, `allowance_total` (the outputs, cache hits included) and `allowance_spent` (this run's calls), `balance_spent` (this run's calls billed elsewhere), `billing_sources` (this run's calls per source) and `not_go_calls` / `not_go_attempts`; the printed summary shows the allowance per model and a WARNING when any call was not billed to `go`. A pilot adds `estimated_full_allowance`, the full run's allowance in dollars (`null` unless every pilot call that answered reported one). Cache entries and results tables from before these fields still load: a cost of 0 recorded there (OpenCode Go's 0, misread as a price) loads as unknown, and they add nothing to the allowance. `uv run python scripts/opencode_go_usage.py --spent [CACHE_DIR]` sums the allowance recorded in a cache (default `.cache/llm`, `repeats/` included) per model without any request, and counts the entries without the field as uncounted. The remaining 5-hour / weekly / monthly budget is shown only in the OpenCode console.
- **Determinism and stability:** every call uses the design's `temperature` (0) and `seed` (sent to the provider, honoured where the model supports it: check with `hcai check`). With `[run] repeat_fraction > 0`, that share of CV variants and Ranker pools per condition and model (drawn by seed) is re-run `repeats` times with the identical request, cached in `.cache/llm/repeats/r<k>` so reruns stay free. The re-runs go to `repeats.csv` (`repeat` = 1..k) and `stability.json` reports, per model × condition × language × agent, the share of repeats that match the original answer, and for scores and ranks the mean absolute difference and mean SD.

- **Results table:** `results/results.csv`, one row per base CV × identity × language × condition × model × agent (`screener`, `evaluator`, `ranker`, plus `blinder` under C1/C3 and `judge` under C2/C3; Ranker rows also per pool), with the outcome (`screen_pass`, `score`, or `rank`/`top3`), rationale, token counts, cost, a `cached` flag, the number of `attempts` and an `ok`/`missing`/`rejected` status. Rejected CVs are not scored or ranked. Ranker rows add `pool`, `pool_kind`, `input_position` (1-based position in the Ranker prompt) and `order_seed`; a Ranker call's tokens and cost sit on its `input_position == 1` row. Under C2/C3, Evaluator rows add `evidence`, `judge_flag`, `judge_verdicts`, `escalated` and `escalation_reason`, and judge rows add `judge_verdicts`, `judge_flag` and `judge_model` (see *Conditions C2 and C3* below). The printed summary shows what this run actually spent (cache hits cost nothing).
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

## Name manipulation check

Before the full run, check that every name of the bank signals its intended gender and origin to the audited models (spec #1, story 11):

```sh
uv run hcai names configs/core.toml                      # every name x pipeline model, 5 answers each
uv run hcai names configs/core.toml --repeats 10 --threshold 0.9 --out results/names
```

- **What is asked:** each pipeline model (`[design] models`; not the generator or the judge) sees one name at a time, as "a job applicant in Portugal", and answers in JSON with the perceived gender (male / female / unsure) and the most likely origin (Portuguese; Brazilian; Cape Verdean or other PALOP; Ukrainian or other Eastern European; Other or unsure). The prompt is the same for every name but the name: it never states the intended gender or origin. The option order changes per repeat (male and female alternate first; the origins are shuffled, never opening with the bank's origins in bank order; "unsure" and "Other or unsure" always last), and each repeat has its own seed, so the repeats are separate samples, each cached on its own. A rerun with the same settings makes no calls. The temperature is the design's (`--temperature` to override).
- **Outputs** (in `--out`, default `results/names`): `perceptions.csv` (one row per answer: name, intended and perceived gender and origin, model, repeat; blank when the answer is missing), `summary.csv` (per name x model: answers, missing, gender and origin hit rates, pass) and `cells.csv` (per gender x origin cell and model, plus `all`: the mean hit rates over its names with a valid answer, how many those are, and how many pass). The printed report shows the cells and lists the failing names.
- **Pass rule:** a name passes when, for every model, both its gender and its origin hit rate are at least `--threshold` (default 0.8) and none of its answers is missing. Rates are over the valid answers; a missing answer (a failed call, or one outside the options) is not cached, so rerunning asks it again. The command exits 2 when a name fails. The report names failed calls with their error; 10 failed calls in a row to one model (e.g. a wrong model id) stop the check with exit 1, and a provider limit stops it with exit 3, as in `hcai run`: rerun to resume from the cache. It never edits the name bank: replace failing names in `data/names/name_bank.toml` by hand and rerun (see `docs/adr/0009-name-manipulation-check.md`).

## Generating the corpus

`configs/core.toml` is the core design: 3 roles (software engineer, nurse, accountant) × 3 quality levels (strong, average, weak) × `base_cvs_per_role` base CVs per role and quality level. Its `[generator]` section sets the generator model (a third model family; generation refuses a model that is also one of the audited pipeline `models`), its temperature, and how many borderline base CVs to write per role and quality level.

```sh
uv run hcai corpus configs/core.toml              # real generation (the generator's provider)
uv run hcai corpus configs/core.toml --fake-llm   # smoke test: fake generator, writes data/corpus/v1-fake
```

- **Job ads:** one per role, with explicit mandatory and desirable requirements.
- **Base CVs:** written from fixed per-quality templates (levers: years above the minimum, desirable requirements met, quantified achievements, certifications) and identity-free: each is a slot template for the variant renderer above, with `$name`, `Nationality: $nationality` and `$work_eligibility` in a fixed header. An answer that breaks the slot rules (e.g. a pronoun outside a slot) is regenerated. `core.toml` takes its 8 identities from the name bank. Each CV must quote, for every mandatory requirement, evidence from its own body (or cite `header` for a requirement only the fixed header meets, such as fluent Portuguese); the check is that the quote exists, not that it satisfies the minimum, so the planned hand review still matters. Answers that fail it are regenerated (up to 3 attempts, then generation fails and nothing is written).
- **PT-PT:** for every design language other than EN (`core.toml`: `["en", "pt"]`), the same generator model translates each job ad and base CV (at temperature 0) into European Portuguese, written next to the EN one with the same stable id and quality level (`job_ad.pt.md`, `cvNN.pt.md`, `cvNN.pt.json`; the PT-PT header and section headings are fixed). A translation is regenerated unless it keeps the original's structure (jobs, bullets, degrees, certifications, skills) and every number (dates, years, percentages, requirement minimums), puts every gendered word in a `{{masc|fem}}` slot and passes the diff check for both genders (a differing word of a slot may not reappear outside it), and uses no common PT-BR term. The PT-BR check (`src/hcai/ptpt.py`: e.g. usuário, equipe, gerenciamento, ensino médio, registro, "está fazendo") also rejects any PT-PT template in `hcai variants` and `hcai run`; it is a list of common terms, not a dialect classifier, so the hand review still matters. To add PT-PT to a corpus already frozen in EN only, run `hcai corpus <config> --translate`: it writes the PT-PT files and records them, and the translating model (`translated_by`), in the manifest without touching the EN files. `--force` rewrites existing translations, but cached answers are reused: point `--cache` elsewhere or delete it to get fresh translations.
- **Borderline CVs** meet every mandatory requirement but only just; the last N CVs of a role × quality level are borderline, per `[generator] borderline`, and carry a note saying which requirement is borderline.
- **Failures and reruns:** each document gets `[generator] max_attempts` (default 4) new generator calls per run; a gendered word right after a candidate slot (`{{Enfermeiro|Enfermeira}} dedicado`) is slotted automatically when that passes every check, otherwise the retry names each leaked word and its fix. The English corpus is frozen as soon as it is valid; valid translations are kept per document in `<corpus_dir>/partial/<lang>/` until the language is complete. If documents still fail, `hcai corpus` lists them with the reason and exits 1; rerun the same command to replay everything valid from the cache at no cost and ask only the failing documents again, with new attempts (see `docs/adr/0006-corpus-reruns-continue-with-fresh-attempts.md`).
- **Language check:** an English job ad or base CV must read as English and a PT-PT translation must not read as English (`src/hcai/language.py`: a count of English vs Portuguese function words over the fields, words inside a capitalised name such as "Ordem dos Contabilistas Certificados" not counted, plus a rule for Portuguese job titles; a translation may not copy a field verbatim); otherwise it is retried, with the offending fields named. To repair a corpus frozen before this check, run `hcai corpus <config> --recheck`: it keeps every frozen document that passes today's checks (no call), regenerates the failing ones (a failing job ad with all its base CVs) and translates again only the regenerated and failing documents (see `docs/adr/0008-generated-documents-are-checked-for-their-language.md`).
- **Frozen and versioned:** the corpus is written to `corpus_dir` (e.g. `data/corpus/v1`) with a `manifest.json` holding each CV's stable id (`<role>/<quality>/cvNN`), its ground-truth quality level and borderline tag. Running again refuses to overwrite it; pass `--force`, or point `corpus_dir` to a new version. Each CV is also stored as structured JSON next to its `.en.md` template.

## Condition C1: blinding

Add `"C1"` to a config's `conditions` (e.g. `conditions = ["C0", "C1"]`) to run every CV through a **Blinder** agent before the crew. Rows of that condition have `condition = "C1"`.

- **Blinder:** a one-agent CrewAI crew on the same pipeline model as the crew (cached, retried and faked like the other agents). It is asked to remove the name, nationality, age cues and gender markers (pronouns, gendered words and their agreement) and keep everything else. Its row in the results table has `agent = "blinder"`, its tokens and cost, and in `rationale` its own answer (before redaction).
- **Redaction:** the Blinder's answer is checked against the demographic tokens: every name of the design's identities (full names and their parts, in their own case or all caps), every identity's nationality in the CV's language (in PT-PT also its masculine form), and both forms of every gendered pair of the template and each word in which a phrase slot's forms differ (plus EN pronouns and titles). Any token it left in becomes `[redacted]`, so the Screener, Evaluator and Ranker only ever see the redacted blinded text. Lines that render the same for every identity (the right-to-work and proficiency lines, say) are left as they are. If the Blinder fails after its retries, the CV's Screener, Evaluator and Ranker rows are `missing` and the CV is not sent. See `docs/adr/0003-blinding-redacts-what-the-blinder-leaves.md`.
- **Proxy leaks:** `results/proxy_leaks.csv` has one row per blinded CV: `redacted` (cues the Blinder left in; a `[redacted]` it wrote itself does not count), `own_gender_words` / `other_gender_words` (PT-PT gendered job titles, adjectives and participles, still in the crew's text; a lower bound, as the word list is not exhaustive; in EN the gendered words are all tokens, so only `redacted` measures gender leaks), and how many of the CV's universities and other languages, and whether its header location, survive verbatim. `proxy_leaks.json` and the printed table summarise them per model × condition × language.
- `--fake-llm` uses a Blinder that returns the CV unchanged, so the redaction step does all the blinding.

## Conditions C2 and C3: rubric, judge and escalation

Add `"C2"` to `conditions` to swap in the **rubric Evaluator**, and `"C3"` for C1 + C2 (blinding, then the rubric). `core.toml` runs C0, C1 and C2; C3 is a stretch goal.

- **Rubric:** the Evaluator scores each criterion of the job ad (one per bullet line) from 0 to 10 and quotes the CV text behind each score. The Evaluator row's `score` is the 0–100 total, and `evidence` holds the per-criterion scores and quotes as JSON.
- **Judge:** a third-family model (`[oversight] judge_model`, default `[generator] model`) checks each quote against the CV the Evaluator saw (under C3, the blinded text). It writes a `judge` row with one verdict per criterion (`faithful`/`unfaithful` and a reason) in `judge_verdicts`, and `judge_flag`. The row is billed to `judge_model`.
- **Escalation:** an Evaluator decision is `escalated` when a quote is flagged, the judge gave no valid verdict, the CV got no score although it was not screened out (the Evaluator, Screener or Blinder failed), or the score is within `[oversight] margin` of the cutoff (`cutoff = "median"`, the median score of the role × language × condition × model, or a fixed score). Escalated rows also go to `results/review_queue.csv` and are counted under `oversight` in `run_summary.json`. See `docs/adr/0004-rubric-evaluator-judge-rows-and-escalation.md`.

## Analysis

```sh
uv run python -m hcai.analysis results/results.csv --reference pt_m --out results/analysis
```

Writes one CSV per table to `results/analysis/` and prints a summary. Fairness (RQ1–RQ3): `gaps`, `equivalence`, `impact_ratios`, `model_terms`, `identity_contrasts`, `rank_identity`, `rank_position` (see the module docstring of `src/hcai/analysis.py`). Also:

- **Ranking quality (RQ3), `ranking_quality.csv`:** the Spearman correlation between Evaluator scores and the base CV's true quality level, and the Ranker's pairwise accuracy on mixed-quality pools (the share of pairs of ranked CVs of different quality put in the right order), per condition and model and pooled (`model = all`), with 95% cluster-bootstrap CIs (base CVs resampled within their quality level; pools). `delta` is the change from C0, with a CI from paired resamples; `within_tolerance` checks the pre-registered success criterion (at most 5 pp of pairwise accuracy lost, on the point estimate). Quality levels come from the base CV id, best first by `--quality-order` (default `strong,average,weak`); screened-out CVs are neither scored nor ranked, so both measures cover screened-in CVs only.
- **Oversight (RQ4), `oversight.csv`:** per rubric condition and model, the Evaluator decisions, how many the judge gave a verdict on and flagged (`flag_rate`), the escalations to the human review queue (`escalation_rate`, the cost of oversight) with their reasons, and the judge's calls and cost.

## Human audit of the judge (RQ4)

```sh
uv run python -m hcai.audit export results/results.csv --config configs/core.toml --out results/audit
# two raters each fill in the `label` column of a copy of audit_items.csv, independently
uv run python -m hcai.audit agreement results/audit/audit_key.csv labels_a.csv labels_b.csv --out results/audit
```

- **Export:** 50 rubric justifications (`--n`), 25 flagged by the judge and 25 not, drawn by `--seed` and shuffled. `audit_items.csv` (UTF-8 CSV, opens in a spreadsheet) has, per item, the CV the Evaluator and the judge saw (under C3 the redacted blinded text), each criterion's score and quote, and empty `label` and `comment` columns; it shows no judge verdict, flag or escalation. `audit_key.csv` holds each item's provenance and the judge's verdicts: keep it from the raters. `INSTRUCTIONS.md` tells the raters what to do. The config must be the one that produced the results (the CVs are rendered from its corpus).
- **Labels:** `faithful` (every quote is in the CV and supports its score) or `unfaithful` (at least one is not), any case. Each sheet is checked against the key: every item labelled exactly once, no unknown ids, no other labels; any problem is listed and nothing is computed.
- **Agreement:** `agreement.csv` has Cohen's κ of the judge against each rater (`judge-human`) and of the raters against each other (`human-human`, the reference), with the observed and chance agreement, the 2×2 counts and a bootstrap 95% CI over items. Raters are named by their file's stem. κ is undefined (empty) when both sides used one same label throughout. The sample is stratified 50/50 by the judge's flag, so κ describes agreement on that balanced sample, not at the judge's real flag rate (see `oversight.csv`). Label sheets may be saved by a spreadsheet as UTF-8 or Windows-1252, comma- or semicolon-separated; blank rows are ignored.

## Tests

```sh
uv run pytest
uv run pyright
```

Tests inject a fake LLM client and never touch the network (a fixture blocks socket connections).
