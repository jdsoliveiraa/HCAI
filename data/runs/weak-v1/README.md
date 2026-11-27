# Weak-model run, version 1 (post hoc)

The frozen outputs of `hcai run configs/weak.toml` over corpus `corpus-v1`. **Post hoc, not pre-registered:** added
after the core results were known, to check whether weaker models show the demographic bias that `gpt-5.6-luna` and
`qwen3.8-flash` did not. The analysis is in `analysis/` (`hcai reproduce data/runs/weak-v1 --per-model --post-hoc
--compare data/runs/core-v1`).

## The models are proxies, not weak models

OpenCode Go (the only provider used) lists none of the planned classes: an older closed model (GPT-3.5/4o-mini era),
a small open 7-8B model, a quantized 7-8B model, or a tiny 1-3B model. Every chat model in its `/models` listing
(37 ids, 2026-11-26) is a 2026 release. The run uses the cheapest or smallest open chat tiers that pass `hcai check`
(chat endpoint, JSON schema; `docs/pilot/check-weak-2026-11-26.json`):

| Planned class | On Go | Used here |
|---|---|---|
| older closed (GPT-3.5/4o-mini era) | no | none (`muse-spark-1.2-contributor` needs an opt-in to endpoints that train on request data; `gpt-6-luna` passes the check but is newer than `gpt-5.6-luna`) |
| small open 7-8B | no (`qwen3-embedding-8b` is an embedding model) | `mimo-v2.6-flash` (open weights, the cheapest open chat tier; no size published) |
| quantized 7-8B | no | none |
| tiny open 1-3B | no | `beam-free` (open weights, mixture of experts, 23B active parameters: the only size Go states), `hy4-flash-free` (open weights, "lower-cost tier" of Hy4; no size published) |

`glm-5.3-flash` (open weights) also passes the check but was not offered for selection: the choice was limited to 4
options, and it wrote the most tokens in the check (8.9k against 1.0k–7.0k), so it would have been the slowest.
None of the three is known to be weaker than the core models. Read the results as "three more 2026 low-cost
models", not as "weak models".

## Facts

| Fact | Value |
|---|---|
| Design | 3 roles (nurse first) × 9 base CVs × 8 identities × EN and PT-PT × C0 only × 3 models; Screener, Evaluator, Ranker (equal-quality pools and 2 mixed pools per role × language); same corpus, name bank (the same name for every base CV × identity × language as core-v1), prompts, temperature 0, seed 42, `[pools]` and repeat fraction as `core.toml` |
| Differences from `core.toml` | models, role order, conditions (C0 only); `[run] concurrency` 6 (core 3; requests and cache keys do not depend on it); `[pilot]` slice; no `[oversight]` (unused in C0) or `[site]`. The Ranker's input order and the repeat draw hash the model name, so they differ from core-v1's per model, as they differ between the two core models |
| Rows | 4,176 (4,106 ok, 70 rejected: the CVs the Screener rejected, with their Evaluator and Ranker rows); missing 0 (0.00 %, target under 2 %) |
| Rejected by model | `mimo-v2.6-flash` 23 CVs, `beam-free` 8, `hy4-flash-free` 0 (of 432 each) |
| Calls | 2,947 records in `calls.jsonl` (pilot 55 + full run 2,892; the full run replayed the 55 pilot calls from the cache): 2,890 fresh answers, 1 failed call (`beam-free` Ranker, 32,768 completion tokens, cut at the length limit; the retry answered), 1 rate-limit wait (5 s) |
| Cost | **USD 0.19 of OpenCode Go allowance** (pilot 0.0047 + full run 0.1849; all `mimo-v2.6-flash`; `beam-free` and `hy4-flash-free` report 0), balance (paid) USD 0; every call billed to `go`. The model checks (about 80 small calls) are not logged per call; their tokens priced at list rates are under USD 0.01 |
| Cost control | budget USD 5, checked by hand before the full run: the pilot (1 base CV × 8 identities × EN × 3 models, `[pilot]` in `weak.toml`) estimated USD 0.28. No code stops a run at a dollar amount. The guards in code are `[run] allowed_billing = ["go"]` (the run stops at the first call billed to the paid balance) and Go's own caps (about USD 11 per 5 h) |
| Time | 26 Nov 2026: pilot 21:25–21:26 UTC; full run 21:27–22:19 UTC (52 min), `[run] concurrency = 6`, 1 start, no restart |
| Provenance | the run log names commit `1e01a57` "(clean)", but `configs/weak.toml` was then an untracked file (the clean check ignores untracked files). It was committed unchanged in `974d08f`; its sha256 matches the log. No runner or prompt code changed after the run |
| Stability | 5 % repeat sample (196 repeat rows): `repeats.csv`, `stability.json`. The sample is drawn per model (the draw hashes the model name), so it holds other CVs than core-v1's |

## Files

- `results.csv`: one row per agent decision (same columns as core-v1).
- `repeats.csv`, `stability.json`: the 5 % repeat sample and its agreement / score SD.
- `run_summary.json`: totals and missing share per model × language; `pilot_run_summary.json`: the pilot's, with
  the cost estimate (`pilot.estimated_full_allowance`).
- `calls.jsonl.gz`: one record per LLM call of the pilot and the full run (no prompt or answer text).
- `run.log.gz`: the pilot's and the full run's logs, in that order.
- `weak.toml`: the config used (sha256 `73b8f6d9…`, as logged at the start of both runs).
- No `proxy_leaks` or `review_queue`: C0 has no Blinder and no judge.
