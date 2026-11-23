# Report notes

Input for the final report. The proposal (`proposal/ijcai27.tex`) is final and does not change: the report must
state each difference between what was proposed and what was done. Keep this file short: one line per fact,
with a pointer to the record. Add to it when a ticket changes something the report must mention.

## Proposal vs. done

| Topic | Proposal | Done | Why | Record |
|---|---|---|---|---|
| LLM access | OpenRouter | OpenCode Go subscription; OpenRouter kept as a one-line fallback | cost: the subscription covers the runs | ADR 0005, #11 |
| Determinism | temperature 0, fixed seed | temperature 0 and a seed are still sent, but no model is deterministic at T=0 or honours the seed | provider behaviour, measured by probes | `docs/pilot/check-2026-11-14.json` |
| Stability | "a small sample is repeated" | 5% repeat sample per run; agreement and score SD reported | now the only control for run-to-run variation | `[run] repeat_fraction`, `stability.json` |
| Models | one commercial + one open-weight, third family as generator/judge | `gpt-5.6-luna` (proprietary), `qwen3.8-flash` (open-weight); `gemini-4-argon` generator and judge | JSON-schema support, low tokens per call (Go budget) | `configs/core.toml`, #11 |
| Base CVs | about 10 per role, about 480 CV texts | 9 per role (3 per quality level), 27 base CVs, 432 variants (27 × 8 × 2) | budget; borderline CVs: 1 average + 1 weak per role | `configs/core.toml`, `data/corpus/v1` |
| PT-PT CVs | generated from templates in 2 languages | English generated first, then translated to PT-PT by the generator with `{{masc\|fem}}` gender slots | one base CV, same content in both languages | ADR 0006–0008 |
| PALOP group | PALOP | represented by Angola; Eastern Europe by Ukraine | Cape Verdean names read as Portuguese/Brazilian in the name check | `data/names/name_bank.toml`, `docs/pilot/name-check.md` |
| Names | several names per identity | 4 full names per gender × origin cell (32), each first name and surname unique | no shared name cue between identities | `name_bank.toml` |
| Manipulation check | "confirms that the names signal" | probabilities per option from both pipeline models, 5 repeats, hedges left out, strict rule, all-hedge answers asked again (10); all 32 pass | single-choice answers hid partial signal | ADR 0009, 0011; `docs/pilot/name-check.md` |
| Realism check | sample checked by hand | 6 job ads + 6 CVs reviewed; 74 recorded text edits (generator's "(fictitious)" labels, a real hospital brand, 3 ad fixes), no regeneration; then frozen (tag `corpus-v1`) | edits are identity-neutral (made before variants) | `docs/pilot/cv-realism-review.md`, `docs/pilot/corpus-corrections.md` |
| Blinding (C1) | an agent removes demographic cues | LLM Blinder on the model under test, then a fixed redaction of any demographic token it left in, the same for every identity | a leak must not make missingness depend on identity | ADR 0003 |
| Judge / escalation | judge in C2/C3; escalate flagged or near-cutoff | judge in C2; escalation = judge flag or score within 5 points of the median cutoff | — | ADR 0004, `[oversight]` |
| C3 (both) | stretch | not run (`conditions = C0, C1, C2`) | budget | `configs/core.toml` |
| Schedule | pilot 15 Nov, final 27 Nov | pilot done 23 Nov | name check and corpus fixes took longer | #11 |

## Numbers to cite

- Pilot (1 role × 1 base CV per level): 1,092 rows, 0 missing (after `[run] request_timeout = 600`; qwen's Blinder
  writes up to 11k tokens on PT CVs).
- Full-run cost estimate: $29.70 of Go allowance (judge $12.8). Actual cost: fill in after #12.
- Name check: lowest cell, female Brazilian, mean p(origin) 0.648; all others ≥ 0.68.

## Fill in later

- #12: actual calls, cost, missing rows, run time, rate-limit waits.
- #13: the analysis actually used (proposal: mixed-effects model with the formula in §Evaluation; the validation
  report notes it cannot answer RQ3 as written), the human audit (proposal: 50 justifications, two raters, κ).
