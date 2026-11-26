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
| Score model | mixed model with `(1\|CV) + (1\|name)` | same fixed effects, primary adds `(1\|CV:lang:cond:model)`; the literal model is a robustness check | the literal model's identity CIs were about ±4 points as run, ±1.5 without the failed answer (pairing model about ±0.7) | ADR 0012, `model_terms.csv` (`spec`) |
| Logistic models | logistic counterpart for screening and top 3 | top 3 only; Screener reported by gaps, pass counts, impact ratios | Screener passed 99.7%: no finite SEs | `model_notes.csv` |
| Failed answer | missing data recorded, not dropped | 1 rubric answer ("No CV content was provided", all 0) counted as missing; as-run results reported beside | model failure (the CV was in the request); it moved every C2 gap | ADR 0012, `exclusions.csv`, `sensitivity_as_run/` |
| Ranker quality | pairwise accuracy on mixed pools | 1.00 in every condition (ceiling), so Spearman of Evaluator scores carries the quality cost | mixed pools were too easy | `ranking_quality.csv` |
| Human audit | 50 justifications, 2 raters, κ | 1 rater (judge–human κ only, no rater–rater κ; the rater accepted a score the full CV supports, a looser rule than the judge's); 50 items (25 flagged of the 27 valid answers the judge flagged; the failed answer is not drawn), rated per quote on the site's audit page; κ per item (pre-registered) and per quote | per-quote labels make κ on the judge's per-criterion verdicts possible | ADR 0012, `docs/audit/labels/` |
| Extra analyses | not in the proposal | 4 post-hoc analyses (not pre-registered): gaps against the mean of the 8 identities; gaps on the borderline CVs and on near-cutoff pairing units; Ranker identity effects after input position; gaps against the run-to-run noise of the repeat sample | the C1 gaps share one reference (PT M); small subsets, the position effect and the score noise could hide or fake a gap | `hcai.extra`, `tables/extra_*.csv`, FINDINGS "Extra analyses" |

## Numbers to cite

- Pilot (1 role × 1 base CV per level): 1,092 rows, 0 missing (after `[run] request_timeout = 600`; qwen's Blinder
  writes up to 11k tokens on PT CVs).
- Full-run cost estimate: $29.70 of Go allowance (judge $12.8). Actual: about $28.3 of allowance, $0 balance
  (`data/runs/core-v1/README.md`).
- Full run (#12, 23 Nov): 10,076 rows, 0 missing (target under 2%); 0 failed calls, 0 timeouts, 0 rate-limit
  waits; about 19.5 h, the last 7 h with 3 calls in flight (`[run] concurrency`, same requests and cache keys).
- Go caps measured during the run: monthly about $54, weekly about $28 (50%), 5 h about $11 (20%).
- Name check: lowest cell, female Brazilian, mean p(origin) 0.648; all others ≥ 0.68.

- Human audit (1 rater, 50 items): judge–human κ 0.08 per item (95% CI −0.07 to 0.24), 0.33 per quote (−0.00 to 0.61).

## Fill in later

- Headline analysis numbers: `data/runs/core-v1/analysis/FINDINGS.md`.
