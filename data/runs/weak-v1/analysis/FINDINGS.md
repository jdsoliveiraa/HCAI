# Findings: weak-model run v1 (post hoc)

**Post hoc, not pre-registered.** Added on 26 Nov 2026, after the core results were known, to ask: do weaker models
show the demographic bias that `gpt-5.6-luna` and `qwen3.8-flash` did not? Read every result here as exploratory.

Made by `uv run hcai reproduce data/runs/weak-v1 --per-model --post-hoc --compare data/runs/core-v1` (offline, seed 0, 2,000
bootstrap resamples, 200 refits of the mixed model, 5,000 permutations). Every number below is in `tables/` (pooled
over the 3 models), `tables/per_model/` (each model alone; Holm within the same families as the core run, e.g. the 7
gaps of one agent, now within one model) or `headline.json`. Gaps are
against PT M (Portuguese man), in score points (0–100) or percentage points (pp). "Survives Holm" means a
Holm-adjusted p < 0.05 within its family.

## The models are not weak models

OpenCode Go (the only provider allowed) has none of the planned classes: no GPT-3.5/4o-mini-era closed model, no
7-8B open model, no quantized model, no 1-3B model (`hcai models`, 37 ids; see `../README.md`). The run uses three
low-cost 2026 open chat tiers that pass `hcai check`: `mimo-v2.6-flash`, `beam-free` (MoE, 23B active) and
`hy4-flash-free`. None is known to be weaker than the core models. Per model, their Spearman correlation of Evaluator
score with true quality (0.925–0.937, `tables/ranking_quality.csv`) is in the range of core-v1 C0's per-model values
(gpt-5.6-luna 0.926, qwen3.8-flash 0.900; core-v1 `headline.json`). **This run does not test weak models.** It tests
three more models of the same generation. `glm-5.3-flash` also passed the check and was not run (`../README.md`).

**Multiplicity.** Holm does not correct across the 3 models. A post-hoc look at 3 models triples the chance that one
of them shows a false positive. `summary.csv` therefore also gives the survivors with p_holm × 3 < 0.05
(`gaps_surviving_holm_x_models`).

## Data

- 3 roles × 9 base CVs × 8 identities × EN and PT-PT × C0 × 3 models: 4,176 rows, 4,106 ok, 70 rejected (the CVs
  the Screener rejected, with their Evaluator and Ranker rows), **0 missing**.
- Failed or invalid structured outputs (`tables/per_model/outputs.csv`): 0 malformed answers (parse errors) for
  every model and agent; 0 failed rubric answers (no rubric in C0). 1 failed call: a `beam-free` Ranker answer
  ran to the 32,768-token length limit with no answer; the retry answered, so no row is missing.
- Ranker tests use complete equal-quality pools only: 53 (beam-free), 54 (hy4-flash-free), 49 (mimo-v2.6-flash) of 54.

## Per model

| | mimo-v2.6-flash | beam-free | hy4-flash-free |
|---|---|---|---|
| Evaluator mean (SD), distinct scores | 89.4 (7.7), 9 | 87.3 (10.0), 11 | 89.4 (8.1), 18 |
| Evaluator gaps vs PT M | +0.33 to +1.45 | −0.66 to +0.06 | +0.22 to +0.35 |
| Equivalence (±2): largest \|90% CI\| | **2.18, not shown** | 1.26, equivalent | 0.69, equivalent |
| Gaps surviving Holm (21: 7 per agent) | **9** (Evaluator 3, Ranker 6) | 0 | 0 |
| … also with p_holm × 3 models < 0.05 | **8** (Evaluator 2, Ranker 6) | 0 | 0 |
| Evaluator vs the mean of the 8 | −0.78 (PT M) to +0.67; 0 survive Holm | −0.50 to +0.21; 0 | −0.24 to +0.11; 0 |
| Screener pass rate (per identity) | 94.7% (90.7–98.1%) | 98.1% (all 98.1%) | 100% |
| Screener gaps surviving Holm | 0 | 0 | 0 |
| Ranker: complete pools returned in input order | 20% | 100% | 100% |
| Ranker identity effect after position (rank) | −2.83 to +1.29 places; 8 of 8 survive; joint p 2e-18 | not testable (input order) | not testable (input order) |
| Ranker top-3 rate, PT F / PT M / others (`ranker_top3`) | **98% / 94% / 10–31%** | by position only | by position only |
| Equal-pool rationales naming nationality (`ranker_rationales`) | **44 of 54** | 1 of 53 | 4 of 54 |
| Repeat SD of one score (items) | 0.94 (31) | 1.61 (20) | 0.55 (10) |
| Gaps vs PT M inside run-to-run noise | 1 of 7 | 6 of 7 | 0 of 7 |
| Missing rows / retried decisions | 0 / 0 | 0 / 1 | 0 / 0 |

### mimo-v2.6-flash: the Ranker discriminates by origin, openly

- In equal-quality pools (8 copies of one CV, only the name and nationality differ), mimo puts both Portuguese
  identities in the top 3 in nearly every pool: PT F 98%, PT M 94% of the 49 complete pools. The others get 10–31%
  (per language 8–33%; EN and PT-PT alike; `per_model/ranker_top3.csv`).
  Against PT M, the 6 non-Portuguese top-3 gaps are −63 to −84 pp, and all survive Holm. PT F vs PT M (+4 pp) does not.
  Against the mean of the 8: PT F +60 pp, PT M +56 pp, and 4 others −19 to −27 pp all survive Holm.
- It is not input position. Mimo keeps the input order in only 20% of the complete pools. After position, the identity effects on rank
  are PT F −2.83 and PT M −2.41 places (better), and the others +0.41 to +1.29. All 8 survive Holm, joint F p = 2e-18.
- Its rationales say why (`results.csv`, Ranker rows, `input_position = 1`). Of its 54 equal-quality rationales, 44
  match the nationality pattern (`NATIONALITY_PATTERN` in `hcai.permodel`; beam-free 1, hy4-flash-free 4;
  `per_model/ranker_rationales.csv`). The tie-break pattern matches all three models (they all see the tie), so
  it does not separate them. Example: "ranking is based on tie-breakers: Portuguese
  nationality (1, 4) avoids any credential-recognition questions for foreign degrees; then other Portuguese-speaking
  countries … (Angola …; Brazil …); then non-Lusophone candidates (Ukraine …)". The CVs state the same Portuguese
  degree, licence and C2 Portuguese for every identity.
- This bias needs a tie. In mixed-quality pools (distinct CVs), pairwise accuracy is 0.991, and 1 of the 12 rationales
  matches the nationality pattern (hy4-flash-free also 1, beam-free 0). With identical candidates, "nationality as tie-breaker" decides the shortlist.
- Its Evaluator gaps are small and run the other way. Every identity scores 0.3–1.45 points above PT M, and 3 survive
  Holm (BR F +1.35, EE F +1.27, EE M +1.45); with the ×3 correction only EE F and EE M (0.048), so this is weak
  evidence. The equivalence check fails (2.18). Against the mean, nothing survives
  (PT M −0.78 is the lowest). This is the core C1 pattern: one reference identity that scores a little lower, not an
  origin ranking. Its Screener rejects PT M most often (90.7% pass) but no Screener gap survives Holm.
- **Verdict:** clear origin bias in the Ranker on ties (Portuguese first), invisible in the Evaluator and Screener.

### beam-free and hy4-flash-free: no bias found where it can be tested

- Evaluator: no gap survives Holm, and both pass equivalence (1.26, 0.69). Screener: no gap. beam-free rejects the same
  base CVs for every identity, and hy4-flash-free rejects none.
- Ranker: both return every equal-quality pool in its input order (mean rank 1.0 at position 1, 8.0 at position 8).
  The order is shuffled independently of identity, so their raw identity gaps are shuffle chance. One raw
  reference-free top-3 gap survives Holm (hy4-flash-free PALOP F −17 pp: its mean input position is 5.09). After
  position there is no residual, so the identity test cannot run (`input_order_only`, p left blank). This is a
  numerical limit, not a null result. The Ranker does not look at the CVs in these pools.
- Noise: hy4-flash-free's gaps vs PT M (+0.22 to +0.35) are outside call noise (z 2.1–3.3), but against the mean only
  PT M stands out (−0.24). Its repeat SD comes from 10 items only (95% CI 0 to 0.84), so the noise test is weak.
- **Verdict:** no demographic bias of practical size in the Screener and Evaluator. The Ranker cannot be tested: it
  ignores the CVs and copies a random input order, so its output does not depend on identity, but it also gives
  no evidence about how it would treat identity if it read the pools.

## Pooled over the 3 models (`tables/`)

- Evaluator gaps +0.05 to +0.57, equivalent (largest |90% CI| 0.95), 0 survive Holm. Ranker top-3 gaps vs PT M: 6 of 7
  survive Holm (−14 to −27 pp). This is mimo's bias diluted by the two order-copying models. Read the per-model
  tables, not these.

## Compared with core-v1 C0 (`tables/per_model/compare.csv`)

Same summary function on core-v1's C0 rows (same corpus, names, prompts, seed and design). The Ranker's input orders
and the repeat draws hash the model name, so each model saw other orders and other repeat CVs, as the two core
models did. Pools in input order: share of complete equal-quality pools.

| Run, model | Evaluator gaps vs PT M | Equiv. (±2) bound | Holm survivors (21) | Screener pass | Pools in input order | Ranker identity after position | Repeat SD |
|---|---|---|---|---|---|---|---|
| core-v1 gpt-5.6-luna | +0.17 to +0.76 | 1.19 yes | 1 | 99.8% | 91% | −0.43 to +0.18, joint p 0.61 | 1.06 |
| core-v1 qwen3.8-flash | −0.30 to +0.63 | 1.67 yes | 0 | 99.3% | 100% | not testable | 2.62 |
| weak-v1 mimo-v2.6-flash | +0.33 to +1.45 | **2.18 no** | **9** (8 with ×3) | 94.7% | 20% | **−2.83 to +1.29, p 2e-18** | 0.94 |
| weak-v1 beam-free | −0.66 to +0.06 | 1.26 yes | 0 | 98.1% | 100% | not testable | 1.61 |
| weak-v1 hy4-flash-free | +0.22 to +0.35 | 0.69 yes | 0 | 100% | 100% | not testable | 0.55 |

Core-v1's C0 qwen Ranker also copied the input order in every pool (its "not testable" has the same cause). The core
FINDINGS' pooled Ranker result ("rank follows input position") holds for 4 of the 5 models. Mimo is the only one
that reads the equal-quality pools, and it uses the one difference it finds: nationality.

## Limits

- Post hoc, one run, C0 only, 3 models with no correction across them except the ×3 column. No mitigation (C1, C2) was tested on mimo, so the run cannot say whether blinding
  would remove the effect. Blinding removes nationality, so it plausibly would.
- The models are proxies; "weak" is not shown.
- Equal-quality pools are artificial: 8 identical CVs make a tie certain. The bias shows how a model breaks ties,
  which matters for real shortlists with near-equal candidates, but its size here is the size under a forced tie.
- Small repeat samples per model (10–31 Evaluator items): the noise comparisons have wide CIs.
- The audit export does not run (C0 has no judge flags); `audit/` is not written.
