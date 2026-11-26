# Findings: core run v1

Made by `uv run hcai reproduce data/runs/core-v1` (offline, seed 0, 2,000 bootstrap resamples, 200 refits of the
mixed model, 5,000 permutations). Every number below is in `tables/` or `headline.json`. Gaps are against the
reference identity PT M (Portuguese man), in score points (0–100) or percentage points (pp). "Survives Holm"
means a Holm-adjusted p < 0.05 within its family.

## Data

- 10,076 rows: 10,056 ok, 20 rejected (10 CVs screened out, each with its Evaluator and Ranker rows), 0 missing.
- 1 failed answer counted as missing (ADR 0012, `tables/exclusions.csv`): qwen3.8-flash, C2, PT-PT,
  accountant/weak/cv03, PT M. It said "No CV content was provided" and scored all criteria 0. The request held
  the full CV (1,212 prompt tokens, like the other 7 identities): a model failure, not a pipeline bug.
- Ranker equal-chance tests use complete equal-quality pools only: 104, 106 and 104 of 108 (C0, C1, C2).

## RQ1: are there gaps by gender and origin?

- **No gap at all as large as the 2-point margin.** Evaluator gaps: C0 +0.06 to +0.49, C1 +0.81 to +1.20,
  C2 +0.09 to +0.22 points. Every condition passes the equivalence check (largest |90% CI| bound: C0 1.02, C1 1.76,
  C2 0.53).
- 1 of 63 paired gaps survives Holm: C1 UA F +1.20 [0.59, 1.86] (see *Looks wrong*). Within models, roles and
  languages (`gaps_by`), 2 of the stratum gaps survive: C1 BR F with gpt-5.6-luna +0.72, and C1 UA F in PT-PT +1.65.
  All are under 2 points and all are in C1. Within a single stratum the check has fewer pairs, and some bounds
  pass 2 points: C1 in each role, each language and qwen3.8-flash (2.2–2.6), and C0 for software engineer (2.4).
  These cells are not shown equivalent. The pre-registered check is the pooled one.
- Impact ratios: no ratio below 0.8 survives Holm in any agent. The Screener ratios are ≥ 0.98, and the Evaluator
  ratios (median rule) are ≥ 0.79, with 1 of 24 below 0.8. The Ranker's top-3 ratios fall below 0.8 for 3, 3 and 7 of
  8 identities (C0, C1, C2; lowest 0.53), but none survives Holm (lowest adjusted p 0.058). With about 100 pools per
  condition they are noise (see RQ2).
- Ranker mean rank by identity: none differs from 4.5 (lowest Holm p 0.68). Top-3 gaps range from −15 to +13 pp,
  and none survives Holm (lowest 0.31).
- Primary mixed model (pairing unit): identity gaps C0 +0.05 to +0.49 (none survives Holm), C2 +0.04 to +0.54
  (none), C1 +0.82 to +1.21 (UA F, UA M and AO F survive). The proposal's literal model finds nothing (SE 0.76–1.04
  against 0.31–0.40).

## RQ2: at which agent?

- **Screener:** passes 99.7% of CVs (10 rejections of 2,592). Its gaps are 0 to −1.9 pp, and its logit cannot be
  fitted (`model_notes`).
- **Evaluator:** see RQ1. The gaps are under 1.3 points everywhere.
- **Ranker:** in equal-quality pools, rank follows the input position, not the identity. In C0 the CV shown
  first gets mean rank 1.08 and the one shown last 7.77. In C2 the values are 1.08 and 7.94. In C1 the effect is
  weaker (2.55 and 5.95). 17 of 24 position tests survive Holm; the middle positions (4–5, and 3 in C1) do not,
  as expected when rank follows position. Because the order is shuffled independently of
  identity, this adds noise to the identity results but no bias. The Ranker mostly keeps the order it is given.

## RQ3: do the mitigations help, and at what cost?

- No baseline bias to reduce: the success criterion "largest gap halved" does not apply. The null result is
  reported through the equivalence check, as pre-registered.
- Identity × condition contrasts (primary model) do not survive Holm. C1 vs C0 is +0.42 to +0.81, and C2 vs C0 is
  −0.33 to +0.17.
- **Quality:** Spearman of Evaluator score with the true quality level: C0 0.867 [0.827, 0.905], C1 0.857 (Δ −0.010
  [−0.027, +0.007]), C2 0.961 (Δ **+0.093** [+0.057, +0.135]). The rubric improves ordering by quality for both
  models (gpt +0.037, qwen +0.058). Blinding costs nothing measurable.
- Pairwise accuracy on mixed pools is 1.00 in C0 and C2 and 0.998 in C1: at the ceiling, so it cannot show a cost.
  Every condition is within the 5 pp tolerance.

## RQ4: oversight

- The judge flagged 28 of 860 C2 justifications (3.3%; gpt 14, qwen 14). 306 decisions (35.6%) were escalated:
  285 near the cutoff, 28 for a flagged quote (7 for both). This cost about USD 12.6 of allowance (USD 0.015 per
  decision).
- Most of the cost of oversight comes from the near-cutoff rule, not the judge. C2 scores cluster (many 75s and
  100s), so many sit within 5 points of the median.
- Human audit: 50 items are exported (25 of the 27 flagged valid answers, 25 unflagged; the failed answer is not
  drawn) and are on the site's audit page. One rater (`labels_joao_oliveira`), so there is no rater-to-rater κ (a limitation).
  The key and `results.csv` are in the repository, so the raters must not open them.
- Judge vs. the human rater, per item: κ 0.08 (95% CI −0.07 to 0.24), agreement 54% (27 of 50). Of the 25 items the
  judge flagged, the rater found 3 unfaithful; of the 25 it did not flag, 1. The judge flags far more than the rater.
- Per quote (400): κ 0.33 (95% CI −0.00 to 0.61), agreement 92%. The judge marked 40 quotes unfaithful, the rater 10;
  9 of the rater's 10 are among the judge's 40. The judge catches what the rater sees, with many false alarms.
- Rule caveat: the rater labelled a quote faithful when the full CV supports its score, even when the quote itself
  does not (comment on J042). The judge and the protocol need the quote itself to support the score. Part of the low
  κ comes from this looser rule, so the κ values are a lower bound on agreement under one shared rule.

## Stability (5% repeat sample)

- The Evaluator gives the same score again in only 52% (gpt) and 55% (qwen) of repeats, with mean |Δ| 0.79 and
  1.62 points (weighted by pairs). The Screener (99–100%) and the Ranker (gpt 100%, qwen 91%) are more stable. The run-to-run noise of one score is
  about as large as the identity gaps. Read the gaps as averages over many pairs.

## Looks wrong, or could not run

- **C1 gaps are a reference effect, not a pattern.** In C1 every identity, PT F included, scores 0.8–1.2 above
  PT M. Each variant is blinded in its own call, and the blinded texts of PT M are no different in length or
  redactions (`proxy_leaks`), so the crew cannot see who PT M is. PT M's blinded versions happened to score lower.
  The 7 gaps share that one reference, so they move together. This explains the C1 survivors above. In the
  reference-free contrast (post hoc, see *Extra analyses*) PT M is the only C1 identity away from the mean
  (−0.81, not surviving Holm); the others are within ±0.39.
- **As run (sensitivity, `tables/sensitivity_as_run/`):** with the failed answer kept as a score of 0, the C2 gaps
  are 0.51–0.92 points larger, and C2 is not shown equivalent (largest |90% CI| bound 3.07). C0 and C1 do not change.
- In the same qwen C2 cell, AO F scored 100 on a weak CV where the others got 55–62. It is a real answer with
  quotes, so it stays.
- The Screener's logistic model could not run (separation). The logistic model for top 3 runs, but its result is
  null.
- Pairwise accuracy is at the ceiling, so it cannot measure the quality cost of a mitigation.
- Blinding leaks: qwen kept the header location in about half of the CVs (EN 50%, PT-PT 53%, against gpt 99% and
  95%). Universities and languages survive in nearly all CVs. Redaction removed what the Blinder left
  (PT-PT about 22–24 tokens per CV).
- The Ranker shows a strong position effect (primacy) in every condition.
- The audit export renders the CVs from `data/corpus/v1` and does not check them against the run. The corpus is
  frozen by tag `corpus-v1` and must not change.

## Extra analyses (not pre-registered)

Post hoc: added after the results above were known, not in the proposal (`hcai.extra`, `tables/extra_*.csv`,
`headline.json` → `extra_not_preregistered`). Same exclusion rule (ADR 0012), seed 0, 2,000 bootstrap
resamples over base CVs, Holm within each family. Read them as checks of the null result, not as new tests.
CIs below are not Holm-adjusted.

1. **Each identity against the mean of all 8** (`extra_gaps_reference_free`, complete pairing units only:
   103–106 per condition). Evaluator deviations: C0 −0.31 to +0.30, C1 −0.81 to +0.38, C2 −0.14 to +0.09 points.
   0 of 72 survive Holm (all agents). The lowest is C1 PT M −0.81 (95% CI −1.31 to −0.36, Holm p 0.085); no
   other C1 identity is beyond ±0.39. The 90% CIs of all deviations lie within ±2 (largest bound 0.71, 1.23,
   0.30). This check is looser than the pre-registered one: a deviation of ±2 allows a gap of up to 4 between
   two identities, and a deviation has less noise than a paired gap. **Verdict:** the C1 "gaps" come from PT M
   scoring a little lower, not from 7 identities scoring higher. No identity differs from the average after Holm.
2. **Subsets** (`extra_gaps_subsets`, `extra_equivalence_subsets`). Borderline base CVs (the corpus label,
   6 CVs: 1 average + 1 weak per role): 22–24 Evaluator pairs per identity and condition, all three agents.
   Near the cutoff (Evaluator pairing units whose mean score is within 5 points of their role × language ×
   model × condition median, the escalation rule's cutoff): 40, 42 and 34 complete units from 12, 14 and 9 base
   CVs (C0, C1, C2), 35–44 pairs against PT M. 0 of 180 gaps survive Holm. The largest borderline gaps are in
   C1 against PT M (+1.4 to +3.5 points, lowest Holm p 0.14), the same PT M pattern as above. Equivalence near
   the cutoff passes in every cell except C1 against PT M (bound 2.47). On the borderline CVs the bounds are
   1.2–5.0. With only 6 clusters, a percentile bootstrap gives CIs that are too narrow, so the borderline
   equivalence result is uninformative in every condition, including C2's pass (1.86). **Verdict:** no gap
   survives where a small bias would matter most. The borderline subset is too small to show equivalence
   (absence of evidence, not evidence of absence).
3. **Ranker after input position** (`extra_rank_adjusted`, `_joint`; OLS of rank and top 3 on position +
   identity, SEs clustered by pool, t and F tests with pools − 1 df, 104–106 pools). Identity effects against
   the identity mean, on rank: −0.32 to +0.27 places after position (−0.45 to +0.48 before); on top 3: −5.3 to
   +4.5 pp (−10.6 to +13.5 before). They shrink by half or more in C0 and C2, and by 17–44% in C1, where the
   position effect is weaker. 0 of 48 survive Holm (lowest 0.62). Joint test of all identity terms per
   condition: p 0.40–0.75. Position is far stronger (Wald χ² 147 in C1, over 41,000 in C0 and C2). **Verdict:**
   after position, no identity effect on the Ranker. Most of its raw identity spread comes from the input order.
4. **Gaps against run-to-run noise** (`extra_repeat_noise`, `extra_noise_vs_gap`). The repeat SD of one Evaluator
   score (the two models weighted equally): C0 1.99 (95% CI 1.43–2.56), C1 1.82 (1.22–2.31), C2 0.78
   (0.47–1.06) points (qwen 2.4–2.6 in C0/C1, gpt 0.9–1.1). One paired comparison against PT M has a noise SD of
   2.82, 2.57 and 1.10: each gap is at most 0.17, 0.47 and 0.20 of it, so no single decision can show a gap.
   Over about 100 pairs, noise alone gives the mean gap an SE of 0.27, 0.25 and 0.11. Against that SE, 31 of 45
   gaps are inside noise (|z| < 1.96). Outside: all 7 C1 gaps against PT M (z 3.2–4.9) and C1 PT M and EE F
   against the mean (z −4.9, +2.3); in C2, 4 gaps against PT M (+0.21 to +0.22, z 2.02–2.03) and PT M against
   the mean (z −1.98). This test leaves out the CV-to-CV spread, so it is the most lenient one. "Outside"
   means more than Evaluator call noise, not bias. In C1, the repeats do not re-run the Blinder, but each
   identity's text is blinded in its own call. A systematic C1 difference beyond call noise is consistent with
   PT M's blinded texts scoring lower, and the noise variance of one comparison is only 43–65% of the observed
   pair-level variance there (C0 60–109%, C2 33–118%; above 100% means σ is uncertain, not a share).
   **Verdict:** apart from C1, the gaps are of the size measurement noise gives. The C1 PT M difference is real
   for this run's blinded texts. It is under 1 point, does not survive Holm with the CV spread, and is not tied
   to gender or origin, because PT M is the only identity it touches.

**Assessment for the main question (is there demographic bias?).** All four checks agree with the
pre-registered result: no demographic bias of practical size in any agent or condition. The one recurring
signal is C1 PT M, under 1 point. It does not survive Holm against the mean, and no other Portuguese or male
identity shows it. The limits are power on the borderline CVs (6), the lenient noise test, and the post-hoc
status of these analyses.
