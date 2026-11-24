# 12. The analysis of the frozen core run: pairing-unit model, failed answers, descriptive Screener, per-quote audit

Date: 2026-11-24 · Status: accepted (#13)

## Context

The analysis code (#9, #10) was tested on planted effects only. On the frozen core run (`data/runs/core-v1`)
four things did not work as the proposal states them:

- The proposal's mixed model, `score ~ gender × origin × cond + (gender + origin) × (role + lang) + model +
  (1|CV) + (1|name)`, gave identity CIs of about ±4 points, against about ±0.5 for the paired gaps. The
  variance of a CV's level under each language, condition and model (C2 SD 17 against C0 SD 6; the two
  models 5 points apart) stays in the residual. With such CIs no identity effect under 4 points could be found,
  so the model cannot answer RQ3.
- The Screener passed 99.7% of CVs, so its logit has no finite SEs.
- One rubric Evaluator answer (qwen3.8-flash, C2, PT-PT, accountant/weak/cv03, identity pt_m) said "No CV content
  was provided" and scored all 8 criteria 0 with empty quotes. The cached request holds the full CV, and its
  prompt tokens (1,212) match the other 7 identities (1,211–1,218): a model failure, not a pipeline bug. As pt_m is
  the reference identity, this one row moved every C2 gap by about 0.55 points and made C2 "not shown equivalent".
- The raters work on the project site (GitHub Pages), not in a spreadsheet.

## Decision

- **Primary score model:** the proposal's fixed effects with `(1|CV) + (1|CV:language:condition:model) + (1|name)`.
  The added intercept is the pairing unit of the paired gaps. Its identity × condition contrasts answer RQ3.
  The literal proposal model is kept as a robustness check (`spec = "proposal"`). The paired gaps (cluster
  bootstrap over base CVs) give the forest plots and the ±2-point equivalence check, as the proposal states.
  On 12 null seeds of the synthetic tests, neither model had a contrast that survived Holm.
- **Failed answers:** a rubric answer with every criterion 0 and every quote empty is a failed call and counts
  as missing (`hcai.analysis.failed_answers`), like an answer that fails validation (ADR 0002). The rule depends
  only on the answer's content, not on its score's effect. The frozen results table is not changed. `exclusions`
  lists the rows, the oversight table (RQ4) keeps them, and `tables/sensitivity_as_run/` gives the as-run
  results.
- **Screener:** reported by its paired gaps, pass counts and impact ratios; its logit is recorded as not fitted in
  `model_notes`, with the reason.
- **Human audit:** 50 items (25 flagged, 25 not, seed 0), rated on the site's `audit.html`, which shows no judge
  verdict and does not highlight quotes in the CV. Each rater labels every quote. The item label is derived
  (unfaithful if any quote is), so the pre-registered item-level κ is kept and a per-quote κ is added.
- **One master command:** `hcai reproduce <run>` reads only the frozen run, the corpus and the label sheets, with
  every socket refused, and writes `<run>/analysis/`.

## Consequences

The report must state the changed random effects and the one excluded answer, with both results. The equivalence
claim for C2 depends on the exclusion rule: the report gives the as-run number beside it. The Ranker's quality
measure (pairwise accuracy on the mixed pools) is 1.00 in every condition, so it cannot show a cost of a
mitigation; Spearman of the Evaluator's scores carries the quality side of RQ3.
