# 3. Rubric Evaluator, judge rows and the escalation rule

Date: 2026-11-10 · Status: accepted (ticket #8)

## Context

Condition C2 swaps in a rubric Evaluator that scores each job criterion separately and quotes the CV text behind each score. An LLM judge from the third model family checks the quotes, and some decisions are escalated to a human review queue (RQ4). Several choices shape what the C2 rows measure and cost.

## Decision

- **Criteria come from the job ad**, one per bullet line (mandatory, then desirable), so every identity of a base CV is scored on the same criteria. A rubric condition fails before any LLM call if a job ad lists none.
- **Each criterion is scored 0–10.** A positive score needs a non-empty quote, otherwise the answer is malformed and retried (ADR 0002). A 0 may have an empty quote, which means "no evidence".
- **We compute the total, not the model.** The total is the sum over the maximum, on 0–100 and rounded half to even. It is stored in `score`, so the C0 analysis applies unchanged, and the per-criterion scores and quotes go in `evidence` (JSON).
- **One judge call per scored Evaluator decision**, with one verdict per criterion (`faithful`/`unfaithful` and a reason). It is stored as a `judge` row, keyed like its Evaluator row (same `model`, the pipeline model it judged), with the judge's model in `judge_model`. `run_totals` bills that row to `judge_model`, so calls, cache hits and costs per model stay right. The judge sees the CV text the Evaluator saw.
- **Escalation** applies to scored Evaluator rows in rubric conditions only. A row is escalated when the judge flags any justification, when the judge gives no valid verdict ("judge missing": a human has to check it), or when the score is within `[oversight] margin` of the shortlist cutoff. The cutoff is `"median"` (the default: the median Evaluator score of the same role × language × condition × model, which is the analysis's primary selection rule) or a fixed score. The flag and the escalation sit on the Evaluator row, and escalated rows also go to `review_queue.csv` and are counted under `oversight` in `run_summary.json`.
- **The judge defaults to `[generator] model`**, the third family. `[oversight] judge_model` overrides it.
- **C3 is listed in `[design] conditions` like any other condition.** It uses the rubric here, and the blinding half comes from C1 (#7).
- The stability sample re-runs the Screener and Evaluator, not the judge.

## Consequences

C0 and C2 rows share the `score` column, so the mitigation effect shows up in the identity × condition terms without special cases. Judge rows add one call per screened-in CV in C2, at the judge model's price. A median cutoff depends on the other scores in its group, so adding CVs to a design can change which rows are near the cutoff.
