# 4. Rubric Evaluator, judge rows and the escalation rule

Date: 2026-11-10 · Status: accepted (ticket #8)

## Context

Condition C2 swaps in a rubric Evaluator that scores each job criterion separately and quotes the CV text behind each score. An LLM judge from the third model family checks the quotes, and some decisions are escalated to a human review queue (RQ4). Several choices shape what the C2 rows measure and cost.

## Decision

- **Criteria come from the job ad**, one per bullet line (mandatory, then desirable), so every identity of a base CV is scored on the same criteria. A rubric condition fails before any LLM call if a job ad lists none.
- **Each criterion is scored 0–10.** A positive score needs a non-empty quote, otherwise the answer is malformed and retried (ADR 0002). A 0 may have an empty quote, which means "no evidence".
- **We compute the total, not the model.** The total is the sum over the maximum, on 0–100 and rounded half to even. It is stored in `score`, so the C0 analysis applies unchanged, and the per-criterion scores and quotes go in `evidence` (JSON).
- **One judge call per valid (`ok`) rubric Evaluator answer**, with one verdict per criterion (`faithful`/`unfaithful` and a reason). It is stored as a `judge` row, keyed like its Evaluator row (same `model`, the pipeline model it judged), with the judge's model in `judge_model`. `run_totals` bills that row to `judge_model`, so calls, cache hits and costs per model stay right. The judge checks the quotes against the CV text the Evaluator saw. The judge's verdicts are also copied onto the Evaluator row, so the review queue is self-contained.
- **Escalation** applies to scored Evaluator rows in rubric conditions only. A row is escalated when the judge flags any justification, when the judge gives no valid verdict ("judge missing": a human has to check it), or when the score is within `[oversight] margin` of the shortlist cutoff. A CV with no Evaluator score that was not screened out is escalated as well, because no decision was made. Its reason names the agent that failed: "evaluator missing", "screener missing" (no screening decision) or "blinding failed" (C3). Screened-out CVs are not decisions here. The cutoff is `"median"` (the default: the median Evaluator score of the same role × language × condition × model, which is the analysis's primary selection rule) or a fixed score. The flag and the escalation sit on the Evaluator row, and escalated rows also go to `review_queue.csv` and are counted under `oversight` in `run_summary.json`.
- **The judge defaults to `[generator] model`**, the third family. `[oversight] judge_model` overrides it, and a run refuses a judge that is one of the audited pipeline models.
- **C3 = C1 + C2.** It is in both `BLINDING_CONDITIONS` (`hcai.blinding`) and `RUBRIC_CONDITIONS` (`hcai.oversight`): the Blinder runs first (ADR 0003), and the rubric Evaluator, and then the judge, get the redacted blinded text. A quote of that text is faithful, and a quote of identity text that blinding removed is not. A C3 CV whose blinding failed has no Evaluator decision, so it is escalated as "blinding failed".
- The stability sample re-runs the Screener and Evaluator, not the judge.
- **Shared requests are shared answers.** C2 changes only the Evaluator, so its Screener requests are byte-identical to C0's, and so is a Ranker request whenever the pool's shuffled order happens to match. C3's Blinder and Screener requests are likewise identical to C1's. The cache key does not include the condition, so within one run these are cache hits (`cache_hits` > 0 on a fresh cache), and the C2-vs-C0 Screener gap is zero by construction. That is intended: the conditions differ only in the agent they change.

## Consequences

C0 and C2 rows share the `score` column, so the mitigation effect shows up in the identity × condition terms without special cases. Judge rows add one call per valid rubric answer in C2 and C3, at the judge model's price. A median cutoff depends on the other scores in its group, so adding CVs to a design can change which rows are near the cutoff. In tiny groups (a pilot) it escalates more than usual: a group with one scored row always escalates that row, so pilot escalation counts are not representative.
