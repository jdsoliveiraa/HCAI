# 1. The Ranker sees the screened-in CV texts only

Date: 2026-11-09 · Status: accepted (ticket #5)

## Context

The crew is Screener → Evaluator → Ranker. The Ranker orders a pool (8 identities of one base CV, or a mixed-quality pool) and shortlists the top 3. Two choices shape what its rows measure: which pool members it sees, and whether it sees the Evaluator's scores.

## Decision

- Pool members the Screener rejected (or whose screening is missing) are not sent to the Ranker. Their Ranker rows are kept with status `rejected` (or `missing`) and no rank or input position, so every pool still has one Ranker row per member.
- The Ranker gets the CV texts only, not the Evaluator's score or rationale.
- The CVs are shown as `Candidate 1..n` in an order shuffled by a seed derived from (design seed, pool, language, condition, model). The seed and each CV's 1-based `input_position` are stored on its Ranker row, so position effects can be separated from identity effects. Reruns reproduce the order, so they hit the cache.
- A Ranker call covers a whole pool; its tokens and cost sit on the `input_position == 1` row so summing the cost column counts each call once.
- The Evaluator does not see the Screener's reason (it judges the CV itself); it only runs on CVs the Screener passed.

## Consequences

Ranker bias is the Ranker's own and is not a copy of the Evaluator's score gaps, which keeps the per-agent measurement (RQ2) clean. The trade-off is that the pipeline is less "chained" than a crew that passes scores forward; a variant that shows the Ranker the Evaluator's output could be added as a separate condition if needed.
