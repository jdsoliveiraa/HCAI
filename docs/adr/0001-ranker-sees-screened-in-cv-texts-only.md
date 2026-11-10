# 1. The Ranker sees the screened-in CV texts only

Date: 2026-11-09 · Status: accepted (ticket #5)

## Context

The crew is Screener → Evaluator → Ranker. The Ranker orders a pool (8 identities of one base CV, or a mixed-quality pool) and shortlists the top 3. Two choices shape what its rows measure: which pool members it sees, and whether it sees the Evaluator's scores.

## Decision

- Pool members the Screener rejected (or whose screening is missing) are not sent to the Ranker. Their Ranker rows are kept with status `rejected` (or `missing`) and no rank or input position, so every pool still has one Ranker row per member.
- The Ranker gets the CV texts only, not the Evaluator's score or rationale.
- The CVs are shown as `Candidate 1..n` in an order shuffled by a seed derived from (design seed, pool, language, condition, model). The seed and each CV's 1-based `input_position` are stored on its Ranker row, so position effects can be separated from identity effects. Reruns reproduce the order, so they hit the cache.
- A Ranker call covers a whole pool; its tokens and cost sit on the `input_position == 1` row so summing the cost column counts each call once.
- Two sequential crews, not one: per CV a Screener → Evaluator crew (the Evaluator is a CrewAI `ConditionalTask` gated on a pass), then per pool a Ranker crew. Pools cut across CVs (a CV can sit in an equal- and a mixed-quality pool), so one crew per pool would screen and score the same CV twice; per-CV crews keep exactly one Screener and one Evaluator row per CV.
- Ranker rows are keyed by (base CV, identity, name, language, condition, model, agent, **pool**): a CV in both an equal- and a mixed-quality pool has one Ranker row per pool.
- "Shuffled per run" means each Ranker call (pool × language × condition × model) gets its own order; a rerun of the same design reproduces it, which keeps the cache valid. Change the design `seed` for fresh orders.
- If the Screener rejects members, the Ranker ranks fewer than 8 (with 3 or fewer, everyone is in the top 3). The analysis should use pools with all members screened in for equal-chance tests, or condition on pool size.
- The Evaluator does not see the Screener's reason (it judges the CV itself); it only runs on CVs the Screener passed.

## Consequences

Ranker bias is the Ranker's own and is not a copy of the Evaluator's score gaps, which keeps the per-agent measurement (RQ2) clean. The trade-off is that the pipeline is less "chained" than a crew that passes scores forward; a variant that shows the Ranker the Evaluator's output could be added as a separate condition if needed.
